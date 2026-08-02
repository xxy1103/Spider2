"""Standalone entrypoint for metadata-only Schema Router evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

from agent.schema_router import (
    SchemaRouterAgent,
    SchemaRouterCatalog,
    SchemaRouterTools,
    build_router_user_prompt,
    load_external_knowledge,
)
from agent.schema_router_config import (
    SchemaRouterConfig,
    SchemaRouterConfigError,
    bind_schema_router_task_set,
    load_schema_router_config,
    prepare_schema_router_run,
)
from agent.schema_router_evaluator import (
    aggregate_scores,
    extract_official_sql_labels,
    labels_sha256,
    load_tasks,
    make_task_set,
    render_report,
    score_rollout,
    threshold_status,
)
from agent.progress import TaskProgressReporter


logger = logging.getLogger(__name__)
_WRITE_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the standalone metadata-only Schema Router evaluator"
    )
    parser.add_argument("--config", required=True, help="Path to one YAML config")
    return parser.parse_args()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _configure_logging(config: SchemaRouterConfig) -> Path:
    log_path = config.experiment_dir / "run.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return log_path


def _routing_path(
    config: SchemaRouterConfig, instance_id: str, rollout_idx: int
) -> Path:
    return (
        config.experiment_dir
        / "routing"
        / instance_id
        / f"rollout-{rollout_idx}.json"
    )


def _load_completed(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        isinstance(value, dict)
        and value.get("completed") is True
        and isinstance(value.get("selection"), dict)
        and value["selection"].get("valid") is True
    ):
        return value
    return None


def _run_one(
    *,
    config: SchemaRouterConfig,
    agent: SchemaRouterAgent,
    catalog: SchemaRouterCatalog,
    task: dict[str, Any],
    rollout_idx: int,
) -> dict[str, Any]:
    path = _routing_path(config, task["instance_id"], rollout_idx)
    existing = _load_completed(path)
    if existing is not None:
        logger.info(
            "Resume: skipping completed %s rollout %s",
            task["instance_id"],
            rollout_idx,
        )
        return existing
    external = load_external_knowledge(task, config.paths["documents"])
    tools = SchemaRouterTools(
        catalog,
        database_id=task["db_id"],
        instance_id=task["instance_id"],
        sample_rows=config.raw["schema_router"]["sample_rows"],
        max_sample_chars=config.raw["schema_router"]["max_sample_chars"],
    )
    prompt = build_router_user_prompt(
        task,
        external_knowledge=external,
        family_overview=catalog.family_overview(task["db_id"]),
    )
    logger.info("Routing %s rollout %s", task["instance_id"], rollout_idx)
    result = agent.run(
        item=task,
        rollout_idx=rollout_idx,
        tools=tools,
        user_prompt=prompt,
    )
    with _WRITE_LOCK:
        _write_json_atomic(path, result)
    return result


def _run_one_with_progress(
    *, reporter: TaskProgressReporter, **kwargs: Any
) -> dict[str, Any]:
    reporter.task_started()
    try:
        result = _run_one(**kwargs)
    except Exception:
        reporter.task_finished(success=False)
        raise
    artifact = _routing_path(
        kwargs["config"],
        kwargs["task"]["instance_id"],
        kwargs["rollout_idx"],
    )
    reporter.task_finished(success=_load_completed(artifact) is not None)
    return result


def _partition_jobs_for_resume(
    config: SchemaRouterConfig,
    jobs: list[tuple[dict[str, Any], int]],
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], int]]]:
    completed_results = []
    pending_jobs = []
    for task, rollout_idx in jobs:
        existing = _load_completed(
            _routing_path(config, task["instance_id"], rollout_idx)
        )
        if existing is not None:
            completed_results.append(existing)
        else:
            pending_jobs.append((task, rollout_idx))
    return completed_results, pending_jobs


def _build_jobs(
    tasks: dict[str, dict[str, Any]],
    instance_ids: list[str],
    rollout_count: int,
) -> list[tuple[dict[str, Any], int]]:
    return [
        (tasks[instance_id], rollout_idx)
        for instance_id in instance_ids
        for rollout_idx in range(rollout_count)
    ]


def _validate_or_write_task_set(
    *,
    config: SchemaRouterConfig,
    task_set: dict[str, Any],
) -> None:
    path = config.experiment_dir / "schema-router-task-set.json"
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Invalid existing task-set manifest: {path}") from exc
        if existing != task_set:
            raise RuntimeError(
                "Existing task-set manifest differs from current labels/catalog; "
                "use a new experiment name."
            )
        return
    _write_json_atomic(path, task_set)


def _evaluation_exit_code(threshold: dict[str, Any]) -> int:
    return 0 if threshold.get("passed") is True else 2


def execute(config: SchemaRouterConfig) -> int:
    tasks = load_tasks(config.paths["input_file"])
    sql_ids = {path.stem for path in config.paths["official_sql_dir"].glob("*.sql")}
    if not sql_ids:
        raise RuntimeError("No official SQL files were found")
    missing_tasks = sorted(sql_ids - set(tasks))
    if missing_tasks:
        raise RuntimeError(
            f"Official SQL task ids are missing from input: {missing_tasks}"
        )
    database_ids = {tasks[instance_id]["db_id"] for instance_id in sql_ids}
    catalog = SchemaRouterCatalog(config.paths["databases"], database_ids)
    labels = extract_official_sql_labels(
        sql_dir=config.paths["official_sql_dir"],
        tasks=tasks,
        catalog=catalog,
    )
    evaluation = config.raw["evaluation"]
    task_set = make_task_set(labels=labels, catalog=catalog)
    config = bind_schema_router_task_set(config, task_set)
    config = prepare_schema_router_run(config)
    log_path = _configure_logging(config)
    _validate_or_write_task_set(config=config, task_set=task_set)
    selected_ids = task_set["instance_ids"]
    rollout_count = evaluation["rollouts"]
    label_map = {label["instance_id"]: label for label in labels}
    labels_path = config.experiment_dir / "schema-router-labels.jsonl"
    labels_text = "".join(
        json.dumps(label, ensure_ascii=False) + "\n"
        for label in (label_map[instance_id] for instance_id in selected_ids)
    )
    _write_text_atomic(labels_path, labels_text)
    logger.info(
        "Full label preflight passed: %s official SQL files, tasks=%s",
        len(labels),
        len(selected_ids),
    )

    model = config.raw["schema_router"]["model"]
    client = OpenAI(
        base_url=config.secrets["model_api"]["base_url"],
        api_key=config.secrets["model_api"]["api_key"],
        timeout=model["request_timeout_seconds"],
    )
    agent = SchemaRouterAgent(
        model_client=client,
        model_config=model,
        system_prompt=config.paths["router_prompt"].read_text(
            encoding="utf-8"
        ).strip(),
        max_rounds=config.raw["schema_router"]["max_rounds"],
        max_tool_calls=config.raw["schema_router"]["max_tool_calls"],
    )

    jobs = _build_jobs(tasks, selected_ids, rollout_count)
    results, pending_jobs = _partition_jobs_for_resume(config, jobs)
    logger.info(
        "Router scheduling: total=%s, skipped=%s, pending=%s",
        len(jobs),
        len(results),
        len(pending_jobs),
    )

    with TaskProgressReporter(
        "Schema Router",
        len(jobs),
        initial_successful=len(results),
        skipped=len(results),
    ) as reporter:
        with ThreadPoolExecutor(
            max_workers=config.raw["schema_router"]["num_threads"]
        ) as executor:
            futures = {
                executor.submit(
                    _run_one_with_progress,
                    reporter=reporter,
                    config=config,
                    agent=agent,
                    catalog=catalog,
                    task=task,
                    rollout_idx=rollout_idx,
                ): (task["instance_id"], rollout_idx)
                for task, rollout_idx in pending_jobs
            }
            for future in as_completed(futures):
                instance_id, rollout_idx = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "Router failed for %s rollout %s",
                        instance_id,
                        rollout_idx,
                    )
                    failed = {
                        "instance_id": instance_id,
                        "rollout_idx": rollout_idx,
                        "completed": False,
                        "error": str(exc).replace(
                            config.secrets["model_api"]["api_key"],
                            "***REDACTED***",
                        ),
                        "performance": {},
                    }
                    _write_json_atomic(
                        _routing_path(config, instance_id, rollout_idx), failed
                    )
                    results.append(failed)

    result_map: dict[tuple[str, int], dict[str, Any]] = {
        (result["instance_id"], int(result["rollout_idx"])): result
        for result in results
    }
    scores = []
    for instance_id in selected_ids:
        for rollout_idx in range(rollout_count):
            scores.append(
                score_rollout(
                    result=result_map.get((instance_id, rollout_idx)),
                    label=label_map[instance_id],
                    catalog=catalog,
                )
            )
    summary = aggregate_scores(
        scores=scores,
        expected_instance_ids=selected_ids,
    )
    summary.update(
        {
            "evaluation_mode": "all",
            "official_sql_files": len(labels),
            "total_tasks": len(selected_ids),
            "rollouts_per_task": rollout_count,
            "expected_rollouts": len(selected_ids) * rollout_count,
            "labels_sha256": labels_sha256(labels),
            "catalog_tables": len(catalog.tables),
            "catalog_families": len(catalog.families),
        }
    )
    threshold = threshold_status(
        summary=summary,
        thresholds=evaluation["thresholds"],
    )
    summary["threshold"] = threshold
    _write_json_atomic(
        config.experiment_dir / "schema-router-summary.json", summary
    )
    _write_text_atomic(
        config.experiment_dir / "schema-router-report.md",
        render_report(
            summary=summary,
            threshold=threshold,
        ),
    )
    print(f"Schema Router 评测完成，详情见 {log_path}")
    return _evaluation_exit_code(threshold)


def main() -> int:
    try:
        args = parse_args()
        config = load_schema_router_config(args.config)
        return execute(config)
    except (SchemaRouterConfigError, RuntimeError, ValueError) as exc:
        print(f"Schema Router 启动失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
