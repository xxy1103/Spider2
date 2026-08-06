"""Run and score a paired DeepSeek V4 Flash thinking-level experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

from agent.auto_evaluator import (
    collect_conversation_stats,
    extract_sql_answers,
    run_evaluation,
)
from agent.model_request import build_model_request_kwargs
from agent.schema_router import SchemaRouterCatalog
from agent.schema_router_evaluator import (
    aggregate_scores,
    extract_official_sql_labels,
    load_tasks,
    score_rollout,
)
from config import ConfigError, LoadedConfig, load_config
from run import check_dependencies, check_snowflake, execute


LEVELS = ("none", "low", "high", "max")
COMPARISON_PROTOCOL_VERSION = 1
_RUN_DIR_PATTERN = re.compile(r"\d{8}-\d{6}(?:-\d{2})?")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a paired DeepSeek V4 Flash thinking-level comparison"
    )
    parser.add_argument("--config", required=True, help="Comparison YAML path")
    return parser.parse_args()


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"Unable to read comparison config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid comparison YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError("Comparison configuration must be a mapping")
    return value


def _repo_path(repo_root: Path, value: str, location: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{location} must be a non-empty repository-relative path")
    path = (repo_root / value).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise ConfigError(f"{location} must remain inside the repository") from exc
    return path


def load_comparison_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).expanduser().resolve()
    raw = _read_yaml(config_path)
    allowed = {
        "experiment",
        "base_config",
        "official_sql_dir",
        "evaluation_standard",
        "conditions",
        "sampling",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"Unknown comparison fields: {', '.join(unknown)}")
    missing = sorted(allowed - set(raw))
    if missing:
        raise ConfigError(f"Missing comparison fields: {', '.join(missing)}")

    repo_root = Path(__file__).resolve().parents[2]
    experiment = raw["experiment"]
    sampling = raw["sampling"]
    if not isinstance(experiment, dict) or set(experiment) != {
        "name",
        "results_root",
        "resume",
    }:
        raise ConfigError(
            "comparison experiment requires exactly name, results_root, and resume"
        )
    if not isinstance(experiment["name"], str) or not experiment["name"].strip():
        raise ConfigError("experiment.name must be a non-empty string")
    if not isinstance(experiment["resume"], bool):
        raise ConfigError("experiment.resume must be true or false")
    _repo_path(repo_root, experiment["results_root"], "experiment.results_root")
    if not isinstance(sampling, dict) or set(sampling) != {"sample_size", "seed"}:
        raise ConfigError("sampling requires exactly sample_size and seed")
    if sampling["sample_size"] != 50:
        raise ConfigError("sampling.sample_size must be exactly 50 for this experiment")
    if not isinstance(sampling["seed"], int) or isinstance(sampling["seed"], bool):
        raise ConfigError("sampling.seed must be an integer")
    conditions = raw["conditions"]
    if not isinstance(conditions, list) or tuple(conditions) != LEVELS:
        raise ConfigError("conditions must be exactly: none, low, high, max")

    for key in ("base_config", "official_sql_dir", "evaluation_standard"):
        resolved = _repo_path(repo_root, raw[key], key)
        if key.endswith("dir"):
            if not resolved.is_dir():
                raise ConfigError(f"{key} is not a directory: {resolved}")
        elif not resolved.is_file():
            raise ConfigError(f"{key} is not a file: {resolved}")
        raw[key] = str(resolved)
    raw["config_path"] = str(config_path)
    raw["repo_root"] = str(repo_root)
    return config_path, raw


def _load_jsonl_ids(path: Path) -> set[str]:
    values: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            instance_id = value.get("instance_id") if isinstance(value, dict) else None
            if not isinstance(instance_id, str) or not instance_id:
                raise ConfigError(f"Invalid instance_id at {path}:{line_number}")
            values.add(instance_id)
    return values


def build_fixed_task_set(raw: dict[str, Any]) -> dict[str, Any]:
    repo_root = Path(raw["repo_root"])
    base_raw = _read_yaml(Path(raw["base_config"]))
    input_path = _repo_path(
        repo_root, base_raw["paths"]["input_file"], "base paths.input_file"
    )
    input_ids = _load_jsonl_ids(input_path)
    evaluation_ids = _load_jsonl_ids(Path(raw["evaluation_standard"]))
    sql_paths = sorted(Path(raw["official_sql_dir"]).glob("*.sql"))
    official_ids = [path.stem for path in sql_paths]
    if len(official_ids) != 120 or len(set(official_ids)) != 120:
        raise ConfigError(
            f"Expected exactly 120 unique official SQL files, found {len(official_ids)}"
        )
    missing_input = sorted(set(official_ids) - input_ids)
    missing_eval = sorted(set(official_ids) - evaluation_ids)
    if missing_input or missing_eval:
        raise ConfigError(
            f"Official SQL coverage mismatch; missing_input={missing_input}, "
            f"missing_evaluation={missing_eval}"
        )
    seed = raw["sampling"]["seed"]
    sampled = random.Random(seed).sample(official_ids, raw["sampling"]["sample_size"])
    sql_hashes = {
        path.stem: hashlib.sha256(path.read_bytes()).hexdigest() for path in sql_paths
    }
    return {
        "source_task_count": len(official_ids),
        "source_instance_ids": official_ids,
        "sample_size": len(sampled),
        "seed": seed,
        "sampled_instance_ids": sampled,
        "sampled_instance_ids_sorted": sorted(sampled),
        "official_sql_sha256": sql_hashes,
    }


def _comparison_fingerprint(raw: dict[str, Any], task_set: dict[str, Any]) -> str:
    implementation_files = (
        Path(__file__),
        Path(__file__).parent / "agent" / "model_request.py",
        Path(__file__).parent / "agent" / "schema_router_evaluator.py",
        Path(__file__).parent / "agent" / "auto_evaluator.py",
    )
    payload = {
        "protocol_version": COMPARISON_PROTOCOL_VERSION,
        "comparison": {
            key: value
            for key, value in raw.items()
            if key not in {"config_path", "repo_root"}
        },
        "base_config_sha256": hashlib.sha256(
            Path(raw["base_config"]).read_bytes()
        ).hexdigest(),
        "task_set": task_set,
        "implementation_sha256": {
            path.relative_to(Path(raw["repo_root"])).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in implementation_files
        },
        "python": sys.version.split()[0],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _select_parent_dir(raw: dict[str, Any], fingerprint: str) -> Path:
    repo_root = Path(raw["repo_root"])
    experiment = raw["experiment"]
    group = (
        repo_root / experiment["results_root"] / experiment["name"]
    ).resolve()
    if experiment["resume"] and group.is_dir():
        for candidate in sorted(group.iterdir(), reverse=True):
            if not candidate.is_dir() or not _RUN_DIR_PATTERN.fullmatch(candidate.name):
                continue
            if (candidate / "comparison-summary.json").is_file():
                continue
            manifest_path = candidate / "comparison-manifest.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("fingerprint") == fingerprint:
                return candidate
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = group / stamp
    suffix = 1
    while candidate.exists():
        candidate = group / f"{stamp}-{suffix:02d}"
        suffix += 1
    return candidate


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _build_child_raw(
    *,
    base_raw: dict[str, Any],
    raw: dict[str, Any],
    parent_dir: Path,
    task_set: dict[str, Any],
    level: str,
) -> dict[str, Any]:
    child = copy.deepcopy(base_raw)
    repo_root = Path(raw["repo_root"])
    relative_parent = parent_dir.relative_to(repo_root).as_posix()
    child["experiment"] = {
        "name": f"condition-{level}",
        "results_root": f"{relative_parent}/condition-runs",
        "resume": True,
    }
    child["tasks"] = {
        "instance_ids": task_set["sampled_instance_ids"],
        "index_ranges": [],
        "databases": [],
        "sample_size": None,
        "seed": raw["sampling"]["seed"],
        "order": "seeded_shuffle",
    }
    for model in (child["model"], child["schema_router"]["model"]):
        model["name"] = "deepseek-v4-flash"
        model["provider"] = "deepseek"
        model["thinking_level"] = level
        model["temperature"] = 0
    child["agent"]["rollout_number"] = 1
    child["schema_router"]["enabled"] = True
    child["schema_router"]["integration"]["mode"] = "strict"
    child["schema_router"]["integration"]["failure_policy"] = "fail_task"
    child["tools"]["sql"]["mode"] = "live"
    child["preflight"] = {"check_model": False, "check_snowflake": False}
    child["auto_evaluate"] = {
        "enabled": False,
        "timeout": child.get("auto_evaluate", {}).get("timeout", 300),
        "max_workers": child.get("auto_evaluate", {}).get("max_workers", 4),
    }
    return child


def _write_child_config(parent_dir: Path, level: str, child: dict[str, Any]) -> Path:
    path = parent_dir / "generated-configs" / f"{level}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(child, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return path


def _probe_model(config: LoadedConfig, level: str) -> dict[str, Any]:
    model = config.raw["model"]
    response = OpenAI(
        base_url=config.secrets["model_api"]["base_url"],
        api_key=config.secrets["model_api"]["api_key"],
        timeout=model["request_timeout_seconds"],
    ).chat.completions.create(
        model=model["name"],
        messages=[{"role": "user", "content": "Reply with exactly: ok"}],
        temperature=0,
        max_tokens=64,
        n=1,
        **build_model_request_kwargs(model),
    )
    returned_model = getattr(response, "model", None)
    if returned_model != "deepseek-v4-flash":
        raise RuntimeError(
            f"Model fallback detected for {level}: returned {returned_model!r}"
        )
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    return {
        "requested_model": model["name"],
        "returned_model": returned_model,
        "thinking_level": level,
        "request_kwargs": build_model_request_kwargs(model),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "reasoning_tokens": int(getattr(details, "reasoning_tokens", 0) or 0),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def _find_completed_child(config: LoadedConfig) -> Path | None:
    group = config.experiment_group_dir
    if not group.is_dir():
        return None
    for candidate in sorted(group.iterdir(), reverse=True):
        manifest_path = candidate / "run-manifest.json"
        if not candidate.is_dir() or not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("fingerprint") == config.fingerprint
            and (candidate / "run-summary.json").is_file()
        ):
            return candidate
    return None


def _score_integrated_router(
    config: LoadedConfig, official_sql_dir: Path
) -> dict[str, Any]:
    tasks = load_tasks(config.paths["input_file"])
    gold_ids = {path.stem for path in official_sql_dir.glob("*.sql")}
    gold_db_ids = {tasks[instance_id]["db_id"] for instance_id in gold_ids}
    catalog = SchemaRouterCatalog(config.paths["databases"], gold_db_ids)
    labels = extract_official_sql_labels(
        sql_dir=official_sql_dir,
        tasks=tasks,
        catalog=catalog,
    )
    labels_by_id = {label["instance_id"]: label for label in labels}
    selected_ids = [item["instance_id"] for item in config.selected_items]
    scores = []
    for instance_id in selected_ids:
        result_path = (
            config.experiment_dir
            / "routing"
            / instance_id
            / "rollout-0.json"
        )
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            result = None
        scores.append(
            score_rollout(
                result=result,
                label=labels_by_id[instance_id],
                catalog=catalog,
            )
        )
    return aggregate_scores(scores=scores, expected_instance_ids=selected_ids)


def _build_final_scores(
    selected_ids: list[str], eval_results: list[dict[str, Any]], submission_dir: Path
) -> dict[str, Any]:
    by_id = {result["instance_id"]: result for result in eval_results}
    tasks = []
    for instance_id in selected_ids:
        submitted = (submission_dir / f"{instance_id}.sql").is_file()
        result = by_id.get(instance_id)
        tasks.append(
            {
                "instance_id": instance_id,
                "submitted": submitted,
                "evaluated": result is not None,
                "score": int(result.get("score", 0) == 1) if result else 0,
                "error": (
                    result.get("error_info") or result.get("error")
                    if result
                    else "Missing submitted SQL"
                ),
            }
        )
    total = len(tasks)
    submitted_count = sum(task["submitted"] for task in tasks)
    evaluated_count = sum(task["evaluated"] for task in tasks)
    correct = sum(task["score"] for task in tasks)
    return {
        "total_tasks": total,
        "submitted_tasks": submitted_count,
        "evaluated_tasks": evaluated_count,
        "correct_tasks": correct,
        "completion_rate": submitted_count / total if total else 0.0,
        "submitted_answer_accuracy": (
            correct / evaluated_count if evaluated_count else 0.0
        ),
        "end_to_end_accuracy": correct / total if total else 0.0,
        "tasks": tasks,
    }


def evaluate_condition(
    config: LoadedConfig, *, level: str, official_sql_dir: Path
) -> dict[str, Any]:
    extraction = extract_sql_answers(config.experiment_dir)
    auto = config.raw["auto_evaluate"]
    eval_results = run_evaluation(
        extraction["submission_dir"],
        config.repo_root,
        timeout=auto["timeout"],
        max_workers=auto["max_workers"],
    )
    selected_ids = [item["instance_id"] for item in config.selected_items]
    run_summary = json.loads(
        (config.experiment_dir / "run-summary.json").read_text(encoding="utf-8")
    )
    payload = {
        "thinking_level": level,
        "run_dir": str(config.experiment_dir),
        "router": _score_integrated_router(config, official_sql_dir),
        "final": _build_final_scores(
            selected_ids, eval_results, extraction["submission_dir"]
        ),
        "conversation_stats": collect_conversation_stats(config.experiment_dir),
        "run_performance": run_summary.get("performance", {}),
        "run_summary": {
            "total_tasks": run_summary.get("total_tasks", 0),
            "successful_tasks": run_summary.get("successful_tasks", 0),
            "failed_tasks": run_summary.get("failed_tasks", 0),
            "schema_router": run_summary.get("schema_router", {}),
        },
    }
    _atomic_json(config.experiment_dir / "condition-evaluation.json", payload)
    (config.experiment_dir / "condition-evaluation.md").write_text(
        render_condition_report(payload), encoding="utf-8"
    )
    return payload


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_condition_report(payload: dict[str, Any]) -> str:
    router = payload["router"]
    final = payload["final"]
    return "\n".join(
        [
            f"# DeepSeek V4 Flash `{payload['thinking_level']}` 条件报告",
            "",
            "## Router",
            "",
            "| 指标 | 结果 |",
            "|---|---:|",
            f"| 完成 Rollout | {router['completed_rollouts']}/{router['scored_rollouts']} |",
            f"| 物理表题级全覆盖率 | {_pct(router['physical_task_full_coverage'])} |",
            f"| 物理表 Micro Recall | {_pct(router['physical_micro_recall'])} |",
            f"| 物理表 Macro Recall | {_pct(router['physical_macro_recall'])} |",
            f"| 表族题级全覆盖率 | {_pct(router['family_task_full_coverage'])} |",
            f"| 平均物理表压缩率 | {_pct(router['average_physical_compression'])} |",
            f"| 非法引用 | {router['invalid_references']} |",
            "",
            "## 最终 SQL",
            "",
            "| 指标 | 结果 |",
            "|---|---:|",
            f"| SQL 提交率 | {final['submitted_tasks']}/{final['total_tasks']} ({_pct(final['completion_rate'])}) |",
            f"| 已提交答案准确率 | {final['correct_tasks']}/{final['evaluated_tasks']} ({_pct(final['submitted_answer_accuracy'])}) |",
            f"| 端到端官方正确率 | {final['correct_tasks']}/{final['total_tasks']} ({_pct(final['end_to_end_accuracy'])}) |",
            "",
        ]
    )


def build_comparison_summary(
    *, task_set: dict[str, Any], conditions: dict[str, dict[str, Any]], probes: dict[str, Any]
) -> dict[str, Any]:
    high_final = {
        task["instance_id"]: task["score"]
        for task in conditions["high"]["final"]["tasks"]
    }
    high_router = {
        task["instance_id"]: int(task["physical_full_coverage"])
        for task in conditions["high"]["router"]["tasks"]
    }
    pairwise = {}
    for level in LEVELS:
        final = {
            task["instance_id"]: task["score"]
            for task in conditions[level]["final"]["tasks"]
        }
        router = {
            task["instance_id"]: int(task["physical_full_coverage"])
            for task in conditions[level]["router"]["tasks"]
        }
        pairwise[level] = {
            "final_wrong_to_right_vs_high": sum(
                high_final[key] == 0 and final[key] == 1 for key in high_final
            ),
            "final_right_to_wrong_vs_high": sum(
                high_final[key] == 1 and final[key] == 0 for key in high_final
            ),
            "router_miss_to_full_vs_high": sum(
                high_router[key] == 0 and router[key] == 1 for key in high_router
            ),
            "router_full_to_miss_vs_high": sum(
                high_router[key] == 1 and router[key] == 0 for key in high_router
            ),
        }
    return {
        "experiment_type": "paired_pilot_single_rollout",
        "task_set": task_set,
        "probes": probes,
        "conditions": conditions,
        "pairwise_vs_high": pairwise,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def render_comparison_report(summary: dict[str, Any]) -> str:
    lines = [
        "# DeepSeek V4 Flash 四种思考状态对比实验",
        "",
        "- 设计：固定随机 50/120 道 Gold SQL 题，每题每档 1 次",
        "- 状态：`none`、`low`、`high`、`max`",
        "- Router 与主 Agent 同档，temperature=0",
        "- 本报告是单次 rollout 先导实验，不代表稳定性或统计显著性",
        "",
        "## 核心结果",
        "",
        "| 档位 | Router 全覆盖 | Router Micro Recall | Router 压缩率 | SQL 提交率 | 已提交准确率 | 端到端正确率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for level in LEVELS:
        condition = summary["conditions"][level]
        router = condition["router"]
        final = condition["final"]
        lines.append(
            f"| {level} | {_pct(router['physical_task_full_coverage'])} | "
            f"{_pct(router['physical_micro_recall'])} | "
            f"{_pct(router['average_physical_compression'])} | "
            f"{_pct(final['completion_rate'])} | "
            f"{_pct(final['submitted_answer_accuracy'])} | "
            f"{_pct(final['end_to_end_accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "## Token 与耗时",
            "",
            "| 档位 | Router Token | Router 推理 Token | Agent Token | Agent 推理 Token | Router 累计秒 | Agent 墙钟秒 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for level in LEVELS:
        condition = summary["conditions"][level]
        router_performance = condition["router"].get("performance", {})
        agent_performance = condition.get("run_performance", {})
        lines.append(
            f"| {level} | {int(router_performance.get('total_tokens', 0))} | "
            f"{int(router_performance.get('reasoning_tokens', 0))} | "
            f"{int(agent_performance.get('total_tokens', 0))} | "
            f"{int(agent_performance.get('reasoning_tokens', 0))} | "
            f"{float(router_performance.get('duration_seconds', 0)):.2f} | "
            f"{float(agent_performance.get('agent_wall_clock_seconds', 0)):.2f} |"
        )
    lines.extend(
        [
            "",
            "## 相对 high 的逐题翻转",
            "",
            "| 档位 | 最终 错→对 | 最终 对→错 | Router 漏→全 | Router 全→漏 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for level in LEVELS:
        pair = summary["pairwise_vs_high"][level]
        lines.append(
            f"| {level} | {pair['final_wrong_to_right_vs_high']} | "
            f"{pair['final_right_to_wrong_vs_high']} | "
            f"{pair['router_miss_to_full_vs_high']} | "
            f"{pair['router_full_to_miss_vs_high']} |"
        )
    lines.extend(
        [
            "",
            "## 每题最终正确性",
            "",
            "| 题目 | none | low | high | max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    task_ids = summary["task_set"]["sampled_instance_ids_sorted"]
    score_maps = {
        level: {
            task["instance_id"]: task["score"]
            for task in summary["conditions"][level]["final"]["tasks"]
        }
        for level in LEVELS
    }
    for instance_id in task_ids:
        lines.append(
            f"| {instance_id} | "
            + " | ".join(str(score_maps[level][instance_id]) for level in LEVELS)
            + " |"
        )
    return "\n".join(lines) + "\n"


def run_comparison(config_path: str | Path) -> Path:
    _, raw = load_comparison_config(config_path)
    task_set = build_fixed_task_set(raw)
    fingerprint = _comparison_fingerprint(raw, task_set)
    parent_dir = _select_parent_dir(raw, fingerprint)
    parent_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = parent_dir / "comparison-manifest.json"
    manifest = {
        "fingerprint": fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "comparison_config": raw["config_path"],
        "base_config": raw["base_config"],
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            raise RuntimeError("Existing comparison manifest fingerprint differs")
        manifest["created_at"] = existing.get("created_at", manifest["created_at"])
    manifest["last_started_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(manifest_path, manifest)
    _atomic_json(parent_dir / "task-set.json", task_set)

    base_raw = _read_yaml(Path(raw["base_config"]))
    child_configs: dict[str, LoadedConfig] = {}
    selected_order: list[str] | None = None
    for level in LEVELS:
        child_raw = _build_child_raw(
            base_raw=base_raw,
            raw=raw,
            parent_dir=parent_dir,
            task_set=task_set,
            level=level,
        )
        path = _write_child_config(parent_dir, level, child_raw)
        config = load_config(path)
        ids = [item["instance_id"] for item in config.selected_items]
        if set(ids) != set(task_set["sampled_instance_ids"]) or len(ids) != 50:
            raise RuntimeError(f"Generated {level} task set differs from fixed sample")
        if selected_order is None:
            selected_order = ids
        elif ids != selected_order:
            raise RuntimeError("Condition execution orders differ")
        child_configs[level] = config
    task_set["execution_order"] = selected_order
    _atomic_json(parent_dir / "task-set.json", task_set)

    print(f"对比实验目录：{parent_dir}")
    print("离线依赖检查中...")
    check_dependencies()
    print("Snowflake 连通性检查中...")
    check_snowflake(child_configs[LEVELS[0]])

    state_path = parent_dir / "comparison-state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {"probes": {}, "conditions": {}}
    )
    probes = state.setdefault("probes", {})
    conditions = state.setdefault("conditions", {})
    condition_order = list(LEVELS)
    random.Random(raw["sampling"]["seed"]).shuffle(condition_order)
    state["condition_order"] = condition_order
    _atomic_json(state_path, state)

    evaluated: dict[str, dict[str, Any]] = {}
    for level in condition_order:
        print(f"[{level}] 条件开始")
        config = child_configs[level]
        if level not in probes:
            probes[level] = _probe_model(config, level)
            _atomic_json(state_path, state)
        condition_state = conditions.setdefault(level, {})
        saved_run = condition_state.get("run_dir")
        if saved_run and (Path(saved_run) / "run-summary.json").is_file():
            config = replace(config, run_dir=Path(saved_run))
        else:
            recovered = _find_completed_child(config)
            if recovered is not None:
                config = replace(config, run_dir=recovered)
                status = 0
            else:
                status = execute(config)
            condition_state["run_status"] = status
            condition_state["run_dir"] = str(config.experiment_dir)
            _atomic_json(state_path, state)
        evaluation_path = config.experiment_dir / "condition-evaluation.json"
        if evaluation_path.is_file():
            payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
        else:
            payload = evaluate_condition(
                config,
                level=level,
                official_sql_dir=Path(raw["official_sql_dir"]),
            )
        condition_state["evaluated"] = True
        _atomic_json(state_path, state)
        evaluated[level] = payload
        partial = {key: evaluated[key] for key in LEVELS if key in evaluated}
        _atomic_json(parent_dir / "comparison-progress.json", partial)

    summary = build_comparison_summary(
        task_set=task_set,
        conditions={level: evaluated[level] for level in LEVELS},
        probes=probes,
    )
    _atomic_json(parent_dir / "comparison-summary.json", summary)
    (parent_dir / "comparison-report.md").write_text(
        render_comparison_report(summary), encoding="utf-8"
    )
    print(f"对比实验完成：{parent_dir / 'comparison-report.md'}")
    return parent_dir


def main() -> int:
    try:
        run_comparison(_parse_args().config)
        return 0
    except KeyboardInterrupt:
        print("实验已中断，可使用同一配置继续")
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"对比实验失败：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
