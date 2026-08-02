"""Integrated Schema Router stage and strict routing-artifact access."""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

from .progress import TaskProgressReporter
from .schema_router import (
    SchemaRouterAgent,
    SchemaRouterCatalog,
    SchemaRouterTools,
    build_router_user_prompt,
    load_external_knowledge,
)


_WRITE_LOCK = threading.Lock()


def route_key(instance_id: str, rollout_idx: int) -> str:
    return f"{instance_id}::{rollout_idx}"


def routing_path(run_dir: Path, instance_id: str, rollout_idx: int) -> Path:
    return run_dir / "routing" / instance_id / f"rollout-{rollout_idx}.json"


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def is_valid_route(result: Any) -> bool:
    return bool(
        isinstance(result, dict)
        and result.get("completed") is True
        and isinstance(result.get("selection"), dict)
        and result["selection"].get("valid") is True
        and result["selection"].get("resolved_physical_tables")
    )


def _canonical_selection_tables(
    catalog: SchemaRouterCatalog,
    task: dict[str, Any],
    result: Any,
) -> list[str] | None:
    if not is_valid_route(result):
        return None
    canonical = []
    for table in dict.fromkeys(result["selection"]["resolved_physical_tables"]):
        actual = catalog.tables_upper.get(str(table).upper())
        if actual is None or not actual.upper().startswith(
            f"{task['db_id']}.".upper()
        ):
            return None
        canonical.append(actual)
    return canonical or None


def load_routing_index(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(run_dir) / "routing-index.json"
    if not path.is_file():
        raise RuntimeError(f"Routing index is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    routes = value.get("routes") if isinstance(value, dict) else None
    if not isinstance(routes, dict):
        raise RuntimeError(f"Routing index is invalid: {path}")
    return routes


def get_route(
    run_dir: str | Path, instance_id: str, rollout_idx: int
) -> dict[str, Any]:
    routes = load_routing_index(run_dir)
    route = routes.get(route_key(instance_id, rollout_idx))
    if not isinstance(route, dict):
        raise RuntimeError(
            f"No valid Schema Router selection for {instance_id} rollout {rollout_idx}"
        )
    return route


def _run_one(
    *,
    agent: SchemaRouterAgent,
    catalog: SchemaRouterCatalog,
    task: dict[str, Any],
    rollout_idx: int,
    run_dir: Path,
    documents_path: Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
    path = routing_path(run_dir, task["instance_id"], rollout_idx)
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if _canonical_selection_tables(catalog, task, existing):
            return existing
    external = load_external_knowledge(task, documents_path)
    tools = SchemaRouterTools(
        catalog,
        database_id=task["db_id"],
        instance_id=task["instance_id"],
        sample_rows=settings["sample_rows"],
        max_sample_chars=settings["max_sample_chars"],
    )
    result = agent.run(
        item=task,
        rollout_idx=rollout_idx,
        tools=tools,
        user_prompt=build_router_user_prompt(
            task,
            external_knowledge=external,
            family_overview=catalog.family_overview(task["db_id"]),
        ),
    )
    with _WRITE_LOCK:
        _write_json_atomic(path, result)
    return result


def run_integrated_schema_router(
    config: Any, catalog: SchemaRouterCatalog
) -> dict[str, dict[str, Any]]:
    """Run every task/rollout before the main Agent and build strict routes."""
    settings = config.raw["schema_router"]
    model = settings["model"]
    agent = SchemaRouterAgent(
        model_client=OpenAI(
            base_url=config.secrets["model_api"]["base_url"],
            api_key=config.secrets["model_api"]["api_key"],
            timeout=model["request_timeout_seconds"],
        ),
        model_config=model,
        system_prompt=config.paths["router_prompt"].read_text(
            encoding="utf-8"
        ).strip(),
        max_rounds=settings["max_rounds"],
        max_tool_calls=settings["max_tool_calls"],
    )
    jobs = [
        (task, rollout_idx)
        for task in config.selected_items
        for rollout_idx in range(config.raw["agent"]["rollout_number"])
    ]
    results: dict[str, dict[str, Any]] = {}
    router_performance: list[dict[str, Any]] = []
    with TaskProgressReporter("Schema Router", len(jobs)) as reporter:
        with ThreadPoolExecutor(max_workers=settings["num_threads"]) as executor:
            futures = {}
            for task, rollout_idx in jobs:
                reporter.task_started()
                future = executor.submit(
                    _run_one,
                    agent=agent,
                    catalog=catalog,
                    task=task,
                    rollout_idx=rollout_idx,
                    run_dir=config.experiment_dir,
                    documents_path=config.paths["documents"],
                    settings=settings,
                )
                futures[future] = (task, rollout_idx)
            for future in as_completed(futures):
                task, rollout_idx = futures[future]
                key = route_key(task["instance_id"], rollout_idx)
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "instance_id": task["instance_id"],
                        "rollout_idx": rollout_idx,
                        "completed": False,
                        "error": str(exc).replace(
                            config.secrets["model_api"]["api_key"],
                            "***REDACTED***",
                        ),
                    }
                    _write_json_atomic(
                        routing_path(
                            config.experiment_dir, task["instance_id"], rollout_idx
                        ),
                        result,
                    )
                if is_valid_route(result):
                    canonical = _canonical_selection_tables(catalog, task, result)
                    if canonical is not None:
                        results[key] = {
                            "instance_id": task["instance_id"],
                            "rollout_idx": rollout_idx,
                            "schema_scope": "routed",
                            "allowed_database": task["db_id"],
                            "allowed_physical_tables": canonical,
                            "candidates": [
                                {
                                    "rank": candidate.get("rank"),
                                    "tier": candidate.get("tier"),
                                    "roles": candidate.get("roles", []),
                                    "resolved_physical_tables": candidate.get(
                                        "resolved_physical_tables", []
                                    ),
                                }
                                for candidate in result["selection"].get(
                                    "candidates", []
                                )
                            ],
                            "available_physical_tables": sum(
                                table.database_id == task["db_id"]
                                for table in catalog.tables.values()
                            ),
                            "routing_artifact": str(
                                routing_path(
                                    config.experiment_dir,
                                    task["instance_id"],
                                    rollout_idx,
                                ).relative_to(config.experiment_dir)
                            ),
                        }
                if isinstance(result.get("performance"), dict):
                    router_performance.append(result["performance"])
                reporter.task_finished(success=key in results)
    expected_keys = {
        route_key(task["instance_id"], rollout_idx)
        for task, rollout_idx in jobs
    }
    performance_fields = (
        "model_calls",
        "model_attempts",
        "tool_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "duration_seconds",
    )
    payload = {
        "mode": "strict",
        "failure_policy": "fail_task",
        "expected_routes": len(jobs),
        "valid_routes": len(results),
        "failed_routes": len(jobs) - len(results),
        "failed_route_keys": sorted(expected_keys - set(results)),
        "performance": {
            field: sum(float(value.get(field, 0)) for value in router_performance)
            for field in performance_fields
        },
        "routes": results,
    }
    _write_json_atomic(config.experiment_dir / "routing-index.json", payload)
    return results
