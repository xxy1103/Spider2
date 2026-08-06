"""Official-SQL labels, full task-set identity, and Schema Router metrics."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .schema_router import SchemaRouterCatalog, SchemaRouterError


K_VALUES = (1, 5, 10, 20)


def load_tasks(input_path: Path) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaRouterError(
                    f"Invalid task JSON at {input_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict) or not isinstance(
                value.get("instance_id"), str
            ):
                raise SchemaRouterError(
                    f"Invalid task object at {input_path}:{line_number}"
                )
            instance_id = value["instance_id"]
            if instance_id in tasks:
                raise SchemaRouterError(f"Duplicate task id: {instance_id}")
            tasks[instance_id] = value
    return tasks


def _canonical_sql_table(
    *,
    catalog: SchemaRouterCatalog,
    task: dict[str, Any],
    catalog_name: str,
    schema_name: str,
    table_name: str,
) -> str:
    database_id = str(task["db_id"])
    if catalog_name:
        candidate = f"{catalog_name}.{schema_name}.{table_name}"
        if catalog_name.upper() != database_id.upper():
            raise SchemaRouterError(
                f"{task['instance_id']} SQL references database outside db_id: "
                f"{candidate}"
            )
        canonical = catalog.canonical_table(candidate)
        if canonical is None:
            raise SchemaRouterError(
                f"{task['instance_id']} SQL table is absent from resource: "
                f"{candidate}"
            )
        return canonical
    if schema_name:
        candidate = f"{database_id}.{schema_name}.{table_name}"
        canonical = catalog.canonical_table(candidate)
        if canonical is None:
            raise SchemaRouterError(
                f"{task['instance_id']} SQL table is absent from resource: "
                f"{candidate}"
            )
        return canonical
    matches = [
        table.full_name
        for table in catalog.tables.values()
        if table.database_id.upper() == database_id.upper()
        and table.table_name.upper() == table_name.upper()
    ]
    if len(matches) != 1:
        raise SchemaRouterError(
            f"{task['instance_id']} unqualified SQL table is "
            f"{'ambiguous' if matches else 'absent'}: {table_name}"
        )
    return matches[0]


def extract_official_sql_labels(
    *,
    sql_dir: Path,
    tasks: dict[str, dict[str, Any]],
    catalog: SchemaRouterCatalog,
) -> list[dict[str, Any]]:
    """Parse every released official SQL file; fail if any label is incomplete."""
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError as exc:
        raise SchemaRouterError(
            "sqlglot is required to extract official SQL labels"
        ) from exc

    sql_paths = sorted(sql_dir.glob("*.sql"))
    if not sql_paths:
        raise SchemaRouterError("No official SQL files were found")
    labels = []
    for path in sql_paths:
        instance_id = path.stem
        task = tasks.get(instance_id)
        if task is None:
            raise SchemaRouterError(
                f"Official SQL has no matching task: {instance_id}"
            )
        sql = path.read_text(encoding="utf-8")
        try:
            expressions = sqlglot.parse(
                sql,
                read="snowflake",
                error_level=sqlglot.ErrorLevel.RAISE,
            )
        except Exception as exc:  # noqa: BLE001
            raise SchemaRouterError(
                f"Unable to parse official SQL for {instance_id}: {exc}"
            ) from exc
        physical: list[str] = []
        for expression in expressions:
            cte_names = {
                cte.alias_or_name.upper()
                for cte in expression.find_all(exp.CTE)
                if cte.alias_or_name
            }
            for table in expression.find_all(exp.Table):
                catalog_name = str(table.catalog or "").strip()
                schema_name = str(table.db or "").strip()
                table_name = str(table.name or "").strip()
                if not table_name:
                    continue
                if (
                    not catalog_name
                    and not schema_name
                    and table_name.upper() in cte_names
                ):
                    continue
                canonical = _canonical_sql_table(
                    catalog=catalog,
                    task=task,
                    catalog_name=catalog_name,
                    schema_name=schema_name,
                    table_name=table_name,
                )
                if canonical not in physical:
                    physical.append(canonical)
        if not physical:
            raise SchemaRouterError(
                f"Official SQL contains no resolvable physical table: {instance_id}"
            )
        families = list(
            dict.fromkeys(catalog.table_to_family[table] for table in physical)
        )
        labels.append(
            {
                "instance_id": instance_id,
                "database_id": task["db_id"],
                "sql_sha256": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
                "physical_tables": physical,
                "families": families,
            }
        )
    return labels


def labels_sha256(labels: list[dict[str, Any]]) -> str:
    ordered = sorted(labels, key=lambda label: label["instance_id"])
    return hashlib.sha256(
        json.dumps(ordered, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def make_task_set(
    *, labels: list[dict[str, Any]], catalog: SchemaRouterCatalog
) -> dict[str, Any]:
    """Build the deterministic identity of the complete official-SQL task set."""
    instance_ids = sorted(label["instance_id"] for label in labels)
    if len(instance_ids) != len(set(instance_ids)):
        raise SchemaRouterError("Official SQL labels contain duplicate task ids")
    catalog_payload = [
        {
            "table": table.full_name,
            "family": catalog.table_to_family[table.full_name],
            "columns": list(table.columns),
        }
        for table in sorted(catalog.tables.values(), key=lambda value: value.full_name)
    ]
    return {
        "task_count": len(instance_ids),
        "instance_ids": instance_ids,
        "labels_sha256": labels_sha256(labels),
        "catalog_sha256": hashlib.sha256(
            json.dumps(
                catalog_payload, ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
        ).hexdigest(),
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _recall(selected: set[str], gold: set[str]) -> float:
    return _safe_divide(len(selected & gold), len(gold))


def score_rollout(
    *,
    result: dict[str, Any] | None,
    label: dict[str, Any],
    catalog: SchemaRouterCatalog,
) -> dict[str, Any]:
    gold_physical = set(label["physical_tables"])
    gold_families = set(label["families"])
    selection = (
        result.get("selection")
        if isinstance(result, dict) and result.get("completed") is True
        else None
    )
    candidates = selection.get("candidates", []) if selection else []
    selected_physical = (
        set(selection.get("resolved_physical_tables", [])) if selection else set()
    )
    selected_families = {
        candidate["family_id"] for candidate in candidates
    }
    database_id = label["database_id"]
    available_physical = sum(
        table.database_id == database_id for table in catalog.tables.values()
    )
    available_families = len(catalog.database_families(database_id))
    family_at_k = {}
    physical_at_k = {}
    for value in K_VALUES:
        prefix = candidates[:value]
        prefix_families = {
            candidate["family_id"] for candidate in prefix
        }
        prefix_physical = {
            table
            for candidate in prefix
            for table in candidate.get("resolved_physical_tables", [])
        }
        family_at_k[str(value)] = _recall(prefix_families, gold_families)
        physical_at_k[str(value)] = _recall(prefix_physical, gold_physical)
    tier_recall = {}
    for tier in ("required", "supporting", "possible"):
        tier_candidates = [
            candidate for candidate in candidates if candidate.get("tier") == tier
        ]
        tier_physical = {
            table
            for candidate in tier_candidates
            for table in candidate.get("resolved_physical_tables", [])
        }
        tier_recall[tier] = _recall(tier_physical, gold_physical)
    invalid_references = 0
    if isinstance(result, dict):
        for round_value in result.get("trace", []):
            for tool_value in round_value.get("tools", []):
                tool_result = tool_value.get("result", {})
                error = (
                    tool_result.get("error", "")
                    if isinstance(tool_result, dict)
                    else ""
                )
                if any(
                    marker in error
                    for marker in (
                        "not present in the allowed database",
                        "not a variant",
                        "Unknown family variant",
                    )
                ):
                    invalid_references += 1
    return {
        "instance_id": label["instance_id"],
        "database_id": database_id,
        "rollout_idx": (
            result.get("rollout_idx") if isinstance(result, dict) else None
        ),
        "completed": bool(selection),
        "physical_full_coverage": gold_physical <= selected_physical,
        "family_full_coverage": gold_families <= selected_families,
        "physical_recall": _recall(selected_physical, gold_physical),
        "family_recall": _recall(selected_families, gold_families),
        "physical_true_positives": len(selected_physical & gold_physical),
        "physical_gold": len(gold_physical),
        "family_true_positives": len(selected_families & gold_families),
        "family_gold": len(gold_families),
        "available_physical": available_physical,
        "available_families": available_families,
        "selected_physical": len(selected_physical),
        "selected_families": len(selected_families),
        "physical_compression": 1
        - _safe_divide(len(selected_physical), available_physical),
        "family_compression": 1
        - _safe_divide(len(selected_families), available_families),
        "candidate_rendered_chars": len(
            json.dumps(candidates, ensure_ascii=False)
        ),
        "family_recall_at_k": family_at_k,
        "physical_recall_at_k": physical_at_k,
        "tier_physical_recall": tier_recall,
        "invalid_references": invalid_references,
        "performance": result.get("performance", {}) if result else {},
        "error": result.get("error") if result else "Missing Router result",
    }


def aggregate_scores(
    *,
    scores: list[dict[str, Any]],
    expected_instance_ids: list[str],
) -> dict[str, Any]:
    expected = set(expected_instance_ids)
    observed = {score["instance_id"] for score in scores}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise SchemaRouterError(
            f"Score coverage mismatch; missing={missing}, extra={extra}"
        )
    total = len(scores)
    physical_tp = sum(score["physical_true_positives"] for score in scores)
    physical_gold = sum(score["physical_gold"] for score in scores)
    family_tp = sum(score["family_true_positives"] for score in scores)
    family_gold = sum(score["family_gold"] for score in scores)
    performance_fields = (
        "model_calls",
        "model_attempts",
        "tool_calls",
        "exploration_tool_calls",
        "tool_errors",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
        "forced_submissions",
        "format_repairs",
        "duration_seconds",
    )
    performance = {
        field: round(
            sum(
                float(score.get("performance", {}).get(field, 0))
                for score in scores
            ),
            6,
        )
        for field in performance_fields
    }
    by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for score in scores:
        by_instance[score["instance_id"]].append(score)
    stability = {
        "instances_with_all_rollouts_physical_full": sum(
            all(value["physical_full_coverage"] for value in values)
            for values in by_instance.values()
        ),
        "instances": len(by_instance),
        "worst_rollout_physical_recall": min(
            (score["physical_recall"] for score in scores),
            default=0.0,
        ),
    }
    completed_scores = [score for score in scores if score["completed"]]
    completed_count = len(completed_scores)
    return {
        "scored_rollouts": total,
        "unique_tasks": len(by_instance),
        "completed_rollouts": sum(score["completed"] for score in scores),
        "failed_rollouts": sum(not score["completed"] for score in scores),
        "physical_task_full_coverage": _safe_divide(
            sum(score["physical_full_coverage"] for score in scores), total
        ),
        "physical_micro_recall": _safe_divide(physical_tp, physical_gold),
        "physical_macro_recall": _safe_divide(
            sum(score["physical_recall"] for score in scores), total
        ),
        "family_task_full_coverage": _safe_divide(
            sum(score["family_full_coverage"] for score in scores), total
        ),
        "family_micro_recall": _safe_divide(family_tp, family_gold),
        "family_macro_recall": _safe_divide(
            sum(score["family_recall"] for score in scores), total
        ),
        "average_physical_compression": _safe_divide(
            sum(
                score["physical_compression"] for score in completed_scores
            ),
            completed_count,
        ),
        "average_family_compression": _safe_divide(
            sum(score["family_compression"] for score in completed_scores),
            completed_count,
        ),
        "average_candidate_rendered_chars": _safe_divide(
            sum(
                score["candidate_rendered_chars"]
                for score in completed_scores
            ),
            completed_count,
        ),
        "invalid_references": sum(
            score["invalid_references"] for score in scores
        ),
        "family_recall_at_k": {
            str(value): _safe_divide(
                sum(score["family_recall_at_k"][str(value)] for score in scores),
                total,
            )
            for value in K_VALUES
        },
        "physical_recall_at_k": {
            str(value): _safe_divide(
                sum(
                    score["physical_recall_at_k"][str(value)]
                    for score in scores
                ),
                total,
            )
            for value in K_VALUES
        },
        "tier_physical_recall": {
            tier: _safe_divide(
                sum(score["tier_physical_recall"][tier] for score in scores),
                total,
            )
            for tier in ("required", "supporting", "possible")
        },
        "stability": stability,
        "performance": performance,
        "tasks": scores,
    }


def threshold_status(
    *,
    summary: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "physical_task_full_coverage": (
            summary["physical_task_full_coverage"]
            >= thresholds["physical_task_full_coverage"]
        ),
        "physical_micro_recall": (
            summary["physical_micro_recall"]
            >= thresholds["physical_micro_recall"]
        ),
        "invalid_references": (
            summary["invalid_references"] <= thresholds["invalid_references"]
        ),
    }
    return {"applied": True, "passed": all(checks.values()), "checks": checks}


def render_report(
    *,
    summary: dict[str, Any],
    threshold: dict[str, Any],
) -> str:
    percentage = lambda value: f"{100 * value:.2f}%"  # noqa: E731
    lines = [
        "# Schema Router 官方 SQL 全量评测报告",
        "",
        "- 评测模式：`all`",
        f"- 官方 SQL：{summary['official_sql_files']}",
        f"- 任务数：{summary['total_tasks']}",
        f"- 每题 Rollout：{summary['rollouts_per_task']}",
        f"- 预期 Rollout：{summary['expected_rollouts']}",
        f"- 计分 Rollout：{summary['scored_rollouts']}",
        f"- 完成 Rollout：{summary['completed_rollouts']}",
        f"- 失败 Rollout：{summary['failed_rollouts']}",
        "",
        "## 核心指标",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        (
            "| 物理表题级全覆盖率 | "
            f"{percentage(summary['physical_task_full_coverage'])} |"
        ),
        f"| 物理表 Micro Recall | {percentage(summary['physical_micro_recall'])} |",
        f"| 物理表 Macro Recall | {percentage(summary['physical_macro_recall'])} |",
        (
            "| 表族题级全覆盖率 | "
            f"{percentage(summary['family_task_full_coverage'])} |"
        ),
        f"| 表族 Micro Recall | {percentage(summary['family_micro_recall'])} |",
        f"| 表族 Macro Recall | {percentage(summary['family_macro_recall'])} |",
        f"| 平均物理表压缩率 | {percentage(summary['average_physical_compression'])} |",
        f"| 平均表族压缩率 | {percentage(summary['average_family_compression'])} |",
        f"| 非法/越权引用尝试 | {summary['invalid_references']} |",
        "",
        "## Recall@K",
        "",
        "| K | Family Recall | Physical Recall |",
        "|---:|---:|---:|",
    ]
    for value in K_VALUES:
        lines.append(
            f"| {value} | "
            f"{percentage(summary['family_recall_at_k'][str(value)])} | "
            f"{percentage(summary['physical_recall_at_k'][str(value)])} |"
        )
    lines.extend(
        [
            "",
            "## 候选层级覆盖",
            "",
            "| Tier | Physical Recall |",
            "|---|---:|",
            (
                "| required | "
                f"{percentage(summary['tier_physical_recall']['required'])} |"
            ),
            (
                "| supporting | "
                f"{percentage(summary['tier_physical_recall']['supporting'])} |"
            ),
            (
                "| possible | "
                f"{percentage(summary['tier_physical_recall']['possible'])} |"
            ),
            "",
            "## 稳定性与性能",
            "",
            (
                "- 所有 Rollout 均物理表全覆盖的任务："
                f"{summary['stability']['instances_with_all_rollouts_physical_full']}"
                f"/{summary['stability']['instances']}"
            ),
            (
                "- 最差单次物理表召回："
                f"{percentage(summary['stability']['worst_rollout_physical_recall'])}"
            ),
            f"- 模型调用：{int(summary['performance']['model_calls'])}",
            f"- 工具调用：{int(summary['performance']['tool_calls'])}",
            f"- 总 Token：{int(summary['performance']['total_tokens'])}",
            (
                "- 平均候选渲染字符："
                f"{summary['average_candidate_rendered_chars']:.0f}"
            ),
            (
                "- 总耗时："
                f"{summary['performance']['duration_seconds']:.2f} 秒"
            ),
            "",
            "## 全量验收门槛",
            "",
        ]
    )
    lines.append(f"- 结论：{'通过' if threshold['passed'] else '未通过'}")
    for name, passed in threshold["checks"].items():
        lines.append(f"- `{name}`：{'通过' if passed else '未通过'}")
    lines.extend(
        [
            "",
            "## 每题压缩明细（物理表）",
            "",
            (
                "压缩前表数是该题 allowed database 在本地 Schema Catalog 中的"
                "物理表总数；压缩后表数是 Router 最终候选展开并去重后的物理表数。"
            ),
            "",
            (
                "| 题目 | Rollout | 数据库 | 压缩前 | 压缩后 | 减少 | "
                "压缩率 | Gold 命中 | Recall | 全覆盖 |"
            ),
            "|---|---:|---|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for score in summary["tasks"]:
        available = score["available_physical"]
        selected = score["selected_physical"]
        rollout_idx = score["rollout_idx"]
        lines.append(
            f"| {score['instance_id']} | "
            f"{rollout_idx if rollout_idx is not None else '-'} | "
            f"{score['database_id']} | "
            f"{available} | {selected} | {available - selected} | "
            f"{percentage(score['physical_compression'])} | "
            f"{score['physical_true_positives']}/{score['physical_gold']} | "
            f"{percentage(score['physical_recall'])} | "
            f"{'是' if score['physical_full_coverage'] else '否'} |"
        )
    return "\n".join(lines) + "\n"
