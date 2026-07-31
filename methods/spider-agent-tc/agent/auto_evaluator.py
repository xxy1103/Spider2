"""Automatic evaluation and report generation after agent run.

This module handles:
1. Extracting SQL answers from terminated tasks
2. Running evaluation against gold standard
3. Collecting conversation statistics
4. Generating a comprehensive Markdown report
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from .progress import TaskProgressReporter

logger = logging.getLogger(__name__)

def _load_evaluate_module(evaluate_py_path: Path):
    """Dynamically load the evaluation suite's evaluate.py module."""
    spec = importlib.util.spec_from_file_location("spider2_evaluate", evaluate_py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load evaluate module from {evaluate_py_path}")
    module = importlib.util.module_from_spec(spec)
    
    # The evaluate.py expects to run from its own directory
    original_cwd = os.getcwd()
    try:
        os.chdir(evaluate_py_path.parent)
        spec.loader.exec_module(module)
    finally:
        os.chdir(original_cwd)
    
    return module


def extract_sql_answers(exp_dir: Path) -> dict[str, Any]:
    """Extract SQL answers from terminated records (reuse convert_to_submission_format logic)."""
    submission_dir = exp_dir / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = list(exp_dir.glob("*.json"))
    # Exclude metadata files
    json_files = [f for f in json_files if f.stem not in {"run-manifest", "run-summary", "selected-tasks", "failed-tasks", "effective-config"}]
    
    processed_count = 0
    skipped_count = 0
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                skipped_count += 1
                continue
            
            terminated_records = [r for r in data if r.get("terminated", False)]
            if not terminated_records:
                skipped_count += 1
                continue
            
            # Use the first terminated record
            selected_record = terminated_records[0]
            instance_id = selected_record.get("instance_id")
            conversation = selected_record.get("conversation")
            
            if not instance_id or not conversation:
                skipped_count += 1
                continue
            
            last_item = conversation[-1]
            tool_calls = last_item.get("tool_calls", [])
            
            if not tool_calls:
                skipped_count += 1
                continue
            
            first_tool_call = tool_calls[0]
            if first_tool_call.get("name") != "terminate":
                skipped_count += 1
                continue
            
            arguments = first_tool_call.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            
            answer = arguments.get("answer")
            if not answer:
                skipped_count += 1
                continue
            
            sql_file = submission_dir / f"{instance_id}.sql"
            sql_file.write_bytes(answer.encode("utf-8"))
            
            processed_count += 1
            
        except Exception as e:
            logger.error("Error extracting SQL from %s: %s", json_file.name, e)
            skipped_count += 1
    
    return {
        "submission_dir": submission_dir,
        "processed": processed_count,
        "skipped": skipped_count,
    }


def run_evaluation(
    submission_dir: Path,
    repo_root: Path,
    timeout: int = 300,
    max_workers: int = 4,
) -> list[dict[str, Any]]:
    """Run evaluation using spider2-snow evaluation suite."""
    evaluate_py = repo_root / "spider2-snow" / "evaluation_suite" / "evaluate.py"
    if not evaluate_py.exists():
        raise FileNotFoundError(f"Evaluation script not found: {evaluate_py}")
    
    # Load the evaluate module
    eval_module = _load_evaluate_module(evaluate_py)
    
    # Prepare paths
    eval_suite_dir = evaluate_py.parent
    gold_dir = eval_suite_dir / "gold"
    gold_sql_dir = gold_dir / "sql"
    gold_result_dir = gold_dir / "exec_result"
    
    # Load metadata
    eval_standard_dict = eval_module.load_jsonl_to_dict(gold_dir / "spider2snow_eval.jsonl")
    spider2sql_metadata = eval_module.load_jsonl_to_dict(repo_root / "spider2-snow" / "spider2-snow.jsonl")
    
    # Find SQL files to evaluate
    pred_ids = []
    for file in submission_dir.glob("*.sql"):
        pred_ids.append(file.stem)
    
    gold_ids = list(eval_standard_dict.keys())
    eval_ids = sorted(set(gold_ids).intersection(pred_ids))
    
    if not eval_ids:
        return []
    
    # Create temp directory for results
    with tempfile.TemporaryDirectory() as temp_dir:
        result_csv_dir = submission_dir.parent / "submission_csv"
        if result_csv_dir.exists():
            import shutil
            shutil.rmtree(result_csv_dir)
        result_csv_dir.mkdir(parents=True, exist_ok=True)
        
        # Run evaluation in parallel
        output_results = []
        actual_max_workers = min(max_workers, len(eval_ids))
        
        # Change to evaluation suite directory (required by evaluate.py)
        original_cwd = os.getcwd()
        try:
            os.chdir(eval_suite_dir)
            
            with TaskProgressReporter("自动评分", len(eval_ids)) as reporter:
                with ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
                    future_to_id = {
                        executor.submit(
                            _evaluate_with_progress,
                            reporter,
                            eval_module.evaluate_single_sql_instance,
                            instance_id,
                            eval_standard_dict,
                            spider2sql_metadata,
                            submission_dir,
                            gold_sql_dir,
                            gold_result_dir,
                            temp_dir,
                            result_csv_dir,
                            timeout,
                        ): instance_id
                        for instance_id in eval_ids
                    }

                    for future in as_completed(future_to_id):
                        instance_id = future_to_id[future]
                        try:
                            output_results.append(future.result())
                        except Exception as e:
                            logger.error(
                                "Evaluation failed for %s: %s", instance_id, e
                            )
                            output_results.append(
                                {"instance_id": instance_id, "score": 0, "error": str(e)}
                            )
        
        finally:
            os.chdir(original_cwd)
    
    return output_results


def _evaluate_with_progress(reporter, evaluate_one, instance_id, *args):
    reporter.task_started()
    try:
        result = evaluate_one(instance_id, *args)
        reporter.task_finished(success=result.get("score") == 1)
        return result
    except Exception:
        reporter.task_finished(success=False)
        raise


def collect_conversation_stats(exp_dir: Path) -> dict[str, dict[str, Any]]:
    """Collect statistics from conversation history JSON files."""
    stats = {}
    
    json_files = list(exp_dir.glob("*.json"))
    json_files = [f for f in json_files if f.stem not in {"run-manifest", "run-summary", "selected-tasks", "failed-tasks", "effective-config"}]
    
    for json_file in json_files:
        instance_id = json_file.stem
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if not isinstance(data, list) or not data:
                continue
            
            # Use the first record (rollout_idx=0)
            record = data[0]
            conversation = record.get("conversation", [])
            
            # Count messages by role
            role_counts = Counter(msg["role"] for msg in conversation)
            
            # Count tool calls
            tool_calls = []
            for msg in conversation:
                if msg["role"] == "assistant" and "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        tool_calls.append(tc["name"])
            
            tool_call_counts = Counter(tool_calls)
            
            stats[instance_id] = {
                "total_messages": len(conversation),
                "assistant_turns": role_counts.get("assistant", 0),
                "tool_messages": role_counts.get("tool", 0),
                "total_tool_calls": len(tool_calls),
                "tool_call_distribution": dict(tool_call_counts),
                "terminated": record.get("terminated", False),
            }
            
        except Exception as e:
            logger.error("Error collecting stats from %s: %s", json_file.name, e)
    
    return stats


def generate_report(
    exp_dir: Path,
    config: Any,
    summary: dict[str, Any],
    eval_results: list[dict[str, Any]],
    stats: dict[str, dict[str, Any]],
) -> Path:
    """Generate a comprehensive Markdown report."""
    report_path = exp_dir / "REPORT.md"
    
    # Load manifest for timing info
    manifest_path = exp_dir / "run-manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    
    # Calculate evaluation metrics
    correct_count = sum(1 for r in eval_results if r["score"] == 1)
    total_evaluated = len(eval_results)
    accuracy = correct_count / total_evaluated if total_evaluated > 0 else 0
    real_score = correct_count / 547  # Spider2-Snow total
    
    # Aggregate tool usage
    all_tool_calls = Counter()
    for stat in stats.values():
        all_tool_calls.update(stat.get("tool_call_distribution", {}))
    
    total_tool_calls = sum(all_tool_calls.values())
    
    # Calculate average stats
    if stats:
        avg_turns = sum(s["assistant_turns"] for s in stats.values()) / len(stats)
        avg_tool_calls = sum(s["total_tool_calls"] for s in stats.values()) / len(stats)
        avg_messages = sum(s["total_messages"] for s in stats.values()) / len(stats)
    else:
        avg_turns = avg_tool_calls = avg_messages = 0
    
    # Build report content
    lines = [
        "# Agent 运行报告",
        "",
        f"**实验名称**: `{exp_dir.name}`  ",
        f"**运行时间**: {manifest.get('last_started_at', 'N/A')}  ",
        f"**模型**: `{config.raw['model']['name']}` (temperature={config.raw['model']['temperature']}, max_tokens={config.raw['model']['max_tokens']})  ",
        f"**Agent 配置**: max_rounds={config.raw['agent']['max_rounds']}, num_threads={config.raw['agent']['num_threads']}, rollout_number={config.raw['agent']['rollout_number']}  ",
        "",
        "---",
        "",
        "## 📊 运行概览",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 任务总数 | {summary['total_tasks']} |",
        f"| 成功任务 | {summary['successful_tasks']} ✅ |",
        f"| 失败任务 | {summary['failed_tasks']} ❌ |",
        f"| 成功率 | {summary['successful_tasks']/summary['total_tasks']*100:.2f}% |",
        "",
        "---",
        "",
        "## 🎯 评分结果",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 正确任务 | {correct_count} / {total_evaluated} |",
        f"| 本次准确率 | {accuracy*100:.2f}% |",
        f"| 全数据集得分率 | {real_score*100:.2f}% ({correct_count}/547) |",
        "",
    ]
    
    # Task score details
    if eval_results:
        lines.extend([
            "### 任务得分详情",
            "",
            "| Instance ID | 得分 | 对话轮次 | 工具调用 | 状态 | 错误信息 |",
            "|-------------|------|----------|----------|------|----------|",
        ])
        
        for result in sorted(eval_results, key=lambda x: x["instance_id"]):
            instance_id = result["instance_id"]
            score = result["score"]
            score_icon = "✅" if score == 1 else "❌"
            error_info = result.get("error_info") or "-"
            
            stat = stats.get(instance_id, {})
            turns = stat.get("assistant_turns", "-")
            tool_calls = stat.get("total_tool_calls", "-")
            
            lines.append(
                f"| {instance_id} | {score_icon} {score} | {turns} | {tool_calls} | {'成功' if score == 1 else '失败'} | {error_info} |"
            )
        
        lines.append("")
    
    lines.extend([
        "---",
        "",
        "## 🔧 工具使用统计",
        "",
    ])
    
    if all_tool_calls:
        lines.extend([
            "| 工具名称 | 调用次数 | 占比 |",
            "|----------|----------|------|",
        ])
        for tool_name, count in all_tool_calls.most_common():
            pct = count / total_tool_calls * 100 if total_tool_calls > 0 else 0
            lines.append(f"| {tool_name} | {count} | {pct:.1f}% |")
        lines.append("")
    else:
        lines.append("无工具调用记录。\n")
    
    lines.extend([
        "---",
        "",
        "## 📈 对话统计",
        "",
        "| 指标 | 平均值 |",
        "|------|--------|",
        f"| 对话轮次 | {avg_turns:.1f} |",
        f"| 工具调用次数 | {avg_tool_calls:.1f} |",
        f"| 消息总数 | {avg_messages:.1f} |",
        "",
        "---",
        "",
        "## ❌ 失败分析",
        "",
    ])
    
    failed_tasks = [r for r in eval_results if r["score"] == 0]
    if failed_tasks:
        lines.append(f"共 {len(failed_tasks)} 个任务失败：\n")
        for result in failed_tasks:
            lines.append(f"- **{result['instance_id']}**: {result.get('error_info', '未知错误')}")
        lines.append("")
    else:
        lines.append("无失败任务。🎉\n")
    
    lines.extend([
        "---",
        "",
        "## 📝 配置文件",
        "",
        "```yaml",
        yaml.safe_dump(config.raw, sort_keys=False, allow_unicode=True),
        "```",
        "",
        "---",
        "",
        f"**报告生成时间**: {__import__('datetime').datetime.now().isoformat()}  ",
        f"**结果目录**: `{exp_dir}`  ",
    ])
    
    # Write report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    return report_path


def _check_evaluation_dependencies() -> tuple[bool, str | None]:
    """Check if evaluation dependencies are available."""
    missing = []
    try:
        import google.cloud.bigquery  # noqa: F401
    except ImportError:
        missing.append("google-cloud-bigquery")
    
    try:
        import snowflake.connector  # noqa: F401
    except ImportError:
        missing.append("snowflake-connector-python")
    
    if missing:
        return False, ", ".join(missing)
    return True, None


def run_evaluation_and_report(config: Any, summary: dict[str, Any]) -> None:
    """Main entry point for automatic evaluation and report generation."""
    exp_dir = config.experiment_dir
    
    # Check dependencies first
    deps_ok, missing_deps = _check_evaluation_dependencies()
    if not deps_ok:
        logger.warning("Skipping auto evaluation; missing dependencies: %s", missing_deps)
        print("自动评分已跳过，详情见 run.log")
        return
    
    # Step 1: Extract SQL answers
    extract_result = extract_sql_answers(exp_dir)
    
    if extract_result['processed'] == 0:
        logger.warning("Skipping auto evaluation; no SQL answers were extracted")
        print("自动评分已跳过，详情见 run.log")
        return
    
    # Step 2: Run evaluation
    auto_eval_config = config.raw.get("auto_evaluate", {})
    timeout = auto_eval_config.get("timeout", 300)
    max_workers = auto_eval_config.get("max_workers", 4)
    
    eval_results = run_evaluation(
        extract_result["submission_dir"],
        config.repo_root,
        timeout=timeout,
        max_workers=max_workers,
    )
    
    correct_count = sum(1 for r in eval_results if r["score"] == 1)
    total_evaluated = len(eval_results)
    accuracy = correct_count / total_evaluated * 100 if total_evaluated > 0 else 0
    
    # Step 3: Collect conversation stats
    stats = collect_conversation_stats(exp_dir)
    
    # Step 4: Generate report
    generate_report(exp_dir, config, summary, eval_results, stats)
