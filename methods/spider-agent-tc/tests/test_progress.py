import io
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TC_ROOT))

from agent.auto_evaluator import _evaluate_with_progress
from agent.progress import TaskProgressReporter
from safe_logging import RedactingFilter


def test_non_tty_reports_plain_text_snapshots():
    output = io.StringIO()

    with TaskProgressReporter(
        "Agent", 2, stream=output, is_terminal=False
    ) as reporter:
        reporter.task_started()
        reporter.task_finished(success=True)
        reporter.task_started()
        reporter.task_finished(success=False)

    lines = output.getvalue().splitlines()
    assert lines[0].startswith("Agent: 已结束 1/2  进行中 0  成功 1  失败 0")
    assert lines[1].startswith("Agent: 已结束 2/2  进行中 0  成功 1  失败 1")
    assert all("已运行 " in line for line in lines)
    assert all("速度 " in line for line in lines)
    assert all("平均 " not in line for line in lines)
    assert "\x1b" not in output.getvalue()
    assert reporter.snapshot.completed == 2
    assert reporter.snapshot.successful == 1
    assert reporter.snapshot.failed == 1


def test_progress_counters_are_thread_safe():
    output = io.StringIO()
    total = 40

    with TaskProgressReporter(
        "Agent", total, stream=output, is_terminal=False
    ) as reporter:
        def complete(index):
            reporter.task_started()
            reporter.task_finished(success=index % 2 == 0)

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(complete, range(total)))

    snapshot = reporter.snapshot
    assert snapshot.completed == total
    assert snapshot.running == 0
    assert snapshot.successful == 20
    assert snapshot.failed == 20
    assert snapshot.completed == snapshot.successful + snapshot.failed


def test_terminal_progress_contains_requested_fields():
    output = io.StringIO()

    with TaskProgressReporter(
        "Agent", 1, stream=output, is_terminal=True
    ) as reporter:
        reporter.task_started()
        reporter.task_finished(success=True)

    rendered = output.getvalue()
    assert "Agent" in rendered
    assert any(frame in rendered for frame in ("⠋", "⠙", "⠹", "⠸", "⠼"))
    assert "已结束 1/1" in rendered
    assert "进行中 0" in rendered
    assert "成功 1" in rendered
    assert "失败 0" in rendered
    assert "已运行 " in rendered
    assert "速度 " in rendered
    assert "平均 " not in rendered


def test_progress_calculates_completion_rate(monkeypatch):
    output = io.StringIO()
    clock_values = iter([100.0, 160.0, 160.0])
    monkeypatch.setattr("agent.progress.time.monotonic", lambda: next(clock_values))

    with TaskProgressReporter(
        "Agent", 1, stream=output, is_terminal=False
    ) as reporter:
        reporter.task_started()
        reporter.task_finished(success=True)

    snapshot = reporter.snapshot
    assert snapshot.elapsed_seconds == 60
    assert snapshot.tasks_per_hour == 60
    assert "已运行 01:00" in output.getvalue()
    assert "速度 60.00 题/小时" in output.getvalue()
    assert "平均 " not in output.getvalue()


def test_progress_reports_resumed_skips_without_inflating_rate(monkeypatch):
    output = io.StringIO()
    clock_values = iter([100.0, 160.0, 160.0, 160.0])
    monkeypatch.setattr("agent.progress.time.monotonic", lambda: next(clock_values))

    with TaskProgressReporter(
        "Router",
        3,
        stream=output,
        is_terminal=False,
        initial_successful=2,
        skipped=2,
    ) as reporter:
        reporter.task_started()
        reporter.task_finished(success=True)

    snapshot = reporter.snapshot
    assert snapshot.completed == 3
    assert snapshot.skipped == 2
    assert snapshot.tasks_per_hour == 60
    assert "跳过 2" in output.getvalue()


def test_evaluation_progress_counts_zero_score_as_failure():
    output = io.StringIO()

    with TaskProgressReporter(
        "自动评分", 1, stream=output, is_terminal=False
    ) as reporter:
        result = _evaluate_with_progress(
            reporter,
            lambda instance_id: {"instance_id": instance_id, "score": 0},
            "task-1",
        )

    assert result["score"] == 0
    assert reporter.snapshot.failed == 1


def test_evaluation_progress_counts_exception_as_failure():
    output = io.StringIO()

    def fail(_instance_id):
        raise RuntimeError("evaluation failed")

    with TaskProgressReporter(
        "自动评分", 1, stream=output, is_terminal=False
    ) as reporter:
        try:
            _evaluate_with_progress(reporter, fail, "task-1")
        except RuntimeError:
            pass

    assert reporter.snapshot.completed == 1
    assert reporter.snapshot.failed == 1
    assert reporter.snapshot.running == 0


def test_log_filter_redacts_credentials():
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.addFilter(RedactingFilter(["model-secret", "snowflake-secret"]))
    test_logger = logging.getLogger("test-redacting-filter")
    test_logger.handlers = [handler]
    test_logger.propagate = False
    test_logger.setLevel(logging.INFO)

    test_logger.info("failed with %s and model-secret", "snowflake-secret")

    assert output.getvalue() == (
        "failed with ***REDACTED*** and ***REDACTED***\n"
    )
