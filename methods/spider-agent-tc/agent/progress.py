"""Thread-safe terminal progress reporting for concurrent task execution."""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import TextIO

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
)
from rich.text import Text


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _format_rate(tasks_per_hour: float | None) -> str:
    if tasks_per_hour is None:
        return "--"
    return f"{tasks_per_hour:.2f} 题/小时"


class RuntimeMetricsColumn(ProgressColumn):
    """Render live elapsed time and task completion rate."""

    def render(self, task: Task) -> Text:
        elapsed_seconds = task.elapsed or 0.0
        completed = int(task.completed)
        newly_completed = completed - int(task.fields.get("skipped", 0))
        if newly_completed > 0 and elapsed_seconds > 0:
            tasks_per_hour = newly_completed * 3600 / elapsed_seconds
        else:
            tasks_per_hour = None
        return Text(
            f"已运行 {_format_duration(elapsed_seconds)}  "
            f"速度 {_format_rate(tasks_per_hour)}"
        )


@dataclass(frozen=True)
class ProgressSnapshot:
    completed: int
    total: int
    running: int
    successful: int
    failed: int
    skipped: int
    elapsed_seconds: float
    tasks_per_hour: float | None


class TaskProgressReporter:
    """Render one aggregate progress row and maintain consistent counters."""

    def __init__(
        self,
        label: str,
        total: int,
        *,
        stream: TextIO = sys.stdout,
        is_terminal: bool | None = None,
        initial_successful: int = 0,
        initial_failed: int = 0,
        skipped: int = 0,
    ) -> None:
        if min(initial_successful, initial_failed, skipped) < 0:
            raise ValueError("Initial progress counters must not be negative")
        if skipped > initial_successful:
            raise ValueError("Skipped tasks must be included in initial successes")
        if initial_successful + initial_failed > total:
            raise ValueError("Initial completed tasks exceed configured total")
        self.label = label
        self.total = total
        self._stream = stream
        self._lock = threading.Lock()
        self._running = 0
        self._successful = initial_successful
        self._failed = initial_failed
        self._skipped = skipped
        self._started_at: float | None = None
        self._console = Console(file=stream, force_terminal=is_terminal)
        self._is_terminal = (
            self._console.is_terminal if is_terminal is None else is_terminal
        )
        self._progress: Progress | None = None
        self._task_id = None

    @property
    def snapshot(self) -> ProgressSnapshot:
        with self._lock:
            return self._snapshot_unlocked()

    def __enter__(self) -> "TaskProgressReporter":
        self._started_at = time.monotonic()
        if self._is_terminal:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn(
                    "已结束 {task.completed:.0f}/{task.total:.0f}  "
                    "进行中 {task.fields[running]}  "
                    "成功 {task.fields[successful]}  "
                    "失败 {task.fields[failed]}  "
                    "跳过 {task.fields[skipped]}"
                ),
                RuntimeMetricsColumn(),
                console=self._console,
                transient=False,
            )
            self._progress.start()
            self._task_id = self._progress.add_task(
                self.label,
                total=self.total,
                completed=self._successful + self._failed,
                running=0,
                successful=self._successful,
                failed=self._failed,
                skipped=self._skipped,
            )
        elif self.total == 0 or self._successful or self._failed:
            self._write_snapshot(self._snapshot_unlocked())
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._progress is not None:
            self._progress.stop()

    def task_started(self) -> None:
        with self._lock:
            self._running += 1
            self._refresh_unlocked()

    def task_finished(self, *, success: bool) -> None:
        with self._lock:
            if self._running <= 0:
                raise RuntimeError("Cannot finish a task that is not running")
            self._running -= 1
            if success:
                self._successful += 1
            else:
                self._failed += 1
            snapshot = self._snapshot_unlocked()
            if snapshot.completed > self.total:
                raise RuntimeError("Completed task count exceeds configured total")
            self._refresh_unlocked()
            if not self._is_terminal:
                self._write_snapshot(snapshot)

    def _snapshot_unlocked(self) -> ProgressSnapshot:
        completed = self._successful + self._failed
        elapsed_seconds = (
            max(0.0, time.monotonic() - self._started_at)
            if self._started_at is not None
            else 0.0
        )
        newly_completed = completed - self._skipped
        tasks_per_hour = (
            newly_completed * 3600 / elapsed_seconds
            if newly_completed > 0 and elapsed_seconds > 0
            else None
        )
        return ProgressSnapshot(
            completed=completed,
            total=self.total,
            running=self._running,
            successful=self._successful,
            failed=self._failed,
            skipped=self._skipped,
            elapsed_seconds=elapsed_seconds,
            tasks_per_hour=tasks_per_hour,
        )

    def _refresh_unlocked(self) -> None:
        if self._progress is None or self._task_id is None:
            return
        snapshot = self._snapshot_unlocked()
        self._progress.update(
            self._task_id,
            completed=snapshot.completed,
            running=snapshot.running,
            successful=snapshot.successful,
            failed=snapshot.failed,
            skipped=snapshot.skipped,
            refresh=True,
        )

    def _write_snapshot(self, snapshot: ProgressSnapshot) -> None:
        print(
            f"{self.label}: 已结束 {snapshot.completed}/{snapshot.total}  "
            f"进行中 {snapshot.running}  成功 {snapshot.successful}  "
            f"失败 {snapshot.failed}  跳过 {snapshot.skipped}  "
            f"已运行 {_format_duration(snapshot.elapsed_seconds)}  "
            f"速度 {_format_rate(snapshot.tasks_per_hour)}",
            file=self._stream,
            flush=True,
        )
