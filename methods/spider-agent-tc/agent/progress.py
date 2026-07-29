"""Thread-safe terminal progress reporting for concurrent task execution."""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import TextIO

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)


@dataclass(frozen=True)
class ProgressSnapshot:
    completed: int
    total: int
    running: int
    successful: int
    failed: int


class TaskProgressReporter:
    """Render one aggregate progress row and maintain consistent counters."""

    def __init__(
        self,
        label: str,
        total: int,
        *,
        stream: TextIO = sys.stdout,
        is_terminal: bool | None = None,
    ) -> None:
        self.label = label
        self.total = total
        self._stream = stream
        self._lock = threading.Lock()
        self._running = 0
        self._successful = 0
        self._failed = 0
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
                    "失败 {task.fields[failed]}"
                ),
                console=self._console,
                transient=False,
            )
            self._progress.start()
            self._task_id = self._progress.add_task(
                self.label,
                total=self.total,
                running=0,
                successful=0,
                failed=0,
            )
        elif self.total == 0:
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
        return ProgressSnapshot(
            completed=self._successful + self._failed,
            total=self.total,
            running=self._running,
            successful=self._successful,
            failed=self._failed,
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
            refresh=True,
        )

    def _write_snapshot(self, snapshot: ProgressSnapshot) -> None:
        print(
            f"{self.label}: 已结束 {snapshot.completed}/{snapshot.total}  "
            f"进行中 {snapshot.running}  成功 {snapshot.successful}  "
            f"失败 {snapshot.failed}",
            file=self._stream,
            flush=True,
        )
