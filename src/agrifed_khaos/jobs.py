"""Asynchronous Job Runner & FIFO Execution Queue for AgriFed-KHAOS MLOps."""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("agrifed.jobs")


def _atomic_write(target_path: Path, data: bytes) -> None:
    """Safely and atomically persist data to target_path."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_path_str = tempfile.mkstemp(dir=target_path.parent, prefix="atomic_")
    temp_path = Path(temp_path_str)
    try:
        with os.fdopen(temp_fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, target_path)
    finally:
        temp_path.unlink(missing_ok=True)


@dataclass
class QueuedJob:
    """Representation of an enqueued background job."""

    job_id: str
    action: str
    cmd: list[str] | list[list[str]]
    on_success: Any | None = None
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


class JobManager:
    """Manages asynchronous training/aggregation tasks with a sequential FIFO execution queue.

    Guarantees strict concurrency = 1 (preventing CPU/GPU starvation and OOM crashes),
    while accepting subsequent jobs into an ordered queue instead of throwing 409 Conflict.
    Persists pending queue to disk to survive server/container restarts.
    """

    def __init__(self, queue_file: Path | str | None = None) -> None:
        self.lock = asyncio.Lock()
        self.process: asyncio.subprocess.Process | None = None
        self.action: str | None = None
        self.active_job_id: str | None = None
        self.started_at: str | None = None
        self.last_exit_code: int | None = None
        self.logs: collections.deque[str] = collections.deque(maxlen=300)
        self.queue: collections.deque[QueuedJob] = collections.deque()
        self.history: collections.deque[dict[str, Any]] = collections.deque(maxlen=20)
        self._abort_requested: bool = False
        self.queue_file: Path = (
            Path(queue_file) if queue_file else Path("models/job_queue.json")
        )

    def _save_queue_locked(self) -> None:
        """Persist pending queue to disk atomically."""
        try:
            self.queue_file.parent.mkdir(parents=True, exist_ok=True)
            data = [
                {
                    "job_id": q.job_id,
                    "action": q.action,
                    "cmd": q.cmd,
                    "created_at": q.created_at,
                }
                for q in self.queue
            ]
            _atomic_write(
                self.queue_file, json.dumps(data, indent=2).encode("utf-8")
            )
        except (OSError, TypeError) as exc:
            msg = f"[!] Warning: failed to persist job queue to disk: {exc}"
            logger.warning(msg)
            self.logs.append(msg)

    def _load_queue_from_disk(self) -> list[QueuedJob]:
        """Read pending jobs from disk if present."""
        if not self.queue_file.is_file():
            return []
        try:
            raw = self.queue_file.read_text(encoding="utf-8")
            items = json.loads(raw)
            if not isinstance(items, list):
                return []
            result: list[QueuedJob] = []
            for it in items:
                if isinstance(it, dict) and "action" in it and "cmd" in it:
                    result.append(
                        QueuedJob(
                            job_id=it.get("job_id")
                            or f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}",
                            action=it["action"],
                            cmd=it["cmd"],
                            created_at=it.get("created_at")
                            or time.strftime("%Y-%m-%d %H:%M:%S"),
                        )
                    )
            return result
        except (json.JSONDecodeError, OSError) as exc:
            msg = f"[!] Warning: failed to read persisted job queue: {exc}"
            logger.warning(msg)
            self.logs.append(msg)
            return []

    async def recover_queue(self) -> int:
        """Recover any pending jobs from disk and resume execution if idle."""
        async with self.lock:
            recovered = self._load_queue_from_disk()
            if not recovered:
                return 0
            self.queue.clear()
            for job in recovered:
                self.queue.append(job)

            self.logs.append(
                f"[*] [JobManager] Recuperados {len(recovered)} trabajo(s) "
                f"pendientes de la cola en disco."
            )

            if not self.is_running and self.queue:
                next_item = self.queue.popleft()
                self._save_queue_locked()
                await self._start_job_locked(next_item)

            return len(recovered)

    @property
    def is_running(self) -> bool:
        return self.active_job_id is not None

    async def enqueue_or_run(
        self,
        action: str,
        cmd: list[str] | list[list[str]],
        on_success: Any | None = None,
    ) -> dict[str, Any]:
        """Enqueue a job or run it immediately if no other job is active."""
        async with self.lock:
            job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            item = QueuedJob(
                job_id=job_id,
                action=action,
                cmd=cmd,
                on_success=on_success,
            )

            if not self.is_running:
                await self._start_job_locked(item)
                self._save_queue_locked()
                return {
                    "status": "started",
                    "job_id": job_id,
                    "action": action,
                    "queue_position": 0,
                    "message": f"Trabajo '{action}' iniciado en ejecución.",
                }
            else:
                self.queue.append(item)
                self._save_queue_locked()
                pos = len(self.queue)
                msg = (
                    f"Trabajo '{action}' añadido a la cola en la posición {pos}. "
                    f"Se ejecutará automáticamente al finalizar '{self.action}'."
                )
                self.logs.append(f"[*] {msg}")
                return {
                    "status": "queued",
                    "job_id": job_id,
                    "action": action,
                    "queue_position": pos,
                    "message": msg,
                }

    async def _start_job_locked(self, item: QueuedJob) -> None:
        """Internal helper to initiate execution for item. Must be called while holding lock."""
        self.action = item.action
        self.active_job_id = item.job_id
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.last_exit_code = None
        self._abort_requested = False

        asyncio.create_task(self._execute_job_task(item))

    async def _execute_job_task(self, item: QueuedJob) -> None:
        if not item.cmd:
            commands: list[list[str]] = []
        elif isinstance(item.cmd[0], list):
            commands = item.cmd  # type: ignore[assignment]
        else:
            commands = [item.cmd]  # type: ignore[list-item]

        exit_code = 0
        total_steps = len(commands)

        try:
            for step_idx, step_cmd in enumerate(commands, start=1):
                if self._abort_requested:
                    exit_code = -1
                    break

                if total_steps > 1:
                    self.logs.append(
                        f"[*] [{item.action}] Paso {step_idx}/{total_steps}: {' '.join(step_cmd)}"
                    )
                else:
                    self.logs.append(
                        f"[*] Starting job '{item.action}' ({item.job_id}): {' '.join(step_cmd)}"
                    )

                try:
                    proc = await asyncio.create_subprocess_exec(
                        *step_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                    )
                    self.process = proc
                except OSError as exc:
                    self.logs.append(f"[!] Failed to spawn process: {exc}")
                    logger.error("Failed to spawn process for job '%s': %s", item.action, exc)
                    exit_code = 1
                    break

                assert proc.stdout is not None
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    text = line.decode("utf-8", errors="replace").rstrip()
                    if text:
                        self.logs.append(text)

                step_exit = await proc.wait()
                if step_exit != 0:
                    exit_code = step_exit
                    self.logs.append(
                        f"[!] Paso {step_idx} terminó con código de error {step_exit}"
                    )
                    break

            self.last_exit_code = exit_code
            self.logs.append(
                f"[*] Job '{item.action}' terminated with exit code {exit_code}"
            )

            if exit_code == 0 and item.on_success is not None:
                try:
                    if asyncio.iscoroutinefunction(item.on_success):
                        await item.on_success()
                    else:
                        item.on_success()
                except Exception as exc:
                    self.logs.append(f"[!] Error in post-job callback: {exc}")
                    logger.exception("Error in post-job callback for '%s': %s", item.action, exc)

            self.history.append(
                {
                    "job_id": item.job_id,
                    "action": item.action,
                    "exit_code": exit_code,
                    "started_at": self.started_at,
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        finally:
            async with self.lock:
                self._abort_requested = False
                if self.queue:
                    next_item = self.queue.popleft()
                    self._save_queue_locked()
                    pos_remaining = len(self.queue)
                    self.logs.append(
                        f"[*] Dequeuing next job '{next_item.action}' "
                        f"({pos_remaining} remaining in queue)..."
                    )
                    await self._start_job_locked(next_item)
                else:
                    self.process = None
                    self.active_job_id = None
                    self._save_queue_locked()

    async def cancel_job(self, job_id_or_action: str) -> dict[str, Any]:
        """Cancel a running job or remove a pending job from the queue."""
        async with self.lock:
            if (
                self.is_running
                and (
                    self.active_job_id == job_id_or_action
                    or self.action == job_id_or_action
                )
            ):
                cancelling_action = self.action
                self._abort_requested = True
                if self.process:
                    try:
                        self.process.terminate()
                    except ProcessLookupError as exc:
                        logger.debug("Process already terminated when cancelling: %s", exc)
                self.logs.append(
                    f"[!] Trabajo activo '{cancelling_action}' cancelado a petición."
                )
                return {
                    "cancelled": True,
                    "target": cancelling_action,
                    "type": "running_terminated",
                }

            for idx, item in enumerate(self.queue):
                if item.job_id == job_id_or_action or item.action == job_id_or_action:
                    del self.queue[idx]
                    self._save_queue_locked()
                    self.logs.append(
                        f"[*] Trabajo en cola '{item.action}' ({item.job_id}) eliminado de la cola."
                    )
                    return {
                        "cancelled": True,
                        "target": item.action,
                        "type": "queued_removed",
                    }

            return {
                "cancelled": False,
                "detail": (
                    f"No se encontró ningún trabajo activo o en cola con "
                    f"identificador '{job_id_or_action}'."
                ),
            }

    async def run_command(
        self,
        action: str,
        cmd: list[str],
        on_success: Any | None = None,
    ) -> dict[str, Any]:
        """Compatibility wrapper: delegates to enqueue_or_run."""
        return await self.enqueue_or_run(action=action, cmd=cmd, on_success=on_success)

    async def run_pipeline(
        self,
        action: str,
        commands: list[list[str]],
        on_success: Any | None = None,
    ) -> dict[str, Any]:
        """Run multiple commands sequentially via FIFO queue."""
        return await self.enqueue_or_run(action=action, cmd=commands, on_success=on_success)

    def get_status(self) -> dict[str, Any]:
        queue_list = [
            {
                "job_id": q.job_id,
                "action": q.action,
                "created_at": q.created_at,
                "position": idx + 1,
            }
            for idx, q in enumerate(self.queue)
        ]
        return {
            "is_running": self.is_running,
            "action": self.action,
            "active_job_id": self.active_job_id,
            "started_at": self.started_at,
            "last_exit_code": self.last_exit_code,
            "pid": self.process.pid if self.is_running and self.process else None,
            "logs": list(self.logs),
            "queue": queue_list,
            "queue_length": len(queue_list),
            "history": list(self.history),
        }

