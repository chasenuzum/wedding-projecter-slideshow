"""Background moderation queue.

Uploads return immediately; this decouples the slow MLX inference from the HTTP
request. Workers pull photo records off an ``asyncio.Queue``, fetch the image
bytes from object storage, run the (blocking) moderation in a thread executor,
move the object to its destination state, and broadcast the outcome.

    SAFE             -> approved/ -> projector clients
    UNSAFE / UNKNOWN -> review/   -> admin clients
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from .moderation import SAFE, ModerationResult, WeddingModerator
from .storage import PhotoRecord, Storage

logger = logging.getLogger("omaha.queue")


class ModerationQueue:
    def __init__(self, storage: Storage, moderator: WeddingModerator, manager):
        self.storage = storage
        self.moderator = moderator
        self.manager = manager  # ConnectionManager (broadcast_projector / broadcast_admin)
        self._queue: asyncio.Queue[PhotoRecord] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._running = False

    def start(self, count: int) -> None:
        self._running = True
        for i in range(max(1, count)):
            self._workers.append(asyncio.create_task(self._run(i), name=f"moderation-worker-{i}"))
        logger.info("started %d moderation worker(s)", len(self._workers))

    async def stop(self) -> None:
        self._running = False
        for task in self._workers:
            task.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def enqueue(self, record: PhotoRecord) -> None:
        await self._queue.put(record)

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    async def _run(self, worker_id: int) -> None:
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                record = await self._queue.get()
            except asyncio.CancelledError:
                break
            try:
                jpeg = await loop.run_in_executor(None, self.storage.image_bytes, record)
                result: ModerationResult = await loop.run_in_executor(
                    None, self.moderator.moderate, jpeg
                )
                await self._handle(record, result)
            except asyncio.CancelledError:
                break
            except Exception:  # pragma: no cover - keep the worker alive
                logger.exception("worker %d failed on %s", worker_id, record.id)
            finally:
                self._queue.task_done()

    async def _handle(self, record: PhotoRecord, result: ModerationResult) -> None:
        updates = dict(
            verdict=result.verdict,
            moderation_source=result.source,
            reason=result.reason,
            latency_ms=round(result.latency_ms, 1),
            moderated_at=datetime.now(timezone.utc).isoformat(),
        )
        if result.verdict == SAFE:
            record = self.storage.move(record, "approved", **updates)
            await self.manager.broadcast_projector({"type": "new_photo", "photo": record.public_dict()})
            logger.info("APPROVED %s via %s (%.0fms)", record.id, result.source, result.latency_ms)
        else:
            record = self.storage.move(record, "review", **updates)
            await self.manager.broadcast_admin({"type": "new_review", "photo": record.public_dict()})
            logger.info(
                "HELD %s (%s) via %s (%.0fms) :: %s",
                record.id, result.verdict, result.source, result.latency_ms, result.reason,
            )
