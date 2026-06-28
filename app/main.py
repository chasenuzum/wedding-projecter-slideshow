"""FastAPI entry point: guest uploads, projector + admin WebSockets, and the
admin review API.

Build the app with ``create_app()`` so tests can inject a stub moderator. The
module also exposes a ready ``app`` for ``uvicorn app.main:app``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from .config import Settings, get_settings
from .moderation import WeddingModerator
from .queue import ModerationQueue
from .storage import Storage, UploadError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("omaha.main")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR.parent / "static"
PUBLIC_STATES = {"approved"}  # served without an admin token


def _static_version() -> str:
    """Cache-buster token = newest static-file mtime, so editing any css/js and
    reloading the page is enough to bust the browser cache (no app restart)."""
    try:
        return str(int(max(p.stat().st_mtime for p in STATIC_DIR.rglob("*") if p.is_file())))
    except ValueError:
        return "0"


class ConnectionManager:
    """Tracks projector and admin WebSocket clients and broadcasts to each group."""

    def __init__(self) -> None:
        self.projector: set[WebSocket] = set()
        self.admin: set[WebSocket] = set()

    async def connect(self, ws: WebSocket, group: str) -> None:
        await ws.accept()
        self._group(group).add(ws)

    def disconnect(self, ws: WebSocket, group: str) -> None:
        self._group(group).discard(ws)

    def _group(self, group: str) -> set[WebSocket]:
        return self.projector if group == "projector" else self.admin

    async def _broadcast(self, clients: set[WebSocket], message: dict) -> None:
        dead = []
        for ws in list(clients):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)

    async def broadcast_projector(self, message: dict) -> None:
        await self._broadcast(self.projector, message)

    async def broadcast_admin(self, message: dict) -> None:
        await self._broadcast(self.admin, message)


def create_app(
    settings: Settings | None = None,
    moderator: WeddingModerator | None = None,
    storage: Storage | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    storage = storage or Storage(settings)
    moderator = moderator or WeddingModerator(settings)
    manager = ConnectionManager()
    mod_queue = ModerationQueue(storage, moderator, manager)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        storage.ensure_bucket()
        moderator.startup()
        mod_queue.start(settings.worker_count)
        logger.info(
            "Omaha-88 up for %s (%s) — store: s3://%s",
            settings.event_name, settings.event_date, settings.archive_bucket,
        )
        try:
            yield
        finally:
            await mod_queue.stop()

    app = FastAPI(title="Omaha-88 Wedding Slideshow", lifespan=lifespan)
    app.state.settings = settings
    app.state.storage = storage
    app.state.moderator = moderator
    app.state.queue = mod_queue
    app.state.manager = manager

    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    static_dir = BASE_DIR.parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def require_admin(token: str | None, cookie: str | None = None) -> None:
        supplied = token or cookie
        if supplied != settings.admin_token:
            raise HTTPException(status_code=403, detail="invalid admin token")

    def ctx(**extra) -> dict:
        base = {
            "event_name": settings.event_name,
            "event_date": settings.event_date,
            "venue": settings.venue,
            "public_domain": settings.public_domain,
            "static_v": _static_version(),
        }
        base.update(extra)
        return base

    # --- pages ------------------------------------------------------------
    @app.get("/")
    async def index():
        return RedirectResponse(url="/upload")

    @app.get("/upload")
    async def upload_page(request: Request):
        return templates.TemplateResponse(request, "upload.html", ctx())

    @app.get("/projector")
    async def projector_page(request: Request):
        return templates.TemplateResponse(request, "projector.html", ctx())

    @app.get("/booth")
    async def booth_page(request: Request):
        return templates.TemplateResponse(request, "booth.html", ctx())

    # --- public download gallery -----------------------------------------
    @app.get("/gallery")
    async def gallery_page(request: Request):
        photos = [r.public_dict() for r in storage.list_state("approved")]
        return templates.TemplateResponse(request, "gallery.html", ctx(photos=photos, count=len(photos)))

    @app.get("/gallery/download/{photo_id}")
    async def gallery_download(photo_id: str):
        record = storage.get(photo_id)
        if record is None or record.state != "approved":
            raise HTTPException(404)
        data = await run_in_threadpool(storage.image_bytes, record)
        return Response(
            content=data,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f'attachment; filename="{settings.archive_key_prefix}-{photo_id}.jpg"'
            },
        )

    @app.get("/gallery/all.zip")
    async def gallery_zip():
        import os
        import tempfile
        import zipfile

        records = storage.list_state("approved")
        tmp = tempfile.NamedTemporaryFile(prefix="wedding-", suffix=".zip", delete=False)
        tmp.close()

        def build() -> None:
            # JPEGs are already compressed; STORED is faster and just as small.
            with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_STORED) as zf:
                for r in records:
                    data = storage.object_bytes(r.state, r.filename)
                    if data is not None:
                        zf.writestr(r.filename, data)

        await run_in_threadpool(build)
        return FileResponse(
            tmp.name,
            media_type="application/zip",
            filename=f"{settings.archive_key_prefix}-photos.zip",
            background=BackgroundTask(os.unlink, tmp.name),
        )

    @app.get("/admin")
    async def admin_page(request: Request, token: str | None = None):
        require_admin(token, request.cookies.get("omaha_admin"))
        resp = templates.TemplateResponse(request, "admin.html", ctx(token=token or ""))
        if token:
            resp.set_cookie("omaha_admin", token, httponly=True, samesite="lax")
        return resp

    # --- uploads ----------------------------------------------------------
    @app.post("/upload")
    async def upload(
        files: list[UploadFile] = File(default=[]),
        file: UploadFile | None = File(default=None),
        source: str = Form("guest"),
    ):
        # Accept both "files" (multi, guest web) and "file" (single, booth /
        # webhook clients) so any uploader works.
        items = list(files)
        if file is not None:
            items.append(file)
        if not items:
            return JSONResponse(
                {"accepted": [], "rejected": [{"error": "no file provided"}]},
                status_code=400,
            )
        accepted, rejected = [], []
        for upload_file in items:
            try:
                raw = await upload_file.read()
                record = storage.save_upload(raw, upload_file.content_type, source=source)
                await mod_queue.enqueue(record)
                accepted.append(record.id)
            except UploadError as exc:
                rejected.append({"filename": upload_file.filename, "error": str(exc)})
            except Exception as exc:  # pragma: no cover
                logger.exception("upload failed")
                rejected.append({"filename": upload_file.filename, "error": "internal error"})
        status = 202 if accepted else 400
        return JSONResponse({"accepted": accepted, "rejected": rejected}, status_code=status)

    # --- media serving (proxied from object storage) ----------------------
    @app.get("/media/{state}/{filename}")
    async def media(state: str, filename: str, request: Request, token: str | None = None):
        if state not in ("approved", "review", "rejected", "uploads"):
            raise HTTPException(404)
        if "/" in filename or ".." in filename:
            raise HTTPException(404)
        if state not in PUBLIC_STATES:
            require_admin(token, request.cookies.get("omaha_admin"))
        data = await run_in_threadpool(storage.object_bytes, state, filename)
        if data is None:
            raise HTTPException(404)
        return Response(
            content=data,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=300"},
        )

    # --- admin decisions --------------------------------------------------
    @app.post("/admin/decision")
    async def admin_decision(request: Request, token: str | None = None):
        require_admin(token, request.cookies.get("omaha_admin"))
        body = await request.json()
        photo_id = body.get("id")
        action = body.get("action")
        if action not in ("approve", "reject"):
            raise HTTPException(400, "action must be approve or reject")
        record = storage.get(photo_id)
        if record is None or record.state != "review":
            raise HTTPException(404, "photo not in review queue")

        from datetime import datetime, timezone

        decided_at = datetime.now(timezone.utc).isoformat()
        if action == "approve":
            record = storage.move(record, "approved", decided_at=decided_at, decided_by="admin")
            await manager.broadcast_projector({"type": "new_photo", "photo": record.public_dict()})
        else:
            record = storage.move(record, "rejected", decided_at=decided_at, decided_by="admin")
        await manager.broadcast_admin({"type": "resolved", "id": photo_id, "action": action})
        return {"ok": True, "id": photo_id, "state": record.state}

    # --- health -----------------------------------------------------------
    @app.get("/healthz")
    async def healthz():
        return {
            "status": "ok",
            "event": settings.event_name,
            "moderation": moderator.health(),
            "storage": {
                "backend": "s3",
                "bucket": settings.archive_bucket or None,
                "endpoint": settings.archive_endpoint_url or None,
            },
            "queue_depth": mod_queue.depth,
            "counts": {s: storage.count_state(s) for s in ("approved", "review", "rejected")},
        }

    # --- websockets -------------------------------------------------------
    @app.websocket("/ws/projector")
    async def ws_projector(ws: WebSocket):
        await manager.connect(ws, "projector")
        try:
            gallery = [r.public_dict() for r in storage.list_state("approved")[:60]]
            await ws.send_json({"type": "gallery", "photos": gallery})
            while True:
                await ws.receive_text()  # keepalive / client pings
        except WebSocketDisconnect:
            pass
        finally:
            manager.disconnect(ws, "projector")

    @app.websocket("/ws/admin")
    async def ws_admin(ws: WebSocket, token: str | None = None):
        if (token or ws.cookies.get("omaha_admin")) != settings.admin_token:
            await ws.close(code=4403)
            return
        await manager.connect(ws, "admin")
        try:
            queue = [r.public_dict() for r in storage.list_state("review")]
            await ws.send_json({"type": "queue", "photos": queue})
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            manager.disconnect(ws, "admin")

    return app


app = create_app()
