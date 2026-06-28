import io
import zipfile

from fastapi.testclient import TestClient

from app.main import create_app
from app.moderation import WeddingModerator
from tests.conftest import make_image_bytes


def _approve_one(storage):
    """Put a single approved photo in the store."""
    rec = storage.save_upload(make_image_bytes(), "image/jpeg")
    storage.move(rec, "approved", verdict="SAFE")
    return rec


def test_gallery_lists_and_downloads(settings, storage):
    rec = _approve_one(storage)
    app = create_app(settings, moderator=WeddingModerator(settings, backends=[]), storage=storage)
    with TestClient(app) as client:
        page = client.get("/gallery")
        assert page.status_code == 200
        assert rec.id in page.text  # the photo appears in the grid

        dl = client.get(f"/gallery/download/{rec.id}")
        assert dl.status_code == 200
        assert dl.headers["content-type"] == "image/jpeg"
        assert "attachment" in dl.headers.get("content-disposition", "")
        assert dl.content[:2] == b"\xff\xd8"  # real JPEG bytes


def test_gallery_zip_contains_approved(settings, storage):
    _approve_one(storage)
    _approve_one(storage)
    app = create_app(settings, moderator=WeddingModerator(settings, backends=[]), storage=storage)
    with TestClient(app) as client:
        resp = client.get("/gallery/all.zip")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            assert len(zf.namelist()) == 2


def test_media_route_serves_approved(settings, storage):
    rec = _approve_one(storage)
    app = create_app(settings, moderator=WeddingModerator(settings, backends=[]), storage=storage)
    with TestClient(app) as client:
        resp = client.get(f"/media/approved/{rec.filename}")
        assert resp.status_code == 200
        assert resp.content[:2] == b"\xff\xd8"


def test_media_review_requires_admin(settings, storage):
    rec = storage.save_upload(make_image_bytes(), "image/jpeg")
    storage.move(rec, "review", verdict="UNSAFE")
    app = create_app(settings, moderator=WeddingModerator(settings, backends=[]), storage=storage)
    with TestClient(app) as client:
        assert client.get(f"/media/review/{rec.filename}").status_code == 403
        ok = client.get(f"/media/review/{rec.filename}?token=testtoken")
        assert ok.status_code == 200


def test_gallery_download_404_for_unknown(settings, storage):
    app = create_app(settings, moderator=WeddingModerator(settings, backends=[]), storage=storage)
    with TestClient(app) as client:
        assert client.get("/gallery/download/nope").status_code == 404
