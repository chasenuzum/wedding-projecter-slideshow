import time

from fastapi.testclient import TestClient

from app.main import create_app
from app.moderation import WeddingModerator
from tests.conftest import StubBackend, make_image_bytes


def _wait_for(client, predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        counts = client.get("/healthz").json()["counts"]
        if predicate(counts):
            return counts
        time.sleep(0.05)
    raise AssertionError(f"condition not met; last counts={counts}")


def test_upload_safe_photo_reaches_approved(settings):
    moderator = WeddingModerator(settings, backends=[StubBackend("stub", "SAFE")])
    app = create_app(settings, moderator=moderator)
    with TestClient(app) as client:
        resp = client.post("/upload", files={"files": ("a.jpg", make_image_bytes(), "image/jpeg")})
        assert resp.status_code == 202
        assert len(resp.json()["accepted"]) == 1

        counts = _wait_for(client, lambda c: c["approved"] >= 1)
        assert counts["approved"] == 1
        assert counts["review"] == 0


def test_upload_unknown_photo_held_for_review(settings):
    # no available backend -> UNKNOWN -> review queue
    moderator = WeddingModerator(settings, backends=[StubBackend("stub", "SAFE", available=False)])
    app = create_app(settings, moderator=moderator)
    with TestClient(app) as client:
        resp = client.post("/upload", files={"files": ("a.jpg", make_image_bytes(), "image/jpeg")})
        assert resp.status_code == 202
        _wait_for(client, lambda c: c["review"] >= 1)


def test_admin_requires_token(settings):
    app = create_app(settings, moderator=WeddingModerator(settings, backends=[]))
    with TestClient(app) as client:
        assert client.get("/admin").status_code == 403
        assert client.get("/admin?token=wrong").status_code == 403
        assert client.get("/admin?token=testtoken").status_code == 200


def test_admin_decision_approves_held_photo(settings):
    moderator = WeddingModerator(settings, backends=[StubBackend("stub", "UNSAFE")])
    app = create_app(settings, moderator=moderator)
    with TestClient(app) as client:
        resp = client.post("/upload", files={"files": ("a.jpg", make_image_bytes(), "image/jpeg")})
        photo_id = resp.json()["accepted"][0]
        _wait_for(client, lambda c: c["review"] >= 1)

        decision = client.post(
            "/admin/decision?token=testtoken",
            json={"id": photo_id, "action": "approve"},
        )
        assert decision.status_code == 200
        assert decision.json()["state"] == "approved"
        _wait_for(client, lambda c: c["approved"] >= 1 and c["review"] == 0)


def test_upload_accepts_singular_file_field(settings):
    # Booth / webhook clients post field name "file" (singular).
    moderator = WeddingModerator(settings, backends=[StubBackend("stub", "no")])  # "no" => SAFE
    app = create_app(settings, moderator=moderator)
    with TestClient(app) as client:
        resp = client.post(
            "/upload",
            data={"source": "booth"},
            files={"file": ("booth.jpg", make_image_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 202
        assert len(resp.json()["accepted"]) == 1
        _wait_for(client, lambda c: c["approved"] >= 1)


def test_upload_rejects_non_image(settings):
    app = create_app(settings, moderator=WeddingModerator(settings, backends=[]))
    with TestClient(app) as client:
        resp = client.post("/upload", files={"files": ("a.txt", b"hello", "text/plain")})
        assert resp.status_code == 400
        assert resp.json()["rejected"]
