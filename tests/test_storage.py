import pytest

from app.storage import UploadError
from tests.conftest import make_image_bytes


def test_save_upload_normalizes_and_records(storage):
    record = storage.save_upload(make_image_bytes(), "image/jpeg", source="guest")

    assert record.state == "uploads"
    assert record.filename.endswith(".jpg")
    assert record.width > 0 and record.height > 0
    # image + metadata are both readable back from the store
    assert storage.image_bytes(record)[:2] == b"\xff\xd8"  # JPEG magic
    assert storage.get(record.id) is not None


def test_save_upload_rejects_oversize(settings, storage):
    settings.max_upload_mb = 0  # everything is "too large"
    with pytest.raises(UploadError):
        storage.save_upload(make_image_bytes(), "image/jpeg")


def test_save_upload_rejects_bad_type(storage):
    with pytest.raises(UploadError):
        storage.save_upload(b"not an image", "application/pdf")


def test_state_transitions(storage):
    record = storage.save_upload(make_image_bytes(), "image/jpeg")

    moved = storage.move(record, "review", verdict="UNSAFE")
    assert moved.state == "review"
    assert storage.image_bytes(moved)  # present in review
    # old location is gone
    assert storage.object_bytes("uploads", record.filename) is None
    assert storage.get(record.id).state == "review"

    approved = storage.move(moved, "approved", decided_by="admin")
    assert approved.state == "approved"
    assert [r.id for r in storage.list_state("approved")] == [record.id]
    assert storage.list_state("review") == []
    assert storage.count_state("approved") == 1


def test_get_returns_none_for_unknown(storage):
    assert storage.get("does-not-exist") is None
