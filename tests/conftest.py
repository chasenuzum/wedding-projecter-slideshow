import io

import pytest
from moto import mock_aws
from PIL import Image

from app.config import Settings
from app.moderation import Backend, WeddingModerator
from app.storage import Storage


def make_image_bytes(color=(120, 160, 200), size=(640, 480), fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        models_dir=tmp_path / "models",
        admin_token="testtoken",
        openrouter_api_key="",  # keep cloud fallback disabled in tests
        use_nsfw_classifier=False,  # don't download a model during tests
        worker_count=1,
        # In-memory S3 (moto). Empty endpoint => default AWS endpoint, which
        # moto intercepts. Dummy region/creds.
        archive_bucket="test-bucket",
        archive_endpoint_url="",
        archive_region="us-east-1",
        archive_access_key_id="testing",
        archive_secret_access_key="testing",
    )


@pytest.fixture(autouse=True)
def s3(settings):
    """Mock S3 for every test and create the bucket."""
    with mock_aws():
        storage = Storage(settings)
        storage.ensure_bucket()
        yield storage


@pytest.fixture
def storage(settings, s3) -> Storage:
    return s3


class StubBackend(Backend):
    """Deterministic backend for tests."""

    def __init__(
        self, name: str, reply: str | None, available: bool = True, structured: bool = False
    ):
        self._name = name
        self._reply = reply
        self._available = available
        self.structured = structured

    @property
    def name(self) -> str:
        return self._name

    @property
    def available(self) -> bool:
        return self._available

    def classify(self, image, jpeg_bytes):
        if self._reply is None:
            raise RuntimeError("stub backend forced failure")
        return self._reply


class StubNSFW:
    """Deterministic NSFW classifier for tests (no model download)."""

    name = "nsfw"

    def __init__(self, score: float, available: bool = True):
        self._score = score
        self._available = available

    @property
    def available(self) -> bool:
        return self._available

    def load(self) -> bool:
        return self._available

    def score(self, image) -> float:
        return self._score


@pytest.fixture
def safe_moderator(settings):
    return WeddingModerator(settings, backends=[StubBackend("stub", "no")])  # "no" => SAFE
