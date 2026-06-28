"""S3-backed photo storage.

Photos live entirely in object storage (MinIO / S3) — there is no local data
directory. Each photo is an object ``{state}/{id}.jpg`` with a sibling
``{state}/{id}.json`` metadata object. State changes (uploads -> review ->
approved / rejected) are a server-side copy + delete (S3 has no rename).

The bucket/credentials come from the OMAHA_ARCHIVE_* settings (the same ones the
old archiver used — now the primary store).
"""

from __future__ import annotations

import io
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from PIL import Image, ImageOps, UnidentifiedImageError

# Register the HEIC/HEIF opener so iPhone / iPad photos decode.
try:  # pragma: no cover - exercised on real installs
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover - pillow-heif optional at import time
    pass

from .config import Settings

VALID_STATES = ("uploads", "review", "approved", "rejected")

# Longest edge for the stored/projected image. Keeps the projector snappy and
# inference fast without visible quality loss on a screen.
MAX_EDGE = 2048
JPEG_QUALITY = 88


class UploadError(ValueError):
    """Raised when an upload is rejected before it ever reaches moderation."""


class StorageError(RuntimeError):
    """Raised when the object store is misconfigured or unreachable."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PhotoRecord:
    id: str
    state: str
    filename: str
    source: str = "guest"  # "guest" | "booth"
    created_at: str = field(default_factory=_utcnow)
    width: int = 0
    height: int = 0
    # Moderation outcome.
    verdict: str | None = None  # "SAFE" | "UNSAFE" | "UNKNOWN"
    moderation_source: str | None = None  # "nsfw" | "moondream" | "openrouter" | ...
    reason: str | None = None
    latency_ms: float | None = None
    moderated_at: str | None = None
    # Human review outcome (admin dashboard).
    decided_at: str | None = None
    decided_by: str | None = None

    def public_dict(self) -> dict:
        """Subset safe to send to projector / admin clients."""
        return {
            "id": self.id,
            "state": self.state,
            "url": f"/media/{self.state}/{self.filename}",
            "source": self.source,
            "created_at": self.created_at,
            "width": self.width,
            "height": self.height,
            "verdict": self.verdict,
            "reason": self.reason,
            "moderation_source": self.moderation_source,
        }


class Storage:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.bucket = settings.archive_bucket
        self._client = None  # built lazily so import/construction needs no bucket

    # --- client / bucket --------------------------------------------------
    @property
    def client(self):
        if self._client is None:
            if not self.bucket:
                raise StorageError(
                    "Object storage is required: set OMAHA_ARCHIVE_BUCKET (+ endpoint "
                    "and credentials), e.g. Cloudflare R2 or `docker compose up -d` MinIO."
                )
            self._client = self._build_client()
        return self._client

    def _build_client(self):
        import boto3
        from botocore.config import Config

        return boto3.client(
            "s3",
            endpoint_url=self.settings.archive_endpoint_url or None,
            region_name=self.settings.archive_region or None,
            aws_access_key_id=self.settings.archive_access_key_id or None,
            aws_secret_access_key=self.settings.archive_secret_access_key or None,
            config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
        )

    def ensure_bucket(self) -> None:
        """Create the bucket if it doesn't exist (idempotent)."""
        from botocore.exceptions import ClientError

        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket)

    # --- keys -------------------------------------------------------------
    def _img_key(self, state: str, filename: str) -> str:
        return f"{state}/{filename}"

    def _meta_key(self, state: str, photo_id: str) -> str:
        return f"{state}/{photo_id}.json"

    # --- create -----------------------------------------------------------
    def save_upload(self, raw: bytes, content_type: str | None, source: str = "guest") -> PhotoRecord:
        if len(raw) == 0:
            raise UploadError("empty file")
        if len(raw) > self.settings.max_upload_bytes:
            raise UploadError(
                f"file too large ({len(raw) // (1024 * 1024)}MB > "
                f"{self.settings.max_upload_mb}MB)"
            )
        if content_type and content_type not in self.settings.allowed_types:
            raise UploadError(f"unsupported content type: {content_type}")

        try:
            img = Image.open(io.BytesIO(raw))
            img = ImageOps.exif_transpose(img)  # honor phone orientation
            img = img.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise UploadError(f"could not decode image: {exc}") from exc

        img.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        jpeg = buf.getvalue()

        photo_id = uuid.uuid4().hex
        record = PhotoRecord(
            id=photo_id,
            state="uploads",
            filename=f"{photo_id}.jpg",
            source=source,
            width=img.width,
            height=img.height,
        )
        self._put_image(record.state, record.filename, jpeg)
        self._put_meta(record)
        return record

    # --- objects ----------------------------------------------------------
    def _put_image(self, state: str, filename: str, data: bytes) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._img_key(state, filename),
            Body=data,
            ContentType="image/jpeg",
        )

    def _put_meta(self, record: PhotoRecord) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._meta_key(record.state, record.id),
            Body=json.dumps(asdict(record), indent=2).encode(),
            ContentType="application/json",
        )

    def _get_meta(self, state: str, photo_id: str) -> PhotoRecord | None:
        from botocore.exceptions import ClientError

        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=self._meta_key(state, photo_id))
        except ClientError:
            return None
        return PhotoRecord(**json.loads(obj["Body"].read()))

    def object_bytes(self, state: str, filename: str) -> bytes | None:
        """Raw image bytes for a state/filename, or None if missing."""
        from botocore.exceptions import ClientError

        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=self._img_key(state, filename))
        except ClientError:
            return None
        return obj["Body"].read()

    def image_bytes(self, record: PhotoRecord) -> bytes:
        data = self.object_bytes(record.state, record.filename)
        if data is None:
            raise StorageError(f"image missing for {record.id} in {record.state}")
        return data

    # --- transitions ------------------------------------------------------
    def move(self, record: PhotoRecord, new_state: str, **updates) -> PhotoRecord:
        if new_state not in VALID_STATES:
            raise ValueError(f"unknown state: {new_state}")
        old_state = record.state
        old_img = self._img_key(old_state, record.filename)
        old_meta = self._meta_key(old_state, record.id)

        # Copy the image to the new prefix.
        self.client.copy_object(
            Bucket=self.bucket,
            CopySource={"Bucket": self.bucket, "Key": old_img},
            Key=self._img_key(new_state, record.filename),
            ContentType="image/jpeg",
            MetadataDirective="REPLACE",
        )
        # Update the record + write its metadata in the new prefix.
        record.state = new_state
        for key, value in updates.items():
            setattr(record, key, value)
        self._put_meta(record)

        # Delete the old objects.
        if old_state != new_state:
            self.client.delete_object(Bucket=self.bucket, Key=old_img)
            self.client.delete_object(Bucket=self.bucket, Key=old_meta)
        return record

    # --- queries ----------------------------------------------------------
    def get(self, photo_id: str) -> PhotoRecord | None:
        for state in VALID_STATES:
            record = self._get_meta(state, photo_id)
            if record is not None:
                return record
        return None

    def _iter_keys(self, prefix: str):
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"]

    def count_state(self, state: str) -> int:
        """Cheap count of photos in a state (counts .jpg keys, no metadata reads)."""
        return sum(1 for k in self._iter_keys(f"{state}/") if k.endswith(".jpg"))

    def list_state(self, state: str, *, newest_first: bool = True) -> list[PhotoRecord]:
        records: list[PhotoRecord] = []
        for key in self._iter_keys(f"{state}/"):
            if not key.endswith(".json"):
                continue
            photo_id = key.rsplit("/", 1)[-1][: -len(".json")]
            record = self._get_meta(state, photo_id)
            if record is not None:
                records.append(record)
        records.sort(key=lambda r: r.created_at, reverse=newest_first)
        return records
