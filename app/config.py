"""Runtime configuration.

All values are overridable via environment variables (prefix ``SLIDESHOW_``) or a
local ``.env`` file, so the same code runs on a dev laptop and at the venue.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = parent of the ``app`` package.
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SLIDESHOW_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Event identity (override these in .env) --------------------------
    event_name: str = "Our Wedding"
    event_date: str = ""
    venue: str = ""
    public_domain: str = "http://localhost:8000"

    # --- Server -----------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Storage ----------------------------------------------------------
    # Photos live entirely in object storage (see SLIDESHOW_ARCHIVE_* below); the
    # only local path is the model cache.
    models_dir: Path = BASE_DIR / "models"
    max_upload_mb: int = 25
    allowed_types: tuple[str, ...] = (
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/heic",
        "image/heif",
    )

    # --- Moderation: local Moondream 3 (primary) --------------------------
    # Photon model name (NOT an HF repo id — a "/" is parsed as a fine-tune
    # adapter). Weights are fetched/cached by the moondream/kestrel runtime.
    model_id: str = "moondream3-preview"
    # Worker concurrency. MLX inference is single-stream on the GPU, so 1 is the
    # safe default; bump only if you have measured headroom.
    worker_count: int = 1
    # DETECTION framing (not subjective judgment) — VLMs reliably answer "does
    # this contain X?", but over-refuse "is this appropriate?". So we ask what's
    # present and treat "yes" as UNSAFE.
    moderation_question: str = (
        "Look carefully. Does this photo contain any nudity, exposed breasts, "
        "genitals or buttocks, underwear/lingerie, sexual activity, an obscene "
        "hand gesture, graphic violence, or gore? If you are not sure, answer yes. "
        "Answer with only one word: yes or no."
    )
    # When True, an affirmative ("yes") answer means the photo is UNSAFE.
    moderation_yes_means_unsafe: bool = True
    # Moondream's reasoning mode is slower but more accurate for nuanced calls
    # like nudity. Worth the few hundred ms for the one thing we must not miss.
    moderation_reasoning: bool = True

    # --- Dedicated NSFW classifier (primary nudity gate) ------------------
    # A purpose-built image classifier is far more reliable for nudity than a
    # general VLM. Runs locally on the Apple GPU. If it flags (score >=
    # threshold) the photo is held regardless of what Moondream says.
    use_nsfw_classifier: bool = True
    nsfw_model_id: str = "Marqo/nsfw-image-detection-384"
    nsfw_threshold: float = 0.5

    # --- Moderation: OpenRouter Gemini Flash (cloud fallback) -------------
    # Used only when the local model is unavailable. Empty key disables it.
    openrouter_api_key: str = ""
    openrouter_model: str = "google/gemini-2.5-flash"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # --- Admin review dashboard ------------------------------------------
    admin_token: str = "change-me-before-the-wedding"

    # --- Object storage: the PRIMARY photo store (S3-compatible) ----------
    # ALL photos live here (uploads/review/approved/rejected as object keys).
    # Required. Works with Cloudflare R2, MinIO, AWS S3, Backblaze B2.
    #   R2:    archive_endpoint_url=https://<accountid>.r2.cloudflarestorage.com,
    #          archive_region="auto"
    #   MinIO: archive_endpoint_url=http://localhost:9000, archive_region="us-east-1"
    archive_bucket: str = ""
    archive_endpoint_url: str = ""  # blank = real AWS S3
    archive_access_key_id: str = ""
    archive_secret_access_key: str = ""
    archive_region: str = "auto"
    archive_prefix: str = ""  # key prefix; defaults to a slug of the event name

    @property
    def archive_enabled(self) -> bool:
        return bool(
            self.archive_bucket
            and self.archive_access_key_id
            and self.archive_secret_access_key
        )

    @property
    def archive_key_prefix(self) -> str:
        if self.archive_prefix:
            return self.archive_prefix.strip("/")
        slug = "".join(c if c.isalnum() else "-" for c in self.event_name.lower())
        return "-".join(filter(None, slug.split("-")))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def openrouter_enabled(self) -> bool:
        return bool(self.openrouter_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
