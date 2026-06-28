"""VLM moderation harness.

``WeddingModerator`` runs an ordered chain of backends and returns the first
usable verdict:

    1. Moondream 3 locally (MLX/Metal) — primary, on-device, sub-second.
    2. OpenRouter Gemini Flash — cloud fallback when the local model is down.
    3. UNKNOWN — both unavailable; the worker holds the photo for human review.

Each backend answers with the word SAFE or UNSAFE; we parse the reply. A verdict
that can't be parsed is treated as UNSAFE (fail toward review, never toward the
projector).
"""

from __future__ import annotations

import base64
import io
import logging
import re
import time
from dataclasses import dataclass

from PIL import Image

from .config import Settings

logger = logging.getLogger("omaha.moderation")

SAFE = "SAFE"
UNSAFE = "UNSAFE"
UNKNOWN = "UNKNOWN"


@dataclass
class ModerationResult:
    verdict: str  # SAFE | UNSAFE | UNKNOWN
    source: str  # backend name, or "none"
    reason: str
    latency_ms: float
    raw: str = ""


_YES_WORDS = {"yes", "y", "yeah", "yep", "true"}
_NO_WORDS = {"no", "n", "nope", "false"}


def parse_verdict(answer: str, yes_means_unsafe: bool = True) -> str:
    """Map a free-text model reply to SAFE / UNSAFE.

    The detection question asks for a single word (yes/no), but small VLMs often
    pad with a sentence. We key on the FIRST meaningful word ("No, just a person"
    -> safe; "Yes, nudity" -> unsafe), with ``yes_means_unsafe`` setting the
    polarity. Explicit "safe"/"unsafe" words always win (used by stubs/other
    models). Unparseable replies fail toward UNSAFE so anything ambiguous lands
    in the review queue rather than on the projector.
    """
    text = (answer or "").strip().lower()
    tokens = re.findall(r"[a-z]+", text)
    if not tokens:
        return UNSAFE
    first = tokens[0]
    # Explicit verdict words override polarity.
    if first in ("unsafe", "nsfw"):
        return UNSAFE
    if first in ("safe", "sfw"):
        return SAFE
    if first in _YES_WORDS:
        return UNSAFE if yes_means_unsafe else SAFE
    if first in _NO_WORDS:
        return SAFE if yes_means_unsafe else UNSAFE
    # Fallback: explicit verdict word anywhere in the reply.
    if "unsafe" in text or "nsfw" in text:
        return UNSAFE
    if "safe" in text or "sfw" in text:
        return SAFE
    return UNSAFE


class Backend:
    name = "base"

    @property
    def available(self) -> bool:
        return False

    def classify(self, image: Image.Image, jpeg_bytes: bytes) -> str:
        """Return the raw model reply string, or raise on failure."""
        raise NotImplementedError


class MoondreamLocalBackend(Backend):
    name = "moondream"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._loaded = False

    @property
    def available(self) -> bool:
        return self._loaded and self._model is not None

    def load(self) -> bool:
        """Load Moondream 3 locally (Photon/MPS) and warm the kernels.

        ``local=True`` is required, otherwise ``md.vl()`` builds a *cloud* client
        that 401s without an API key. ``model`` is a Photon model name like
        ``moondream3-preview`` (note: a ``/`` in the name is parsed as a
        fine-tune adapter, so HF repo ids like ``moondream/md3p-int4`` are wrong
        here). Returns True on success.
        """
        try:
            import moondream as md
        except Exception as exc:  # pragma: no cover - depends on optional extra
            logger.warning("moondream package not installed (%s); skipping local backend", exc)
            return False

        try:
            self._model = md.vl(local=True, model=self.settings.model_id)
        except Exception as exc:  # missing kestrel, no accelerator, bad model id…
            logger.warning(
                "could not load local Moondream (model=%s): %s; local backend disabled",
                self.settings.model_id, exc,
            )
            return False

        self._loaded = True
        logger.info("Moondream 3 loaded locally (model=%s)", self.settings.model_id)
        self._warmup()
        return True

    def _warmup(self) -> None:
        try:
            dummy = Image.new("RGB", (64, 64), (128, 128, 128))
            t0 = time.perf_counter()
            self._query(dummy)
            logger.info("Moondream warmup done in %.0f ms", (time.perf_counter() - t0) * 1000)
        except Exception as exc:  # pragma: no cover
            logger.warning("Moondream warmup failed: %s", exc)

    def _query(self, image: Image.Image) -> str:
        # reasoning=True is slower but more accurate for nuanced moderation calls.
        try:
            result = self._model.query(
                image, self.settings.moderation_question,
                reasoning=self.settings.moderation_reasoning,
            )
        except TypeError:
            # Older signatures without the reasoning kwarg.
            result = self._model.query(image, self.settings.moderation_question)
        if isinstance(result, dict):
            return str(result.get("answer", ""))
        return str(result)

    def classify(self, image: Image.Image, jpeg_bytes: bytes) -> str:
        return self._query(image)


class OpenRouterGeminiBackend(Backend):
    name = "openrouter"

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def available(self) -> bool:
        return self.settings.openrouter_enabled

    def classify(self, image: Image.Image, jpeg_bytes: bytes) -> str:
        import httpx

        b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        payload = {
            "model": self.settings.openrouter_model,
            "max_tokens": 8,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.settings.moderation_question},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "HTTP-Referer": self.settings.public_domain,
            "X-Title": "Omaha-88 Wedding Moderation",
        }
        url = f"{self.settings.openrouter_base_url.rstrip('/')}/chat/completions"
        with httpx.Client(timeout=20.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]


_NSFW_LABELS = {"nsfw", "porn", "sexy", "hentai", "explicit"}


class NSFWClassifier:
    """Dedicated local NSFW image classifier — the primary nudity gate.

    Purpose-built classifiers are far more reliable for nudity than a general
    VLM. Runs on the Apple GPU via transformers. Lazy-imports so the web harness
    and tests don't need transformers/torch installed.
    """

    name = "nsfw"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipe = None

    @property
    def available(self) -> bool:
        return self._pipe is not None

    def load(self) -> bool:
        if not self.settings.use_nsfw_classifier:
            logger.info("NSFW classifier disabled")
            return False
        try:
            import torch
            from transformers import pipeline

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._pipe = pipeline(
                "image-classification",
                model=self.settings.nsfw_model_id,
                device=device,
            )
            self._pipe(Image.new("RGB", (64, 64), (128, 128, 128)))  # warmup
            logger.info(
                "NSFW classifier loaded (%s on %s)", self.settings.nsfw_model_id, device
            )
            return True
        except Exception as exc:  # pragma: no cover - optional dep / load failure
            logger.warning("could not load NSFW classifier: %s", exc)
            self._pipe = None
            return False

    def score(self, image: Image.Image) -> float:
        """Return P(unsafe) — sum of scores over NSFW-ish labels."""
        preds = self._pipe(image)
        return float(sum(p["score"] for p in preds if p["label"].lower() in _NSFW_LABELS))


class WeddingModerator:
    def __init__(
        self,
        settings: Settings,
        backends: list[Backend] | None = None,
        nsfw: NSFWClassifier | None = None,
    ):
        self.settings = settings
        if backends is None:
            backends = [
                MoondreamLocalBackend(settings),
                OpenRouterGeminiBackend(settings),
            ]
        self.backends = backends
        # Dedicated nudity gate, run before the VLM. Default-constructed but only
        # active once loaded at startup (so tests stay dependency-free).
        self.nsfw = nsfw if nsfw is not None else NSFWClassifier(settings)

    def startup(self) -> None:
        """Load the NSFW gate and any backends that need it (once, at startup)."""
        self.nsfw.load()
        for backend in self.backends:
            loader = getattr(backend, "load", None)
            if callable(loader):
                loader()
        active = [b.name for b in self.backends if b.available]
        logger.info(
            "Moderation active: nsfw=%s backends=%s",
            self.nsfw.available, active or "NONE",
        )

    def health(self) -> dict:
        return {
            "nsfw_classifier": self.nsfw.available,
            "backends": [
                {"name": b.name, "available": b.available} for b in self.backends
            ],
            "primary": next((b.name for b in self.backends if b.available), None),
        }

    def moderate(self, jpeg_bytes: bytes) -> ModerationResult:
        image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")

        # 1. Dedicated NSFW gate — the reliable nudity check, runs first.
        if self.nsfw.available:
            t0 = time.perf_counter()
            try:
                score = self.nsfw.score(image)
                latency_ms = (time.perf_counter() - t0) * 1000
                if score >= self.settings.nsfw_threshold:
                    reason = f"nsfw classifier {score:.2f} >= {self.settings.nsfw_threshold}"
                    return ModerationResult(UNSAFE, "nsfw", reason, latency_ms, raw=f"score={score:.3f}")
            except Exception as exc:
                logger.warning("NSFW classifier failed: %s", exc)

        # 2. VLM chain for everything else (gestures, violence, …) + backup.
        last_error = "no backend available"
        for backend in self.backends:
            if not backend.available:
                continue
            t0 = time.perf_counter()
            try:
                raw = backend.classify(image, jpeg_bytes)
            except Exception as exc:
                last_error = f"{backend.name}: {exc}"
                logger.warning("backend %s failed: %s", backend.name, exc)
                continue
            latency_ms = (time.perf_counter() - t0) * 1000
            verdict = parse_verdict(raw, self.settings.moderation_yes_means_unsafe)
            snippet = " ".join(str(raw).split())[:120]
            reason = f'{backend.name}: "{snippet}" -> {verdict}'
            return ModerationResult(verdict, backend.name, reason, latency_ms, raw=str(raw)[:200])

        # Nothing could classify it: hold for human review.
        return ModerationResult(UNKNOWN, "none", last_error, 0.0)
