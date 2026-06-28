"""Re-run moderation against an image — a tuning tool.

    uv run python scripts/test_moderation.py <path-or-photo-id> [more...]
    uv run python scripts/test_moderation.py nudity.jpg

Prints the raw NSFW score, Moondream's raw answer, and the final verdict for
each image so you can pick OMAHA_NSFW_THRESHOLD with real numbers. A local file
path works without object storage; a photo id is fetched from the S3 store.

Tip: stop the running app first, or this loads a second copy of the models.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.moderation import WeddingModerator  # noqa: E402


def load_bytes(settings, arg: str) -> bytes | None:
    p = Path(arg)
    if p.exists():
        return p.read_bytes()
    # Otherwise treat it as a photo id in the object store.
    from app.storage import Storage

    record = Storage(settings).get(arg)
    if record is not None:
        from app.storage import Storage as S

        return S(settings).image_bytes(record)
    return None


def main(args: list[str]) -> None:
    settings = get_settings()
    mod = WeddingModerator(settings)

    print("Loading models (first run downloads weights)…", flush=True)
    mod.startup()
    print(
        f"nsfw_available={mod.nsfw.available}  threshold={settings.nsfw_threshold}  "
        f"reasoning={settings.moderation_reasoning}\n"
    )

    md_backend = next((b for b in mod.backends if b.name == "moondream" and b.available), None)

    for arg in args:
        jpeg = load_bytes(settings, arg)
        if jpeg is None:
            print(f"  ! not found: {arg}")
            continue
        image = Image.open(io.BytesIO(jpeg)).convert("RGB")

        nsfw_score = "n/a"
        if mod.nsfw.available:
            try:
                nsfw_score = f"{mod.nsfw.score(image):.3f}"
            except Exception as exc:
                nsfw_score = f"err: {exc}"

        md_raw = "n/a"
        if md_backend is not None:
            try:
                md_raw = repr(md_backend.classify(image, jpeg))
            except Exception as exc:
                md_raw = f"err: {exc}"

        result = mod.moderate(jpeg)
        flag = "🚫 HELD" if result.verdict != "SAFE" else "✅ SAFE"
        print(f"{arg}")
        print(f"  nsfw score : {nsfw_score}  (>= {settings.nsfw_threshold} = held)")
        print(f"  moondream  : {md_raw}")
        print(f"  --> {flag}  via {result.source}  ({result.latency_ms:.0f}ms)")
        print(f"      {result.reason}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: uv run python scripts/test_moderation.py <path-or-photo-id> [...]")
        raise SystemExit(1)
    main(sys.argv[1:])
