"""Generate the print-ready QR code (SVG + PNG) for the table cards.

    uv run python scripts/generate_qr.py

Pulls the target URL from settings (SLIDESHOW_PUBLIC_DOMAIN) and writes into the
repo root, regardless of where the script is invoked from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import segno

# Make the `app` package importable when run as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.config import get_settings  # noqa: E402


def generate_wedding_qr() -> None:
    settings = get_settings()
    # Point straight at the upload page (avoids relying on the / -> /upload
    # redirect, which some QR scanners handle poorly).
    target_url = settings.public_domain.rstrip("/") + "/upload"
    qr = segno.make_qr(target_url, error="H")  # high error correction for print

    svg_path = REPO_ROOT / "wedding_slideshow_qr.svg"
    png_path = REPO_ROOT / "wedding_slideshow_qr.png"
    qr.save(svg_path, scale=10, border=4, dark="#1b1722", light="#ffffff")
    qr.save(png_path, scale=12, border=4, dark="#1b1722", light="#ffffff")

    print(f"Target URL : {target_url}")
    print(f"Wrote      : {svg_path.relative_to(REPO_ROOT)}")
    print(f"Wrote      : {png_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    generate_wedding_qr()
