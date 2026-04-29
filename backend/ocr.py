"""Local OCR using Tesseract (pytesseract + PIL). No external APIs."""

from __future__ import annotations

import re
from io import BytesIO

import pytesseract
from PIL import Image

from backend.tools import get_logger

logger = get_logger("backend.ocr")


def tesseract_available() -> bool:
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001
        return False


def clean_ocr_text(raw: str) -> str:
    """Remove noise: empty lines, collapse whitespace, drop junk single-char lines."""
    if not raw:
        return ""
    lines: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        s = re.sub(r"\s+", " ", s)
        if len(s) == 1 and not s.isalnum():
            continue
        lines.append(s)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_image(file_bytes: bytes) -> str:
    """
    Run Tesseract OCR on image bytes (PNG, JPEG, etc.).
    Requires `tesseract` binary on PATH (e.g. brew install tesseract).
    """
    image = Image.open(BytesIO(file_bytes))
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    raw = pytesseract.image_to_string(image)
    logger.info(
        "ocr_raw_extracted chars=%d text_for_debug=%s",
        len(raw),
        raw[:4000].replace("\n", " | "),
    )

    cleaned = clean_ocr_text(raw)
    logger.info(
        "ocr_cleaned chars=%d text_for_debug=%s",
        len(cleaned),
        cleaned[:4000].replace("\n", " | "),
    )

    if not cleaned:
        logger.warning("ocr_empty_after_clean")
        return (
            "[No text could be extracted from this image. "
            "Try a sharper photo, better lighting, or verify Tesseract is installed.]"
        )

    return cleaned
