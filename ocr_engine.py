"""OCR engine abstraction layer for HVAC duct detection.

Defines the ``OcrEngine`` protocol and a Tesseract implementation.
A ``create_ocr_engine`` factory function instantiates the requested backend.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OcrEngine protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class OcrEngine(Protocol):
    """Common interface for OCR text-extraction backends."""

    def extract_text(
        self,
        image: np.ndarray,
        region: tuple[int, int, int, int] | None = None,
    ) -> str: ...

    def extract_text_with_boxes(
        self,
        image: np.ndarray,
        region: tuple[int, int, int, int] | None = None,
    ) -> list[tuple[str, tuple[int, int, int, int], float]]: ...


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _crop_to_region(
    image: np.ndarray, region: tuple[int, int, int, int]
) -> np.ndarray:
    """Return the sub-image defined by ``(x, y, w, h)``."""
    x, y, w, h = region
    return image[y : y + h, x : x + w]


# ---------------------------------------------------------------------------
# TesseractEngine
# ---------------------------------------------------------------------------


class TesseractEngine:
    """Tesseract OCR backend via *pytesseract*."""

    def __init__(self, lang: str = "eng", config: str = "--psm 6") -> None:
        import shutil
        import pytesseract

        if not shutil.which("tesseract"):
            import os
            win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.isfile(win_path):
                pytesseract.pytesseract.tesseract_cmd = win_path

        self.lang = lang
        self.config = config

    def extract_text(
        self,
        image: np.ndarray,
        region: tuple[int, int, int, int] | None = None,
    ) -> str:
        import pytesseract
        img = _crop_to_region(image, region) if region is not None else image
        text: str = pytesseract.image_to_string(img, lang=self.lang, config=self.config)
        return text.strip()

    def extract_text_with_boxes(
        self,
        image: np.ndarray,
        region: tuple[int, int, int, int] | None = None,
    ) -> list[tuple[str, tuple[int, int, int, int], float]]:
        import pytesseract
        img = _crop_to_region(image, region) if region is not None else image
        data = pytesseract.image_to_data(
            img, lang=self.lang, config=self.config, output_type=pytesseract.Output.DICT
        )
        x_off = region[0] if region is not None else 0
        y_off = region[1] if region is not None else 0
        results: list[tuple[str, tuple[int, int, int, int], float]] = []
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            conf = float(data["conf"][i])
            if not text or conf < 0:
                continue
            bx = int(data["left"][i]) + x_off
            by = int(data["top"][i]) + y_off
            bw = int(data["width"][i])
            bh = int(data["height"][i])
            results.append((text, (bx, by, bw, bh), conf))
        return results


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_ocr_engine(engine_name: str = "tesseract") -> OcrEngine:
    """Create an OCR engine instance by name."""
    if engine_name == "tesseract":
        return TesseractEngine()
    raise ValueError(f"Unknown OCR engine {engine_name!r}. Only 'tesseract' is supported.")
