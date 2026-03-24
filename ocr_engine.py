"""OCR engine abstraction layer for HVAC duct detection.

Defines the ``OcrEngine`` protocol and two concrete implementations:
* ``TesseractEngine`` – Tesseract OCR via *pytesseract*.
* ``Florence2Engine`` – Microsoft Florence-2 VLM via Hugging Face *transformers*.

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
    """Common interface for OCR / VLM text-extraction backends."""

    def extract_text(
        self,
        image: np.ndarray,
        region: tuple[int, int, int, int] | None = None,
    ) -> str:
        """Extract all text from an image or a cropped region.

        Args:
            image: BGR numpy array.
            region: Optional ``(x, y, w, h)`` crop rectangle.

        Returns:
            Extracted text string.
        """
        ...

    def extract_text_with_boxes(
        self,
        image: np.ndarray,
        region: tuple[int, int, int, int] | None = None,
    ) -> list[tuple[str, tuple[int, int, int, int], float]]:
        """Extract text with bounding boxes and confidence scores.

        Args:
            image: BGR numpy array.
            region: Optional ``(x, y, w, h)`` crop rectangle.

        Returns:
            List of ``(text, (x, y, w, h), confidence)`` tuples.
        """
        ...


# ---------------------------------------------------------------------------
# Helper – crop image to region
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
    """Tesseract OCR backend via *pytesseract*.

    Args:
        lang: Tesseract language code (default ``"eng"``).
        config: Tesseract CLI config string (default ``"--psm 6"``).
    """

    def __init__(self, lang: str = "eng", config: str = "--psm 6") -> None:
        import shutil

        import pytesseract

        # Auto-detect Tesseract on Windows if not already on PATH
        if not shutil.which("tesseract"):
            import os
            win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.isfile(win_path):
                pytesseract.pytesseract.tesseract_cmd = win_path

        self.lang = lang
        self.config = config

    # -- extract_text -------------------------------------------------------

    def extract_text(
        self,
        image: np.ndarray,
        region: tuple[int, int, int, int] | None = None,
    ) -> str:
        """Extract all text from *image* (or a cropped *region*) using Tesseract.

        Args:
            image: BGR numpy array.
            region: Optional ``(x, y, w, h)`` crop rectangle.

        Returns:
            Extracted text string.
        """
        import pytesseract

        img = _crop_to_region(image, region) if region is not None else image
        text: str = pytesseract.image_to_string(img, lang=self.lang, config=self.config)
        return text.strip()

    # -- extract_text_with_boxes --------------------------------------------

    def extract_text_with_boxes(
        self,
        image: np.ndarray,
        region: tuple[int, int, int, int] | None = None,
    ) -> list[tuple[str, tuple[int, int, int, int], float]]:
        """Extract text with bounding boxes and confidence scores.

        Uses ``pytesseract.image_to_data`` and filters out empty or
        low-confidence (< 0) results.

        Args:
            image: BGR numpy array.
            region: Optional ``(x, y, w, h)`` crop rectangle.

        Returns:
            List of ``(text, (x, y, w, h), confidence)`` tuples.
        """
        import pytesseract

        img = _crop_to_region(image, region) if region is not None else image

        data = pytesseract.image_to_data(
            img, lang=self.lang, config=self.config, output_type=pytesseract.Output.DICT
        )

        # Offset boxes back to full-image coordinates when a region was used.
        x_off = region[0] if region is not None else 0
        y_off = region[1] if region is not None else 0

        results: list[tuple[str, tuple[int, int, int, int], float]] = []
        n_boxes = len(data["text"])
        for i in range(n_boxes):
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
# Florence2Engine
# ---------------------------------------------------------------------------


class Florence2Engine:
    """Florence-2 VLM backend via Hugging Face *transformers*.

    Uses the ``microsoft/Florence-2-base`` model (0.23 B params).  The model
    and processor are loaded lazily on the first call to avoid startup cost
    when the engine is instantiated but never used.

    Args:
        model_name: Hugging Face model identifier.
        device: PyTorch device string (``"cpu"`` or ``"cuda"``).  When set to
            ``"cpu"`` the engine will still attempt to use CUDA if available,
            falling back to CPU on failure.
    """

    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-base",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self._requested_device = device
        self._model = None
        self._processor = None
        self._device: str | None = None

    # -- lazy loading -------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load model and processor on first use."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Florence2Engine requires the 'transformers' and 'torch' "
                "packages.  Install them with:  pip install transformers torch"
            ) from exc

        # Determine device – prefer CUDA when available.
        device = self._requested_device
        if device == "cpu" and torch.cuda.is_available():
            device = "cuda"

        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True
            ).to(device)
            self._device = device
            logger.info("Florence-2 loaded on %s", device)
        except RuntimeError:
            # GPU memory insufficient – fall back to CPU.
            logger.warning(
                "Failed to load Florence-2 on %s, falling back to CPU", device
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True
            ).to("cpu")
            self._device = "cpu"

    # -- internal inference --------------------------------------------------

    def _run_task(
        self, image: np.ndarray, task_prompt: str
    ) -> dict:
        """Run a Florence-2 task and return the parsed result dict."""
        self._ensure_loaded()

        from PIL import Image

        # Florence-2 expects an RGB PIL Image.
        if image.ndim == 3 and image.shape[2] == 3:
            rgb = image[:, :, ::-1]  # BGR → RGB
        else:
            rgb = image
        pil_image = Image.fromarray(rgb)

        import torch

        inputs = self._processor(
            text=task_prompt, images=pil_image, return_tensors="pt"
        ).to(self._device)

        with torch.inference_mode():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )

        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed = self._processor.post_process_generation(
            generated_text, task=task_prompt, image_size=pil_image.size
        )
        return parsed

    # -- extract_text -------------------------------------------------------

    def extract_text(
        self,
        image: np.ndarray,
        region: tuple[int, int, int, int] | None = None,
    ) -> str:
        """Extract all text from *image* using Florence-2's OCR task.

        Args:
            image: BGR numpy array.
            region: Optional ``(x, y, w, h)`` crop rectangle.

        Returns:
            Extracted text string.
        """
        img = _crop_to_region(image, region) if region is not None else image
        result = self._run_task(img, "<OCR>")
        return result.get("<OCR>", "").strip()

    # -- extract_text_with_boxes --------------------------------------------

    def extract_text_with_boxes(
        self,
        image: np.ndarray,
        region: tuple[int, int, int, int] | None = None,
    ) -> list[tuple[str, tuple[int, int, int, int], float]]:
        """Extract text with bounding boxes using Florence-2's OCR_WITH_REGION task.

        Args:
            image: BGR numpy array.
            region: Optional ``(x, y, w, h)`` crop rectangle.

        Returns:
            List of ``(text, (x, y, w, h), confidence)`` tuples.
            Florence-2 does not provide per-word confidence, so confidence
            is set to ``1.0`` for every detection.
        """
        img = _crop_to_region(image, region) if region is not None else image
        result = self._run_task(img, "<OCR_WITH_REGION>")

        ocr_data = result.get("<OCR_WITH_REGION>", {})
        labels: list[str] = ocr_data.get("labels", [])
        quad_boxes: list = ocr_data.get("quad_boxes", [])

        x_off = region[0] if region is not None else 0
        y_off = region[1] if region is not None else 0

        results: list[tuple[str, tuple[int, int, int, int], float]] = []
        for label, quad in zip(labels, quad_boxes):
            text = label.strip()
            if not text:
                continue
            # quad_boxes are [x1, y1, x2, y2, x3, y3, x4, y4] – convert to
            # axis-aligned (x, y, w, h).
            xs = [quad[i] for i in range(0, len(quad), 2)]
            ys = [quad[i] for i in range(1, len(quad), 2)]
            bx = int(min(xs)) + x_off
            by = int(min(ys)) + y_off
            bw = int(max(xs) - min(xs))
            bh = int(max(ys) - min(ys))
            results.append((text, (bx, by, bw, bh), 1.0))

        return results


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_ocr_engine(engine_name: str) -> OcrEngine:
    """Create an OCR engine instance by name.

    Args:
        engine_name: ``"tesseract"`` or ``"florence2"``.

    Returns:
        An ``OcrEngine``-conformant object.

    Raises:
        ValueError: If *engine_name* is not recognised.
    """
    if engine_name == "tesseract":
        return TesseractEngine()
    if engine_name == "florence2":
        return Florence2Engine()
    raise ValueError(
        f"Unknown OCR engine {engine_name!r}. Choose 'tesseract' or 'florence2'."
    )
