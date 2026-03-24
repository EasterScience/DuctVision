"""Dimension extractor for HVAC duct detection.

Parses duct dimension labels (round and rectangular formats) from OCR text
and associates them with detected duct segments by proximity matching.
"""

from __future__ import annotations

import logging
import math
import re

from models import Dimension, DuctSegment, DuctShape, PageImage
from ocr_engine import OcrEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for dimension text
# ---------------------------------------------------------------------------

# Round: 14"⌀, 14"Ø, 14" ⌀, 14⌀, 14Ø, 14"dia, 14"D, 14"o, 14" DIA, 14"0 (OCR misreads)
_ROUND_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*"?\s*(?:[⌀Ø]|[Dd][Ii][Aa]\.?|[Dd](?:ia)?\.?|[Oo0](?!\d))\s*$'
)

# Also match standalone numbers followed by inch mark that appear near ducts
# e.g., 14", 8", 10" — common in HVAC drawings as duct sizes
_SIMPLE_ROUND_PATTERN = re.compile(
    r'^(\d+(?:\.\d+)?)\s*"?\s*$'
)

# Rectangular: 12"x8", 12" x 8", 12x8, 12"X8"
_RECT_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*"?\s*[xX×]\s*(\d+(?:\.\d+)?)\s*"?\s*$'
)


# ---------------------------------------------------------------------------
# parse_dimension_text
# ---------------------------------------------------------------------------


def parse_dimension_text(text: str) -> Dimension:
    """Parse dimension text into a Dimension object.

    Supported formats:
    - Round:  ``14"⌀``, ``14"Ø``, ``14" ⌀``, ``14⌀``, ``14Ø``
    - Rectangular: ``12"x8"``, ``12" x 8"``, ``12x8``, ``12"X8"``

    Args:
        text: Raw dimension string from OCR.

    Returns:
        A :class:`Dimension` with the parsed shape and values.

    Raises:
        ValueError: If *text* does not match any known dimension format.
    """
    cleaned = text.strip()

    m = _ROUND_PATTERN.search(cleaned)
    if m:
        diameter = float(m.group(1))
        return Dimension(raw_text=cleaned, shape=DuctShape.ROUND, values=[diameter])

    m = _RECT_PATTERN.search(cleaned)
    if m:
        width = float(m.group(1))
        height = float(m.group(2))
        return Dimension(
            raw_text=cleaned, shape=DuctShape.RECTANGULAR, values=[width, height]
        )

    raise ValueError(f"Cannot parse dimension text: {text!r}")


# ---------------------------------------------------------------------------
# format_dimension
# ---------------------------------------------------------------------------


def format_dimension(dim: Dimension) -> str:
    """Format a Dimension object to its canonical text representation.

    Round dimensions produce ``{value}"⌀`` and rectangular dimensions
    produce ``{w}"x{h}"``.  Integer values omit the decimal point.

    The output is designed to round-trip through :func:`parse_dimension_text`.
    """
    def _fmt(v: float) -> str:
        return str(int(v)) if v == int(v) else str(v)

    if dim.shape is DuctShape.ROUND:
        return f'{_fmt(dim.values[0])}"⌀'
    else:
        return f'{_fmt(dim.values[0])}"x{_fmt(dim.values[1])}"'


# ---------------------------------------------------------------------------
# Proximity helpers
# ---------------------------------------------------------------------------


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """Return the centre (cx, cy) of an (x, y, w, h) bounding box."""
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _is_dimension_text(text: str) -> bool:
    """Return True if *text* looks like a dimension label."""
    try:
        parse_dimension_text(text)
        return True
    except ValueError:
        pass

    # Also accept standalone numbers with inch marks that are common duct sizes
    cleaned = text.strip()
    m = _SIMPLE_ROUND_PATTERN.match(cleaned)
    if m:
        val = float(m.group(1))
        # Common HVAC duct sizes range from 4" to 60"
        if 4 <= val <= 60 and val == int(val):
            return True
    return False


# ---------------------------------------------------------------------------
# Per-duct region OCR fallback
# ---------------------------------------------------------------------------

_REGION_PAD = 200  # px padding around each duct for region-based OCR

# Regex for finding dimension patterns in raw OCR text
_RAW_DIM_PATTERN = re.compile(
    r'(\d+)\s*"?\s*(?:[⌀Ø@]|[Dd][Ii][Aa]\.?|[Dd]\.?|[Oo0](?!\d))'  # round
    r'|(\d+)\s*"?\s*[xX×]\s*(\d+)'                                     # rectangular
    r'|(\d+)\s*"?\s*[bB]'                                               # duct size with b suffix
    r'|(\d+)\s*"'                                                        # plain inches
)


def _find_dimension_in_text(text: str) -> Dimension | None:
    """Try to extract a duct dimension from raw OCR text."""
    for m in _RAW_DIM_PATTERN.finditer(text):
        # Round duct (group 1)
        if m.group(1):
            val = float(m.group(1))
            if 4 <= val <= 60:
                return Dimension(raw_text=m.group(0), shape=DuctShape.ROUND, values=[val])
        # Rectangular (groups 2, 3)
        elif m.group(2) and m.group(3):
            w, h = float(m.group(2)), float(m.group(3))
            if 4 <= w <= 60 and 4 <= h <= 60:
                return Dimension(raw_text=m.group(0), shape=DuctShape.RECTANGULAR, values=[w, h])
        # Size with b suffix (group 4)
        elif m.group(4):
            val = float(m.group(4))
            if 4 <= val <= 60:
                return Dimension(raw_text=m.group(0), shape=DuctShape.RECTANGULAR, values=[val])
        # Plain inches (group 5)
        elif m.group(5):
            val = float(m.group(5))
            if 4 <= val <= 60 and val == int(val):
                return Dimension(raw_text=m.group(0), shape=DuctShape.ROUND, values=[val])
    return None


# ---------------------------------------------------------------------------
# extract_dimensions
# ---------------------------------------------------------------------------


def extract_dimensions(
    page_image: PageImage,
    ducts: list[DuctSegment],
    ocr: OcrEngine,
    proximity_threshold: float = 150.0,
) -> list[DuctSegment]:
    """Associate dimension labels with duct segments via OCR proximity matching.

    Uses a two-pass approach:
    1. Global OCR with bounding boxes — match by proximity.
    2. Per-duct region OCR fallback — crop a padded region around each
       unmatched duct and search for dimension patterns in the raw text.

    Args:
        page_image: The rendered page image.
        ducts: Detected duct segments (dimension field is ignored).
        ocr: An :class:`OcrEngine` instance for text extraction.
        proximity_threshold: Maximum pixel distance for a label match.

    Returns:
        A **new** list of :class:`DuctSegment` objects with the ``dimension``
        field populated where a matching label was found.
    """
    image = page_image.image
    h_img, w_img = image.shape[:2]

    # --- Pass 1: Global OCR with bounding boxes ---
    ocr_results = ocr.extract_text_with_boxes(image)

    # Try sparse text mode (--psm 11) for better label detection on drawings
    from ocr_engine import TesseractEngine
    if isinstance(ocr, TesseractEngine):
        sparse_ocr = TesseractEngine(config="--psm 11")
        sparse_results = sparse_ocr.extract_text_with_boxes(image)
        existing_positions = {(b[0] // 10, b[1] // 10) for _, b, _ in ocr_results}
        for text, bbox, conf in sparse_results:
            key = (bbox[0] // 10, bbox[1] // 10)
            if key not in existing_positions:
                ocr_results.append((text, bbox, conf))
                existing_positions.add(key)

    # Pre-filter to only dimension-like text boxes
    dim_boxes: list[tuple[Dimension, tuple[float, float]]] = []
    for text, bbox, _conf in ocr_results:
        if _is_dimension_text(text):
            try:
                dim = parse_dimension_text(text)
            except ValueError:
                cleaned = text.strip()
                m = _SIMPLE_ROUND_PATTERN.match(cleaned)
                if m:
                    val = float(m.group(1))
                    if 4 <= val <= 60 and val == int(val):
                        dim = Dimension(
                            raw_text=cleaned,
                            shape=DuctShape.ROUND,
                            values=[val],
                        )
                    else:
                        continue
                else:
                    continue
            center = _bbox_center(bbox)
            dim_boxes.append((dim, center))

    updated: list[DuctSegment] = []
    for duct in ducts:
        if duct.bounding_box is not None:
            duct_center = _bbox_center(duct.bounding_box)
        else:
            xs = [p[0] for p in duct.polyline]
            ys = [p[1] for p in duct.polyline]
            duct_center = (sum(xs) / len(xs), sum(ys) / len(ys))

        # Pass 1: proximity match from global OCR
        best_dim: Dimension | None = None
        best_dist = float("inf")
        for dim, text_center in dim_boxes:
            dist = _euclidean(duct_center, text_center)
            if dist <= proximity_threshold and dist < best_dist:
                best_dim = dim
                best_dist = dist

        # Pass 2: per-duct region OCR fallback
        if best_dim is None:
            cx, cy = int(duct_center[0]), int(duct_center[1])
            rx = max(0, cx - _REGION_PAD)
            ry = max(0, cy - _REGION_PAD)
            rw = min(w_img, cx + _REGION_PAD) - rx
            rh = min(h_img, cy + _REGION_PAD) - ry
            if rw > 0 and rh > 0:
                region_text = ocr.extract_text(image, region=(rx, ry, rw, rh))
                best_dim = _find_dimension_in_text(region_text)

        if best_dim is None:
            logger.warning(
                "Duct %d: no dimension label found within %dpx",
                duct.id, _REGION_PAD,
            )

        updated.append(
            DuctSegment(
                id=duct.id,
                polyline=duct.polyline,
                shape=duct.shape,
                dimension=best_dim,
                pressure_class=duct.pressure_class,
                bounding_box=duct.bounding_box,
            )
        )

    return updated
