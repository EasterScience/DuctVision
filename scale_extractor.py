"""Drawing scale extraction and dimension conversion for HVAC duct detection.

Locates scale notations in engineering drawings (e.g. ``1/4" = 1'-0"``),
parses them into numeric ratios, and converts drawing-space measurements
to real-world feet-and-inches strings.
"""

from __future__ import annotations

import logging
import math
import re

from models import DrawingScale, PageImage
from ocr_engine import OcrEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Matches scale patterns like:
#   1/4" = 1'-0"    3/8" = 1'-0"    1" = 1'-0"
#   1-1/2" = 1'-0"  1/2" = 1'-0"    3/16" = 1'-0"
# Also handles OCR misreads: ° for ", garbled spacing, etc.
_SCALE_PATTERN = re.compile(
    r"""
    (\d+(?:-\d+/\d+|\d*/\d+)?)   # drawing side: integer, fraction, or mixed
    \s*["″°]?\s*                   # inch mark (optional — OCR may drop it)
    =\s*                           # equals sign
    (\d+)                          # feet value
    \s*['\u2032]?\s*-?\s*          # foot mark (optional)
    (\d+)                          # inches value
    \s*["″°]?                      # inch mark (optional)
    """,
    re.VERBOSE,
)

# Common scales: mapping from canonical text to ratio (drawing_inches / real_inches)
_COMMON_SCALES: list[tuple[str, float]] = [
    ('1/4" = 1\'-0"', 0.25 / 12.0),
    ('1/8" = 1\'-0"', 0.125 / 12.0),
    ('3/8" = 1\'-0"', 0.375 / 12.0),
    ('1/2" = 1\'-0"', 0.5 / 12.0),
    ('1" = 1\'-0"', 1.0 / 12.0),
    ('3/16" = 1\'-0"', (3.0 / 16.0) / 12.0),
    ('1-1/2" = 1\'-0"', 1.5 / 12.0),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_drawing_inches(text: str) -> float:
    """Parse the drawing-side measurement text into inches.

    Supports:
      - Simple integer: ``1``  → 1.0
      - Simple fraction: ``1/4`` → 0.25
      - Mixed number: ``1-1/2`` → 1.5
    """
    text = text.strip()

    # Mixed number: e.g. "1-1/2"
    mixed_match = re.fullmatch(r"(\d+)-(\d+)/(\d+)", text)
    if mixed_match:
        whole = int(mixed_match.group(1))
        num = int(mixed_match.group(2))
        den = int(mixed_match.group(3))
        if den == 0:
            raise ValueError(f"Zero denominator in mixed number: {text!r}")
        return whole + num / den

    # Simple fraction: e.g. "1/4"
    frac_match = re.fullmatch(r"(\d+)/(\d+)", text)
    if frac_match:
        num = int(frac_match.group(1))
        den = int(frac_match.group(2))
        if den == 0:
            raise ValueError(f"Zero denominator in fraction: {text!r}")
        return num / den

    # Simple integer: e.g. "1"
    int_match = re.fullmatch(r"(\d+)", text)
    if int_match:
        return float(int(int_match.group(1)))

    raise ValueError(f"Cannot parse drawing measurement: {text!r}")



def _ratio_to_drawing_fraction(ratio: float) -> str:
    """Convert a scale ratio back to the drawing-side fraction string.

    Finds the closest common fraction representation for the drawing inches
    value implied by the ratio (assuming 1'-0" = 12" on the real-world side).

    Returns strings like ``1/4``, ``3/16``, ``1-1/2``, or ``1``.
    """
    drawing_inches = ratio * 12.0  # ratio = drawing / real, real = 12"

    # Check common fractions (denominator up to 16)
    best_text: str | None = None
    best_err = float("inf")

    # Try simple fractions with denominators 1, 2, 4, 8, 16
    for den in (1, 2, 4, 8, 16):
        for num in range(1, den * 4 + 1):  # up to 4 inches drawing side
            val = num / den
            err = abs(val - drawing_inches)
            if err < best_err:
                best_err = err
                if den == 1:
                    best_text = str(num)
                else:
                    whole = num // den
                    remainder = num % den
                    if whole > 0 and remainder > 0:
                        best_text = f"{whole}-{remainder}/{den}"
                    elif whole > 0:
                        best_text = str(whole)
                    else:
                        best_text = f"{num}/{den}"

    return best_text or "1"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_scale_text(text: str) -> DrawingScale:
    """Parse a scale string into a ``DrawingScale`` object.

    Supported formats::

        1/4" = 1'-0"
        1/8" = 1'-0"
        3/8" = 1'-0"
        1/2" = 1'-0"
        1" = 1'-0"
        3/16" = 1'-0"
        1-1/2" = 1'-0"

    The left side is the drawing measurement in inches, the right side is
    the real-world measurement in feet-inches.

    Args:
        text: Scale notation string.

    Returns:
        ``DrawingScale`` with ``status="found"`` and computed ratio.

    Raises:
        ValueError: If *text* cannot be parsed as a valid scale.
    """
    text = text.strip()
    m = _SCALE_PATTERN.search(text)
    if not m:
        raise ValueError(f"Cannot parse scale text: {text!r}")

    drawing_part = m.group(1)
    feet = int(m.group(2))
    inches = int(m.group(3))

    drawing_inches = _parse_drawing_inches(drawing_part)
    real_inches = feet * 12.0 + inches

    if real_inches == 0:
        raise ValueError(f"Real-world measurement is zero in scale: {text!r}")
    if drawing_inches <= 0:
        raise ValueError(f"Drawing measurement must be positive: {text!r}")

    ratio = drawing_inches / real_inches

    return DrawingScale(raw_text=text, ratio=ratio, status="found")


def format_scale(scale: DrawingScale) -> str:
    """Format a ``DrawingScale`` back to canonical text representation.

    Produces strings like ``1/4" = 1'-0"`` that can be round-tripped
    through ``parse_scale_text``.

    Args:
        scale: A ``DrawingScale`` with a known ratio.

    Returns:
        Canonical scale string.
    """
    if scale.ratio is None or scale.status == "unknown":
        return scale.raw_text

    frac = _ratio_to_drawing_fraction(scale.ratio)
    return f'{frac}" = 1\'-0"'


def extract_scale(page_image: PageImage, ocr: OcrEngine) -> DrawingScale:
    """Locate and parse the drawing scale notation from a page image.

    Scans the bottom 20% of the page (title block area) for scale text
    using the provided OCR engine.

    Args:
        page_image: Rendered page image.
        ocr: OCR engine instance.

    Returns:
        ``DrawingScale`` with parsed ratio, or ``status="unknown"`` if
        no scale notation is found.
    """
    h, w = page_image.image.shape[:2]

    # Search regions where scale notation commonly appears:
    # 1. Title strip just below drawing area (60-75% down the page)
    # 2. Bottom-right quadrant (traditional title block)
    # 3. Bottom 25% of the page
    # 4. Right 35% of the page
    search_regions = [
        (int(w * 0.5), int(h * 0.60), int(w * 0.5), int(h * 0.15)),  # title strip below drawing
        (int(w * 0.5), int(h * 0.75), int(w * 0.5), int(h * 0.25)),  # bottom-right
        (0, int(h * 0.75), w, int(h * 0.25)),                          # bottom 25%
        (int(w * 0.65), 0, int(w * 0.35), h),                          # right 35%
    ]

    text = ""
    for region in search_regions:
        region_text = ocr.extract_text(page_image.image, region=region)
        if region_text:
            text += " " + region_text

    # Also try sparse text mode (--psm 11) which handles scattered labels better
    from ocr_engine import TesseractEngine
    if isinstance(ocr, TesseractEngine):
        sparse_ocr = TesseractEngine(config="--psm 11")
        for region in search_regions:
            sparse_text = sparse_ocr.extract_text(page_image.image, region=region)
            if sparse_text:
                text += " " + sparse_text

    if not text:
        logger.warning(
            "No text found in title block region of page %d", page_image.page_number
        )
        return DrawingScale(raw_text="", ratio=None, status="unknown")

    # Search for scale patterns in the extracted text
    m = _SCALE_PATTERN.search(text)
    if m:
        scale_text = m.group(0)
        try:
            result = parse_scale_text(scale_text)
            logger.info(
                "Found scale %r (ratio=%.6f) on page %d",
                result.raw_text,
                result.ratio,
                page_image.page_number,
            )
            return result
        except ValueError:
            logger.warning(
                "Matched scale pattern but failed to parse: %r", scale_text
            )

    logger.warning("No scale notation found on page %d", page_image.page_number)
    return DrawingScale(raw_text="", ratio=None, status="unknown")


def convert_to_real_world(drawing_measurement: float, scale: DrawingScale) -> str:
    """Convert a drawing-space measurement to real-world feet-and-inches.

    Args:
        drawing_measurement: Measurement in drawing units (inches on paper).
        scale: The drawing scale to apply.

    Returns:
        A string like ``3'-6"`` for known scales, or
        ``"{measurement} drawing units"`` when the scale is unknown.
    """
    if scale.status == "unknown" or scale.ratio is None:
        return f"{drawing_measurement} drawing units"

    # real_inches = drawing_measurement / scale.ratio
    # because ratio = drawing_inches / real_inches
    real_inches = drawing_measurement / scale.ratio

    total_inches = round(real_inches)
    feet = total_inches // 12
    inches = total_inches % 12

    return f"{feet}'-{inches}\""
