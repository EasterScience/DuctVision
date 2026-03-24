"""Pressure classifier for HVAC duct segments.

Determines the pressure class for each duct segment using drawing cues
(line weight, line style) from the page image and optionally notes-derived
DuctSpecification records. Notes-derived values take precedence over
drawing cues when conflicts arise.
"""

import logging

import cv2
import numpy as np

from models import DuctSegment, DuctSpecification, PageImage, PressureClass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds for drawing-cue analysis
# ---------------------------------------------------------------------------

# Mean line thickness (in pixels) above which we consider the duct high-pressure
_HIGH_PRESSURE_THICKNESS = 4.0
# Mean line thickness above which we consider medium-pressure
_MEDIUM_PRESSURE_THICKNESS = 2.5

# Fraction of edge pixels that must be "gapped" to consider the line dashed
_DASHED_GAP_RATIO = 0.25

# Minimum fraction of edge pixels in the double-line detector
_DOUBLE_LINE_EDGE_RATIO = 0.10


def _classify_from_drawing_cues(
    duct: DuctSegment, page_image: PageImage
) -> PressureClass:
    """Classify pressure from visual drawing cues in the page image.

    Analyses the region around the duct's polyline for:
    - Line thickness (thicker → higher pressure)
    - Dashed lines (typically medium pressure)
    - Double lines (typically high pressure)

    Returns PressureClass.UNKNOWN when the region is too small or featureless.
    """
    if len(duct.polyline) < 2:
        return PressureClass.UNKNOWN

    # Compute bounding box of the polyline with a small margin
    xs = [p[0] for p in duct.polyline]
    ys = [p[1] for p in duct.polyline]
    margin = 10
    x_min = max(0, min(xs) - margin)
    y_min = max(0, min(ys) - margin)
    x_max = min(page_image.width, max(xs) + margin)
    y_max = min(page_image.height, max(ys) + margin)

    if x_max - x_min < 3 or y_max - y_min < 3:
        return PressureClass.UNKNOWN

    # Crop the region of interest from the page image
    roi = page_image.image[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return PressureClass.UNKNOWN

    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    edges = cv2.Canny(gray, 50, 150)

    edge_pixels = np.count_nonzero(edges)
    total_pixels = edges.size
    if total_pixels == 0 or edge_pixels == 0:
        return PressureClass.UNKNOWN

    # --- Double-line detection ---
    # Use morphological operations to detect parallel lines (double-wall ducts)
    edge_ratio = edge_pixels / total_pixels
    if edge_ratio > _DOUBLE_LINE_EDGE_RATIO:
        # Look for two distinct contour groups separated by a gap
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) >= 2:
            # Check if the two largest contours are roughly parallel
            areas = sorted([cv2.contourArea(c) for c in contours], reverse=True)
            if len(areas) >= 2 and areas[1] > 0.2 * areas[0]:
                return PressureClass.HIGH

    # --- Dashed-line detection ---
    # Project edges along the dominant axis and look for periodic gaps
    h, w = edges.shape
    if w >= h:
        projection = np.sum(edges, axis=0)
    else:
        projection = np.sum(edges, axis=1)

    if len(projection) > 0:
        max_proj = np.max(projection)
        if max_proj > 0:
            gap_count = np.count_nonzero(projection == 0)
            gap_ratio = gap_count / len(projection)
            if gap_ratio > _DASHED_GAP_RATIO:
                return PressureClass.MEDIUM

    # --- Line thickness estimation ---
    # Use distance transform on the inverted binary image
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    line_pixels = dist[dist > 0]
    if len(line_pixels) > 0:
        mean_thickness = float(np.mean(line_pixels)) * 2  # diameter from radius
        if mean_thickness >= _HIGH_PRESSURE_THICKNESS:
            return PressureClass.HIGH
        if mean_thickness >= _MEDIUM_PRESSURE_THICKNESS:
            return PressureClass.MEDIUM
        return PressureClass.LOW

    return PressureClass.UNKNOWN


def _classify_from_specs(
    duct: DuctSegment, specs: list[DuctSpecification]
) -> PressureClass | None:
    """Classify pressure using notes-derived DuctSpecification records.

    Matches the duct against specifications by type (round/rectangular) and
    size range. Returns the matching pressure class, or ``None`` if no
    specification applies.
    """
    if not specs:
        return None

    # Determine duct type from dimension if available
    duct_type: str | None = None
    duct_size: float | None = None
    if duct.dimension is not None:
        from models import DuctShape

        duct_type = duct.dimension.shape.value  # "round" or "rectangular"
        # Use the first (or largest) value as the representative size
        duct_size = max(duct.dimension.values) if duct.dimension.values else None

    for spec in specs:
        # Check type match: "all" matches everything
        if spec.duct_type != "all":
            if duct_type is not None and spec.duct_type != duct_type:
                continue

        # Check size range match
        if spec.size_range is not None and duct_size is not None:
            if not _size_in_range(duct_size, spec.size_range):
                continue

        # If the spec has a meaningful pressure class, use it
        if spec.pressure_class != PressureClass.UNKNOWN:
            return spec.pressure_class

    return None


def _size_in_range(size: float, size_range: str) -> bool:
    """Check whether a duct size (in inches) falls within a size range string.

    Supported formats:
    - 'up to N"'       → size <= N
    - 'N" to M"'       → N <= size <= M
    - 'N" and larger'  → size >= N
    """
    import re

    size_range_lower = size_range.lower().strip()

    # "up to N"
    m = re.match(r"up\s+to\s+(\d+)", size_range_lower)
    if m:
        return size <= float(m.group(1))

    # "N" to M""
    m = re.match(r"(\d+)\"?\s+to\s+(\d+)", size_range_lower)
    if m:
        return float(m.group(1)) <= size <= float(m.group(2))

    # "N" and larger"
    m = re.match(r"(\d+)\"?\s+and\s+larger", size_range_lower)
    if m:
        return size >= float(m.group(1))

    # Unknown format — assume it matches
    return True


def classify_pressure(
    ducts: list[DuctSegment],
    page_image: PageImage,
    duct_specs: list[DuctSpecification] | None = None,
) -> list[DuctSegment]:
    """Determine the pressure class for each duct segment.

    Uses drawing cues (line weight, style) and optionally notes-derived
    ``DuctSpecification`` records. Notes-derived values take precedence over
    drawing cues when conflicts arise.

    Returns new ``DuctSegment`` instances — originals are not mutated.
    """
    from dataclasses import replace

    classified: list[DuctSegment] = []

    for duct in ducts:
        cue_class = _classify_from_drawing_cues(duct, page_image)
        spec_class = (
            _classify_from_specs(duct, duct_specs) if duct_specs else None
        )

        if spec_class is not None and cue_class not in (
            PressureClass.UNKNOWN,
            spec_class,
        ):
            logger.info(
                "Duct %d: notes-derived pressure class (%s) overrides "
                "drawing-cue class (%s)",
                duct.id,
                spec_class.value,
                cue_class.value,
            )

        # Resolve final pressure class
        if spec_class is not None:
            final_class = spec_class
        elif cue_class != PressureClass.UNKNOWN:
            final_class = cue_class
        else:
            logger.warning(
                "Duct %d: no pressure cues available, defaulting to UNKNOWN",
                duct.id,
            )
            final_class = PressureClass.UNKNOWN

        classified.append(replace(duct, pressure_class=final_class))

    return classified
