"""Duct segment detection for HVAC mechanical drawings.

Two detection strategies for a monochrome (black & white) drawing:

1. **Round ducts** — thick-walled capsule/oval shapes with ø labels.
   Detected via distance transform thresholding + contour analysis.

2. **Rectangular ducts** — thin double-line pairs with dimension labels.
   Detected via parallel-pair matching on Hough lines.

The drawing area is isolated first to exclude notes, title block, and
legends.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from models import DuctSegment, DuctShape, PageImage
from ocr_engine import OcrEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hough parameters for rectangular duct detection
_HOUGH_RHO = 1
_HOUGH_THETA = np.pi / 180
_HOUGH_THRESHOLD = 50
_HOUGH_MIN_LINE_LENGTH = 40
_HOUGH_MAX_LINE_GAP = 15

_MIN_SEGMENT_LENGTH = 100
_CARDINAL_ANGLE_TOL = 15.0
_MERGE_DISTANCE = 35
_PARALLEL_ANGLE_TOL = 10.0

# Parallel pair spacing calibrated for HVAC drawings at 300 DPI
# 1/4" = 1'-0" scale → 1 pixel ≈ 0.16 real inches
# Realistic duct widths: 6" to 36" → ~37px to ~225px spacing
# Building walls are typically wider (>250px) or very thin (<5px)
_PARALLEL_DIST_MIN = 10
_PARALLEL_DIST_MAX = 100

# Round duct detection
_ROUND_DIST_THRESHOLD = 5   # distance transform threshold for thick walls
_ROUND_MIN_AREA = 800       # minimum contour area (filter small artifacts)
_ROUND_MIN_ASPECT = 2.5     # minimum aspect ratio (capsules are elongated)
_ROUND_MAX_WIDTH = 80       # max width of a round duct cross-section indicator
_ROUND_MIN_LENGTH = 100     # minimum polyline length in pixels (~1.3 ft at 300 DPI)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _line_length(x1: int, y1: int, x2: int, y2: int) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def _line_angle(x1: int, y1: int, x2: int, y2: int) -> float:
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180


def _perpendicular_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float,
) -> float:
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return math.hypot(px - x1, py - y1)
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / length


def _midpoint(x1: int, y1: int, x2: int, y2: int) -> tuple[float, float]:
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _angle_diff(a: float, b: float) -> float:
    diff = abs(a - b) % 180
    return min(diff, 180 - diff)


def _is_cardinal(angle: float) -> bool:
    if angle <= _CARDINAL_ANGLE_TOL or angle >= (180 - _CARDINAL_ANGLE_TOL):
        return True
    if abs(angle - 90) <= _CARDINAL_ANGLE_TOL:
        return True
    return False


# ---------------------------------------------------------------------------
# Drawing area detection
# ---------------------------------------------------------------------------


def _find_drawing_area(image: np.ndarray) -> tuple[int, int, int, int]:
    """Find the main plan area using section divider detection."""
    h, w = image.shape[:2]

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect long horizontal dividers (>50% of page width)
    h_klen = max(w // 2, 100)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_klen, 1))
    h_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Detect long vertical dividers (>50% of page height)
    v_klen = max(h // 2, 100)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_klen))
    v_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    h_proj = np.sum(h_mask, axis=1) / 255
    h_positions = _find_line_positions(h_proj, w * 0.3, min_gap=30)

    v_proj = np.sum(v_mask, axis=0) / 255
    v_positions = _find_line_positions(v_proj, h * 0.3, min_gap=30)

    h_positions = sorted(set([0] + h_positions + [h]))
    v_positions = sorted(set([0] + v_positions + [w]))

    logger.debug(
        "Dividers: H=%s, V=%s", h_positions[1:-1], v_positions[1:-1],
    )

    # Pick the largest section
    best_rect = (0, 0, w, h)
    best_area = 0
    for i in range(len(h_positions) - 1):
        for j in range(len(v_positions) - 1):
            ry, ry2 = h_positions[i], h_positions[i + 1]
            rx, rx2 = v_positions[j], v_positions[j + 1]
            rw, rh = rx2 - rx, ry2 - ry
            if rw < w * 0.10 or rh < h * 0.10:
                continue
            area = rw * rh
            if area > best_area:
                best_area = area
                best_rect = (rx, ry, rw, rh)

    margin = 30
    rx, ry, rw, rh = best_rect
    rx += margin
    ry += margin
    rw = max(rw - 2 * margin, 1)
    rh = max(rh - 2 * margin, 1)

    # Trim extra right margin to exclude logos/title block elements
    # that may intrude into the drawing area's right edge
    right_trim = int(rw * 0.08)
    rw = max(rw - right_trim, 1)

    logger.info(
        "Drawing area: x=%d y=%d w=%d h=%d (%.0f%% of page)",
        rx, ry, rw, rh, (rw * rh) / (w * h) * 100,
    )
    return (rx, ry, rw, rh)


def _find_line_positions(
    projection: np.ndarray, threshold: float, min_gap: int = 20,
) -> list[int]:
    positions: list[int] = []
    in_line = False
    line_start = 0
    for i in range(len(projection)):
        if projection[i] > threshold:
            if not in_line:
                line_start = i
                in_line = True
        else:
            if in_line:
                pos = (line_start + i) // 2
                if not positions or pos - positions[-1] > min_gap:
                    positions.append(pos)
                in_line = False
    if in_line:
        pos = (line_start + len(projection) - 1) // 2
        if not positions or pos - positions[-1] > min_gap:
            positions.append(pos)
    return positions


# ---------------------------------------------------------------------------
# Round duct detection (thick-walled capsule shapes)
# ---------------------------------------------------------------------------


def _detect_round_ducts(
    binary: np.ndarray, roi: tuple[int, int, int, int],
) -> list[DuctSegment]:
    """Detect round ducts using distance transform + contour analysis.

    Round grease ducts appear as thick gray-filled capsule shapes.
    The distance transform isolates thick-walled features, then contour
    analysis finds elongated shapes (capsules).
    """
    rx, ry, rw, rh = roi
    region = binary[ry : ry + rh, rx : rx + rw]

    # Distance transform — highlights thick features
    dist = cv2.distanceTransform(region, cv2.DIST_L2, 5)

    # Threshold to keep only thick-walled features
    _, thick_mask = cv2.threshold(
        dist, _ROUND_DIST_THRESHOLD, 255, cv2.THRESH_BINARY
    )
    thick_mask = thick_mask.astype(np.uint8)

    # Dilate to reconnect fragments
    dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thick_mask = cv2.dilate(thick_mask, dk, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(
        thick_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    segments: list[DuctSegment] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < _ROUND_MIN_AREA:
            continue

        # Fit a rotated rectangle
        rect = cv2.minAreaRect(contour)
        (cx, cy), (w_rect, h_rect), angle = rect

        if w_rect == 0 or h_rect == 0:
            continue

        # Aspect ratio — capsules are elongated
        aspect = max(w_rect, h_rect) / min(w_rect, h_rect)
        if aspect < _ROUND_MIN_ASPECT:
            continue

        # Width of the narrow side should be reasonable for a duct
        narrow = min(w_rect, h_rect)
        if narrow > _ROUND_MAX_WIDTH:
            continue

        # Build polyline along the long axis
        box = cv2.boxPoints(rect)
        box = box.astype(int)

        # Sort box points to find the long axis endpoints
        # The long axis connects the midpoints of the short sides
        dists = [np.linalg.norm(box[(i + 1) % 4] - box[i]) for i in range(4)]
        if dists[0] >= dists[1]:
            # sides 0-1 and 2-3 are long
            p1 = ((box[0] + box[3]) // 2).tolist()
            p2 = ((box[1] + box[2]) // 2).tolist()
        else:
            # sides 1-2 and 3-0 are long
            p1 = ((box[0] + box[1]) // 2).tolist()
            p2 = ((box[2] + box[3]) // 2).tolist()

        # Offset back to full image coordinates
        p1 = (p1[0] + rx, p1[1] + ry)
        p2 = (p2[0] + rx, p2[1] + ry)

        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if length < _ROUND_MIN_LENGTH:
            continue

        x_min = min(p1[0], p2[0]) - 5
        y_min = min(p1[1], p2[1]) - 5
        bbox_w = abs(p2[0] - p1[0]) + 10
        bbox_h = abs(p2[1] - p1[1]) + 10

        segments.append(
            DuctSegment(
                id=0,  # assigned later
                polyline=[p1, p2],
                shape=DuctShape.ROUND,
                bounding_box=(max(0, x_min), max(0, y_min), bbox_w, bbox_h),
            )
        )

    logger.info("Round ducts detected: %d", len(segments))
    return segments


# ---------------------------------------------------------------------------
# Rectangular duct detection (parallel line pairs)
# ---------------------------------------------------------------------------


def _preprocess_for_lines(image: np.ndarray) -> np.ndarray:
    """Binarise and extract line structures for rectangular duct detection."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4,
    )

    # Remove thin text strokes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

    # Extract horizontal line structures
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Extract vertical line structures
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    combined = cv2.bitwise_or(horiz, vert)

    dk = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.dilate(combined, dk, iterations=1)

    return combined


def _detect_hough_lines(binary: np.ndarray) -> list[tuple[int, int, int, int]]:
    raw = cv2.HoughLinesP(
        binary,
        rho=_HOUGH_RHO,
        theta=_HOUGH_THETA,
        threshold=_HOUGH_THRESHOLD,
        minLineLength=_HOUGH_MIN_LINE_LENGTH,
        maxLineGap=_HOUGH_MAX_LINE_GAP,
    )
    if raw is None:
        return []
    return [(int(l[0][0]), int(l[0][1]), int(l[0][2]), int(l[0][3])) for l in raw]


def _filter_in_roi(
    lines: list[tuple[int, int, int, int]], roi: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]]:
    rx, ry, rw, rh = roi
    rx2, ry2 = rx + rw, ry + rh
    return [
        (x1, y1, x2, y2)
        for x1, y1, x2, y2 in lines
        if rx <= x1 <= rx2 and ry <= y1 <= ry2 and rx <= x2 <= rx2 and ry <= y2 <= ry2
    ]


def _merge_collinear(
    lines: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    if not lines:
        return []

    used = [False] * len(lines)
    merged: list[tuple[int, int, int, int]] = []
    angles = [_line_angle(*s) for s in lines]

    for i in range(len(lines)):
        if used[i]:
            continue
        used[i] = True
        pts = [lines[i][:2], lines[i][2:]]

        changed = True
        while changed:
            changed = False
            for j in range(len(lines)):
                if used[j]:
                    continue
                if _angle_diff(angles[i], angles[j]) > _PARALLEL_ANGLE_TOL:
                    continue
                mid_j = _midpoint(*lines[j])
                perp = _perpendicular_distance(
                    mid_j[0], mid_j[1],
                    float(lines[i][0]), float(lines[i][1]),
                    float(lines[i][2]), float(lines[i][3]),
                )
                if perp > 8:
                    continue
                min_d = min(
                    math.hypot(px - qx, py - qy)
                    for px, py in pts
                    for qx, qy in [lines[j][:2], lines[j][2:]]
                )
                if min_d <= _MERGE_DISTANCE:
                    pts.extend([lines[j][:2], lines[j][2:]])
                    used[j] = True
                    changed = True

        if len(pts) >= 2:
            ar = math.radians(angles[i])
            ca, sa = math.cos(ar), math.sin(ar)
            proj = sorted((px * ca + py * sa, px, py) for px, py in pts)
            merged.append((int(proj[0][1]), int(proj[0][2]),
                           int(proj[-1][1]), int(proj[-1][2])))

    return merged


def _find_parallel_pairs(
    lines: list[tuple[int, int, int, int]],
) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    if not lines:
        return []

    angles = [_line_angle(*s) for s in lines]
    used = [False] * len(lines)
    pairs = []

    for i in range(len(lines)):
        if used[i]:
            continue
        best_j, best_perp = -1, float("inf")

        for j in range(i + 1, len(lines)):
            if used[j]:
                continue
            if _angle_diff(angles[i], angles[j]) > _PARALLEL_ANGLE_TOL:
                continue

            mid_j = _midpoint(*lines[j])
            perp = _perpendicular_distance(
                mid_j[0], mid_j[1],
                float(lines[i][0]), float(lines[i][1]),
                float(lines[i][2]), float(lines[i][3]),
            )
            if not (_PARALLEL_DIST_MIN <= perp <= _PARALLEL_DIST_MAX):
                continue

            mid_i = _midpoint(*lines[i])
            len_i = _line_length(*lines[i])
            len_j = _line_length(*lines[j])
            avg_len = (len_i + len_j) / 2
            mid_dist = math.hypot(mid_i[0] - mid_j[0], mid_i[1] - mid_j[1])

            if mid_dist < avg_len * 0.9 and perp < best_perp:
                best_j = j
                best_perp = perp

        if best_j >= 0:
            used[i] = True
            used[best_j] = True
            pairs.append((lines[i], lines[best_j]))

    return pairs


# ---------------------------------------------------------------------------
# Dimension-proximity validation for rectangular ducts
# ---------------------------------------------------------------------------

import re as _re

# Patterns that indicate a duct dimension label nearby
_DIM_LABEL_PATTERN = _re.compile(
    r'(?:'
    r'\d+\s*"'                       # e.g. 14", 8"
    r'|\d+\s*[⌀Ø]'                  # e.g. 18⌀
    r'|\d+\s*"?\s*[xX×]\s*\d+'      # e.g. 12x8, 12"x8"
    r'|\d+\s*"?\s*[bBdD]'           # e.g. 14"b, 8"B (duct size labels)
    r'|\d+\s*[⌀Ø]'                  # standalone diameter
    r')'
)

_OCR_SEARCH_PAD = 200  # px padding around each pair for OCR search


def _has_dimension_label_nearby(
    pair: tuple[tuple[int, int, int, int], tuple[int, int, int, int]],
    image: np.ndarray,
    ocr: OcrEngine,
) -> bool:
    """Check if a parallel line pair has a duct dimension label nearby.

    Crops a padded region around the pair, runs OCR, and checks for
    dimension-like text patterns.
    """
    a, b = pair
    h_img, w_img = image.shape[:2]

    # Bounding box of both lines + padding
    all_x = [a[0], a[2], b[0], b[2]]
    all_y = [a[1], a[3], b[1], b[3]]
    x_min = max(0, min(all_x) - _OCR_SEARCH_PAD)
    y_min = max(0, min(all_y) - _OCR_SEARCH_PAD)
    x_max = min(w_img, max(all_x) + _OCR_SEARCH_PAD)
    y_max = min(h_img, max(all_y) + _OCR_SEARCH_PAD)

    region = (x_min, y_min, x_max - x_min, y_max - y_min)
    if region[2] <= 0 or region[3] <= 0:
        return False

    text = ocr.extract_text(image, region=region)
    return bool(_DIM_LABEL_PATTERN.search(text))


def _detect_rectangular_ducts(
    image: np.ndarray,
    roi: tuple[int, int, int, int],
    ocr: OcrEngine | None = None,
) -> list[DuctSegment]:
    """Detect rectangular ducts as parallel line pairs within the ROI.

    When *ocr* is provided, each candidate pair is validated by checking
    for a duct dimension label in the surrounding region.  Pairs without
    a nearby label (likely building walls) are discarded.
    """
    binary = _preprocess_for_lines(image)

    lines = _detect_hough_lines(binary)
    if not lines:
        return []

    # Cardinal only
    lines = [s for s in lines if _is_cardinal(_line_angle(*s))]

    # Within ROI
    lines = _filter_in_roi(lines, roi)

    if not lines:
        return []

    # Merge collinear
    lines = _merge_collinear(lines)

    # Min length
    lines = [s for s in lines if _line_length(*s) >= _MIN_SEGMENT_LENGTH]

    if not lines:
        return []

    # Find parallel pairs
    pairs = _find_parallel_pairs(lines)
    logger.info("Parallel pairs found (pre-validation): %d", len(pairs))

    # Dimension-proximity validation: keep only pairs near a dimension label
    if ocr is not None and pairs:
        validated = []
        for pair in pairs:
            if _has_dimension_label_nearby(pair, image, ocr):
                validated.append(pair)
            else:
                logger.debug(
                    "Discarding pair — no dimension label nearby: %s / %s",
                    pair[0], pair[1],
                )
        logger.info(
            "Pairs after dimension-proximity validation: %d / %d",
            len(validated), len(pairs),
        )
        pairs = validated

    segments: list[DuctSegment] = []
    for a, b in pairs:
        cx1 = int((a[0] + b[0]) / 2)
        cy1 = int((a[1] + b[1]) / 2)
        cx2 = int((a[2] + b[2]) / 2)
        cy2 = int((a[3] + b[3]) / 2)
        polyline = [(cx1, cy1), (cx2, cy2)]

        xs = [cx1, cx2]
        ys = [cy1, cy2]
        pad = 10
        bbox = (
            max(0, min(xs) - pad),
            max(0, min(ys) - pad),
            abs(cx2 - cx1) + 2 * pad,
            abs(cy2 - cy1) + 2 * pad,
        )

        segments.append(
            DuctSegment(
                id=0,
                polyline=polyline,
                shape=DuctShape.RECTANGULAR,
                bounding_box=bbox,
            )
        )

    logger.info("Rectangular ducts detected: %d", len(segments))
    return segments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_ducts(
    page_image: PageImage,
    ocr: OcrEngine | None = None,
) -> list[DuctSegment]:
    """Detect duct segments using dual strategy:

    1. Round ducts via distance transform + contour analysis.
    2. Rectangular ducts via parallel line pair matching.

    Both are restricted to the detected drawing area.

    When *ocr* is provided, rectangular duct candidates are validated
    by checking for nearby dimension labels — pairs without labels
    (likely building walls) are discarded.
    """
    image = page_image.image
    if image is None or image.size == 0:
        logger.info("Page %d: empty image.", page_image.page_number)
        return []

    h, w = image.shape[:2]

    # Binarise for round duct detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find drawing area
    roi = _find_drawing_area(image)

    # Detect both types
    round_ducts = _detect_round_ducts(binary, roi)
    rect_ducts = _detect_rectangular_ducts(image, roi, ocr=ocr)

    # Combine and assign IDs
    all_ducts: list[DuctSegment] = []
    for idx, duct in enumerate(round_ducts + rect_ducts, start=1):
        duct.id = idx
    all_ducts = round_ducts + rect_ducts

    if not all_ducts:
        logger.info("Page %d: no ducts found.", page_image.page_number)
    else:
        logger.info(
            "Page %d: %d ducts (%d round, %d rectangular).",
            page_image.page_number,
            len(all_ducts),
            len(round_ducts),
            len(rect_ducts),
        )

    return all_ducts
