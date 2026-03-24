"""VLM-assisted duct detection for HVAC mechanical drawings.

Hybrid approach:
1. Distance-transform capsule detection finds thick-walled duct symbols
2. Line tracing extends from capsule endpoints to find full pipe runs
3. OCR finds duct size labels and matches them to detected pipes
4. VLM (optional) provides additional context for validation

This replaces the pure-CV approach which had too many false positives.
"""

from __future__ import annotations

import logging
import math
import re

import cv2
import numpy as np

from models import DuctSegment, DuctShape, PageImage
from ocr_engine import OcrEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DRAWING_MARGIN = 30

# Capsule detection
_DIST_THRESHOLD = 3
_MIN_CAPSULE_AREA = 150
_MIN_ASPECT = 2.0
_MAX_NARROW = 50
_MIN_LONG = 50

# Line tracing
_TRACE_MAX_STEPS = 2000
_TRACE_BAND_WIDTH = 20
_TRACE_MAX_GAP = 25
_MIN_PIPE_LENGTH = 120  # minimum total pipe length in pixels

# Label matching
_LABEL_MATCH_RADIUS = 500

# OCR patterns (Tesseract reads ø as 6, 9, 0, o, O)
_DUCT_LABEL_RE = re.compile(r'(\d{1,2})\s*["\'"°]\s*([69oO0])')
_DUCT_LABEL_RE2 = re.compile(r'^(\d{1,2})["\'"°]([69oO0])$')
_DUCT_DIM_RE = re.compile(r'(\d{1,2})\s*["\'"°]?\s*[xX×]\s*(\d{1,2})')


# ---------------------------------------------------------------------------
# Drawing area detection
# ---------------------------------------------------------------------------

def _find_drawing_area(image: np.ndarray) -> tuple[int, int, int, int]:
    h, w = image.shape[:2]
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h_klen = max(w // 2, 100)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_klen, 1))
    h_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    v_klen = max(h // 2, 100)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_klen))
    v_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    h_proj = np.sum(h_mask, axis=1) / 255
    h_positions = _find_line_positions(h_proj, w * 0.3, min_gap=30)
    v_proj = np.sum(v_mask, axis=0) / 255
    v_positions = _find_line_positions(v_proj, h * 0.3, min_gap=30)

    h_positions = sorted(set([0] + h_positions + [h]))
    v_positions = sorted(set([0] + v_positions + [w]))

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

    rx, ry, rw, rh = best_rect
    rx += _DRAWING_MARGIN
    ry += _DRAWING_MARGIN
    rw = max(rw - 2 * _DRAWING_MARGIN, 1)
    rh = max(rh - 2 * _DRAWING_MARGIN, 1)
    right_trim = int(rw * 0.08)
    rw = max(rw - right_trim, 1)

    logger.info("Drawing area: x=%d y=%d w=%d h=%d", rx, ry, rw, rh)
    return (rx, ry, rw, rh)


def _find_line_positions(projection, threshold, min_gap=20):
    positions = []
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
# Capsule detection (thick-walled duct symbols)
# ---------------------------------------------------------------------------

def _detect_capsules(
    binary: np.ndarray,
    roi: tuple[int, int, int, int],
) -> list[dict]:
    """Detect capsule-shaped features using distance transform."""
    rx, ry, rw, rh = roi
    region = binary[ry:ry+rh, rx:rx+rw]

    dist = cv2.distanceTransform(region, cv2.DIST_L2, 5)
    _, thick = cv2.threshold(dist, _DIST_THRESHOLD, 255, cv2.THRESH_BINARY)
    thick = thick.astype(np.uint8)
    dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thick = cv2.dilate(thick, dk, iterations=1)

    contours, _ = cv2.findContours(thick, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    capsules = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < _MIN_CAPSULE_AREA:
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        if w == 0 or h == 0:
            continue
        aspect = max(w, h) / min(w, h)
        narrow = min(w, h)
        long_side = max(w, h)
        if aspect < _MIN_ASPECT or narrow > _MAX_NARROW or long_side < _MIN_LONG:
            continue

        box = cv2.boxPoints(rect).astype(int)
        dists = [np.linalg.norm(box[(i+1)%4] - box[i]) for i in range(4)]
        if dists[0] >= dists[1]:
            p1 = ((box[0] + box[3]) // 2).tolist()
            p2 = ((box[1] + box[2]) // 2).tolist()
        else:
            p1 = ((box[0] + box[1]) // 2).tolist()
            p2 = ((box[2] + box[3]) // 2).tolist()

        p1 = (p1[0] + rx, p1[1] + ry)
        p2 = (p2[0] + rx, p2[1] + ry)
        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])

        capsules.append({
            'p1': p1, 'p2': p2,
            'cx': int(cx + rx), 'cy': int(cy + ry),
            'length': length, 'width': narrow,
            'area': area, 'aspect': aspect,
        })

    logger.info("Capsules detected: %d", len(capsules))
    return capsules


# ---------------------------------------------------------------------------
# Line tracing from capsule endpoints
# ---------------------------------------------------------------------------

def _trace_line(
    binary: np.ndarray,
    sx: int, sy: int,
    dx: float, dy: float,
) -> tuple[int, int]:
    """Trace from (sx,sy) in direction (dx,dy) following any black pixels.
    
    Returns the furthest point reached along the line.
    """
    h, w = binary.shape[:2]
    x, y = float(sx), float(sy)
    last_x, last_y = sx, sy
    empty_count = 0

    for _ in range(_TRACE_MAX_STEPS):
        x += dx
        y += dy
        ix, iy = int(round(x)), int(round(y))

        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            break

        # Check perpendicular band for black pixels
        found = False
        half_band = _TRACE_BAND_WIDTH // 2
        for offset in range(-half_band, half_band + 1):
            px = ix + int(round(-dy * offset))
            py = iy + int(round(dx * offset))
            if 0 <= px < w and 0 <= py < h and binary[py, px] > 0:
                found = True
                break

        if found:
            last_x, last_y = ix, iy
            empty_count = 0
        else:
            empty_count += 1
            if empty_count > _TRACE_MAX_GAP:
                break

    return last_x, last_y


def _extend_capsules(
    capsules: list[dict],
    binary: np.ndarray,
) -> list[dict]:
    """Extend each capsule by tracing connecting lines from its endpoints."""
    extended = []

    for cap in capsules:
        p1, p2 = cap['p1'], cap['p2']
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length == 0:
            extended.append(cap)
            continue
        dx /= length
        dy /= length

        # Trace backwards from p1
        ext1 = _trace_line(binary, p1[0], p1[1], -dx, -dy)
        # Trace forwards from p2
        ext2 = _trace_line(binary, p2[0], p2[1], dx, dy)

        total = math.hypot(ext2[0] - ext1[0], ext2[1] - ext1[1])

        extended.append({
            'p1': ext1, 'p2': ext2,
            'cx': (ext1[0] + ext2[0]) // 2,
            'cy': (ext1[1] + ext2[1]) // 2,
            'length': total,
            'width': cap['width'],
            'area': cap['area'],
            'aspect': cap['aspect'],
            'capsule_length': cap['length'],
        })

    logger.info("Extended %d capsules", len(extended))
    return extended


# ---------------------------------------------------------------------------
# OCR-based duct label finder
# ---------------------------------------------------------------------------

def _find_duct_labels(
    image: np.ndarray,
    roi: tuple[int, int, int, int],
    ocr: OcrEngine,
) -> list[dict]:
    """Find duct size labels using OCR with bounding boxes."""
    import pytesseract

    rx, ry, rw, rh = roi
    region = image[ry:ry+rh, rx:rx+rw]
    if len(region.shape) == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = region.copy()

    try:
        data = pytesseract.image_to_data(
            gray, output_type=pytesseract.Output.DICT,
            config='--psm 11 --oem 3'
        )
    except Exception as e:
        logger.warning("pytesseract failed: %s", e)
        return []

    labels = []
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if not text or int(data['conf'][i]) < 20:
            continue

        size = None
        shape = DuctShape.ROUND

        m = _DUCT_LABEL_RE.search(text) or _DUCT_LABEL_RE2.search(text)
        if m:
            size = int(m.group(1))

        if not m:
            m2 = _DUCT_DIM_RE.search(text)
            if m2:
                size = int(m2.group(1))
                shape = DuctShape.RECTANGULAR

        if size and 4 <= size <= 48:
            x = data['left'][i] + rx
            y = data['top'][i] + ry
            w = data['width'][i]
            h = data['height'][i]
            labels.append({
                'text': text, 'size': size, 'shape': shape,
                'cx': x + w // 2, 'cy': y + h // 2,
            })

    logger.info("Duct labels found: %d", len(labels))
    return labels


# ---------------------------------------------------------------------------
# Label-to-pipe matching
# ---------------------------------------------------------------------------

def _match_labels_to_pipes(
    labels: list[dict],
    pipes: list[dict],
) -> list[tuple[dict | None, dict]]:
    """Match duct labels to nearby pipes. Returns (label, pipe) pairs."""
    used = set()
    matches = []

    for label in labels:
        lx, ly = label['cx'], label['cy']
        best_idx = -1
        best_dist = float('inf')

        for idx, pipe in enumerate(pipes):
            if idx in used:
                continue
            dist = math.hypot(pipe['cx'] - lx, pipe['cy'] - ly)
            if dist < best_dist and dist <= _LABEL_MATCH_RADIUS:
                best_dist = dist
                best_idx = idx

        if best_idx >= 0:
            used.add(best_idx)
            matches.append((label, pipes[best_idx]))

    # Include unmatched pipes that are long enough
    for idx, pipe in enumerate(pipes):
        if idx not in used and pipe['length'] >= _MIN_PIPE_LENGTH * 2:
            matches.append((None, pipe))

    logger.info("Label-pipe matches: %d (labeled: %d, unlabeled: %d)",
                len(matches),
                sum(1 for m in matches if m[0] is not None),
                sum(1 for m in matches if m[0] is None))
    return matches


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _deduplicate_pipes(
    matches: list[tuple[dict | None, dict]],
    min_overlap_dist: float = 150,
) -> list[tuple[dict | None, dict]]:
    """Remove duplicate/overlapping pipe detections."""
    if not matches:
        return matches

    # Sort by: labeled first, then by length descending
    matches.sort(key=lambda m: (0 if m[0] else 1, -m[1]['length']))
    
    keep = []
    for label, pipe in matches:
        is_dup = False
        for _, kept_pipe in keep:
            # Check if centers are close
            dist = math.hypot(pipe['cx'] - kept_pipe['cx'],
                            pipe['cy'] - kept_pipe['cy'])
            if dist < min_overlap_dist:
                is_dup = True
                break
            # Also check if endpoints overlap
            for ep in [pipe['p1'], pipe['p2']]:
                for kep in [kept_pipe['p1'], kept_pipe['p2']]:
                    if math.hypot(ep[0]-kep[0], ep[1]-kep[1]) < 80:
                        is_dup = True
                        break
                if is_dup:
                    break
        if not is_dup:
            keep.append((label, pipe))

    logger.info("After dedup: %d pipes (removed %d)", len(keep), len(matches) - len(keep))
    return keep


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_ducts(
    page_image: PageImage,
    ocr: OcrEngine | None = None,
    vlm_url: str | None = None,
    vlm_model: str | None = None,
) -> list[DuctSegment]:
    """Detect duct segments using capsule detection + line tracing.

    1. Find drawing area
    2. Detect capsule shapes (thick-walled duct symbols)
    3. Trace connecting lines from capsule endpoints
    4. Match with OCR-detected duct size labels
    5. Build DuctSegment list
    """
    image = page_image.image
    if image is None or image.size == 0:
        return []

    h, w = image.shape[:2]

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find drawing area
    roi = _find_drawing_area(image)

    # Detect capsules
    capsules = _detect_capsules(binary, roi)

    # Extend capsules by tracing connecting lines
    pipes = _extend_capsules(capsules, binary)

    # Filter by minimum length
    pipes = [p for p in pipes if p['length'] >= _MIN_PIPE_LENGTH]
    logger.info("Pipes after length filter: %d", len(pipes))

    # Find duct labels
    labels = []
    if ocr is not None:
        labels = _find_duct_labels(image, roi, ocr)

    # Match labels to pipes
    matches = _match_labels_to_pipes(labels, pipes)

    # Deduplicate
    matches = _deduplicate_pipes(matches)

    # Build DuctSegment list
    segments: list[DuctSegment] = []
    for idx, (label, pipe) in enumerate(matches, start=1):
        shape = label['shape'] if label else DuctShape.ROUND
        p1 = pipe['p1']
        p2 = pipe['p2']

        x_min = min(p1[0], p2[0]) - 5
        y_min = min(p1[1], p2[1]) - 5
        bbox_w = abs(p2[0] - p1[0]) + 10
        bbox_h = abs(p2[1] - p1[1]) + 10

        segments.append(DuctSegment(
            id=idx,
            polyline=[p1, p2],
            shape=shape,
            bounding_box=(max(0, x_min), max(0, y_min), bbox_w, bbox_h),
        ))

    round_count = sum(1 for s in segments if s.shape == DuctShape.ROUND)
    rect_count = sum(1 for s in segments if s.shape == DuctShape.RECTANGULAR)
    logger.info(
        "Page %d: %d ducts (%d round, %d rectangular)",
        page_image.page_number, len(segments), round_count, rect_count
    )

    return segments
