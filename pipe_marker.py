"""Pipe/duct detection and blue-fill marking for HVAC mechanical drawings.

Hybrid approach:
1. CV finds white channels between black lines (pipe interiors) — primary
2. CV finds parallel line pairs via Hough — secondary
3. Merges both, deduplicates, fills with solid blue
"""

from __future__ import annotations

import json
import logging
import math
import os
import re

import cv2
import numpy as np

from pdf_renderer import render_pdf

logger = logging.getLogger(__name__)

PIPE_BLUE = (255, 100, 0)  # BGR
PIPE_THICKNESS = 22       # Uniform blue line width in pixels (at 300 DPI)


# ---------------------------------------------------------------------------
# Drawing area detection (downscaled for speed)
# ---------------------------------------------------------------------------

def _find_line_positions(projection, threshold, min_gap=8):
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


def _find_drawing_area(image: np.ndarray) -> tuple[int, int, int, int]:
    """Find the main plan area by detecting section dividers.
    Uses 0.25x downscale for speed on large images."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    scale = 0.25
    small = cv2.resize(gray, (int(w * scale), int(h * scale)))
    sh, sw = small.shape

    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h_klen = max(sw // 2, 100)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_klen, 1))
    h_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    v_klen = max(sh // 2, 100)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_klen))
    v_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    h_proj = np.sum(h_mask, axis=1) / 255
    h_positions = _find_line_positions(h_proj, sw * 0.3)
    v_proj = np.sum(v_mask, axis=0) / 255
    v_positions = _find_line_positions(v_proj, sh * 0.3)

    # Scale back
    h_positions = [int(p / scale) for p in h_positions]
    v_positions = [int(p / scale) for p in v_positions]

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
            if rw * rh > best_area:
                best_area = rw * rh
                best_rect = (rx, ry, rw, rh)

    margin = 30
    rx, ry, rw, rh = best_rect
    rx += margin; ry += margin
    rw = max(rw - 2 * margin, 1)
    rh = max(rh - 2 * margin, 1)

    logger.info("Drawing area: x=%d y=%d w=%d h=%d (%.0f%% of page)",
                rx, ry, rw, rh, (rw * rh) / (w * h) * 100)
    return (rx, ry, rw, rh)


# ---------------------------------------------------------------------------
# PRIMARY: White-channel detection (find pipe interiors directly)
# ---------------------------------------------------------------------------

def _check_black_borders(black_mask, bx, by, bw, bh, is_horizontal, ih, iw,
                         border_w=5, min_ratio=0.25):
    """Check if a white channel has black borders on its narrow sides."""
    if is_horizontal:
        top = black_mask[max(0, by - border_w):by, bx:bx + bw]
        bot = black_mask[by + bh:min(ih, by + bh + border_w), bx:bx + bw]
        if top.size == 0 or bot.size == 0:
            return False
        return (np.sum(top > 0) / top.size > min_ratio and
                np.sum(bot > 0) / bot.size > min_ratio)
    else:
        left = black_mask[by:by + bh, max(0, bx - border_w):bx]
        right = black_mask[by:by + bh, bx + bw:min(iw, bx + bw + border_w)]
        if left.size == 0 or right.size == 0:
            return False
        return (np.sum(left > 0) / left.size > min_ratio and
                np.sum(right > 0) / right.size > min_ratio)



def _measure_border_thickness(black_mask, bx, by, bw, bh, ih, iw, max_scan=20):
    """Measure the actual black line thickness on all 4 sides of a channel.
    Returns (top, bottom, left, right) thicknesses."""
    n_samples = 7

    def _median_thickness(starts, direction):
        thicknesses = []
        for sy, sx in starts:
            t = 0
            for step in range(max_scan):
                ny = sy + direction[0] * step
                nx = sx + direction[1] * step
                if ny < 0 or ny >= ih or nx < 0 or nx >= iw:
                    break
                if black_mask[ny, nx] > 0:
                    t += 1
                else:
                    if t > 0:
                        break
            thicknesses.append(t)
        if not thicknesses:
            return 0
        thicknesses.sort()
        return thicknesses[len(thicknesses) // 2]

    # Sample positions along each edge
    xs = [bx + int(bw * (i + 1) / (n_samples + 1)) for i in range(n_samples)]
    ys = [by + int(bh * (i + 1) / (n_samples + 1)) for i in range(n_samples)]

    # Top: scan upward from top edge
    top_starts = [(by - 1, x) for x in xs if 0 <= x < iw and by - 1 >= 0]
    t_top = _median_thickness(top_starts, (-1, 0))

    # Bottom: scan downward from bottom edge
    bot_starts = [(by + bh, x) for x in xs if 0 <= x < iw and by + bh < ih]
    t_bot = _median_thickness(bot_starts, (1, 0))

    # Left: scan leftward from left edge
    left_starts = [(y, bx - 1) for y in ys if 0 <= y < ih and bx - 1 >= 0]
    t_left = _median_thickness(left_starts, (0, -1))

    # Right: scan rightward from right edge
    right_starts = [(y, bx + bw) for y in ys if 0 <= y < ih and bx + bw < iw]
    t_right = _median_thickness(right_starts, (0, 1))

    return t_top, t_bot, t_left, t_right




def _sort_box_points(box):
    """Sort 4 box points from minAreaRect into order: p0-p1 is a long edge,
    p2-p3 is the opposite long edge. Returns array of 4 points."""
    # box is 4 points from cv2.boxPoints
    # Find the two longest edges
    dists = []
    for i in range(4):
        j = (i + 1) % 4
        d = math.hypot(box[j][0] - box[i][0], box[j][1] - box[i][1])
        dists.append(d)
    # If edge 0-1 is longer, long edges are 0-1 and 2-3
    if dists[0] >= dists[1]:
        return np.array([box[0], box[1], box[2], box[3]])
    else:
        return np.array([box[1], box[2], box[3], box[0]])


def _check_rotated_borders(mask, box, narrow_dim, ih, iw, n_samples=9, min_ratio=0.25):
    """Check if a rotated rectangle has dark borders on its narrow (short) sides.
    box: 4 absolute-coordinate points from cv2.boxPoints.
    Samples pixels just outside the two short edges."""
    # Find short edges
    dists = []
    for i in range(4):
        j = (i + 1) % 4
        d = math.hypot(box[j][0] - box[i][0], box[j][1] - box[i][1])
        dists.append((d, i))
    dists.sort()
    # Two shortest edges
    short_edges = [(dists[0][1], (dists[0][1] + 1) % 4),
                   (dists[1][1], (dists[1][1] + 1) % 4)]

    border_w = 5
    hits = 0
    total = 0
    for ei, ej in short_edges:
        p1 = box[ei]
        p2 = box[ej]
        # Direction outward from the rectangle center
        mid = np.mean(box, axis=0)
        edge_mid = (p1 + p2) / 2
        outward = edge_mid - mid
        out_len = np.linalg.norm(outward)
        if out_len < 1:
            continue
        outward = outward / out_len

        for s in range(n_samples):
            t = (s + 1) / (n_samples + 1)
            pt = p1 * (1 - t) + p2 * t
            for step in range(1, border_w + 1):
                sx = int(pt[0] + outward[0] * step)
                sy = int(pt[1] + outward[1] * step)
                if 0 <= sy < ih and 0 <= sx < iw:
                    total += 1
                    if mask[sy, sx] > 0:
                        hits += 1

    if total == 0:
        return False
    return (hits / total) >= min_ratio


def _detect_white_channels(image, roi):
    """Find elongated white channels bounded by black lines (pipe interiors).
    Works only within ROI for speed. Returns (channels, labels)."""
    rx, ry, rw, rh = roi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    ih, iw = gray.shape

    roi_gray = gray[ry:ry + rh, rx:rx + rw]

    # White mask: pipe interiors are white (>200)
    _, white_mask = cv2.threshold(roi_gray, 200, 255, cv2.THRESH_BINARY)
    _, black_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    # Gray border mask: catches pipes with gray walls (value < 140)
    _, gray_border_mask = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

    ek = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    white_eroded = cv2.erode(white_mask, ek, iterations=1)

    # Close small gaps from text labels inside pipes
    ck = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    white_closed = cv2.morphologyEx(white_eroded, cv2.MORPH_CLOSE, ck)

    # Additional aggressive close for diagonal pipe detection — REMOVED (merges with background)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        white_closed, connectivity=8)
    logger.info("White CC: %d components in ROI", n_labels)

    channels = []
    _rej = {"fill": 0, "large": 0, "rect": 0, "angle": 0, "poly": 0,
            "border": 0, "wall_thin": 0, "wall_ratio": 0}
    for i in range(1, n_labels):
        bx = stats[i, cv2.CC_STAT_LEFT]
        by = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        aspect = max(bw, bh) / max(min(bw, bh), 1)
        narrow = min(bw, bh)

        if narrow < 8 or narrow > 150 or max(bw, bh) < 120:
            continue
        # Allow lower aspect ratio for larger pipes (e.g. 22x14)
        min_aspect = 1.5 if narrow >= 30 else 2.0
        if aspect < min_aspect:
            continue
        fill = area / max(bw * bh, 1)
        if fill < 0.35:
            _rej["fill"] += 1
            continue
        if area > 150000:
            _rej["large"] += 1
            continue

        comp_mask = (labels[by:by + bh, bx:bx + bw] == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(cnt)
            rect_area = rect[1][0] * rect[1][1]
            cnt_area = cv2.contourArea(cnt)
            if rect_area > 0:
                rectangularity = cnt_area / rect_area
            else:
                rectangularity = 0

            # Determine if this is a diagonal shape using minAreaRect angle
            mar_angle = rect[2]  # angle from minAreaRect (-90..0]
            mar_w, mar_h = rect[1]
            # Normalize: make mar_long/mar_narrow consistent
            mar_long = max(mar_w, mar_h)
            mar_narrow = min(mar_w, mar_h)
            is_diagonal = not (mar_angle > -5 or mar_angle < -85 or
                               (mar_w > mar_h and mar_angle > -5) or
                               (mar_h > mar_w and mar_angle < -85))
            # Simpler diagonal check: if axis-aligned fill is much lower than
            # minAreaRect fill, the shape is rotated
            aa_fill = cnt_area / max(bw * bh, 1)
            is_diagonal = is_diagonal or (rectangularity > 0.70 and aa_fill < 0.65)

            if is_diagonal:
                # For diagonal pipes, use minAreaRect dimensions for all checks
                d_narrow = mar_narrow
                d_long = mar_long
                d_aspect = d_long / max(d_narrow, 1)
                d_fill = rectangularity  # contour area / rotated rect area

                # Apply same size filters but using rotated dimensions
                if d_narrow < 8 or d_narrow > 150 or d_long < 120:
                    continue
                min_d_aspect = 1.5 if d_narrow >= 30 else 2.0
                if d_aspect < min_d_aspect:
                    continue
                if d_fill < 0.60:
                    _rej["fill"] += 1
                    continue
                # Rectangularity can be lower for diagonal pipes with text inside
                if rectangularity < 0.55:
                    _rej["rect"] += 1
                    continue
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) > 10:
                    _rej["poly"] += 1
                    continue

                # For diagonal pipes, check borders along the rotated rectangle edges
                abs_x, abs_y = bx + rx, by + ry
                # Use rotated rect box points to check borders
                box = cv2.boxPoints(rect)
                box[:, 0] += rx  # offset to absolute coords
                box[:, 1] += ry
                use_gray_borders = False
                if not _check_rotated_borders(black_mask, box, mar_narrow, ih, iw):
                    if _check_rotated_borders(gray_border_mask, box, mar_narrow, ih, iw):
                        use_gray_borders = True
                    else:
                        _rej["border"] += 1
                        continue

                # Build line pair from the rotated rectangle for drawing
                # Sort box points to find the long edges
                box_sorted = _sort_box_points(box)
                # Long edges: (p0-p1) and (p3-p2)
                a_line = (int(box_sorted[0][0]), int(box_sorted[0][1]),
                          int(box_sorted[1][0]), int(box_sorted[1][1]))
                b_line = (int(box_sorted[3][0]), int(box_sorted[3][1]),
                          int(box_sorted[2][0]), int(box_sorted[2][1]))

                channels.append({
                    "x": abs_x, "y": abs_y, "w": bw, "h": bh,
                    "orient": "D",
                    "label_id": i, "narrow": d_narrow, "aspect": d_aspect,
                    "roi_x": bx, "roi_y": by,
                    "border_t": [0, 0, 0, 0],
                    "diag_pair": _snap_pair_to_endwalls(a_line, b_line, gray),
                })
                logger.info("Diagonal pipe found: pos=(%d,%d) %dx%d mar=%.0fx%.0f "
                            "rect=%.2f aa_fill=%.2f angle=%.1f",
                            abs_x, abs_y, bw, bh, mar_long, mar_narrow,
                            rectangularity, aa_fill, mar_angle)
                continue

            # Non-diagonal: standard checks
            if rectangularity < 0.88:
                _rej["rect"] += 1
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) > 8:
                _rej["poly"] += 1
                continue
        else:
            continue

        is_h = bw > bh
        abs_x, abs_y = bx + rx, by + ry

        # Require black borders on narrow sides — fall back to gray border mask
        min_border = 0.25 if aspect >= 3.0 else 0.40
        use_gray_borders = False
        if not _check_black_borders(black_mask, abs_x, abs_y, bw, bh, is_h, ih, iw,
                                     min_ratio=min_border):
            # Try with gray border mask for pipes with gray walls
            if _check_black_borders(gray_border_mask, abs_x, abs_y, bw, bh, is_h, ih, iw,
                                     min_ratio=min_border):
                use_gray_borders = True
            else:
                _rej["border"] += 1
                continue

        # Measure wall thickness using appropriate mask
        border_mask = gray_border_mask if use_gray_borders else black_mask
        t_top, t_bot, t_left, t_right = _measure_border_thickness(
            border_mask, abs_x, abs_y, bw, bh, ih, iw)

        # For pipes, the two parallel walls (narrow sides) must have consistent thickness.
        if is_h:
            wall_thicknesses = [t_top, t_bot]
        else:
            wall_thicknesses = [t_left, t_right]

        # Both walls must be >= 2px
        if min(wall_thicknesses) < 2:
            _rej["wall_thin"] += 1
            continue

        # Both walls must be similar thickness (ratio <= 2.5)
        if max(wall_thicknesses) / max(min(wall_thicknesses), 1) > 2.5:
            _rej["wall_ratio"] += 1
            continue

        # End-walls (the short sides) must also have some thickness
        # to confirm the shape is a closed rectangle, not just parallel lines
        if is_h:
            end_walls = [t_left, t_right]
        else:
            end_walls = [t_top, t_bot]
        if min(end_walls) < 2:
            _rej["wall_thin"] += 1
            continue

        # Wall darkness: both parallel walls must actually be dark lines,
        # not white/gray background. Sample median brightness along each wall.
        if is_h:
            wall_regions = [
                gray[max(0, abs_y - 4):abs_y + 1, abs_x:abs_x + bw],       # top wall
                gray[abs_y + bh - 1:min(ih, abs_y + bh + 4), abs_x:abs_x + bw],  # bot wall
            ]
        else:
            wall_regions = [
                gray[abs_y:abs_y + bh, max(0, abs_x - 4):abs_x + 1],       # left wall
                gray[abs_y:abs_y + bh, abs_x + bw - 1:min(iw, abs_x + bw + 4)],  # right wall
            ]
        wall_too_bright = False
        for wr in wall_regions:
            if wr.size > 0 and np.median(wr) > 150:
                wall_too_bright = True
                break
        if wall_too_bright:
            _rej["wall_thin"] += 1
            continue

        channels.append({
            "x": abs_x, "y": abs_y, "w": bw, "h": bh,
            "orient": "H" if is_h else "V",
            "label_id": i, "narrow": narrow, "aspect": aspect,
            "roi_x": bx, "roi_y": by,
            "border_t": [t_top, t_bot, t_left, t_right],
        })

    logger.info("White-channel detection: %d pipe interiors", len(channels))
    logger.info("Rejection reasons: %s", _rej)
    if channels:
        narrows = [c["narrow"] for c in channels]
        all_borders = [c["border_t"] for c in channels]
        logger.info("White channel narrow dims: min=%d max=%d median=%d",
                    min(narrows), max(narrows), sorted(narrows)[len(narrows)//2])
        logger.info("White channel border thickness [T,B,L,R]: %s",
                    [t for t in all_borders])

    return channels, labels


def _detect_gray_channels(image, roi, existing_channels):
    """Find gray-filled pipe interiors (gray value ~80-150).
    Some pipes have gray fill instead of white. Returns additional channels."""
    rx, ry, rw, rh = roi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    ih, iw = gray.shape

    roi_gray = gray[ry:ry + rh, rx:rx + rw]

    # Gray band: 80-160 (not white, not black)
    gray_mask = ((roi_gray >= 80) & (roi_gray <= 160)).astype(np.uint8) * 255

    ek = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    gray_eroded = cv2.erode(gray_mask, ek, iterations=1)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        gray_eroded, connectivity=8)

    # Build set of existing channel centers for dedup
    existing_centers = set()
    for ch in existing_channels:
        existing_centers.add((ch["x"] // 60, ch["y"] // 60))

    channels = []
    for i in range(1, n_labels):
        bx = stats[i, cv2.CC_STAT_LEFT]
        by = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        aspect = max(bw, bh) / max(min(bw, bh), 1)
        narrow = min(bw, bh)

        if aspect < 3.0 or narrow < 10 or narrow > 80 or max(bw, bh) < 80:
            continue
        if area / max(bw * bh, 1) < 0.5:
            continue
        if area > 100000:
            continue

        # --- Rectangularity filter: reject half-circles / curved shapes ---
        comp_mask = (labels[by:by + bh, bx:bx + bw] == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(cnt)
            rect_area = rect[1][0] * rect[1][1]
            if rect_area > 0:
                rectangularity = cv2.contourArea(cnt) / rect_area
                if rectangularity < 0.88:
                    continue
                angle = rect[2]
                if 5 < abs(angle) < 85:
                    continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) > 8:
                continue
        else:
            continue

        is_h = bw > bh
        abs_x, abs_y = bx + rx, by + ry

        # Skip if already covered
        grid_key = (abs_x // 60, abs_y // 60)
        if grid_key in existing_centers:
            continue

        # For gray channels, check that borders are lighter (white) or darker
        # The pipe walls show as a transition zone — check that the region
        # outside the gray band is significantly different (white or black)
        border_ok = _check_gray_borders(gray, abs_x, abs_y, bw, bh, is_h, ih, iw)
        if not border_ok:
            continue

        channels.append({
            "x": abs_x, "y": abs_y, "w": bw, "h": bh,
            "orient": "H" if is_h else "V",
            "label_id": i, "narrow": narrow, "aspect": aspect,
            "roi_x": bx, "roi_y": by,
            "gray": True,
        })
        existing_centers.add(grid_key)

    logger.info("Gray-channel detection: %d additional pipe interiors", len(channels))
    return channels, labels




def _check_gray_borders(gray, bx, by, bw, bh, is_horizontal, ih, iw, border_w=8):
    """Check if a gray channel has distinct borders (lighter or darker)."""
    if is_horizontal:
        # Check above and below — should be white (>200) or have a clear edge
        top = gray[max(0, by - border_w):by, bx:bx + bw]
        bot = gray[by + bh:min(ih, by + bh + border_w), bx:bx + bw]
        if top.size == 0 or bot.size == 0:
            return False
        # The border should be significantly different from the gray interior
        interior_mean = gray[by:by + bh, bx:bx + bw].mean()
        top_mean = top.mean()
        bot_mean = bot.mean()
        # Border should be lighter (white) or have a clear transition
        return (abs(top_mean - interior_mean) > 40 and
                abs(bot_mean - interior_mean) > 40)
    else:
        left = gray[by:by + bh, max(0, bx - border_w):bx]
        right = gray[by:by + bh, bx + bw:min(iw, bx + bw + border_w)]
        if left.size == 0 or right.size == 0:
            return False
        interior_mean = gray[by:by + bh, bx:bx + bw].mean()
        left_mean = left.mean()
        right_mean = right.mean()
        return (abs(left_mean - interior_mean) > 40 and
                abs(right_mean - interior_mean) > 40)


# ---------------------------------------------------------------------------
# DIAGONAL: Detect diagonal pipes via rotated rectangle contours
# ---------------------------------------------------------------------------

def _detect_diagonal_pipes(image, roi, existing_channels):
    """Detect diagonal pipes using Canny edge detection + Hough parallel pairs.
    The standard Hough detector uses morphology at 0.5x scale which loses gray
    diagonal lines. This uses full-resolution Canny edges and allows higher
    median pixel values (gray walls up to 160)."""
    rx, ry, rw, rh = roi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    ih, iw = gray.shape

    roi_gray = gray[ry:ry + rh, rx:rx + rw]

    # Canny edge detection — catches both black and gray wall edges
    edges = cv2.Canny(roi_gray, 40, 120)

    raw = cv2.HoughLinesP(edges, 1, np.pi / 180, 30,
                          minLineLength=80, maxLineGap=10)
    if raw is None:
        logger.info("Diagonal pipe detection: no Canny-Hough lines")
        return []

    # Filter to diagonal-only lines with wall-like darkness
    diag_lines = []
    for l in raw:
        x1, y1, x2, y2 = l[0]
        length = _line_length(x1, y1, x2, y2)
        if length < _MIN_SEG_LEN:
            continue
        angle = _line_angle(x1, y1, x2, y2)
        if _is_cardinal(angle):
            continue  # skip H/V lines — already handled by main Hough

        ax1, ay1 = x1 + rx, y1 + ry
        ax2, ay2 = x2 + rx, y2 + ry

        # Allow gray walls (median < 120) — slightly relaxed for diagonal pipes
        n_pts = max(int(length // 4), 5)
        xs = np.linspace(ax1, ax2, n_pts, dtype=int)
        ys = np.linspace(ay1, ay2, n_pts, dtype=int)
        xs = np.clip(xs, 0, iw - 1)
        ys = np.clip(ys, 0, ih - 1)
        median_val = np.median(gray[ys, xs])
        if median_val > 120:
            continue

        diag_lines.append((ax1, ay1, ax2, ay2))

    diag_lines = _merge_collinear(diag_lines)
    diag_lines = [s for s in diag_lines if _line_length(*s) >= _MIN_SEG_LEN]
    logger.info("Diagonal Canny-Hough: %d diagonal segments", len(diag_lines))

    # Find parallel pairs — use wider gap range for diagonal pipes
    # (some diagonal pipes like 22x14 have gaps up to 150px)
    saved_max = _PARALLEL_DIST_MAX
    pairs = _find_parallel_pairs_diag(diag_lines)
    logger.info("Diagonal parallel pairs (raw): %d", len(pairs))

    # Build existing centers for dedup
    existing_centers = set()
    for ch in existing_channels:
        existing_centers.add((ch["x"] // 50, ch["y"] // 50))

    # Validate and convert to channel dicts
    new_channels = []
    for a, b in pairs:
        mid_b = _midpoint(*b)
        gap = _perp_dist(mid_b[0], mid_b[1],
                         float(a[0]), float(a[1]), float(a[2]), float(a[3]))

        # Diagonal pipes must have a meaningful gap (not hatching/thin lines)
        if gap < 25:
            continue

        cx = (a[0] + a[2] + b[0] + b[2]) / 4
        cy = (a[1] + a[3] + b[1] + b[3]) / 4
        grid_key = (int(cx) // 50, int(cy) // 50)
        if grid_key in existing_centers:
            continue

        # Both lines must be similar length (proper rectangle, not random pair)
        len_a = _line_length(*a)
        len_b = _line_length(*b)
        length_ratio = min(len_a, len_b) / max(len_a, len_b)
        if length_ratio < 0.6:
            continue

        # Check wall darkness: at least one wall must be very dark (<50)
        # and the other reasonably dark (<100) — rejects light grey walls
        def _wall_median(line):
            n = max(int(_line_length(*line) // 4), 5)
            xs_ = np.linspace(line[0], line[2], n, dtype=int)
            ys_ = np.linspace(line[1], line[3], n, dtype=int)
            xs_ = np.clip(xs_, 0, iw - 1)
            ys_ = np.clip(ys_, 0, ih - 1)
            return int(np.median(gray[ys_, xs_]))
        med_a = _wall_median(a)
        med_b = _wall_median(b)
        darkest = min(med_a, med_b)
        lightest = max(med_a, med_b)
        if darkest > 50 or lightest > 100:
            continue

        # Aspect ratio: long dimension / gap must be pipe-like
        avg_len = (len_a + len_b) / 2
        diag_aspect = avg_len / max(gap, 1)
        if diag_aspect < 1.5:
            continue

        # Validate interior is mostly white (pipe interior, not wall/hatching)
        if not _validate_diagonal_interior(gray, a, b):
            continue

        bx = int(min(a[0], a[2], b[0], b[2]))
        by = int(min(a[1], a[3], b[1], b[3]))
        bw = int(max(a[0], a[2], b[0], b[2]) - bx)
        bh = int(max(a[1], a[3], b[1], b[3]) - by)

        new_channels.append({
            "x": bx, "y": by, "w": bw, "h": bh,
            "orient": "D",
            "label_id": -1, "narrow": int(gap), "aspect": 0,
            "roi_x": 0, "roi_y": 0,
            "border_t": [0, 0, 0, 0],
            "diag_pair": _snap_pair_to_endwalls(*_clip_pair_to_overlap(a, b), gray),
        })
        existing_centers.add(grid_key)
        # Compute median darkness of each line for logging
        def _line_median(line):
            n = max(int(_line_length(*line) // 4), 5)
            xs_ = np.linspace(line[0], line[2], n, dtype=int)
            ys_ = np.linspace(line[1], line[3], n, dtype=int)
            xs_ = np.clip(xs_, 0, iw - 1)
            ys_ = np.clip(ys_, 0, ih - 1)
            return int(np.median(gray[ys_, xs_]))
        logger.info("Diagonal pipe: center=(%.0f,%.0f) gap=%.0f med=%d/%d "
                    "lens=%.0f/%.0f A=(%d,%d)-(%d,%d) B=(%d,%d)-(%d,%d)",
                    cx, cy, gap, _line_median(a), _line_median(b),
                    _line_length(*a), _line_length(*b),
                    a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3])

    logger.info("Diagonal pipe detection: %d pipes", len(new_channels))
    return new_channels


def _clip_pair_to_overlap(a, b):
    """Trim two parallel lines to their mutual overlap along the pipe direction.
    Returns clipped (a, b) tuple."""
    pa1 = np.array([a[0], a[1]], dtype=float)
    pa2 = np.array([a[2], a[3]], dtype=float)
    pb1 = np.array([b[0], b[1]], dtype=float)
    pb2 = np.array([b[2], b[3]], dtype=float)

    # Ensure same direction
    dir_a = pa2 - pa1
    dir_b = pb2 - pb1
    if np.dot(dir_a, dir_b) < 0:
        pb1, pb2 = pb2, pb1
        dir_b = -dir_b

    # Average direction
    dir_v = (dir_a / max(np.linalg.norm(dir_a), 1) +
             dir_b / max(np.linalg.norm(dir_b), 1))
    norm = np.linalg.norm(dir_v)
    if norm < 0.01:
        return (a, b)
    dir_v /= norm

    # Project onto pipe direction
    origin = (pa1 + pa2 + pb1 + pb2) / 4.0
    ta1 = np.dot(pa1 - origin, dir_v)
    ta2 = np.dot(pa2 - origin, dir_v)
    tb1 = np.dot(pb1 - origin, dir_v)
    tb2 = np.dot(pb2 - origin, dir_v)

    # Overlap range
    t_start = max(min(ta1, ta2), min(tb1, tb2))
    t_end = min(max(ta1, ta2), max(tb1, tb2))
    if t_end <= t_start:
        return (a, b)  # no overlap, return as-is

    # Perpendicular offsets of each line from origin
    perp = np.array([-dir_v[1], dir_v[0]])
    pa_perp = np.dot((pa1 + pa2) / 2 - origin, perp)
    pb_perp = np.dot((pb1 + pb2) / 2 - origin, perp)

    # Reconstruct clipped lines
    new_a1 = origin + dir_v * t_start + perp * pa_perp
    new_a2 = origin + dir_v * t_end + perp * pa_perp
    new_b1 = origin + dir_v * t_start + perp * pb_perp
    new_b2 = origin + dir_v * t_end + perp * pb_perp

    return (
        (int(new_a1[0]), int(new_a1[1]), int(new_a2[0]), int(new_a2[1])),
        (int(new_b1[0]), int(new_b1[1]), int(new_b2[0]), int(new_b2[1])),
    )


def _snap_pair_to_endwalls(a, b, gray):
    """Scan along a diagonal pipe pair to find actual end-walls (dark bands
    perpendicular to the pipe direction) and adjust the line endpoints.
    This fixes cases where Hough lines don't align with the pipe rectangle."""
    ih, iw = gray.shape
    _, black_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    pa1 = np.array([a[0], a[1]], dtype=float)
    pa2 = np.array([a[2], a[3]], dtype=float)
    pb1 = np.array([b[0], b[1]], dtype=float)
    pb2 = np.array([b[2], b[3]], dtype=float)

    dir_a = pa2 - pa1
    dir_b = pb2 - pb1
    if np.dot(dir_a, dir_b) < 0:
        pb1, pb2 = pb2, pb1

    dir_v = (dir_a / max(np.linalg.norm(dir_a), 1) +
             dir_b / max(np.linalg.norm(dir_b), 1))
    norm = np.linalg.norm(dir_v)
    if norm < 0.01:
        return (a, b)
    dir_v /= norm
    perp = np.array([-dir_v[1], dir_v[0]])

    mid_a = (pa1 + pa2) / 2
    mid_b = (pb1 + pb2) / 2
    pipe_center = (mid_a + mid_b) / 2
    half_gap = np.linalg.norm(mid_b - mid_a) / 2 * 0.8

    # Scan along pipe direction to find dark bands (end-walls)
    n_across = 7
    scan_range = int(max(_line_length(*a), _line_length(*b)) * 0.8)
    dark_positions = []
    for t in range(-scan_range, scan_range + 1, 2):
        dark_count = 0
        total = 0
        for s_idx in range(n_across):
            s = -half_gap + (s_idx / (n_across - 1)) * 2 * half_gap
            px = int(pipe_center[0] + dir_v[0] * t + perp[0] * s)
            py = int(pipe_center[1] + dir_v[1] * t + perp[1] * s)
            if 0 <= py < ih and 0 <= px < iw:
                total += 1
                if black_mask[py, px] > 0:
                    dark_count += 1
        if total > 0 and dark_count / total > 0.5:
            dark_positions.append(t)

    if len(dark_positions) < 2:
        return (a, b)

    # Group into clusters
    clusters = []
    current = [dark_positions[0]]
    for dp in dark_positions[1:]:
        if dp - current[-1] <= 10:
            current.append(dp)
        else:
            clusters.append(current)
            current = [dp]
    clusters.append(current)

    if len(clusters) < 2:
        return (a, b)

    # First and last clusters are the end-walls
    wall1_t = (clusters[0][0] + clusters[0][-1]) / 2
    wall2_t = (clusters[-1][0] + clusters[-1][-1]) / 2

    # Perpendicular offsets
    pa_perp = np.dot(mid_a - pipe_center, perp)
    pb_perp = np.dot(mid_b - pipe_center, perp)

    new_a1 = pipe_center + dir_v * wall1_t + perp * pa_perp
    new_a2 = pipe_center + dir_v * wall2_t + perp * pa_perp
    new_b1 = pipe_center + dir_v * wall1_t + perp * pb_perp
    new_b2 = pipe_center + dir_v * wall2_t + perp * pb_perp

    return (
        (int(new_a1[0]), int(new_a1[1]), int(new_a2[0]), int(new_a2[1])),
        (int(new_b1[0]), int(new_b1[1]), int(new_b2[0]), int(new_b2[1])),
    )


def _find_parallel_pairs_diag(lines):
    """Find parallel pairs among diagonal lines with wider gap tolerance (up to 150px)."""
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
            if _angle_diff(angles[i], angles[j]) > _PARALLEL_TOL:
                continue
            mid_j = _midpoint(*lines[j])
            perp = _perp_dist(mid_j[0], mid_j[1], float(lines[i][0]), float(lines[i][1]),
                               float(lines[i][2]), float(lines[i][3]))
            if not (_PARALLEL_DIST_MIN <= perp <= 150):
                continue
            mid_i = _midpoint(*lines[i])
            avg_len = (_line_length(*lines[i]) + _line_length(*lines[j])) / 2
            mid_dist = math.hypot(mid_i[0] - mid_j[0], mid_i[1] - mid_j[1])
            if mid_dist < avg_len * 0.9 and perp < best_perp:
                best_j = j
                best_perp = perp
        if best_j >= 0:
            len_i = _line_length(*lines[i])
            len_j = _line_length(*lines[best_j])
            ratio = min(len_i, len_j) / max(len_i, len_j)
            if ratio < 0.5:
                continue
            # Midpoint alignment
            mid_i = _midpoint(*lines[i])
            mid_j = _midpoint(*lines[best_j])
            mid_dist = math.hypot(mid_i[0] - mid_j[0], mid_i[1] - mid_j[1])
            avg_len = (_line_length(*lines[i]) + _line_length(*lines[best_j])) / 2
            if mid_dist > avg_len * 0.6:
                continue
            used[i] = True
            used[best_j] = True
            pairs.append((lines[i], lines[best_j]))
    return pairs


def _validate_diagonal_interior(gray, a, b):
    """Check that the region between two diagonal parallel lines has a mostly
    white/light interior (pipe interior). Samples a grid of points between
    the two lines."""
    ih, iw = gray.shape
    mid_a = _midpoint(*a)
    mid_b = _midpoint(*b)

    # Direction along the pipe
    dir_a = np.array([a[2] - a[0], a[3] - a[1]], dtype=float)
    length = np.linalg.norm(dir_a)
    if length < 1:
        return False
    dir_a /= length

    # Perpendicular direction (from line a toward line b)
    perp = np.array([mid_b[0] - mid_a[0], mid_b[1] - mid_a[1]], dtype=float)
    perp_len = np.linalg.norm(perp)
    if perp_len < 1:
        return False
    perp /= perp_len

    center = ((mid_a[0] + mid_b[0]) / 2, (mid_a[1] + mid_b[1]) / 2)

    # Sample a grid: along the pipe length and across the width
    n_along = max(int(length // 15), 8)
    n_across = 5
    white_count = 0
    total = 0
    half_len = length * 0.4  # sample inner 80% of length
    half_gap = perp_len * 0.3  # sample inner 60% of width

    for i in range(n_along):
        t = -half_len + (i / max(n_along - 1, 1)) * (2 * half_len)
        for j in range(n_across):
            s = -half_gap + (j / max(n_across - 1, 1)) * (2 * half_gap)
            px = int(center[0] + dir_a[0] * t + perp[0] * s)
            py = int(center[1] + dir_a[1] * t + perp[1] * s)
            if 0 <= py < ih and 0 <= px < iw:
                total += 1
                if gray[py, px] > 150:
                    white_count += 1

    if total == 0:
        return False
    return (white_count / total) > 0.40


# ---------------------------------------------------------------------------
# TERTIARY: Black-contour rectangle detection (catches text-heavy pipes)
# ---------------------------------------------------------------------------

def _detect_black_rect_pipes(image, roi, existing_pairs, threshold=50):
    """Find rectangular pipe outlines directly from black contours.
    Catches pipes where text fills the interior (e.g. '18"', '22" x 14"').
    These pipes have clear black rectangular borders but broken white interiors."""
    rx, ry, rw, rh = roi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    ih, iw = gray.shape

    roi_gray = gray[ry:ry + rh, rx:rx + rw]

    # Black mask at strict threshold
    _, black_mask = cv2.threshold(roi_gray, 60, 255, cv2.THRESH_BINARY_INV)

    # Dilate to connect nearby black pixels into continuous borders
    dk = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(black_mask, dk, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Build existing centers for dedup
    existing_centers = []
    for a, b in existing_pairs:
        existing_centers.append(((a[0]+a[2]+b[0]+b[2])/4, (a[1]+a[3]+b[1]+b[3])/4))

    new_pairs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000 or area > 200000:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        if w < 1 or h < 1:
            continue

        # Must be elongated
        aspect = max(w, h) / min(w, h)
        if aspect < 2.0:
            continue

        narrow = min(w, h)
        long = max(w, h)
        if narrow < 15 or narrow > 120 or long < 100:
            continue

        # Must be axis-aligned
        if 5 < abs(angle) < 85 and 5 < abs(angle - 90) < 85:
            continue

        # Rectangularity check
        rect_area = w * h
        rectangularity = area / rect_area if rect_area > 0 else 0
        if rectangularity < 0.4:
            continue

        # Convert to absolute coordinates
        abs_cx = cx + rx
        abs_cy = cy + ry

        # Check not already detected
        if any(abs(abs_cx - ecx) < threshold and abs(abs_cy - ecy) < threshold
               for ecx, ecy in existing_centers):
            continue

        # Verify this is a pipe: check that the border lines are thick enough
        # by measuring black pixel density along the contour
        box = cv2.boxPoints(rect).astype(int)

        # Create a mask of just the contour border region
        border_mask = np.zeros_like(roi_gray)
        cv2.drawContours(border_mask, [cnt], -1, 255, 2)
        border_pixels = roi_gray[border_mask > 0]
        if len(border_pixels) == 0:
            continue
        # Border pixels should be dark (black pipe walls)
        if np.median(border_pixels) > 80:
            continue

        # Build line pair from the rectangle
        is_h = w > h if abs(angle) < 45 else h > w
        abs_x = int(abs_cx - max(w, h) / 2)
        abs_y = int(abs_cy - min(w, h) / 2)
        bw_r = int(max(w, h))
        bh_r = int(min(w, h))

        if is_h:
            a = (abs_x, abs_y, abs_x + bw_r, abs_y)
            b = (abs_x, abs_y + bh_r, abs_x + bw_r, abs_y + bh_r)
        else:
            a = (abs_x, abs_y, abs_x, abs_y + bw_r)
            b = (abs_x + bh_r, abs_y, abs_x + bh_r, abs_y + bw_r)

        new_pairs.append((a, b))
        existing_centers.append((abs_cx, abs_cy))

    logger.info("Black-rect detection: %d additional pipes", len(new_pairs))
    return new_pairs


# ---------------------------------------------------------------------------
# SECONDARY: Hough parallel-pair detection
# ---------------------------------------------------------------------------

_CARDINAL_TOL = 12.0
_PARALLEL_TOL = 8.0
_PARALLEL_DIST_MIN = 8
_PARALLEL_DIST_MAX = 100
_MERGE_DIST = 30
_MIN_SEG_LEN = 100


def _line_length(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def _line_angle(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180

def _midpoint(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def _angle_diff(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)

def _perp_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return math.hypot(px - x1, py - y1)
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / length

def _is_cardinal(angle):
    return (angle <= _CARDINAL_TOL or angle >= 180 - _CARDINAL_TOL or
            abs(angle - 90) <= _CARDINAL_TOL)


def _detect_lines(image, roi):
    """Detect line segments via morphology + Hough.
    Uses 0.5x downscale for speed, then scales coords back."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    scale = 0.5
    small = cv2.resize(gray, (int(gray.shape[1] * scale), int(gray.shape[0] * scale)))

    binary = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 4)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk)

    # Diagonal kernels for angled pipes
    diag1 = np.eye(10, dtype=np.uint8)  # 45 degrees
    diag1_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, diag1)
    diag2 = np.fliplr(np.eye(10, dtype=np.uint8))  # 135 degrees
    diag2_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, diag2)

    combined = cv2.bitwise_or(horiz, vert)
    combined = cv2.bitwise_or(combined, diag1_lines)
    combined = cv2.bitwise_or(combined, diag2_lines)
    dk = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.dilate(combined, dk, iterations=1)

    raw = cv2.HoughLinesP(combined, 1, np.pi / 180, 30,
                           minLineLength=40, maxLineGap=8)
    if raw is None:
        return []

    rx, ry, rw, rh = roi
    rx2, ry2 = rx + rw, ry + rh
    lines = []
    for l in raw:
        # Scale back to full resolution
        x1 = int(l[0][0] / scale)
        y1 = int(l[0][1] / scale)
        x2 = int(l[0][2] / scale)
        y2 = int(l[0][3] / scale)
        if rx <= x1 <= rx2 and ry <= y1 <= ry2 and rx <= x2 <= rx2 and ry <= y2 <= ry2:
            # Check line darkness — only accept truly black lines, not gray walls
            n_pts = max(int(_line_length(x1, y1, x2, y2) // 4), 5)
            xs = np.linspace(x1, x2, n_pts, dtype=int)
            ys = np.linspace(y1, y2, n_pts, dtype=int)
            xs = np.clip(xs, 0, gray.shape[1] - 1)
            ys = np.clip(ys, 0, gray.shape[0] - 1)
            median_val = np.median(gray[ys, xs])
            if median_val > 100:
                continue  # gray line, not a pipe wall
            lines.append((x1, y1, x2, y2))

    lines = _merge_collinear(lines)
    lines = [s for s in lines if _line_length(*s) >= _MIN_SEG_LEN]
    logger.info("Hough: %d line segments", len(lines))
    return lines


def _merge_collinear(lines):
    if not lines:
        return []
    used = [False] * len(lines)
    merged = []
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
                if _angle_diff(angles[i], angles[j]) > _PARALLEL_TOL:
                    continue
                mid_j = _midpoint(*lines[j])
                perp = _perp_dist(mid_j[0], mid_j[1], float(lines[i][0]), float(lines[i][1]),
                                   float(lines[i][2]), float(lines[i][3]))
                if perp > 8:
                    continue
                min_d = min(math.hypot(px - qx, py - qy)
                            for px, py in pts for qx, qy in [lines[j][:2], lines[j][2:]])
                if min_d <= _MERGE_DIST:
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


def _find_parallel_pairs(lines):
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
            if _angle_diff(angles[i], angles[j]) > _PARALLEL_TOL:
                continue
            mid_j = _midpoint(*lines[j])
            perp = _perp_dist(mid_j[0], mid_j[1], float(lines[i][0]), float(lines[i][1]),
                               float(lines[i][2]), float(lines[i][3]))
            if not (_PARALLEL_DIST_MIN <= perp <= _PARALLEL_DIST_MAX):
                continue
            mid_i = _midpoint(*lines[i])
            avg_len = (_line_length(*lines[i]) + _line_length(*lines[j])) / 2
            mid_dist = math.hypot(mid_i[0] - mid_j[0], mid_i[1] - mid_j[1])
            if mid_dist < avg_len * 0.9 and perp < best_perp:
                best_j = j
                best_perp = perp
        if best_j >= 0:
            # Reject if line lengths differ too much (not a proper rectangle)
            len_i = _line_length(*lines[i])
            len_j = _line_length(*lines[best_j])
            ratio = min(len_i, len_j) / max(len_i, len_j)
            if ratio < 0.6:
                continue
            # For non-cardinal lines, require tighter midpoint alignment
            if not _is_cardinal(angles[i]):
                mid_i = _midpoint(*lines[i])
                mid_j = _midpoint(*lines[best_j])
                mid_dist = math.hypot(mid_i[0] - mid_j[0], mid_i[1] - mid_j[1])
                avg_len = (_line_length(*lines[i]) + _line_length(*lines[best_j])) / 2
                if mid_dist > avg_len * 0.6:
                    continue
            used[i] = True
            used[best_j] = True
            pairs.append((lines[i], lines[best_j]))
    logger.info("Hough parallel pairs: %d", len(pairs))
    if pairs:
        perps = []
        for a, b in pairs:
            mid_b = _midpoint(*b)
            p = _perp_dist(mid_b[0], mid_b[1], float(a[0]), float(a[1]),
                           float(a[2]), float(a[3]))
            perps.append(int(p))
        logger.info("Hough pair gaps: min=%d max=%d values=%s",
                    min(perps), max(perps), sorted(perps))
    return pairs


# ---------------------------------------------------------------------------
# Merge + deduplicate + fill
# ---------------------------------------------------------------------------

def _channels_to_pairs(channels):
    pairs = []
    for ch in channels:
        if ch["orient"] == "D" and "diag_pair" in ch:
            pairs.append(ch["diag_pair"])
        else:
            x, y, w, h = ch["x"], ch["y"], ch["w"], ch["h"]
            if ch["orient"] == "H":
                pairs.append(((x, y, x + w, y), (x, y + h, x + w, y + h)))
            else:
                pairs.append(((x, y, x, y + h), (x + w, y, x + w, y + h)))
    return pairs


def _validate_hough_pair(gray, a, b):
    """Check that the region between two parallel lines is mostly white/gray
    (pipe interior). Rejects curved shapes like half-circles where the arc
    cuts through the interior region."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ih, iw = gray.shape

    angle = _line_angle(*a) if _line_length(*a) > 0 else 0
    is_h = (angle <= _CARDINAL_TOL or angle >= 180 - _CARDINAL_TOL)

    if is_h:
        # Sort endpoints left-to-right
        if ax1 > ax2: ax1, ay1, ax2, ay2 = ax2, ay2, ax1, ay1
        if bx1 > bx2: bx1, by1, bx2, by2 = bx2, by2, bx1, by1
        xs = max(ax1, bx1)
        xe = min(ax2, bx2)
        if xe <= xs:
            return True  # no overlap, can't check
        y_top = min(ay1, by1)
        y_bot = max(ay1, by1)
        if y_bot - y_top < 4:
            return True
        # Sample the interior strip
        y1c = max(0, min(y_top + 2, ih - 1))
        y2c = max(0, min(y_bot - 2, ih - 1))
        x1c = max(0, min(int(xs), iw - 1))
        x2c = max(0, min(int(xe), iw - 1))
    else:
        # Sort endpoints top-to-bottom
        if ay1 > ay2: ax1, ay1, ax2, ay2 = ax2, ay2, ax1, ay1
        if by1 > by2: bx1, by1, bx2, by2 = bx2, by2, bx1, by1
        ys = max(ay1, by1)
        ye = min(ay2, by2)
        if ye <= ys:
            return True
        x_left = min(ax1, bx1)
        x_right = max(ax1, bx1)
        if x_right - x_left < 4:
            return True
        x1c = max(0, min(x_left + 2, iw - 1))
        x2c = max(0, min(x_right - 2, iw - 1))
        y1c = max(0, min(int(ys), ih - 1))
        y2c = max(0, min(int(ye), ih - 1))

    if y2c <= y1c or x2c <= x1c:
        return True

    strip = gray[y1c:y2c, x1c:x2c]
    if strip.size == 0:
        return True

    # Pipe interiors are white (>180) or gray (>80). Black pixels (<80) indicate
    # curve walls cutting through the interior.
    white_ratio = np.count_nonzero(strip > 80) / strip.size
    return white_ratio > 0.6


def _deduplicate(channel_pairs, hough_pairs, threshold=40):
    """Channel pairs take priority; add Hough pairs only if non-overlapping."""
    all_p = list(channel_pairs)
    centers = []
    for a, b in channel_pairs:
        centers.append(((a[0]+a[2]+b[0]+b[2])/4, (a[1]+a[3]+b[1]+b[3])/4))

    for a, b in hough_pairs:
        cx = (a[0]+a[2]+b[0]+b[2])/4
        cy = (a[1]+a[3]+b[1]+b[3])/4
        if not any(abs(cx-ccx) < threshold and abs(cy-ccy) < threshold
                   for ccx, ccy in centers):
            all_p.append((a, b))
            centers.append((cx, cy))
    return all_p


def _merge_overlapping_pairs(pairs, perp_thresh=40, gap_thresh=30):
    """Merge pipe pairs that are collinear/parallel and overlap or nearly touch.

    Two pipe pairs are merged when:
    1. They have similar angles (parallel).
    2. Their centerlines are close in the perpendicular direction (same lane).
    3. They overlap or have a small gap along the pipe direction.

    Returns a new list with overlapping pipes merged into single longer pipes.
    """
    if len(pairs) <= 1:
        return list(pairs)

    # Represent each pair by its centerline segment and angle
    def _center_seg(pair):
        a, b = pair
        cx1 = (a[0] + b[0]) / 2.0
        cy1 = (a[1] + b[1]) / 2.0
        cx2 = (a[2] + b[2]) / 2.0
        cy2 = (a[3] + b[3]) / 2.0
        return (cx1, cy1, cx2, cy2)

    def _pipe_width(pair):
        a, b = pair
        w1 = math.hypot(a[0] - b[0], a[1] - b[1])
        w2 = math.hypot(a[2] - b[2], a[3] - b[3])
        return (w1 + w2) / 2.0

    centers = [_center_seg(p) for p in pairs]
    angles = [_line_angle(*c) for c in centers]
    widths = [_pipe_width(p) for p in pairs]

    used = [False] * len(pairs)
    merged = []

    for i in range(len(pairs)):
        if used[i]:
            continue
        used[i] = True

        # Collect group of overlapping pipes starting from pipe i
        group_indices = [i]
        changed = True
        while changed:
            changed = False
            for j in range(len(pairs)):
                if used[j]:
                    continue
                # Check against any member of the current group
                for gi in group_indices:
                    if _angle_diff(angles[gi], angles[j]) > _PARALLEL_TOL:
                        continue
                    # Check perpendicular distance between centerlines
                    ci = centers[gi]
                    cj = centers[j]
                    mid_j = _midpoint(*cj)
                    perp = _perp_dist(mid_j[0], mid_j[1],
                                      float(ci[0]), float(ci[1]),
                                      float(ci[2]), float(ci[3]))
                    mid_i = _midpoint(*ci)
                    perp2 = _perp_dist(mid_i[0], mid_i[1],
                                       float(cj[0]), float(cj[1]),
                                       float(cj[2]), float(cj[3]))
                    if min(perp, perp2) > perp_thresh:
                        continue
                    # Check overlap along the pipe direction
                    ang_rad = math.radians(angles[gi])
                    ca, sa = math.cos(ang_rad), math.sin(ang_rad)
                    proj_i = sorted([ci[0]*ca + ci[1]*sa, ci[2]*ca + ci[3]*sa])
                    proj_j = sorted([cj[0]*ca + cj[1]*sa, cj[2]*ca + cj[3]*sa])
                    overlap = min(proj_i[1], proj_j[1]) - max(proj_i[0], proj_j[0])
                    if overlap >= -gap_thresh:
                        # They overlap or are close enough to merge
                        used[j] = True
                        group_indices.append(j)
                        changed = True
                        break

        if len(group_indices) == 1:
            merged.append(pairs[i])
            continue

        # Merge the group: find the dominant angle, project all endpoints,
        # and build a single merged pair spanning the full extent.
        ref_angle = angles[group_indices[0]]
        ang_rad = math.radians(ref_angle)
        ca, sa = math.cos(ang_rad), math.sin(ang_rad)
        perp_x, perp_y = -sa, ca  # perpendicular direction

        # Collect all wall-line endpoints from the group
        all_a_pts = []
        all_b_pts = []
        for gi in group_indices:
            a, b = pairs[gi]
            all_a_pts.extend([(a[0], a[1]), (a[2], a[3])])
            all_b_pts.extend([(b[0], b[1]), (b[2], b[3])])

        # Project onto pipe direction to find extent
        all_pts = all_a_pts + all_b_pts
        projs = [(px * ca + py * sa, px, py) for px, py in all_pts]
        projs.sort()

        # Average perpendicular offset for wall A and wall B
        avg_width = sum(widths[gi] for gi in group_indices) / len(group_indices)
        half_w = avg_width / 2.0

        # Centerline from start to end
        t_min = projs[0][0]
        t_max = projs[-1][0]
        # Average perpendicular position of all centerline midpoints
        perp_positions = []
        for gi in group_indices:
            mid = _midpoint(*centers[gi])
            perp_positions.append(mid[0] * perp_x + mid[1] * perp_y)
        avg_perp = sum(perp_positions) / len(perp_positions)

        # Reconstruct the two wall lines
        # Origin on the average perpendicular line
        origin_perp = avg_perp
        c_start_x = ca * t_min + perp_x * origin_perp
        c_start_y = sa * t_min + perp_y * origin_perp
        c_end_x = ca * t_max + perp_x * origin_perp
        c_end_y = sa * t_max + perp_y * origin_perp

        # Wall A: offset by +half_w perpendicular
        a_line = (int(round(c_start_x + perp_x * half_w)),
                  int(round(c_start_y + perp_y * half_w)),
                  int(round(c_end_x + perp_x * half_w)),
                  int(round(c_end_y + perp_y * half_w)))
        # Wall B: offset by -half_w perpendicular
        b_line = (int(round(c_start_x - perp_x * half_w)),
                  int(round(c_start_y - perp_y * half_w)),
                  int(round(c_end_x - perp_x * half_w)),
                  int(round(c_end_y - perp_y * half_w)))

        merged.append((a_line, b_line))

    logger.info("Merged overlapping pipes: %d -> %d", len(pairs), len(merged))
    return merged


def _draw_centerline(image, a, b, thickness=PIPE_THICKNESS):
    """Draw a uniform-thickness blue rectangle centered between two parallel lines.

    For cardinal (H/V) pipes, draws an axis-aligned rectangle.
    For diagonal pipes, draws a rotated filled polygon.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    angle = _line_angle(*a) if _line_length(*a) > 0 else 0
    is_h = (angle <= _CARDINAL_TOL or angle >= 180 - _CARDINAL_TOL)
    is_v = abs(angle - 90) <= _CARDINAL_TOL
    half = thickness // 2

    if is_h:
        if ax1 > ax2: ax1, ay1, ax2, ay2 = ax2, ay2, ax1, ay1
        if bx1 > bx2: bx1, by1, bx2, by2 = bx2, by2, bx1, by1
        xs = max(ax1, bx1)
        xe = min(ax2, bx2)
        if xe <= xs:
            xs, xe = min(ax1, bx1), max(ax2, bx2)
        cy = int(((ay1 + ay2) / 2 + (by1 + by2) / 2) / 2)
        cv2.rectangle(image, (int(xs), cy - half), (int(xe), cy + half),
                       PIPE_BLUE, thickness=-1)
    elif is_v:
        if ay1 > ay2: ax1, ay1, ax2, ay2 = ax2, ay2, ax1, ay1
        if by1 > by2: bx1, by1, bx2, by2 = bx2, by2, bx1, by1
        ys = max(ay1, by1)
        ye = min(ay2, by2)
        if ye <= ys:
            ys, ye = min(ay1, by1), max(ay2, by2)
        else:
            # Extend slightly beyond overlap to cover pipe caps
            extend = min(abs(ay1 - by1), abs(ay2 - by2), 15)
            ys -= extend
            ye += extend
        cx = int(((ax1 + ax2) / 2 + (bx1 + bx2) / 2) / 2)
        cv2.rectangle(image, (cx - half, int(ys)), (cx + half, int(ye)),
                       PIPE_BLUE, thickness=-1)
    else:
        # Diagonal pipe: use the overlap of the two parallel lines
        pa1 = np.array([ax1, ay1], dtype=float)
        pa2 = np.array([ax2, ay2], dtype=float)
        pb1 = np.array([bx1, by1], dtype=float)
        pb2 = np.array([bx2, by2], dtype=float)

        # Ensure both lines point the same direction
        dir_a = pa2 - pa1
        dir_b = pb2 - pb1
        if np.dot(dir_a, dir_b) < 0:
            pb1, pb2 = pb2, pb1
            dir_b = -dir_b

        # Average direction
        dir_v = (dir_a / max(np.linalg.norm(dir_a), 1) +
                 dir_b / max(np.linalg.norm(dir_b), 1))
        norm = np.linalg.norm(dir_v)
        if norm < 0.01:
            return
        dir_v /= norm
        perp = np.array([-dir_v[1], dir_v[0]])

        # Project all 4 endpoints onto the pipe direction
        origin = (pa1 + pa2 + pb1 + pb2) / 4.0
        ta1 = np.dot(pa1 - origin, dir_v)
        ta2 = np.dot(pa2 - origin, dir_v)
        tb1 = np.dot(pb1 - origin, dir_v)
        tb2 = np.dot(pb2 - origin, dir_v)

        # Overlap: where both lines exist
        t_start = max(min(ta1, ta2), min(tb1, tb2))
        t_end = min(max(ta1, ta2), max(tb1, tb2))
        if t_end <= t_start:
            t_start = min(ta1, ta2, tb1, tb2)
            t_end = max(ta1, ta2, tb1, tb2)

        # Center between the two walls
        mid_a = (pa1 + pa2) / 2
        mid_b = (pb1 + pb2) / 2
        center = (mid_a + mid_b) / 2

        p1 = center + dir_v * (t_start - np.dot(center - origin, dir_v)) + perp * half
        p2 = center + dir_v * (t_end - np.dot(center - origin, dir_v)) + perp * half
        p3 = center + dir_v * (t_end - np.dot(center - origin, dir_v)) - perp * half
        p4 = center + dir_v * (t_start - np.dot(center - origin, dir_v)) - perp * half

        pts = np.array([p1, p2, p3, p4], dtype=np.int32)
        cv2.fillPoly(image, [pts], PIPE_BLUE)



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(input_pdf="input/testset2.pdf", output_dir="output", dpi=300):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    pages = render_pdf(input_pdf, dpi)
    if not pages:
        logger.error("No pages rendered")
        return ""

    page = pages[0]
    image = page.image
    logger.info("Page 1: %dx%d at %d DPI", page.width, page.height, page.dpi)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Drawing area
    roi = _find_drawing_area(image)
    rx, ry, _, _ = roi

    # 2. Primary: white channel detection
    channels, labels = _detect_white_channels(image, roi)

    # 2b. Gray channel detection — skipped
    gray_channels = []

    # 2c. Diagonal pipe detection
    diag_channels = _detect_diagonal_pipes(image, roi, channels)

    all_channels = channels + diag_channels

    channel_pairs = _channels_to_pairs(all_channels)
    logger.info("Primary: %d white + %d gray + %d diagonal = %d channel pipes",
                len(channels), len(gray_channels), len(diag_channels), len(all_channels))

    # 3. Secondary: Hough parallel pairs
    lines = _detect_lines(image, roi)
    hough_pairs = _find_parallel_pairs(lines)

    # Filter Hough pairs: reject curved shapes (half-circles)
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hough_pairs = [p for p in hough_pairs
                   if _validate_hough_pair(gray_full, p[0], p[1])]

    # Reject Hough pairs near the ROI boundary (title block / border artifacts)
    rx2, ry2 = rx + roi[2], ry + roi[3]
    edge_margin = 80  # pixels from ROI edge
    def _pair_inside_margin(pair):
        a, b = pair
        for x, y in [(a[0], a[1]), (a[2], a[3]), (b[0], b[1]), (b[2], b[3])]:
            if x < rx + edge_margin or x > rx2 - edge_margin:
                return False
            if y < ry + edge_margin or y > ry2 - edge_margin:
                return False
        return True
    hough_pairs = [p for p in hough_pairs if _pair_inside_margin(p)]

    # Reject Hough pairs with short average line length (random noise matches)
    def _avg_len_ok(pair):
        a, b = pair
        avg = (_line_length(*a) + _line_length(*b)) / 2
        return avg >= 150
    hough_pairs = [p for p in hough_pairs if _avg_len_ok(p)]

    # Reject Hough pairs with dark/non-white interior (not pipe interiors)
    def _interior_bright(pair):
        a, b = pair
        n_along = 15
        n_across = 5
        white = 0
        total = 0
        for t in np.linspace(0.1, 0.9, n_along):
            for s in np.linspace(0.2, 0.8, n_across):
                px = int(a[0] + (a[2]-a[0])*t + (b[0]-a[0])*s + (b[2]-b[0]-a[2]+a[0])*t*s)
                py = int(a[1] + (a[3]-a[1])*t + (b[1]-a[1])*s + (b[3]-b[1]-a[3]+a[1])*t*s)
                if 0 <= py < gray_full.shape[0] and 0 <= px < gray_full.shape[1]:
                    total += 1
                    if gray_full[py, px] > 150:
                        white += 1
        return total == 0 or (white / total) > 0.50
    hough_pairs = [p for p in hough_pairs if _interior_bright(p)]

    logger.info("Secondary: %d Hough pairs (after all filters)", len(hough_pairs))

    # 4. Merge
    all_pairs = _deduplicate(channel_pairs, hough_pairs)
    logger.info("Total after dedup: %d", len(all_pairs))

    # 5. Tertiary: black-contour rectangle detection for text-heavy pipes
    black_rect_pairs = _detect_black_rect_pipes(image, roi, all_pairs)
    all_pairs = all_pairs + black_rect_pairs
    logger.info("Total with black-rect: %d", len(all_pairs))

    # 6. Merge overlapping pipes into single segments
    all_pairs = _merge_overlapping_pairs(all_pairs)

    # 7. Draw uniform-thickness blue centerlines
    output = image.copy()

    for pair in all_pairs:
        _draw_centerline(output, pair[0], pair[1])

    out_path = os.path.join(output_dir, "pipes_bedrock.png")
    cv2.imwrite(out_path, output)
    logger.info("Saved to %s", out_path)

    print(f"\nDetected {len(all_pairs)} pipes. Output: {out_path}")
    print(f"  {len(all_channels)} from channels ({len(channels)} white + {len(gray_channels)} gray), "
          f"{len(all_pairs) - len(channel_pairs)} from Hough")
    return out_path


if __name__ == "__main__":
    run()
