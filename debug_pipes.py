"""Generate numbered pipe clips and overview image for review."""
import cv2
import numpy as np
import os
from pipe_marker import (
    _find_drawing_area, _detect_white_channels, _detect_diagonal_pipes,
    _channels_to_pairs, _detect_lines, _find_parallel_pairs,
    _validate_hough_pair, _deduplicate, _detect_black_rect_pipes,
    _draw_centerline, _line_length, _line_angle, _midpoint, _perp_dist,
    PIPE_BLUE, PIPE_THICKNESS,
)
from pdf_renderer import render_pdf
import logging

logging.basicConfig(level=logging.WARNING)

pages = render_pdf("input/testset2.pdf", 300)
image = pages[0].image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ih, iw = gray.shape

roi = _find_drawing_area(image)
rx, ry, rw, rh = roi

channels, labels = _detect_white_channels(image, roi)
diag_channels = _detect_diagonal_pipes(image, roi, channels)
all_channels = channels + diag_channels
channel_pairs = _channels_to_pairs(all_channels)

lines = _detect_lines(image, roi)
hough_pairs = _find_parallel_pairs(lines)
gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hough_pairs = [p for p in hough_pairs if _validate_hough_pair(gray_full, p[0], p[1])]

# Edge margin filter (same as in run())
rx2, ry2 = rx + rw, ry + rh
edge_margin = 80
def _pair_inside_margin(pair):
    a, b = pair
    for x, y in [(a[0], a[1]), (a[2], a[3]), (b[0], b[1]), (b[2], b[3])]:
        if x < rx + edge_margin or x > rx2 - edge_margin:
            return False
        if y < ry + edge_margin or y > ry2 - edge_margin:
            return False
    return True

hough_pairs = [p for p in hough_pairs if _pair_inside_margin(p)]

# Avg length filter (same as in run())
hough_pairs = [p for p in hough_pairs
               if (_line_length(*p[0]) + _line_length(*p[1])) / 2 >= 150]

# Interior brightness filter (same as in run())
def _interior_bright(pair):
    a, b = pair
    white = 0; total = 0
    for t in np.linspace(0.1, 0.9, 15):
        for s in np.linspace(0.2, 0.8, 5):
            px = int(a[0] + (a[2]-a[0])*t + (b[0]-a[0])*s + (b[2]-b[0]-a[2]+a[0])*t*s)
            py = int(a[1] + (a[3]-a[1])*t + (b[1]-a[1])*s + (b[3]-b[1]-a[3]+a[1])*t*s)
            if 0 <= py < ih and 0 <= px < iw:
                total += 1
                if gray[py, px] > 150:
                    white += 1
    return total == 0 or (white / total) > 0.50
hough_pairs = [p for p in hough_pairs if _interior_bright(p)]

all_pairs = _deduplicate(channel_pairs, hough_pairs)
black_rect_pairs = _detect_black_rect_pipes(image, roi, all_pairs)
all_pairs = all_pairs + black_rect_pairs

print(f"Total pipes: {len(all_pairs)}")
print(f"  Channels: {len(channel_pairs)} ({len(channels)} white, {len(diag_channels)} diag)")
print(f"  Hough added: {len(all_pairs) - len(channel_pairs)}")

# --- Generate individual clips ---
clip_dir = os.path.join("output", "pipe_clips")
os.makedirs(clip_dir, exist_ok=True)
# Clean old clips
for f in os.listdir(clip_dir):
    os.remove(os.path.join(clip_dir, f))

margin = 60
for idx, (a, b) in enumerate(all_pairs):
    cx = int((a[0] + a[2] + b[0] + b[2]) / 4)
    cy = int((a[1] + a[3] + b[1] + b[3]) / 4)
    # Bounding box of the pair
    xs = [a[0], a[2], b[0], b[2]]
    ys = [a[1], a[3], b[1], b[3]]
    x1 = max(0, min(xs) - margin)
    y1 = max(0, min(ys) - margin)
    x2 = min(iw, max(xs) + margin)
    y2 = min(ih, max(ys) + margin)

    # Draw blue on a copy and crop
    clip_img = image.copy()
    _draw_centerline(clip_img, a, b)
    crop = clip_img[y1:y2, x1:x2]

    source = "ch" if idx < len(channel_pairs) else "hough"
    gap = _perp_dist((b[0]+b[2])/2, (b[1]+b[3])/2,
                     float(a[0]), float(a[1]), float(a[2]), float(a[3]))
    angle = _line_angle(*a)

    fname = f"pipe_{idx:02d}_{source}_gap{int(gap)}_ang{int(angle)}.png"
    cv2.imwrite(os.path.join(clip_dir, fname), crop)

# --- Generate numbered overview ---
overview = image.copy()
for idx, (a, b) in enumerate(all_pairs):
    _draw_centerline(overview, a, b)
    cx = int((a[0] + a[2] + b[0] + b[2]) / 4)
    cy = int((a[1] + a[3] + b[1] + b[3]) / 4)
    cv2.putText(overview, str(idx), (cx - 10, cy - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

cv2.imwrite("output/pipes_numbered.png", overview)
print(f"\nSaved {len(all_pairs)} clips to {clip_dir}/")
print(f"Saved numbered overview to output/pipes_numbered.png")
