"""Annotation engine for rendering duct overlays onto drawing pages.

Draws a blue centerline along detected duct paths and renders
pipe length labels. Saves annotated images as PNG files.
"""

import math
import os

import cv2
import numpy as np

from models import (
    AnnotatedImage,
    DuctSegment,
    DuctShape,
    DrawingScale,
    ExtractedNotes,
    PageImage,
    PressureClass,
)

# Single blue color for all pipe markings (BGR)
PIPE_COLOR = (255, 100, 0)


def _pipe_length_text(duct: DuctSegment, scale: DrawingScale, dpi: int) -> str:
    """Compute real-world pipe length and return a formatted string."""
    total_px = 0.0
    for i in range(len(duct.polyline) - 1):
        x1, y1 = duct.polyline[i]
        x2, y2 = duct.polyline[i + 1]
        total_px += math.hypot(x2 - x1, y2 - y1)

    if scale.ratio and scale.ratio > 0:
        drawing_inches = total_px / dpi
        real_inches = drawing_inches / scale.ratio
        feet = int(real_inches // 12)
        inches = int(round(real_inches % 12))
        if feet > 0:
            return f"{feet}'-{inches}\""
        return f'{inches}"'
    return f"{total_px:.0f}px"


def annotate_page(
    page_image: PageImage,
    ducts: list[DuctSegment],
    scale: DrawingScale,
    notes: ExtractedNotes,
) -> AnnotatedImage:
    """Render blue centerline on each pipe with length label."""
    image = page_image.image.copy()
    dpi = page_image.dpi

    for duct in ducts:
        if len(duct.polyline) >= 2:
            pts = np.array(duct.polyline, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=False, color=PIPE_COLOR, thickness=2)

            # Length label at midpoint
            length_text = _pipe_length_text(duct, scale, dpi)
            mid_idx = len(duct.polyline) // 2
            if len(duct.polyline) % 2 == 0:
                ax, ay = duct.polyline[mid_idx - 1]
                bx, by = duct.polyline[mid_idx]
                lx, ly = (ax + bx) // 2, (ay + by) // 2
            else:
                lx, ly = duct.polyline[mid_idx]

            ly = max(ly - 12, 15)
            cv2.putText(
                image, length_text, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, PIPE_COLOR, 1, cv2.LINE_AA,
            )

    return AnnotatedImage(
        page_number=page_image.page_number,
        image=image,
        ducts=ducts,
    )


def save_annotated_image(image: AnnotatedImage, output_dir: str, page_num: int) -> str:
    """Save the annotated image as a PNG file."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"page_{page_num:03d}.png"
    file_path = os.path.join(output_dir, filename)
    cv2.imwrite(file_path, image.image)
    image.file_path = file_path
    return file_path
