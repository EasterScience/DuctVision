"""Interactive Tkinter viewer for HVAC duct detection results.

Provides a split-panel layout with an annotated image (left) supporting
pan/zoom and click-to-inspect, and an information panel (right) showing
duct details, drawing scale, and extracted notes.
"""

from __future__ import annotations

import math
import os
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageTk

from models import (
    AnnotatedImage,
    DuctSegment,
    PipelineResult,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Standalone hit-test function (testable independently for Property 12)
# ---------------------------------------------------------------------------


def hit_test(
    x: int,
    y: int,
    ducts: list[DuctSegment],
    proximity: float = 20.0,
) -> DuctSegment | None:
    """Find the duct segment nearest to the click point *(x, y)*.

    Algorithm:
    1. Filter ducts that have a bounding box.
    2. For each duct, check if *(x, y)* is inside the bounding box expanded
       by *proximity* pixels on every side.
    3. Among candidates, return the one whose bounding-box centroid is
       closest to *(x, y)*.
    4. Return ``None`` when no duct is within proximity.
    """
    candidates: list[tuple[float, DuctSegment]] = []

    for duct in ducts:
        if duct.bounding_box is None:
            continue

        bx, by, bw, bh = duct.bounding_box

        # Expand the bounding box by the proximity threshold
        if not (
            bx - proximity <= x <= bx + bw + proximity
            and by - proximity <= y <= by + bh + proximity
        ):
            continue

        # Centroid of the bounding box
        cx = bx + bw / 2.0
        cy = by + bh / 2.0
        dist = math.hypot(x - cx, y - cy)
        candidates.append((dist, duct))

    if not candidates:
        return None

    # Return the nearest by centroid distance
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Viewer application
# ---------------------------------------------------------------------------


class _DuctViewer:
    """Internal Tkinter viewer application."""

    def __init__(
        self,
        root: tk.Tk,
        result: PipelineResult,
        annotated_images: list[AnnotatedImage],
    ) -> None:
        self._root = root
        self._result = result
        self._images = annotated_images
        self._current_page = 0

        # Pan / zoom state
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._drag_start: tuple[int, int] | None = None

        # Tk photo reference (prevent GC)
        self._tk_photo: ImageTk.PhotoImage | None = None

        self._build_ui()
        self._show_page(0)

    # ---- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        self._root.title("HVAC Duct Viewer")
        self._root.geometry("1200x800")

        # Main horizontal paned window
        paned = ttk.PanedWindow(self._root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Left panel — canvas
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=3)

        self._canvas = tk.Canvas(left_frame, bg="gray20")
        self._canvas.pack(fill=tk.BOTH, expand=True)

        # Canvas event bindings
        self._canvas.bind("<ButtonPress-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<MouseWheel>", self._on_mousewheel)
        # Linux scroll events
        self._canvas.bind("<Button-4>", lambda e: self._on_mousewheel_linux(e, 1))
        self._canvas.bind("<Button-5>", lambda e: self._on_mousewheel_linux(e, -1))

        # Right panel — info
        right_frame = ttk.Frame(paned, padding=10)
        paned.add(right_frame, weight=1)

        self._info_text = tk.Text(
            right_frame, wrap=tk.WORD, state=tk.DISABLED, font=("TkDefaultFont", 10)
        )
        info_scroll = ttk.Scrollbar(right_frame, command=self._info_text.yview)
        self._info_text.configure(yscrollcommand=info_scroll.set)
        self._info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Navigation bar
        nav_frame = ttk.Frame(self._root, padding=5)
        nav_frame.pack(fill=tk.X)

        self._prev_btn = ttk.Button(nav_frame, text="◀ Previous", command=self._prev_page)
        self._prev_btn.pack(side=tk.LEFT, padx=5)

        self._page_label = ttk.Label(nav_frame, text="")
        self._page_label.pack(side=tk.LEFT, expand=True)

        self._next_btn = ttk.Button(nav_frame, text="Next ▶", command=self._next_page)
        self._next_btn.pack(side=tk.RIGHT, padx=5)

    # ---- Page navigation ---------------------------------------------------

    def _show_page(self, index: int) -> None:
        self._current_page = index
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._redraw_canvas()
        self._update_info_panel()
        self._update_nav()

    def _prev_page(self) -> None:
        if self._current_page > 0:
            self._show_page(self._current_page - 1)

    def _next_page(self) -> None:
        if self._current_page < len(self._images) - 1:
            self._show_page(self._current_page + 1)

    def _update_nav(self) -> None:
        total = len(self._images)
        self._page_label.config(text=f"Page {self._current_page + 1} / {total}")
        self._prev_btn.state(["!disabled"] if self._current_page > 0 else ["disabled"])
        self._next_btn.state(
            ["!disabled"] if self._current_page < total - 1 else ["disabled"]
        )

    # ---- Canvas drawing ----------------------------------------------------

    def _redraw_canvas(self) -> None:
        self._canvas.delete("all")
        ann = self._images[self._current_page]
        img_bgr = ann.image

        # Convert BGR (OpenCV) → RGB → PIL
        img_rgb = img_bgr[:, :, ::-1] if img_bgr.ndim == 3 else img_bgr
        pil_img = Image.fromarray(img_rgb.astype(np.uint8))

        # Apply zoom
        new_w = max(1, int(pil_img.width * self._zoom))
        new_h = max(1, int(pil_img.height * self._zoom))
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        self._tk_photo = ImageTk.PhotoImage(pil_img)
        self._canvas.create_image(
            self._pan_x, self._pan_y, anchor=tk.NW, image=self._tk_photo
        )

    # ---- Pan events --------------------------------------------------------

    def _on_press(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        self._drag_start = (event.x, event.y)
        self._drag_moved = False

    def _on_drag(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if self._drag_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        if abs(dx) > 3 or abs(dy) > 3:
            self._drag_moved = True
        self._pan_x += dx
        self._pan_y += dy
        self._drag_start = (event.x, event.y)
        self._redraw_canvas()

    def _on_release(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if not getattr(self, "_drag_moved", False):
            self._on_click(event)
        self._drag_start = None
        self._drag_moved = False

    # ---- Zoom events -------------------------------------------------------

    def _on_mousewheel(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        factor = 1.1 if event.delta > 0 else 0.9
        self._zoom = max(0.1, min(10.0, self._zoom * factor))
        self._redraw_canvas()

    def _on_mousewheel_linux(self, event: tk.Event, direction: int) -> None:  # type: ignore[type-arg]
        factor = 1.1 if direction > 0 else 0.9
        self._zoom = max(0.1, min(10.0, self._zoom * factor))
        self._redraw_canvas()

    # ---- Click-to-inspect --------------------------------------------------

    def _on_click(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        # Convert canvas coords to image coords
        img_x = int((event.x - self._pan_x) / self._zoom)
        img_y = int((event.y - self._pan_y) / self._zoom)

        ann = self._images[self._current_page]
        duct = hit_test(img_x, img_y, ann.ducts)

        self._update_info_panel(selected_duct=duct)

    # ---- Info panel --------------------------------------------------------

    def _update_info_panel(self, selected_duct: DuctSegment | None = None) -> None:
        self._info_text.config(state=tk.NORMAL)
        self._info_text.delete("1.0", tk.END)

        lines: list[str] = []

        # Selected duct section
        lines.append("═══ SELECTED DUCT ═══\n")
        if selected_duct is not None:
            dim_text = (
                selected_duct.dimension.raw_text
                if selected_duct.dimension
                else "Unknown"
            )
            lines.append(f"  Dimension: {dim_text}")
            lines.append(f"  Pressure Class: {selected_duct.pressure_class.value}")
            if selected_duct.bounding_box:
                bx, by, bw, bh = selected_duct.bounding_box
                lines.append(f"  Bounding Box: ({bx}, {by}, {bw}, {bh})")
            lines.append(f"  ID: {selected_duct.id}")
        else:
            lines.append("  No duct selected")

        lines.append("")

        # Page summary
        page_result = self._result.pages[self._current_page]

        lines.append("═══ DRAWING SCALE ═══\n")
        scale = page_result.scale
        lines.append(f"  {scale.raw_text}  (status: {scale.status})")
        if scale.ratio is not None:
            lines.append(f"  Ratio: {scale.ratio:.6f}")
        lines.append("")

        lines.append("═══ GENERAL NOTES ═══\n")
        if page_result.notes.general_notes:
            for note in page_result.notes.general_notes:
                lines.append(f"  • {note}")
        else:
            lines.append("  (none)")
        lines.append("")

        lines.append("═══ PLAN NOTES ═══\n")
        if page_result.notes.plan_notes:
            for note in page_result.notes.plan_notes:
                lines.append(f"  • {note}")
        else:
            lines.append("  (none)")
        lines.append("")

        lines.append("═══ DUCT SPECIFICATIONS ═══\n")
        if page_result.notes.duct_specifications:
            for spec in page_result.notes.duct_specifications:
                lines.append(f"  Type: {spec.duct_type}")
                if spec.size_range:
                    lines.append(f"    Size Range: {spec.size_range}")
                lines.append(f"    Pressure: {spec.pressure_class.value}")
                if spec.material:
                    lines.append(f"    Material: {spec.material}")
                if spec.gauge:
                    lines.append(f"    Gauge: {spec.gauge}")
                if spec.sealing_class:
                    lines.append(f"    Sealing Class: {spec.sealing_class}")
                lines.append("")
        else:
            lines.append("  (none)")

        self._info_text.insert("1.0", "\n".join(lines))
        self._info_text.config(state=tk.DISABLED)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def launch_viewer(
    result: PipelineResult,
    annotated_images: list[AnnotatedImage],
) -> None:
    """Launch the Tkinter-based interactive viewer.

    Displays a split-panel layout:
    - Left: Annotated image with pan/zoom
    - Right: Information panel (duct details on click, notes, scale)

    Supports multi-page navigation.

    Raises
    ------
    RuntimeError
        If no display is available (headless environment).
    """
    # Check for display availability before creating the Tk root
    if os.environ.get("DISPLAY") is None and os.name != "nt":
        # On non-Windows systems, DISPLAY must be set for Tkinter
        try:
            root = tk.Tk()
            root.destroy()
        except tk.TclError as exc:
            raise RuntimeError(
                "No display available. Run without --ui in headless environments."
            ) from exc
    else:
        root = tk.Tk()

    _DuctViewer(root, result, annotated_images)
    root.mainloop()
