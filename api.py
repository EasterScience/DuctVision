"""FastAPI backend for HVAC pipe viewer/editor."""
from __future__ import annotations

import json
import math
import os
import uuid
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pdf_renderer import render_pdf
from pipe_marker import (
    _find_drawing_area,
    _detect_white_channels,
    _detect_diagonal_pipes,
    _channels_to_pairs,
    _detect_lines,
    _find_parallel_pairs,
    _validate_hough_pair,
    _deduplicate,
    _detect_black_rect_pipes,
    _line_length,
    _line_angle,
    _midpoint,
    _perp_dist,
)
from scale_extractor import extract_scale, format_scale, convert_to_real_world
from ocr_engine import create_ocr_engine
from models import PageImage

app = FastAPI(title="HVAC Pipe Viewer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory state
_state: dict = {}

DATA_FILE = "output/pipes_data.json"


# ── Models ──────────────────────────────────────────────────────────────────

class PipeOut(BaseModel):
    id: str
    x1: int; y1: int; x2: int; y2: int
    width: float
    length_px: float
    angle: float
    source: str  # "channel" or "hough"

class PipeUpdate(BaseModel):
    x1: int; y1: int; x2: int; y2: int
    width: float | None = None

class PipeCreate(BaseModel):
    x1: int; y1: int; x2: int; y2: int
    width: float = 30


# ── Helpers ─────────────────────────────────────────────────────────────────

def _pair_to_pipe(pair, pid: str, source: str) -> dict:
    """Convert a line-pair (a, b) into a centerline-based pipe dict."""
    a, b = pair
    a = tuple(int(x) for x in a)
    b = tuple(int(x) for x in b)
    # Centerline = midpoint of corresponding endpoints
    x1 = int((a[0] + b[0]) / 2)
    y1 = int((a[1] + b[1]) / 2)
    x2 = int((a[2] + b[2]) / 2)
    y2 = int((a[3] + b[3]) / 2)
    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    angle = _line_angle(x1, y1, x2, y2)
    mid_b = _midpoint(*b)
    width = _perp_dist(mid_b[0], mid_b[1], float(a[0]), float(a[1]),
                       float(a[2]), float(a[3]))
    return {
        "id": pid,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "width": round(float(width), 1),
        "length_px": round(float(length), 1),
        "angle": round(float(angle), 1),
        "source": source,
    }


def _pipe_out(p: dict) -> PipeOut:
    return PipeOut(
        id=p["id"],
        x1=p["x1"], y1=p["y1"], x2=p["x2"], y2=p["y2"],
        width=p["width"],
        length_px=p["length_px"],
        angle=p["angle"],
        source=p["source"],
    )


def _save_pipes():
    pipes = _state.get("pipes", [])
    os.makedirs("output", exist_ok=True)
    with open(DATA_FILE, "w") as f:
        json.dump(pipes, f, indent=2)


def _load_pipes():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            pipes = json.load(f)
        # Migrate old format (a/b arrays) to new centerline format
        migrated = []
        for p in pipes:
            if "a" in p and "b" in p:
                a, b = p["a"], p["b"]
                x1 = int((a[0] + b[0]) / 2)
                y1 = int((a[1] + b[1]) / 2)
                x2 = int((a[2] + b[2]) / 2)
                y2 = int((a[3] + b[3]) / 2)
                mid_b = _midpoint(*b)
                width = _perp_dist(mid_b[0], mid_b[1], float(a[0]), float(a[1]),
                                   float(a[2]), float(a[3]))
                length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                migrated.append({
                    "id": p["id"],
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "width": round(float(width), 1),
                    "length_px": round(float(length), 1),
                    "angle": round(float(_line_angle(x1, y1, x2, y2)), 1),
                    "source": p.get("source", "channel"),
                })
            else:
                migrated.append(p)
        return migrated
    return None


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.post("/api/detect")
def detect_pipes(pdf_path: str = "input/testset2.pdf", dpi: int = 300):
    """Run pipe detection on a PDF and store results."""
    if not os.path.exists(pdf_path):
        raise HTTPException(404, f"PDF not found: {pdf_path}")

    pages = render_pdf(pdf_path, dpi)
    if not pages:
        raise HTTPException(500, "Failed to render PDF")

    image = pages[0].image
    _state["image"] = image
    _state["pdf_path"] = pdf_path
    _state["dpi"] = dpi
    _state["width"] = pages[0].width
    _state["height"] = pages[0].height

    roi = _find_drawing_area(image)
    channels, labels = _detect_white_channels(image, roi)
    diag_channels = _detect_diagonal_pipes(image, roi, channels)
    all_channels = channels + diag_channels
    channel_pairs = _channels_to_pairs(all_channels)

    lines = _detect_lines(image, roi)
    hough_pairs = _find_parallel_pairs(lines)
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hough_pairs = [p for p in hough_pairs
                   if _validate_hough_pair(gray_full, p[0], p[1])]

    rx, ry = roi[0], roi[1]
    rx2, ry2 = rx + roi[2], ry + roi[3]
    edge_margin = 80
    hough_pairs = [p for p in hough_pairs if all(
        rx + edge_margin <= x <= rx2 - edge_margin and
        ry + edge_margin <= y <= ry2 - edge_margin
        for x, y in [(p[0][0], p[0][1]), (p[0][2], p[0][3]),
                      (p[1][0], p[1][1]), (p[1][2], p[1][3])]
    )]
    hough_pairs = [p for p in hough_pairs
                   if (_line_length(*p[0]) + _line_length(*p[1])) / 2 >= 150]

    def _interior_bright(pair):
        a, b = pair
        white = 0; total = 0
        for t in np.linspace(0.1, 0.9, 15):
            for s in np.linspace(0.2, 0.8, 5):
                px = int(a[0]+(a[2]-a[0])*t+(b[0]-a[0])*s+(b[2]-b[0]-a[2]+a[0])*t*s)
                py = int(a[1]+(a[3]-a[1])*t+(b[1]-a[1])*s+(b[3]-b[1]-a[3]+a[1])*t*s)
                if 0 <= py < gray_full.shape[0] and 0 <= px < gray_full.shape[1]:
                    total += 1
                    if gray_full[py, px] > 150:
                        white += 1
        return total == 0 or (white / total) > 0.50
    hough_pairs = [p for p in hough_pairs if _interior_bright(p)]

    all_pairs = _deduplicate(channel_pairs, hough_pairs)
    black_rect_pairs = _detect_black_rect_pipes(image, roi, all_pairs)
    all_pairs = all_pairs + black_rect_pairs

    n_ch = len(channel_pairs)
    pipes = []
    for i, pair in enumerate(all_pairs):
        src = "channel" if i < n_ch else "hough"
        pipes.append(_pair_to_pipe(pair, str(uuid.uuid4())[:8], src))

    _state["pipes"] = pipes
    _save_pipes()

    # Extract scale
    page_img = PageImage(page_number=1, image=image, width=_state["width"],
                         height=_state["height"], dpi=dpi)
    try:
        ocr = create_ocr_engine("tesseract")
        scale_info = extract_scale(page_img, ocr)
        _state["scale"] = {
            "raw_text": scale_info.raw_text,
            "ratio": scale_info.ratio,
            "status": scale_info.status,
            "display": format_scale(scale_info),
        }
    except Exception:
        _state["scale"] = {"raw_text": "", "ratio": None, "status": "unknown", "display": "Unknown"}

    return {
        "count": len(pipes),
        "width": _state["width"],
        "height": _state["height"],
        "scale": _state.get("scale", {}),
    }


@app.get("/api/page-image")
def get_page_image():
    """Return the raw PDF page as a JPEG image."""
    if "image" not in _state:
        # Try to load from default PDF
        pages = render_pdf("input/testset2.pdf", 300)
        if pages:
            _state["image"] = pages[0].image
            _state["width"] = pages[0].width
            _state["height"] = pages[0].height
        else:
            raise HTTPException(404, "No image loaded. Call /api/detect first.")

    _, buf = cv2.imencode(".jpg", _state["image"], [cv2.IMWRITE_JPEG_QUALITY, 95])
    return StreamingResponse(BytesIO(buf.tobytes()), media_type="image/jpeg")


@app.get("/api/pipes", response_model=list[PipeOut])
def list_pipes():
    """Return all detected pipes."""
    if "pipes" not in _state:
        saved = _load_pipes()
        if saved:
            _state["pipes"] = saved
        else:
            return []
    return [_pipe_out(p) for p in _state["pipes"]]


@app.put("/api/pipes/{pipe_id}", response_model=PipeOut)
def update_pipe(pipe_id: str, body: PipeUpdate):
    """Update a pipe's centerline coordinates and optionally width."""
    pipes = _state.get("pipes", [])
    for p in pipes:
        if p["id"] == pipe_id:
            p["x1"] = body.x1; p["y1"] = body.y1
            p["x2"] = body.x2; p["y2"] = body.y2
            if body.width is not None:
                p["width"] = round(body.width, 1)
            p["length_px"] = round(math.sqrt(
                (body.x2 - body.x1) ** 2 + (body.y2 - body.y1) ** 2), 1)
            p["angle"] = round(_line_angle(body.x1, body.y1, body.x2, body.y2), 1)
            _save_pipes()
            return _pipe_out(p)
    raise HTTPException(404, "Pipe not found")


@app.post("/api/pipes", response_model=PipeOut)
def create_pipe(body: PipeCreate):
    """Add a new pipe manually."""
    pipes = _state.setdefault("pipes", [])
    length = math.sqrt((body.x2 - body.x1) ** 2 + (body.y2 - body.y1) ** 2)
    angle = _line_angle(body.x1, body.y1, body.x2, body.y2)
    p = {
        "id": str(uuid.uuid4())[:8],
        "x1": body.x1, "y1": body.y1, "x2": body.x2, "y2": body.y2,
        "width": round(body.width, 1),
        "length_px": round(length, 1),
        "angle": round(angle, 1),
        "source": "manual",
    }
    pipes.append(p)
    _save_pipes()
    return _pipe_out(p)


@app.delete("/api/pipes/{pipe_id}")
def delete_pipe(pipe_id: str):
    """Remove a pipe."""
    pipes = _state.get("pipes", [])
    _state["pipes"] = [p for p in pipes if p["id"] != pipe_id]
    _save_pipes()
    return {"ok": True}


@app.post("/api/save")
def save_all():
    """Persist current pipe data to disk."""
    _save_pipes()
    return {"ok": True, "path": DATA_FILE}


@app.get("/api/scale")
def get_scale():
    """Return the extracted drawing scale."""
    if "scale" in _state:
        return _state["scale"]
    return {"raw_text": "", "ratio": None, "status": "unknown", "display": "Unknown"}


@app.get("/api/summary")
def get_summary():
    """Return total pipe length in pixels and real-world units."""
    pipes = _state.get("pipes", [])
    if not pipes:
        saved = _load_pipes()
        if saved:
            pipes = saved
    total_px = sum(p["length_px"] for p in pipes)
    dpi = _state.get("dpi", 300)
    total_drawing_inches = total_px / dpi
    scale = _state.get("scale", {})
    ratio = scale.get("ratio")
    if ratio and ratio > 0:
        real_inches = total_drawing_inches / ratio
        feet = int(real_inches) // 12
        inches = round(real_inches % 12)
        total_real = f"{feet}'-{inches}\""
    else:
        total_real = "N/A"
    return {
        "total_px": round(total_px, 1),
        "total_drawing_inches": round(total_drawing_inches, 2),
        "total_real": total_real,
        "pipe_count": len(pipes),
        "scale": scale,
    }

# Serve frontend static build (for Docker / production)
_frontend_build = Path(__file__).parent / "frontend" / "build"
if _frontend_build.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_build), html=True), name="frontend")
