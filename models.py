"""Data models for the HVAC duct detection pipeline.

Defines all domain objects used throughout the pipeline, including enums for
duct shape and pressure class, and dataclasses for images, scales, dimensions,
duct segments, specifications, notes, and pipeline results.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class DuctShape(Enum):
    """Shape classification for a duct segment."""

    ROUND = "round"
    RECTANGULAR = "rectangular"


class PressureClass(Enum):
    """Pressure class rating for a duct segment."""

    LOW = "Low Pressure"
    MEDIUM = "Medium Pressure"
    HIGH = "High Pressure"
    UNKNOWN = "Unknown"


@dataclass
class PageImage:
    """A single rendered page from a PDF drawing."""

    page_number: int
    image: np.ndarray  # OpenCV BGR image
    width: int
    height: int
    dpi: int


@dataclass
class DrawingScale:
    """Parsed drawing scale information.

    Represents the ratio between drawing dimensions and real-world dimensions,
    e.g. 1/4" = 1'-0" corresponds to a ratio of 1/48.
    """

    raw_text: str  # e.g., '1/4" = 1\'-0"'
    ratio: float | None  # e.g., 0.02083 (1/48) — drawing inches to real inches
    status: str = "found"  # "found" or "unknown"


@dataclass
class Dimension:
    """Parsed duct dimension label.

    Holds the raw OCR text and the parsed numeric values for either
    round (single diameter) or rectangular (width x height) ducts.
    """

    raw_text: str  # e.g., '14"⌀' or '12"x8"'
    shape: DuctShape
    values: list[float]  # [14.0] for round, [12.0, 8.0] for rectangular


@dataclass
class DuctSegment:
    """A detected duct segment in a drawing page.

    Contains the geometric path, optional dimension and pressure class,
    and an optional bounding box for click-to-inspect hit testing.
    """

    id: int
    polyline: list[tuple[int, int]]  # List of (x, y) pixel coordinates
    shape: DuctShape = DuctShape.RECTANGULAR  # Detected shape type
    dimension: Dimension | None = None
    pressure_class: PressureClass = PressureClass.UNKNOWN
    bounding_box: tuple[int, int, int, int] | None = None  # (x, y, w, h)


@dataclass
class DuctSpecification:
    """Duct construction specification extracted from drawing notes.

    Describes requirements such as pressure class, material, gauge,
    and sealing class for a given duct type and size range.
    """

    duct_type: str  # e.g., "round", "rectangular", "all"
    size_range: str | None  # e.g., 'up to 12"' or None for all sizes
    pressure_class: PressureClass = PressureClass.UNKNOWN
    material: str | None = None
    gauge: str | None = None
    sealing_class: str | None = None


@dataclass
class ExtractedNotes:
    """Notes extracted from a drawing page.

    Contains general project-wide notes, page-specific plan notes,
    and any structured duct specifications parsed from the notes text.
    """

    general_notes: list[str]
    plan_notes: list[str]
    duct_specifications: list[DuctSpecification] = field(default_factory=list)


@dataclass
class PageResult:
    """Complete detection results for a single drawing page."""

    page_number: int
    ducts: list[DuctSegment]
    scale: DrawingScale
    notes: ExtractedNotes


@dataclass
class PipelineResult:
    """Aggregated results from processing an entire PDF document."""

    input_path: str
    pages: list[PageResult]


@dataclass
class AnnotatedImage:
    """An annotated drawing page with duct overlays.

    Contains the rendered image with colored duct overlays and labels,
    along with duct segment references for click-to-inspect hit testing.
    """

    page_number: int
    image: np.ndarray
    ducts: list[DuctSegment]  # For click-to-inspect hit testing
    file_path: str | None = None
