"""Shared Hypothesis strategies for HVAC duct detection property-based tests."""

import sys
import os

import numpy as np
from hypothesis import strategies as st
from hypothesis.strategies import composite

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import (
    DuctShape,
    PressureClass,
    DrawingScale,
    Dimension,
    DuctSegment,
    DuctSpecification,
    ExtractedNotes,
    PageResult,
    PipelineResult,
)


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------

_COMMON_SCALE_RATIOS: list[tuple[str, float]] = [
    ('1/4" = 1\'-0"', 1.0 / 48.0),
    ('1/8" = 1\'-0"', 1.0 / 96.0),
    ('3/8" = 1\'-0"', 3.0 / 96.0),
    ('1/2" = 1\'-0"', 1.0 / 24.0),
    ('1" = 1\'-0"', 1.0 / 12.0),
    ('3/16" = 1\'-0"', 3.0 / 192.0),
    ('1-1/2" = 1\'-0"', 1.5 / 12.0),
]

_DUCT_TYPES = ["round", "rectangular", "all"]

_MATERIALS = [None, "galvanized steel", "aluminum", "stainless steel", "fiberglass"]

_GAUGES = [None, "26 ga", "24 ga", "22 ga", "20 ga", "18 ga", "16 ga"]

_SEALING_CLASSES = [None, "A", "B", "C"]


# ---------------------------------------------------------------------------
# Strategy: draw_valid_scale
# ---------------------------------------------------------------------------

@composite
def draw_valid_scale(draw: st.DrawFn) -> DrawingScale:
    """Generate a valid DrawingScale with a known ratio."""
    raw_text, ratio = draw(st.sampled_from(_COMMON_SCALE_RATIOS))
    return DrawingScale(raw_text=raw_text, ratio=ratio, status="found")


# ---------------------------------------------------------------------------
# Strategy: draw_valid_dimension_text
# ---------------------------------------------------------------------------

@composite
def draw_valid_dimension_text(draw: st.DrawFn) -> str:
    """Generate a valid dimension text string like '14"⌀' or '12"x8"'."""
    shape = draw(st.sampled_from(list(DuctShape)))
    if shape is DuctShape.ROUND:
        diameter = draw(st.integers(min_value=4, max_value=60))
        return f'{diameter}"⌀'
    else:
        width = draw(st.integers(min_value=4, max_value=96))
        height = draw(st.integers(min_value=4, max_value=96))
        return f'{width}"x{height}"'


# ---------------------------------------------------------------------------
# Strategy: draw_valid_dimension
# ---------------------------------------------------------------------------

@composite
def draw_valid_dimension(draw: st.DrawFn) -> Dimension:
    """Generate a valid Dimension object (round or rectangular)."""
    shape = draw(st.sampled_from(list(DuctShape)))
    if shape is DuctShape.ROUND:
        diameter = float(draw(st.integers(min_value=4, max_value=60)))
        raw_text = f'{int(diameter)}"⌀'
        values = [diameter]
    else:
        width = float(draw(st.integers(min_value=4, max_value=96)))
        height = float(draw(st.integers(min_value=4, max_value=96)))
        raw_text = f'{int(width)}"x{int(height)}"'
        values = [width, height]
    return Dimension(raw_text=raw_text, shape=shape, values=values)


# ---------------------------------------------------------------------------
# Strategy: draw_pressure_class
# ---------------------------------------------------------------------------

@composite
def draw_pressure_class(draw: st.DrawFn) -> PressureClass:
    """Generate a PressureClass enum value."""
    return draw(st.sampled_from(list(PressureClass)))


# ---------------------------------------------------------------------------
# Strategy: draw_duct_segment
# ---------------------------------------------------------------------------

@composite
def draw_duct_segment(draw: st.DrawFn) -> DuctSegment:
    """Generate a DuctSegment with polyline and optional dimension."""
    seg_id = draw(st.integers(min_value=0, max_value=10_000))

    # Generate a polyline with 2-10 points
    num_points = draw(st.integers(min_value=2, max_value=10))
    polyline = [
        (
            draw(st.integers(min_value=0, max_value=4000)),
            draw(st.integers(min_value=0, max_value=3000)),
        )
        for _ in range(num_points)
    ]

    has_dimension = draw(st.booleans())
    dimension = draw(draw_valid_dimension()) if has_dimension else None

    pressure = draw(draw_pressure_class())

    # Compute bounding box from polyline
    xs = [p[0] for p in polyline]
    ys = [p[1] for p in polyline]
    x_min, y_min = min(xs), min(ys)
    w = max(xs) - x_min
    h = max(ys) - y_min
    bounding_box = (x_min, y_min, max(w, 1), max(h, 1))

    return DuctSegment(
        id=seg_id,
        polyline=polyline,
        dimension=dimension,
        pressure_class=pressure,
        bounding_box=bounding_box,
    )


# ---------------------------------------------------------------------------
# Strategy: draw_duct_spec
# ---------------------------------------------------------------------------

@composite
def draw_duct_spec(draw: st.DrawFn) -> DuctSpecification:
    """Generate a DuctSpecification object."""
    duct_type = draw(st.sampled_from(_DUCT_TYPES))
    size_range = draw(
        st.one_of(
            st.none(),
            st.sampled_from(['up to 12"', '13" to 24"', '25" and larger']),
        )
    )
    pressure = draw(draw_pressure_class())
    material = draw(st.sampled_from(_MATERIALS))
    gauge = draw(st.sampled_from(_GAUGES))
    sealing_class = draw(st.sampled_from(_SEALING_CLASSES))

    return DuctSpecification(
        duct_type=duct_type,
        size_range=size_range,
        pressure_class=pressure,
        material=material,
        gauge=gauge,
        sealing_class=sealing_class,
    )


# ---------------------------------------------------------------------------
# Strategy: draw_pipeline_result
# ---------------------------------------------------------------------------

@composite
def draw_pipeline_result(draw: st.DrawFn) -> PipelineResult:
    """Generate a PipelineResult with 1-5 pages."""
    num_pages = draw(st.integers(min_value=1, max_value=5))
    pages: list[PageResult] = []
    for page_num in range(1, num_pages + 1):
        num_ducts = draw(st.integers(min_value=0, max_value=5))
        ducts = [draw(draw_duct_segment()) for _ in range(num_ducts)]
        scale = draw(draw_valid_scale())
        general = draw(
            st.lists(st.text(min_size=1, max_size=80, alphabet=st.characters(categories=("L", "N", "P", "Z"))), min_size=0, max_size=3)
        )
        plan = draw(
            st.lists(st.text(min_size=1, max_size=80, alphabet=st.characters(categories=("L", "N", "P", "Z"))), min_size=0, max_size=3)
        )
        specs = draw(st.lists(draw_duct_spec(), min_size=0, max_size=3))
        notes = ExtractedNotes(
            general_notes=general,
            plan_notes=plan,
            duct_specifications=specs,
        )
        pages.append(PageResult(page_number=page_num, ducts=ducts, scale=scale, notes=notes))

    input_path = draw(st.text(min_size=1, max_size=100, alphabet=st.characters(categories=("L", "N", "P"))))
    return PipelineResult(input_path=input_path, pages=pages)


# ---------------------------------------------------------------------------
# Strategy: draw_duct_list
# ---------------------------------------------------------------------------

@composite
def draw_duct_list(draw: st.DrawFn) -> list[DuctSegment]:
    """Generate a list of DuctSegment objects with bounding boxes."""
    return draw(st.lists(draw_duct_segment(), min_size=0, max_size=10))


# ---------------------------------------------------------------------------
# Strategy: draw_click_coord
# ---------------------------------------------------------------------------

@composite
def draw_click_coord(draw: st.DrawFn) -> tuple[int, int]:
    """Generate an (x, y) click coordinate."""
    x = draw(st.integers(min_value=0, max_value=4000))
    y = draw(st.integers(min_value=0, max_value=3000))
    return (x, y)


# ---------------------------------------------------------------------------
# Strategy: draw_spec_text
# ---------------------------------------------------------------------------

_SPEC_KEYWORDS = [
    "low pressure",
    "medium pressure",
    "high pressure",
    "galvanized steel",
    "aluminum",
    "sealing class A",
    "sealing class B",
    "sealing class C",
    "26 gauge",
    "24 gauge",
    "22 gauge",
    "round duct",
    "rectangular duct",
]


@composite
def draw_spec_text(draw: st.DrawFn) -> str:
    """Generate note text containing duct specification keywords."""
    num_keywords = draw(st.integers(min_value=1, max_value=4))
    keywords = draw(
        st.lists(
            st.sampled_from(_SPEC_KEYWORDS),
            min_size=num_keywords,
            max_size=num_keywords,
        )
    )
    # Build a realistic note sentence around the keywords
    prefix = draw(
        st.sampled_from([
            "All ducts shall be",
            "Supply ducts up to 12\" shall be",
            "Return ducts shall be",
            "Ductwork shall conform to",
            "All rectangular ducts shall be",
        ])
    )
    return f"{prefix} {', '.join(keywords)}."
