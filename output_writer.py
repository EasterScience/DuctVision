"""Output writer for the HVAC duct detection pipeline.

Provides JSON serialization and deserialization of PipelineResult objects,
enabling persistence and round-trip loading of detection results.
"""

import json
import os
from dataclasses import fields
from enum import Enum

import numpy as np

from models import (
    Dimension,
    DrawingScale,
    DuctSegment,
    DuctShape,
    DuctSpecification,
    ExtractedNotes,
    PageResult,
    PipelineResult,
    PressureClass,
)


def _serialize(obj: object) -> object:
    """Custom serializer for domain objects.

    Handles enums, dataclasses, numpy arrays, and tuples.
    """
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, np.ndarray):
        return None
    if isinstance(obj, tuple):
        return list(obj)
    if hasattr(obj, "__dataclass_fields__"):
        result = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = _serialize(value)
        return result
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


def save_json_summary(result: PipelineResult, output_dir: str) -> str:
    """Export the full pipeline result as a JSON file.

    Serializes the PipelineResult with custom handling for enums (→ value string),
    dataclasses (→ dict), numpy arrays (→ skipped), and tuples (→ lists).
    Auto-creates the output directory if it doesn't exist.

    Args:
        result: The pipeline result to serialize.
        output_dir: Directory where summary.json will be saved.

    Returns:
        The path to the saved JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "summary.json")
    data = _serialize(result)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return output_path


def _deserialize_dimension(d: dict | None) -> Dimension | None:
    """Reconstruct a Dimension from a dict."""
    if d is None:
        return None
    return Dimension(
        raw_text=d["raw_text"],
        shape=DuctShape(d["shape"]),
        values=d["values"],
    )


def _deserialize_duct_segment(d: dict) -> DuctSegment:
    """Reconstruct a DuctSegment from a dict."""
    return DuctSegment(
        id=d["id"],
        polyline=[tuple(pt) for pt in d["polyline"]],
        shape=DuctShape(d.get("shape", "rectangular")),
        dimension=_deserialize_dimension(d.get("dimension")),
        pressure_class=PressureClass(d.get("pressure_class", "Unknown")),
        bounding_box=tuple(d["bounding_box"]) if d.get("bounding_box") else None,
    )


def _deserialize_drawing_scale(d: dict) -> DrawingScale:
    """Reconstruct a DrawingScale from a dict."""
    return DrawingScale(
        raw_text=d["raw_text"],
        ratio=d.get("ratio"),
        status=d.get("status", "found"),
    )


def _deserialize_duct_specification(d: dict) -> DuctSpecification:
    """Reconstruct a DuctSpecification from a dict."""
    return DuctSpecification(
        duct_type=d["duct_type"],
        size_range=d.get("size_range"),
        pressure_class=PressureClass(d.get("pressure_class", "Unknown")),
        material=d.get("material"),
        gauge=d.get("gauge"),
        sealing_class=d.get("sealing_class"),
    )


def _deserialize_extracted_notes(d: dict) -> ExtractedNotes:
    """Reconstruct ExtractedNotes from a dict."""
    return ExtractedNotes(
        general_notes=d.get("general_notes", []),
        plan_notes=d.get("plan_notes", []),
        duct_specifications=[
            _deserialize_duct_specification(spec)
            for spec in d.get("duct_specifications", [])
        ],
    )


def _deserialize_page_result(d: dict) -> PageResult:
    """Reconstruct a PageResult from a dict."""
    return PageResult(
        page_number=d["page_number"],
        ducts=[_deserialize_duct_segment(seg) for seg in d.get("ducts", [])],
        scale=_deserialize_drawing_scale(d["scale"]),
        notes=_deserialize_extracted_notes(d["notes"]),
    )


def load_json_summary(json_path: str) -> PipelineResult:
    """Load a previously saved JSON summary back into a PipelineResult object.

    Reconstructs all nested domain objects including enums and dataclasses.
    Handles missing optional fields gracefully with sensible defaults.

    Args:
        json_path: Path to the summary.json file.

    Returns:
        The deserialized PipelineResult.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return PipelineResult(
        input_path=data["input_path"],
        pages=[_deserialize_page_result(p) for p in data.get("pages", [])],
    )
