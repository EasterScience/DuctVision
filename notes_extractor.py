"""Notes extraction from HVAC mechanical drawing pages.

Identifies General Notes and Plan Notes sections in engineering drawings
using OCR, preserves hierarchical structure, and parses duct specification
information into structured ``DuctSpecification`` records.
"""

from __future__ import annotations

import logging
import re

from models import DuctSpecification, ExtractedNotes, PageImage, PressureClass
from ocr_engine import OcrEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex / keyword helpers
# ---------------------------------------------------------------------------

_GENERAL_NOTES_HEADING = re.compile(r"GENERAL\s+NOTES", re.IGNORECASE)
_PLAN_NOTES_HEADING = re.compile(r"PLAN\s+NOTES", re.IGNORECASE)

# Numbered item pattern: "1.", "1)", "1 -", or similar at start of line
_NUMBERED_ITEM = re.compile(r"^\s*(\d+)\s*[.):\-]", re.MULTILINE)

# Pressure class keywords
_PRESSURE_MAP: list[tuple[re.Pattern[str], PressureClass]] = [
    (re.compile(r"low\s+pressure", re.IGNORECASE), PressureClass.LOW),
    (re.compile(r"medium\s+pressure", re.IGNORECASE), PressureClass.MEDIUM),
    (re.compile(r"high\s+pressure", re.IGNORECASE), PressureClass.HIGH),
]

# Material keywords
_MATERIAL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"galvanized\s+steel", re.IGNORECASE), "galvanized steel"),
    (re.compile(r"stainless\s+steel", re.IGNORECASE), "stainless steel"),
    (re.compile(r"aluminum", re.IGNORECASE), "aluminum"),
]

# Gauge pattern: "NN ga" or "NN gauge"
_GAUGE_PATTERN = re.compile(r"(\d{1,2})\s*(?:ga(?:uge)?)\b", re.IGNORECASE)

# Sealing class pattern: "sealing class A/B/C"
_SEALING_CLASS_PATTERN = re.compile(
    r"sealing\s+class\s+([A-Ca-c])", re.IGNORECASE
)

# Duct type references
_DUCT_TYPE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"round\s+duct", re.IGNORECASE), "round"),
    (re.compile(r"rectangular\s+duct", re.IGNORECASE), "rectangular"),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_numbered_items(text: str) -> list[str]:
    """Split text into numbered items, preserving hierarchy.

    Lines that start with a number followed by a delimiter (``1.``, ``2)``,
    etc.) begin a new item.  Continuation lines are appended to the
    current item.  Returns a list of item strings with leading/trailing
    whitespace stripped.
    """
    items: list[str] = []
    current: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _NUMBERED_ITEM.match(stripped):
            # Flush previous item
            if current:
                items.append(" ".join(current))
                current = []
            current.append(stripped)
        else:
            # Continuation of current item, or standalone line
            if current:
                current.append(stripped)
            else:
                # No numbered item started yet — treat as standalone
                items.append(stripped)

    if current:
        items.append(" ".join(current))

    return items


def _extract_section(full_text: str, heading_pattern: re.Pattern[str]) -> str:
    """Extract the text block following *heading_pattern* up to the next
    major heading or end of text.

    Returns an empty string if the heading is not found.
    """
    match = heading_pattern.search(full_text)
    if not match:
        return ""

    start = match.end()

    # Look for the next major heading (all-caps line of 3+ words, or another
    # known heading) to delimit the section.
    next_heading = re.search(
        r"\n\s*(?:GENERAL\s+NOTES|PLAN\s+NOTES|MECHANICAL\s+SCHEDULE|"
        r"ABBREVIATIONS|LEGEND|SYMBOLS|EQUIPMENT\s+SCHEDULE)\b",
        full_text[start:],
        re.IGNORECASE,
    )
    end = start + next_heading.start() if next_heading else len(full_text)
    return full_text[start:end].strip()


# ---------------------------------------------------------------------------
# Public API — duct specification parsing
# ---------------------------------------------------------------------------


def parse_duct_specifications(text: str) -> list[DuctSpecification]:
    """Parse duct specification information from free-form notes text.

    Scans *text* for keywords related to pressure class, material, gauge,
    sealing class, and duct type, and assembles ``DuctSpecification``
    records.

    This function can be tested independently of OCR.

    Args:
        text: Free-form notes text (may contain multiple sentences/items).

    Returns:
        List of ``DuctSpecification`` records found in the text.
    """
    if not text or not text.strip():
        return []

    specs: list[DuctSpecification] = []

    # Split into sentences / numbered items for finer-grained parsing
    items = _split_numbered_items(text)
    if not items:
        # Fall back to treating the whole text as one block
        items = [text]

    for item in items:
        # Determine pressure class
        pressure = PressureClass.UNKNOWN
        for pattern, pc in _PRESSURE_MAP:
            if pattern.search(item):
                pressure = pc
                break

        # Determine material
        material: str | None = None
        for pattern, mat in _MATERIAL_PATTERNS:
            if pattern.search(item):
                material = mat
                break

        # Determine gauge
        gauge: str | None = None
        gauge_match = _GAUGE_PATTERN.search(item)
        if gauge_match:
            gauge = f"{gauge_match.group(1)} ga"

        # Determine sealing class
        sealing_class: str | None = None
        seal_match = _SEALING_CLASS_PATTERN.search(item)
        if seal_match:
            sealing_class = seal_match.group(1).upper()

        # Determine duct type
        duct_type = "all"
        for pattern, dt in _DUCT_TYPE_PATTERNS:
            if pattern.search(item):
                duct_type = dt
                break

        # Only create a spec if at least one meaningful field was found
        has_info = (
            pressure is not PressureClass.UNKNOWN
            or material is not None
            or gauge is not None
            or sealing_class is not None
        )
        if has_info:
            specs.append(
                DuctSpecification(
                    duct_type=duct_type,
                    size_range=None,
                    pressure_class=pressure,
                    material=material,
                    gauge=gauge,
                    sealing_class=sealing_class,
                )
            )

    return specs


# ---------------------------------------------------------------------------
# Public API — main extraction entry point
# ---------------------------------------------------------------------------


def extract_notes(page_image: PageImage, ocr: OcrEngine) -> ExtractedNotes:
    """Identify and extract General Notes and Plan Notes from a page image.

    Uses the provided ``OcrEngine`` to read text from the full page, then
    locates "GENERAL NOTES" and "PLAN NOTES" section headings to split
    the text into structured note lists.  Duct specification keywords
    within the notes are parsed into ``DuctSpecification`` records.

    Args:
        page_image: Rendered page image.
        ocr: OCR engine instance.

    Returns:
        ``ExtractedNotes`` with general notes, plan notes, and any
        parsed duct specifications.  Returns empty lists when no notes
        sections are found.
    """
    full_text = ocr.extract_text(page_image.image)

    if not full_text or not full_text.strip():
        logger.info(
            "No text found on page %d; returning empty notes",
            page_image.page_number,
        )
        return ExtractedNotes(general_notes=[], plan_notes=[], duct_specifications=[])

    # Extract section text
    general_text = _extract_section(full_text, _GENERAL_NOTES_HEADING)
    plan_text = _extract_section(full_text, _PLAN_NOTES_HEADING)

    # Split into hierarchical items
    general_notes = _split_numbered_items(general_text) if general_text else []
    plan_notes = _split_numbered_items(plan_text) if plan_text else []

    if not general_notes and not plan_notes:
        logger.info(
            "No General Notes or Plan Notes headings found on page %d",
            page_image.page_number,
        )

    # Parse duct specifications from all notes text
    combined_text = f"{general_text}\n{plan_text}".strip()
    duct_specs = parse_duct_specifications(combined_text)

    if duct_specs:
        logger.info(
            "Parsed %d duct specification(s) from notes on page %d",
            len(duct_specs),
            page_image.page_number,
        )

    return ExtractedNotes(
        general_notes=general_notes,
        plan_notes=plan_notes,
        duct_specifications=duct_specs,
    )
