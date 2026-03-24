"""Unit tests for scale_extractor module."""

import math

import numpy as np
import pytest

from models import DrawingScale, PageImage
from scale_extractor import (
    convert_to_real_world,
    extract_scale,
    format_scale,
    parse_scale_text,
)


# ---------------------------------------------------------------------------
# parse_scale_text
# ---------------------------------------------------------------------------


class TestParseScaleText:
    """Tests for parse_scale_text."""

    def test_quarter_inch_scale(self):
        result = parse_scale_text('1/4" = 1\'-0"')
        assert result.status == "found"
        assert result.ratio == pytest.approx(1.0 / 48.0)

    def test_eighth_inch_scale(self):
        result = parse_scale_text('1/8" = 1\'-0"')
        assert result.status == "found"
        assert result.ratio == pytest.approx(1.0 / 96.0)

    def test_three_eighths_inch_scale(self):
        result = parse_scale_text('3/8" = 1\'-0"')
        assert result.status == "found"
        assert result.ratio == pytest.approx(3.0 / 96.0)

    def test_half_inch_scale(self):
        result = parse_scale_text('1/2" = 1\'-0"')
        assert result.status == "found"
        assert result.ratio == pytest.approx(1.0 / 24.0)

    def test_one_inch_scale(self):
        result = parse_scale_text('1" = 1\'-0"')
        assert result.status == "found"
        assert result.ratio == pytest.approx(1.0 / 12.0)

    def test_three_sixteenths_scale(self):
        result = parse_scale_text('3/16" = 1\'-0"')
        assert result.status == "found"
        assert result.ratio == pytest.approx(3.0 / 192.0)

    def test_one_and_half_inch_scale(self):
        result = parse_scale_text('1-1/2" = 1\'-0"')
        assert result.status == "found"
        assert result.ratio == pytest.approx(1.5 / 12.0)

    def test_raw_text_preserved(self):
        text = '1/4" = 1\'-0"'
        result = parse_scale_text(text)
        assert result.raw_text == text

    def test_invalid_text_raises(self):
        with pytest.raises(ValueError):
            parse_scale_text("not a scale")

    def test_empty_text_raises(self):
        with pytest.raises(ValueError):
            parse_scale_text("")


# ---------------------------------------------------------------------------
# format_scale
# ---------------------------------------------------------------------------


class TestFormatScale:
    """Tests for format_scale."""

    def test_quarter_inch_format(self):
        scale = DrawingScale(raw_text='1/4" = 1\'-0"', ratio=1.0 / 48.0, status="found")
        result = format_scale(scale)
        assert '1/4"' in result
        assert "1'-0\"" in result

    def test_unknown_scale_returns_raw(self):
        scale = DrawingScale(raw_text="unknown", ratio=None, status="unknown")
        assert format_scale(scale) == "unknown"

    def test_round_trip_quarter_inch(self):
        original = parse_scale_text('1/4" = 1\'-0"')
        formatted = format_scale(original)
        reparsed = parse_scale_text(formatted)
        assert reparsed.ratio == pytest.approx(original.ratio, rel=1e-4)

    def test_round_trip_eighth_inch(self):
        original = parse_scale_text('1/8" = 1\'-0"')
        formatted = format_scale(original)
        reparsed = parse_scale_text(formatted)
        assert reparsed.ratio == pytest.approx(original.ratio, rel=1e-4)

    def test_round_trip_one_inch(self):
        original = parse_scale_text('1" = 1\'-0"')
        formatted = format_scale(original)
        reparsed = parse_scale_text(formatted)
        assert reparsed.ratio == pytest.approx(original.ratio, rel=1e-4)

    def test_round_trip_mixed_number(self):
        original = parse_scale_text('1-1/2" = 1\'-0"')
        formatted = format_scale(original)
        reparsed = parse_scale_text(formatted)
        assert reparsed.ratio == pytest.approx(original.ratio, rel=1e-4)


# ---------------------------------------------------------------------------
# convert_to_real_world
# ---------------------------------------------------------------------------


class TestConvertToRealWorld:
    """Tests for convert_to_real_world."""

    def test_known_scale_feet_inches(self):
        # 1/4" = 1'-0" means ratio = 1/48
        # 1 drawing inch → 48 real inches = 4'-0"
        scale = DrawingScale(raw_text='1/4" = 1\'-0"', ratio=1.0 / 48.0, status="found")
        result = convert_to_real_world(1.0, scale)
        assert result == "4'-0\""

    def test_known_scale_mixed_result(self):
        # 1/4" = 1'-0", 0.5 drawing inches → 24 real inches = 2'-0"
        scale = DrawingScale(raw_text='1/4" = 1\'-0"', ratio=1.0 / 48.0, status="found")
        result = convert_to_real_world(0.5, scale)
        assert result == "2'-0\""

    def test_unknown_scale_returns_drawing_units(self):
        scale = DrawingScale(raw_text="", ratio=None, status="unknown")
        result = convert_to_real_world(5.0, scale)
        assert result == "5.0 drawing units"

    def test_feet_and_inches_format(self):
        # 1" = 1'-0" means ratio = 1/12
        # 3.5 drawing inches → 42 real inches = 3'-6"
        scale = DrawingScale(raw_text='1" = 1\'-0"', ratio=1.0 / 12.0, status="found")
        result = convert_to_real_world(3.5, scale)
        assert result == "3'-6\""


# ---------------------------------------------------------------------------
# extract_scale (with mock OCR)
# ---------------------------------------------------------------------------


class _MockOcrEngine:
    """Minimal mock OCR engine for testing extract_scale."""

    def __init__(self, text: str = ""):
        self._text = text

    def extract_text(self, image, region=None):
        return self._text

    def extract_text_with_boxes(self, image, region=None):
        return []


class TestExtractScale:
    """Tests for extract_scale with mock OCR."""

    def _make_page_image(self) -> PageImage:
        img = np.zeros((1000, 800, 3), dtype=np.uint8)
        return PageImage(page_number=1, image=img, width=800, height=1000, dpi=200)

    def test_finds_scale_in_text(self):
        ocr = _MockOcrEngine('SCALE: 1/4" = 1\'-0"')
        page = self._make_page_image()
        result = extract_scale(page, ocr)
        assert result.status == "found"
        assert result.ratio == pytest.approx(1.0 / 48.0)

    def test_no_scale_returns_unknown(self):
        ocr = _MockOcrEngine("PROJECT TITLE: HVAC SYSTEM")
        page = self._make_page_image()
        result = extract_scale(page, ocr)
        assert result.status == "unknown"
        assert result.ratio is None

    def test_empty_ocr_returns_unknown(self):
        ocr = _MockOcrEngine("")
        page = self._make_page_image()
        result = extract_scale(page, ocr)
        assert result.status == "unknown"
