"""Unit tests for dimension_extractor module."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from dimension_extractor import parse_dimension_text, format_dimension, extract_dimensions
from models import Dimension, DuctShape, DuctSegment, PageImage
import numpy as np


class TestParseDimensionText:
    """Tests for parse_dimension_text."""

    def test_round_with_quote_and_diameter(self):
        d = parse_dimension_text('14"⌀')
        assert d.shape is DuctShape.ROUND
        assert d.values == [14.0]

    def test_round_with_oslash(self):
        d = parse_dimension_text('14"Ø')
        assert d.shape is DuctShape.ROUND
        assert d.values == [14.0]

    def test_round_with_space_before_symbol(self):
        d = parse_dimension_text('14" ⌀')
        assert d.shape is DuctShape.ROUND
        assert d.values == [14.0]

    def test_round_without_quote(self):
        d = parse_dimension_text('14⌀')
        assert d.shape is DuctShape.ROUND
        assert d.values == [14.0]

    def test_rectangular_basic(self):
        d = parse_dimension_text('12"x8"')
        assert d.shape is DuctShape.RECTANGULAR
        assert d.values == [12.0, 8.0]

    def test_rectangular_with_spaces(self):
        d = parse_dimension_text('12" x 8"')
        assert d.shape is DuctShape.RECTANGULAR
        assert d.values == [12.0, 8.0]

    def test_rectangular_without_quotes(self):
        d = parse_dimension_text('12x8')
        assert d.shape is DuctShape.RECTANGULAR
        assert d.values == [12.0, 8.0]

    def test_rectangular_uppercase_x(self):
        d = parse_dimension_text('12"X8"')
        assert d.shape is DuctShape.RECTANGULAR
        assert d.values == [12.0, 8.0]

    def test_unparseable_raises_valueerror(self):
        with pytest.raises(ValueError):
            parse_dimension_text("hello")

    def test_empty_string_raises_valueerror(self):
        with pytest.raises(ValueError):
            parse_dimension_text("")


class TestFormatDimension:
    """Tests for format_dimension."""

    def test_round_format(self):
        d = Dimension(raw_text='14"⌀', shape=DuctShape.ROUND, values=[14.0])
        assert format_dimension(d) == '14"⌀'

    def test_rectangular_format(self):
        d = Dimension(raw_text='12"x8"', shape=DuctShape.RECTANGULAR, values=[12.0, 8.0])
        assert format_dimension(d) == '12"x8"'

    def test_round_trip_round(self):
        d = Dimension(raw_text='14"⌀', shape=DuctShape.ROUND, values=[14.0])
        txt = format_dimension(d)
        d2 = parse_dimension_text(txt)
        assert d.shape == d2.shape
        assert d.values == d2.values

    def test_round_trip_rectangular(self):
        d = Dimension(raw_text='12"x8"', shape=DuctShape.RECTANGULAR, values=[12.0, 8.0])
        txt = format_dimension(d)
        d2 = parse_dimension_text(txt)
        assert d.shape == d2.shape
        assert d.values == d2.values


class TestExtractDimensions:
    """Tests for extract_dimensions with a mock OCR engine."""

    class MockOcr:
        """Minimal OcrEngine mock returning predefined text boxes."""

        def __init__(self, boxes):
            self._boxes = boxes

        def extract_text(self, image, region=None):
            return ""

        def extract_text_with_boxes(self, image, region=None):
            return self._boxes

    def _make_page(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        return PageImage(page_number=1, image=img, width=100, height=100, dpi=200)

    def test_matches_nearest_dimension(self):
        ocr = self.MockOcr([
            ('14"⌀', (48, 48, 10, 10), 0.9),  # center at (53, 53)
        ])
        duct = DuctSegment(id=1, polyline=[(50, 50), (60, 50)], bounding_box=(50, 50, 10, 10))
        result = extract_dimensions(self._make_page(), [duct], ocr, proximity_threshold=50.0)
        assert len(result) == 1
        assert result[0].dimension is not None
        assert result[0].dimension.shape is DuctShape.ROUND
        assert result[0].dimension.values == [14.0]

    def test_no_match_beyond_threshold(self):
        ocr = self.MockOcr([
            ('14"⌀', (200, 200, 10, 10), 0.9),  # far away
        ])
        duct = DuctSegment(id=1, polyline=[(10, 10), (20, 10)], bounding_box=(10, 10, 10, 10))
        result = extract_dimensions(self._make_page(), [duct], ocr, proximity_threshold=50.0)
        assert len(result) == 1
        assert result[0].dimension is None

    def test_does_not_mutate_original(self):
        ocr = self.MockOcr([
            ('14"⌀', (48, 48, 10, 10), 0.9),
        ])
        duct = DuctSegment(id=1, polyline=[(50, 50), (60, 50)], bounding_box=(50, 50, 10, 10))
        result = extract_dimensions(self._make_page(), [duct], ocr, proximity_threshold=50.0)
        assert duct.dimension is None  # original unchanged
        assert result[0].dimension is not None
        assert result[0] is not duct  # new instance

    def test_empty_ducts_returns_empty(self):
        ocr = self.MockOcr([('14"⌀', (50, 50, 10, 10), 0.9)])
        result = extract_dimensions(self._make_page(), [], ocr, proximity_threshold=50.0)
        assert result == []
