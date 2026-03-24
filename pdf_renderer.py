"""PDF rendering module for the HVAC duct detection pipeline.

Converts PDF pages into raster images suitable for image processing using
PyMuPDF (fitz). Each page is rendered at a configurable DPI (minimum 200)
and returned as a PageImage with a BGR numpy array for OpenCV compatibility.
"""

import logging
import os

import fitz  # PyMuPDF
import numpy as np

from models import PageImage

logger = logging.getLogger(__name__)

MIN_DPI = 200


def render_pdf(pdf_path: str, dpi: int = 200) -> list[PageImage]:
    """Convert each page of a PDF to a raster image.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering (minimum 200). Values below 200 are
            clamped up to 200.

    Returns:
        List of PageImage objects, one per page.

    Raises:
        FileNotFoundError: If pdf_path does not exist.
        ValueError: If the file is not a valid PDF.
    """
    # Validate file existence
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Validate file extension
    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError(f"File is not a PDF (wrong extension): {pdf_path}")

    # Enforce minimum DPI
    effective_dpi = max(dpi, MIN_DPI)
    if dpi < MIN_DPI:
        logger.info("Requested DPI %d is below minimum; using %d", dpi, effective_dpi)

    # Try opening the file to validate it is a real PDF
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise ValueError(f"Unable to open file as PDF: {pdf_path}") from exc

    zoom = effective_dpi / 72.0  # fitz default is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)

    pages: list[PageImage] = []

    for page_index in range(len(doc)):
        try:
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix)

            # Convert pixmap to numpy array (RGB) then to BGR for OpenCV
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )

            # Handle alpha channel if present (RGBA -> BGR)
            if pix.n == 4:
                bgr = img_array[:, :, :3][:, :, ::-1].copy()
            else:
                # RGB -> BGR
                bgr = img_array[:, :, ::-1].copy()

            pages.append(
                PageImage(
                    page_number=page_index + 1,
                    image=bgr,
                    width=pix.width,
                    height=pix.height,
                    dpi=effective_dpi,
                )
            )
        except Exception:
            logger.warning(
                "Skipping corrupted page %d in %s", page_index + 1, pdf_path,
                exc_info=True,
            )
            continue

    doc.close()
    return pages
