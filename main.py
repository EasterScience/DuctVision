"""CLI entry point and pipeline orchestration for HVAC duct detection.

Parses command-line arguments and runs the full detection pipeline:
render PDF → extract scale → extract notes → detect ducts →
extract dimensions → classify pressure → annotate → save outputs.
"""

import argparse
import logging

from models import PageResult, PipelineResult

logger = logging.getLogger(__name__)


def run_pipeline(
    input_path: str,
    output_dir: str,
    dpi: int,
    ocr_engine: str = "tesseract",
) -> PipelineResult:
    """Execute the full detection pipeline on a PDF file.

    Args:
        input_path: Path to the input PDF.
        output_dir: Directory for output files.
        dpi: Rendering DPI.
        ocr_engine: OCR backend — "tesseract" or "florence2".

    Returns:
        A PipelineResult containing all extracted data per page.
    """
    from annotation_engine import annotate_page, save_annotated_image
    from dimension_extractor import extract_dimensions
    from vlm_detector import detect_ducts
    from notes_extractor import extract_notes
    from ocr_engine import create_ocr_engine
    from output_writer import save_json_summary
    from pdf_renderer import render_pdf
    from pressure_classifier import classify_pressure
    from scale_extractor import extract_scale

    ocr = create_ocr_engine(ocr_engine)
    logger.info("Using OCR engine: %s", ocr_engine)

    pages = render_pdf(input_path, dpi)
    logger.info("Rendered %d page(s) from %s at %d DPI", len(pages), input_path, dpi)

    page_results: list[PageResult] = []

    for page in pages:
        logger.info("Processing page %d ...", page.page_number)

        scale = extract_scale(page, ocr)
        notes = extract_notes(page, ocr)
        ducts = detect_ducts(page, ocr=ocr)
        ducts = extract_dimensions(page, ducts, ocr, proximity_threshold=300.0)
        ducts = classify_pressure(ducts, page, notes.duct_specifications)

        annotated = annotate_page(page, ducts, scale, notes)
        save_annotated_image(annotated, output_dir, page.page_number)

        page_results.append(
            PageResult(
                page_number=page.page_number,
                ducts=ducts,
                scale=scale,
                notes=notes,
            )
        )

    result = PipelineResult(input_path=input_path, pages=page_results)
    save_json_summary(result, output_dir)
    logger.info("Pipeline complete. Output saved to %s", output_dir)

    return result


def main() -> None:
    """Parse CLI args and run the detection pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="HVAC duct detection pipeline — detect ducts, extract dimensions and pressure classes from mechanical drawings.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input PDF file",
    )
    parser.add_argument(
        "--output",
        default="output/",
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rendering DPI (default: 300)",
    )
    parser.add_argument(
        "--ocr-engine",
        default="tesseract",
        choices=["tesseract"],
        help="OCR backend (default: tesseract)",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch interactive viewer after processing",
    )

    args = parser.parse_args()

    result = run_pipeline(args.input, args.output, args.dpi, args.ocr_engine)

    total_ducts = sum(len(p.ducts) for p in result.pages)
    print(
        f"Processed {len(result.pages)} page(s), "
        f"detected {total_ducts} duct segment(s). "
        f"Output saved to {args.output}"
    )

    # Calculate and print total pipe length
    import math as _math
    from scale_extractor import convert_to_real_world

    for page_result in result.pages:
        scale = page_result.scale
        total_px = 0.0
        for duct in page_result.ducts:
            for i in range(len(duct.polyline) - 1):
                x1, y1 = duct.polyline[i]
                x2, y2 = duct.polyline[i + 1]
                total_px += _math.hypot(x2 - x1, y2 - y1)

        print(f"\n--- Page {page_result.page_number} ---")
        print(f"  Scale: {scale.raw_text if scale.raw_text else 'not found'}")
        if scale.ratio:
            # total_px is in pixels at the given DPI
            # Convert pixels to drawing inches: px / DPI
            total_drawing_inches = total_px / args.dpi
            # Convert drawing inches to real inches: drawing_inches / ratio
            total_real_inches = total_drawing_inches / scale.ratio
            total_feet = total_real_inches / 12.0
            ft = int(total_feet)
            inches = round((total_feet - ft) * 12)
            print(f"  Total pipe length: {ft}'-{inches}\" ({total_feet:.1f} ft)")
        else:
            print(f"  Total pipe length: {total_px:.0f} pixels (scale unknown)")

        # Per-duct breakdown
        round_count = sum(1 for d in page_result.ducts if d.shape.value == "round")
        rect_count = sum(1 for d in page_result.ducts if d.shape.value == "rectangular")
        print(f"  Ducts: {len(page_result.ducts)} total ({round_count} round, {rect_count} rectangular)")

    if args.ui:
        from viewer_ui import launch_viewer

        # Rebuild annotated images for the viewer
        from annotation_engine import annotate_page
        from ocr_engine import create_ocr_engine
        from pdf_renderer import render_pdf
        from scale_extractor import extract_scale
        from notes_extractor import extract_notes

        ocr = create_ocr_engine(args.ocr_engine)
        page_images = render_pdf(args.input, args.dpi)
        annotated_images = []
        for page_img, page_result in zip(page_images, result.pages):
            annotated = annotate_page(
                page_img,
                page_result.ducts,
                page_result.scale,
                page_result.notes,
            )
            annotated_images.append(annotated)

        launch_viewer(result, annotated_images)


if __name__ == "__main__":
    main()
