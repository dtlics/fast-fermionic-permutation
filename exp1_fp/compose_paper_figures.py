"""Compose the standalone Experiment 1 plots into paper-facing PDFs."""

from __future__ import annotations

import argparse
import io
import os
import shutil
from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf._page import PageObject
from reportlab.pdfgen import canvas


def _single_page(path: Path):
    reader = PdfReader(path)
    if len(reader.pages) != 1:
        raise ValueError(f"expected a one-page PDF: {path}")
    return reader, reader.pages[0]


def compose_depth_spacetime(depth_path: Path, spacetime_path: Path, output: Path) -> None:
    """Stack depth over spacetime while preserving vector content."""
    depth_reader, depth = _single_page(depth_path)
    spacetime_reader, spacetime = _single_page(spacetime_path)
    # Keep readers alive until writing; their streams back the page resources.
    _ = depth_reader, spacetime_reader

    target_width = max(float(depth.mediabox.width), float(spacetime.mediabox.width))
    left_margin, right_margin = 30.0, 8.0
    top_margin, bottom_margin, row_gap = 24.0, 8.0, 30.0
    depth_scale = target_width / float(depth.mediabox.width)
    spacetime_scale = target_width / float(spacetime.mediabox.width)
    depth_height = float(depth.mediabox.height) * depth_scale
    spacetime_height = float(spacetime.mediabox.height) * spacetime_scale
    page_width = left_margin + target_width + right_margin
    page_height = (
        bottom_margin + spacetime_height + row_gap + depth_height + top_margin
    )

    combined = PageObject.create_blank_page(width=page_width, height=page_height)
    spacetime_y = bottom_margin
    depth_y = bottom_margin + spacetime_height + row_gap
    combined.merge_transformed_page(
        spacetime,
        Transformation().scale(spacetime_scale).translate(left_margin, spacetime_y),
    )
    combined.merge_transformed_page(
        depth,
        Transformation().scale(depth_scale).translate(left_margin, depth_y),
    )

    label_stream = io.BytesIO()
    labels = canvas.Canvas(label_stream, pagesize=(page_width, page_height), pdfVersion=(1, 4))
    labels.setFont("Helvetica-Bold", 14)
    labels.drawString(5, depth_y + depth_height + 4, "(a) Circuit Depth")
    labels.drawString(5, spacetime_y + spacetime_height + 4, "(b) Spacetime Volume")
    labels.save()
    label_stream.seek(0)
    combined.merge_page(PdfReader(label_stream).pages[0])

    writer = PdfWriter()
    writer.pdf_header = "%PDF-1.4"
    writer.add_page(combined)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("wb") as stream:
        writer.write(stream)
    os.replace(tmp, output)


def copy_atomic(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    shutil.copyfile(source, tmp)
    os.replace(tmp, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig-dir", type=Path, default=Path("exp1_fp/figures"))
    parser.add_argument("--paper-fig-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    compose_depth_spacetime(
        args.fig_dir / "depth_vs_L.pdf",
        args.fig_dir / "spacetime_vs_N.pdf",
        args.paper_fig_dir / "FP-exp1_depth_spacetime.pdf",
    )
    # Compatibility filename retained because main.tex already references it;
    # the plotted quantity is explicitly labeled all-zero return probability.
    copy_atomic(
        args.fig_dir / "stim_fidelity.pdf",
        args.paper_fig_dir / "FP-exp1_stim_fidelity.pdf",
    )


if __name__ == "__main__":
    main()
