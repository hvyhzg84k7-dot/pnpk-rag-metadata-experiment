from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pdfplumber

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_common_arguments, load_config, resolve_repo_path, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ekstrak PDF PNPK menjadi teks per halaman untuk artefak pages.jsonl."
    )
    add_common_arguments(parser)
    parser.add_argument("--pdf", help="Path PDF PNPK. Default diambil dari konfigurasi.")
    parser.add_argument("--output", help="Path output pages.jsonl. Default diambil dari konfigurasi.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    pdf_path = resolve_repo_path(args.pdf or config["data"]["source_document"])
    output_path = args.output or config["data"]["raw_pages"]

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF tidak ditemukan: {pdf_path}")

    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=1.5, y_tolerance=3) or ""
            rows.append(
                {
                    "page": page_number,
                    "raw_text": text,
                    "char_count": len(text),
                    "line_count": len([line for line in text.splitlines() if line.strip()]),
                    "extraction_notes": "",
                }
            )

    written = write_jsonl(output_path, rows)
    if not args.dry_run:
        print(f"[extract_pdf] pdf: {pdf_path}")
        print(f"[extract_pdf] total_pages: {total_pages}")
        print(f"[extract_pdf] written_pages: {written}")
        print(f"[extract_pdf] output: {resolve_repo_path(output_path)}")


if __name__ == "__main__":
    main()
