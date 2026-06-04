from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_common_arguments, load_config, read_jsonl, resolve_repo_path, write_jsonl


PAGE_MARKER_RE = re.compile(r"^\s*-\s*\d+\s*-\s*$")
CHAPTER_RE = re.compile(r"^\s*BAB\s+([IVXLC]+)\s*$", re.IGNORECASE)
SECTION_RE = re.compile(r"^\s*([A-Z])\.\s+(.+?)\s*$")


LABEL_KEYWORDS = [
    ("definisi dan klasifikasi", ["definisi", "klasifikasi"]),
    ("diagnosis", ["diagnosis", "skrining"]),
    ("diabetes melitus tipe-1", ["tipe-1", "tipe 1", "dm tipe-1", "dm tipe 1"]),
    ("diabetes melitus tipe-2", ["tipe-2", "tipe 2", "dm tipe-2", "dm tipe 2"]),
    ("komplikasi", ["komplikasi", "ketoasidosis", "hipoglikemia", "mikrovaskular", "makrovaskular"]),
    ("masa transisi", ["transisi"]),
    ("tim kesehatan diabetes anak", ["tenaga medis", "tenaga kesehatan"]),
]


def clean_page_text(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if PAGE_MARKER_RE.match(line):
            continue
        if line.lower() == "jdih.kemkes.go.id":
            continue
        lines.append(line)

    joined = "\n".join(lines)
    joined = re.sub(r"(\w)-\n(\w)", r"\1\2", joined)
    joined = re.sub(r"[ \t]+", " ", joined)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


def update_structure(text: str, chapter: str, section: str) -> tuple[str, str]:
    for line in text.splitlines():
        chapter_match = CHAPTER_RE.match(line)
        if chapter_match:
            chapter = f"BAB {chapter_match.group(1).upper()}"
            section = ""
            continue
        section_match = SECTION_RE.match(line)
        if section_match and len(section_match.group(2)) <= 120:
            section = f"{section_match.group(1).upper()}. {section_match.group(2).strip()}"
    return chapter, section


def infer_preliminary_label(text: str, section: str) -> str:
    section_text = section.lower()
    if "definisi" in section_text or "klasifikasi" in section_text:
        return "definisi dan klasifikasi"
    if "diagnosis" in section_text:
        return "diagnosis"
    if "masa transisi" in section_text or "transisi" in section_text:
        return "masa transisi"
    if "tenaga medis" in section_text or "tenaga kesehatan" in section_text:
        return "tim kesehatan diabetes anak"
    if "tipe-1" in section_text or "tipe 1" in section_text:
        return "diabetes melitus tipe-1"
    if "tipe-2" in section_text or "tipe 2" in section_text:
        return "diabetes melitus tipe-2"

    haystack = f"{section}\n{text}".lower()
    if "komplikasi" in haystack or "ketoasidosis" in haystack or "hipoglikemia" in haystack:
        return "komplikasi"
    for label, keywords in LABEL_KEYWORDS:
        if any(keyword in haystack for keyword in keywords):
            return label
    return "lainnya"


def build_report(
    *,
    total_pages: int,
    clean_rows: list[dict],
    clinical_rows: list[dict],
    report_path: str,
    clinical_start: int,
    clinical_end: int,
) -> None:
    low_text_pages = [row["page"] for row in clean_rows if row["char_count"] < 200]
    chapters: dict[str, int] = {}
    sections: dict[str, int] = {}
    preliminary_labels: dict[str, int] = {}
    for row in clinical_rows:
        chapters[row["chapter"]] = chapters.get(row["chapter"], 0) + 1
        sections[row["section"]] = sections.get(row["section"], 0) + 1
        preliminary_labels[row["preliminary_label"]] = preliminary_labels.get(row["preliminary_label"], 0) + 1

    def bullet_counts(values: dict[str, int]) -> str:
        return "\n".join(f"- {key or '(kosong)'}: {value}" for key, value in sorted(values.items()))

    text = f"""# Laporan Ekstraksi PNPK

## Ringkasan

- Total halaman PDF: {total_pages}
- Rentang korpus klinis awal: halaman {clinical_start}-{clinical_end}
- Jumlah halaman teks bersih: {len(clean_rows)}
- Jumlah halaman korpus klinis: {len(clinical_rows)}
- Halaman dengan teks bersih kurang dari 200 karakter: {low_text_pages if low_text_pages else 'tidak ada'}

## Distribusi Bab pada Korpus Klinis

{bullet_counts(chapters)}

## Distribusi Bagian pada Korpus Klinis

{bullet_counts(sections)}

## Distribusi Label Awal pada Korpus Klinis

{bullet_counts(preliminary_labels)}

## Catatan Validasi

- Halaman 1-3 merupakan bagian keputusan/legal dan tidak dimasukkan ke korpus klinis.
- Halaman 4-12 berisi pendahuluan dan metodologi pedoman sehingga tidak dimasukkan ke korpus klinis utama.
- Halaman {clinical_start}-{clinical_end} dipakai sebagai korpus klinis awal karena mencakup Bab III dan Bab IV.
- Halaman 119 merupakan Bab V Penutup dan tidak dimasukkan ke korpus klinis utama.
- `preliminary_label` pada tahap ini bersifat heuristik awal untuk membantu audit; verifikasi final metadata `labels` dilakukan pada tahap chunking.
"""
    output = resolve_repo_path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bersihkan hasil ekstraksi PNPK dan tandai korpus klinis."
    )
    add_common_arguments(parser)
    parser.add_argument("--input", help="Path pages.jsonl.")
    parser.add_argument("--output", help="Path pnpk_clean_pages.jsonl.")
    parser.add_argument("--clinical-output", help="Path pnpk_clinical_corpus.jsonl.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    input_path = args.input or config["data"]["raw_pages"]
    output_path = args.output or config["data"]["clean_pages"]
    clinical_output_path = args.clinical_output or config["data"]["clinical_corpus"]
    report_path = config["results"]["extraction_report"]
    clinical_start = int(config["data"]["clinical_page_start"])
    clinical_end = int(config["data"]["clinical_page_end"])

    pages = read_jsonl(input_path)
    clean_rows = []
    clinical_rows = []
    chapter = ""
    section = ""

    for page in pages:
        page_number = int(page["page"])
        clean_text = clean_page_text(page.get("raw_text", ""))
        chapter, section = update_structure(clean_text, chapter, section)
        is_clinical = clinical_start <= page_number <= clinical_end
        preliminary_label = infer_preliminary_label(clean_text, section) if is_clinical else ""
        row = {
            "page": page_number,
            "clean_text": clean_text,
            "chapter": chapter,
            "section": section,
            "preliminary_label": preliminary_label,
            "is_clinical_corpus": is_clinical,
            "char_count": len(clean_text),
            "line_count": len([line for line in clean_text.splitlines() if line.strip()]),
            "cleaning_notes": "",
        }
        clean_rows.append(row)
        if is_clinical:
            clinical_rows.append(row)

    write_jsonl(output_path, clean_rows)
    write_jsonl(clinical_output_path, clinical_rows)
    build_report(
        total_pages=len(pages),
        clean_rows=clean_rows,
        clinical_rows=clinical_rows,
        report_path=report_path,
        clinical_start=clinical_start,
        clinical_end=clinical_end,
    )

    if not args.dry_run:
        print(f"[clean_text] input_pages: {len(pages)}")
        print(f"[clean_text] clean_pages: {len(clean_rows)} -> {resolve_repo_path(output_path)}")
        print(f"[clean_text] clinical_pages: {len(clinical_rows)} -> {resolve_repo_path(clinical_output_path)}")
        print(f"[clean_text] report: {resolve_repo_path(report_path)}")


if __name__ == "__main__":
    main()
