from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_common_arguments, load_config, read_jsonl, resolve_repo_path
from evaluation.run_dataset import load_dataset


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_page(value: Any) -> str:
    text = normalize_text(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text


def context_matches(row: dict[str, Any], contexts: list[dict[str, Any]]) -> dict[str, Any]:
    expected_page = normalize_page(row.get("source_page"))
    expected_chapter = normalize_text(row.get("chapter"))
    expected_section = normalize_text(row.get("section"))
    expected_labels = {normalize_text(label) for label in row.get("labels", []) if normalize_text(label)}

    page_match = 0
    chapter_match = 0
    section_match = 0
    label_match = 0
    matched_rank = ""
    matched_chunk_id = ""

    for context in contexts:
        metadata = context.get("metadata") or {}
        rank = context.get("rank", "")
        chunk_id = context.get("chunk_id", "")
        context_labels = {
            normalize_text(label)
            for label in metadata.get("labels", [])
            if normalize_text(label)
        }

        current_page_match = int(bool(expected_page) and normalize_page(metadata.get("page")) == expected_page)
        current_chapter_match = int(bool(expected_chapter) and normalize_text(metadata.get("chapter")) == expected_chapter)
        current_section_match = int(bool(expected_section) and normalize_text(metadata.get("section")) == expected_section)
        current_label_match = int(bool(expected_labels) and bool(expected_labels & context_labels))

        page_match = max(page_match, current_page_match)
        chapter_match = max(chapter_match, current_chapter_match)
        section_match = max(section_match, current_section_match)
        label_match = max(label_match, current_label_match)

        if not matched_rank and (
            current_page_match or current_chapter_match or current_section_match or current_label_match
        ):
            matched_rank = str(rank)
            matched_chunk_id = str(chunk_id)

    return {
        "page_match": page_match,
        "chapter_match": chapter_match,
        "section_match": section_match,
        "label_match": label_match,
        "matched_rank": matched_rank,
        "matched_chunk_id": matched_chunk_id,
    }


def write_scores(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = resolve_repo_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "pipeline",
        "top_k",
        "page_match",
        "chapter_match",
        "section_match",
        "label_match",
        "matched_rank",
        "matched_chunk_id",
        "metadata_filter_used",
        "metadata_filter_fallback_used",
        "notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hitung indikator page/chapter/section/label match@k."
    )
    add_common_arguments(parser)
    parser.add_argument("--retrieval", help="Path retrieval_outputs.jsonl.")
    parser.add_argument("--dataset", help="Path evaluation_dataset.csv.")
    parser.add_argument("--output", help="Path traceability_scores.csv.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    retrieval_path = args.retrieval or config["results"]["retrieval_outputs"]
    dataset_path = args.dataset or config["data"]["evaluation_dataset"]
    output_path = args.output or config["results"]["traceability_scores"]

    dataset = {row["question_id"]: row for row in load_dataset(dataset_path)}
    retrieval_rows = read_jsonl(retrieval_path)

    if args.dry_run:
        print(f"[traceability] dry_run dataset: {resolve_repo_path(dataset_path)}")
        print(f"[traceability] dry_run retrieval: {resolve_repo_path(retrieval_path)}")
        print(f"[traceability] dry_run retrieval_rows: {len(retrieval_rows)}")
        print(f"[traceability] dry_run output: {resolve_repo_path(output_path)}")
        return

    score_rows: list[dict[str, Any]] = []
    for retrieval_row in retrieval_rows:
        question_id = retrieval_row.get("question_id")
        reference_row = dataset.get(str(question_id))
        if not reference_row:
            raise ValueError(f"question_id tidak ada di dataset: {question_id}")
        match = context_matches(reference_row, retrieval_row.get("contexts", []))
        notes = ""
        if retrieval_row.get("pipeline") == "baseline":
            notes = "baseline tidak menyimpan metadata terstruktur; nilai match metadata dipertahankan 0 jika field tidak tersedia"
        if retrieval_row.get("metadata_filter_fallback_used"):
            notes = f"{notes}; metadata filter fallback digunakan".strip("; ")
        score_rows.append(
            {
                "question_id": question_id,
                "pipeline": retrieval_row.get("pipeline", ""),
                "top_k": retrieval_row.get("top_k", ""),
                **match,
                "metadata_filter_used": retrieval_row.get("metadata_filter_used", False),
                "metadata_filter_fallback_used": retrieval_row.get("metadata_filter_fallback_used", False),
                "notes": notes,
            }
        )

    write_scores(output_path, score_rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        grouped[str(row["pipeline"])].append(row)

    print(f"[traceability] rows: {len(score_rows)}")
    for pipeline, rows in sorted(grouped.items()):
        count = len(rows)
        means = {
            metric: sum(int(row[metric]) for row in rows) / count if count else 0.0
            for metric in ["page_match", "chapter_match", "section_match", "label_match"]
        }
        fallback_count = sum(1 for row in rows if row["metadata_filter_fallback_used"] in (True, "True", "true", 1))
        print(
            "[traceability] "
            f"{pipeline} n={count} "
            f"page_match@k={means['page_match']:.3f} "
            f"chapter_match@k={means['chapter_match']:.3f} "
            f"section_match@k={means['section_match']:.3f} "
            f"label_match@k={means['label_match']:.3f} "
            f"fallback={fallback_count}"
        )
    print(f"[traceability] output: {resolve_repo_path(output_path)}")


if __name__ == "__main__":
    main()
