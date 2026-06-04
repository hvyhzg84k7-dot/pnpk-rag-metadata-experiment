from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_common_arguments, load_config, read_jsonl, resolve_repo_path
from evaluation.run_dataset import load_dataset


def normalize(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def tokens(value: str) -> set[str]:
    return {token.strip(".,;:()[]{}") for token in normalize(value).split() if len(token.strip(".,;:()[]{}")) >= 4}


def overlap_ratio(expected: str, source: str) -> float:
    expected_tokens = tokens(expected)
    if not expected_tokens:
        return 0.0
    source_tokens = tokens(source)
    return len(expected_tokens & source_tokens) / len(expected_tokens)


def write_review(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = resolve_repo_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "source_page",
        "chapter_match",
        "section_match",
        "reference_context_overlap",
        "reference_answer_overlap",
        "labels_valid",
        "review_status",
        "issues",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit dataset evaluasi terhadap korpus klinis PNPK."
    )
    add_common_arguments(parser)
    parser.add_argument("--dataset", help="Path evaluation_dataset.csv.")
    parser.add_argument("--corpus", help="Path pnpk_clinical_corpus.jsonl.")
    parser.add_argument("--output", help="Path dataset review CSV.")
    parser.add_argument(
        "--min-context-overlap",
        type=float,
        default=0.45,
        help="Ambang minimal token overlap reference_context terhadap halaman PNPK.",
    )
    parser.add_argument(
        "--min-answer-overlap",
        type=float,
        default=0.20,
        help="Ambang minimal token overlap reference_answer terhadap halaman PNPK.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    dataset_path = args.dataset or config["data"]["evaluation_dataset"]
    corpus_path = args.corpus or config["data"]["clinical_corpus"]
    output_path = args.output or config.get("results", {}).get(
        "dataset_review",
        "results/evaluation/dataset_review.csv",
    )
    dataset = load_dataset(dataset_path)
    corpus_by_page = {str(row.get("page")): row for row in read_jsonl(corpus_path)}
    valid_labels = set(config["metadata"]["label_vocabulary"])

    review_rows: list[dict[str, Any]] = []
    issue_count = 0
    for row in dataset:
        issues: list[str] = []
        page = str(row.get("source_page", "")).strip()
        source = corpus_by_page.get(page)
        if not source:
            issues.append("source_page_not_in_corpus")
            source_text = ""
        else:
            source_text = str(source.get("clean_text", ""))

        chapter_match = int(bool(source) and normalize(source.get("chapter")) == normalize(row.get("chapter")))
        section_match = int(bool(source) and normalize(source.get("section")) == normalize(row.get("section")))
        if not chapter_match:
            issues.append("chapter_mismatch")
        if not section_match:
            issues.append("section_mismatch")

        context_overlap = overlap_ratio(str(row.get("reference_context", "")), source_text)
        answer_overlap = overlap_ratio(str(row.get("reference_answer", "")), source_text)
        if context_overlap < args.min_context_overlap:
            issues.append("low_reference_context_overlap")
        if answer_overlap < args.min_answer_overlap:
            issues.append("low_reference_answer_overlap")

        labels = row.get("labels", [])
        labels_valid = int(isinstance(labels, list) and bool(labels) and all(label in valid_labels for label in labels))
        if not labels_valid:
            issues.append("invalid_labels")

        issue_count += bool(issues)
        review_rows.append(
            {
                "question_id": row["question_id"],
                "source_page": page,
                "chapter_match": chapter_match,
                "section_match": section_match,
                "reference_context_overlap": f"{context_overlap:.3f}",
                "reference_answer_overlap": f"{answer_overlap:.3f}",
                "labels_valid": labels_valid,
                "review_status": "needs_review" if issues else "ok",
                "issues": ";".join(issues),
            }
        )

    if args.dry_run:
        print(f"[validate_dataset] dry_run dataset: {resolve_repo_path(dataset_path)}")
        print(f"[validate_dataset] dry_run corpus: {resolve_repo_path(corpus_path)}")
        print(f"[validate_dataset] dry_run rows: {len(dataset)}")
        print(f"[validate_dataset] dry_run output: {resolve_repo_path(output_path)}")
        return

    write_review(output_path, review_rows)
    print(f"[validate_dataset] rows: {len(review_rows)}")
    print(f"[validate_dataset] rows_with_issues: {issue_count}")
    print(f"[validate_dataset] output: {resolve_repo_path(output_path)}")


if __name__ == "__main__":
    main()
