from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_common_arguments, load_config, read_jsonl, resolve_repo_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validasi sampel chunk dan metadata PNPK."
    )
    add_common_arguments(parser)
    parser.add_argument("--chunks", help="Path chunks_metadata.jsonl.")
    parser.add_argument("--review-output", help="Path metadata_review.csv.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    chunks_path = args.chunks or config["data"]["chunks_metadata"]
    review_path = resolve_repo_path(args.review_output or config["data"]["metadata_review"])
    chunks = read_jsonl(chunks_path)

    if not chunks:
        raise ValueError(f"Tidak ada chunk untuk divalidasi: {chunks_path}")

    required_fields = [
        "chunk_id",
        "chunk_text",
        "source_document",
        "page",
        "chapter",
        "section",
        "labels",
        "token_count",
    ]
    valid_labels = set(config["metadata"]["label_vocabulary"])
    max_labels = int(config.get("metadata", {}).get("label_annotation", {}).get("max_labels_per_chunk", 3))
    rows = []
    issue_count = 0
    for chunk in chunks:
        issues = []
        for field in required_fields:
            if chunk.get(field) in ("", None):
                issues.append(f"missing_{field}")
        labels = chunk.get("labels")
        if not isinstance(labels, list) or not labels:
            issues.append("invalid_labels")
            labels = []
        if len(labels) > max_labels:
            issues.append("too_many_labels")
        invalid_labels = [label for label in labels if label not in valid_labels]
        if invalid_labels:
            issues.append("labels_not_in_vocabulary")
        if int(chunk.get("token_count", 0)) < 30:
            issues.append("short_chunk")
        if int(chunk.get("token_count", 0)) > int(config["chunking"]["chunk_size_tokens"]):
            issues.append("over_chunk_size")

        issue_count += bool(issues)
        rows.append(
            {
                "chunk_id": chunk.get("chunk_id", ""),
                "page": chunk.get("page", ""),
                "chapter": chunk.get("chapter", ""),
                "section": chunk.get("section", ""),
                "labels": json.dumps(labels, ensure_ascii=False),
                "token_count": chunk.get("token_count", ""),
                "char_count": len(chunk.get("chunk_text", "")),
                "review_status": "needs_review" if issues else "ok",
                "issues": ";".join(issues),
                "sample_text": chunk.get("chunk_text", "")[:240].replace("\n", " "),
            }
        )

    review_path.parent.mkdir(parents=True, exist_ok=True)
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    token_counts = [int(chunk["token_count"]) for chunk in chunks]
    pages = sorted({int(chunk["page"]) for chunk in chunks})
    label_counts: dict[str, int] = {}
    for chunk in chunks:
        for label in chunk.get("labels", []):
            label_counts[label] = label_counts.get(label, 0) + 1

    print(f"[validate_chunks] chunks: {len(chunks)}")
    print(f"[validate_chunks] pages: {pages[0]}-{pages[-1]} ({len(pages)} pages)")
    print(f"[validate_chunks] token_count_min: {min(token_counts)}")
    print(f"[validate_chunks] token_count_avg: {mean(token_counts):.1f}")
    print(f"[validate_chunks] token_count_max: {max(token_counts)}")
    print(f"[validate_chunks] chunks_with_issues: {issue_count}")
    print(f"[validate_chunks] labels: {json.dumps(label_counts, ensure_ascii=False, sort_keys=True)}")
    print(f"[validate_chunks] review_output: {review_path}")


if __name__ == "__main__":
    main()
