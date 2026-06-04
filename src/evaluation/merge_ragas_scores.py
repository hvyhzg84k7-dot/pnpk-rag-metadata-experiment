from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_common_arguments, read_jsonl, resolve_repo_path, write_jsonl
from evaluation.run_ragas import score_row_complete, write_summary


def score_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("question_id", "")), str(row.get("pipeline", ""))


def sort_key(row: dict[str, Any]) -> tuple[str, int]:
    pipeline_order = {"baseline": 0, "metadata": 1}
    question_id, pipeline = score_key(row)
    return question_id, pipeline_order.get(pipeline, 99)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gabungkan skor RAGAS lama dengan skor incremental."
    )
    add_common_arguments(parser)
    parser.add_argument("--base", required=True, help="Path skor RAGAS lama.")
    parser.add_argument("--incremental", required=True, help="Path skor RAGAS baru.")
    parser.add_argument("--output", help="Path output skor gabungan.")
    parser.add_argument("--summary-output", help="Path output summary gabungan.")
    parser.add_argument(
        "--skip-incomplete",
        action="store_true",
        help="Jangan menimpa skor lama dengan row incremental yang belum lengkap.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = args.output or "results/ragas/ragas_scores.jsonl"
    summary_path = args.summary_output or "results/ragas/ragas_summary.csv"

    base_rows = read_jsonl(args.base)
    incremental_rows = read_jsonl(args.incremental)
    merged = {score_key(row): row for row in base_rows}
    overwritten = 0
    skipped_incomplete = 0
    for row in incremental_rows:
        if args.skip_incomplete and not score_row_complete(row):
            skipped_incomplete += 1
            continue
        key = score_key(row)
        overwritten += int(key in merged)
        merged[key] = row

    output_rows = sorted(merged.values(), key=sort_key)
    if args.dry_run:
        print(f"[merge_ragas_scores] dry_run base_rows: {len(base_rows)}")
        print(f"[merge_ragas_scores] dry_run incremental_rows: {len(incremental_rows)}")
        print(f"[merge_ragas_scores] dry_run overwritten: {overwritten}")
        print(f"[merge_ragas_scores] dry_run skipped_incomplete: {skipped_incomplete}")
        print(f"[merge_ragas_scores] dry_run output_rows: {len(output_rows)}")
        print(f"[merge_ragas_scores] dry_run output: {resolve_repo_path(output_path)}")
        print(f"[merge_ragas_scores] dry_run summary: {resolve_repo_path(summary_path)}")
        return

    count = write_jsonl(output_path, output_rows)
    write_summary(summary_path, output_rows)
    print(f"[merge_ragas_scores] base_rows: {len(base_rows)}")
    print(f"[merge_ragas_scores] incremental_rows: {len(incremental_rows)}")
    print(f"[merge_ragas_scores] overwritten: {overwritten}")
    print(f"[merge_ragas_scores] skipped_incomplete: {skipped_incomplete}")
    print(f"[merge_ragas_scores] output_rows: {count}")
    print(f"[merge_ragas_scores] output: {resolve_repo_path(output_path)}")
    print(f"[merge_ragas_scores] summary: {resolve_repo_path(summary_path)}")


if __name__ == "__main__":
    main()
