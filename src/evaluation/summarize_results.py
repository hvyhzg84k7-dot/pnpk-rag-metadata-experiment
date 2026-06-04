from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_common_arguments, load_config, read_jsonl, resolve_repo_path


RAGAS_METRICS = ["context_relevance", "faithfulness", "answer_relevance"]
TRACEABILITY_METRICS = ["page_match", "chapter_match", "section_match", "label_match"]


def as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    input_path = resolve_repo_path(path)
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_text(path: str | Path, text: str) -> None:
    output_path = resolve_repo_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def summarize_ragas(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    pipelines = sorted({str(row.get("pipeline", "")) for row in rows if row.get("pipeline")})
    summary: list[dict[str, Any]] = []
    missing: list[str] = []
    for metric in RAGAS_METRICS:
        values_by_pipeline: dict[str, list[float]] = {}
        for pipeline in pipelines:
            values: list[float] = []
            for row in rows:
                if row.get("pipeline") != pipeline:
                    continue
                value = as_float(row.get(metric))
                if value is None:
                    missing.append(f"{row.get('question_id', '')}/{pipeline}/{metric}")
                    continue
                values.append(value)
            values_by_pipeline[pipeline] = values
            summary.append(
                {
                    "metric": metric,
                    "pipeline": pipeline,
                    "n": len(values),
                    "mean": sum(values) / len(values) if values else None,
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                    "metadata_minus_baseline": None,
                }
            )

        baseline = values_by_pipeline.get("baseline", [])
        metadata = values_by_pipeline.get("metadata", [])
        if baseline and metadata:
            summary.append(
                {
                    "metric": metric,
                    "pipeline": "comparison",
                    "n": min(len(baseline), len(metadata)),
                    "mean": None,
                    "min": None,
                    "max": None,
                    "metadata_minus_baseline": (sum(metadata) / len(metadata)) - (sum(baseline) / len(baseline)),
                }
            )
    return summary, missing


def summarize_traceability(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("pipeline", ""))].append(row)

    summary: list[dict[str, Any]] = []
    for pipeline, pipeline_rows in sorted(grouped.items()):
        item: dict[str, Any] = {"pipeline": pipeline, "n": len(pipeline_rows)}
        for metric in TRACEABILITY_METRICS:
            values = [int(row.get(metric, 0) or 0) for row in pipeline_rows]
            item[f"{metric}@k"] = sum(values) / len(values) if values else None
        fallback_values = [
            str(row.get("metadata_filter_fallback_used", "")).lower() in {"true", "1"}
            for row in pipeline_rows
        ]
        item["fallback_count"] = sum(fallback_values)
        summary.append(item)
    return summary


def ragas_by_question(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row.get("question_id", "")), str(row.get("pipeline", ""))): row
        for row in rows
    }


def choose_cases(
    ragas_rows: list[dict[str, Any]],
    traceability_rows: list[dict[str, str]],
) -> tuple[list[str], list[str]]:
    ragas_lookup = ragas_by_question(ragas_rows)
    representative: list[str] = []
    failures: list[str] = []

    metadata_trace = [row for row in traceability_rows if row.get("pipeline") == "metadata"]
    for row in metadata_trace:
        question_id = row.get("question_id", "")
        if row.get("page_match") == "1" and row.get("label_match") == "1":
            representative.append(
                f"- {question_id}: metadata retrieval cocok halaman dan label pada rank {row.get('matched_rank', '')} ({row.get('matched_chunk_id', '')})."
            )
            break

    for row in metadata_trace:
        if row.get("page_match") == "0":
            failures.append(
                f"- {row.get('question_id', '')}: metadata tidak mencapai page_match@k; chunk cocok pertama {row.get('matched_chunk_id', '') or '-'}."
            )
            break

    question_ids = sorted({str(row.get("question_id", "")) for row in ragas_rows})
    for metric in RAGAS_METRICS:
        for question_id in question_ids:
            baseline = as_float(ragas_lookup.get((question_id, "baseline"), {}).get(metric))
            metadata = as_float(ragas_lookup.get((question_id, "metadata"), {}).get(metric))
            if baseline is not None and metadata is not None and metadata > baseline:
                representative.append(
                    f"- {question_id}: metadata lebih tinggi dari baseline pada {metric} ({metadata:.3f} vs {baseline:.3f})."
                )
                break
        if len(representative) >= 3:
            break

    for metric in RAGAS_METRICS:
        for question_id in question_ids:
            baseline = as_float(ragas_lookup.get((question_id, "baseline"), {}).get(metric))
            metadata = as_float(ragas_lookup.get((question_id, "metadata"), {}).get(metric))
            if baseline is not None and metadata is not None and metadata < baseline:
                failures.append(
                    f"- {question_id}: metadata lebih rendah dari baseline pada {metric} ({metadata:.3f} vs {baseline:.3f})."
                )
                break
        if len(failures) >= 3:
            break

    for row in ragas_rows:
        missing_metrics = [metric for metric in RAGAS_METRICS if as_float(row.get(metric)) is None]
        if missing_metrics:
            failures.append(
                f"- {row.get('question_id', '')}/{row.get('pipeline', '')}: metrik kosong {', '.join(missing_metrics)}."
            )
        if len(failures) >= 5:
            break

    return representative[:5], failures[:5]


def format_number(value: Any) -> str:
    number = as_float(value)
    return "" if number is None else f"{number:.3f}"


def build_markdown(
    ragas_summary: list[dict[str, Any]],
    traceability_summary: list[dict[str, Any]],
    missing_metrics: list[str],
    representative_cases: list[str],
    failure_cases: list[str],
) -> str:
    lines = [
        "# Ringkasan Evaluasi Final RAG PNPK",
        "",
        "## RAGAS",
        "",
        "| Metric | Pipeline | n | Mean | Min | Max | Metadata - Baseline |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ragas_summary:
        lines.append(
            "| {metric} | {pipeline} | {n} | {mean} | {min} | {max} | {diff} |".format(
                metric=row["metric"],
                pipeline=row["pipeline"],
                n=row["n"],
                mean=format_number(row.get("mean")),
                min=format_number(row.get("min")),
                max=format_number(row.get("max")),
                diff=format_number(row.get("metadata_minus_baseline")),
            )
        )

    lines.extend(
        [
            "",
            "## Keterlacakan Sumber",
            "",
            "| Pipeline | n | page_match@k | chapter_match@k | section_match@k | label_match@k | fallback |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in traceability_summary:
        lines.append(
            "| {pipeline} | {n} | {page} | {chapter} | {section} | {label} | {fallback} |".format(
                pipeline=row["pipeline"],
                n=row["n"],
                page=format_number(row.get("page_match@k")),
                chapter=format_number(row.get("chapter_match@k")),
                section=format_number(row.get("section_match@k")),
                label=format_number(row.get("label_match@k")),
                fallback=row.get("fallback_count", 0),
            )
        )

    lines.extend(["", "## Kasus Representatif", ""])
    lines.extend(representative_cases or ["- Belum ada kasus representatif yang dapat dipilih dari artefak saat ini."])
    lines.extend(["", "## Kasus Gagal atau Perlu Dicermati", ""])
    lines.extend(failure_cases or ["- Tidak ada kasus gagal eksplisit pada artefak saat ini."])
    lines.extend(["", "## Catatan Kelengkapan Metrik", ""])
    if missing_metrics:
        lines.append(f"- Metrik kosong: {len(missing_metrics)} item/metrik.")
        lines.extend(f"- {item}" for item in missing_metrics[:20])
        if len(missing_metrics) > 20:
            lines.append(f"- ... dan {len(missing_metrics) - 20} item/metrik lain.")
    else:
        lines.append("- Semua baris RAGAS memiliki nilai untuk metrik yang dibaca.")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ringkas skor RAGAS, keterlacakan sumber, dan kasus representatif."
    )
    add_common_arguments(parser)
    parser.add_argument("--ragas", help="Path ragas_scores.jsonl.")
    parser.add_argument("--traceability", help="Path traceability_scores.csv.")
    parser.add_argument("--output", help="Path ringkasan hasil.")
    parser.add_argument("--failure-output", help="Path kasus gagal.")
    parser.add_argument("--representative-output", help="Path kasus representatif.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    results_config = config.get("results", {})
    ragas_path = args.ragas or results_config["ragas_scores"]
    traceability_path = args.traceability or results_config["traceability_scores"]
    output_path = args.output or results_config.get(
        "evaluation_summary",
        "results/qualitative/evaluation_summary.md",
    )
    failure_output = args.failure_output or results_config["failure_cases"]
    representative_output = args.representative_output or results_config["representative_cases"]

    ragas_rows = read_jsonl(ragas_path)
    traceability_rows = read_csv_rows(traceability_path)
    ragas_summary, missing_metrics = summarize_ragas(ragas_rows)
    traceability_summary = summarize_traceability(traceability_rows)
    representative_cases, failure_cases = choose_cases(ragas_rows, traceability_rows)
    markdown = build_markdown(
        ragas_summary,
        traceability_summary,
        missing_metrics,
        representative_cases,
        failure_cases,
    )

    if args.dry_run:
        print(f"[summarize_results] dry_run ragas_rows: {len(ragas_rows)}")
        print(f"[summarize_results] dry_run traceability_rows: {len(traceability_rows)}")
        print(f"[summarize_results] dry_run output: {resolve_repo_path(output_path)}")
        return

    write_text(output_path, markdown)
    write_text(
        representative_output,
        "# Kasus Representatif\n\n"
        + "\n".join(representative_cases or ["- Belum ada kasus representatif yang dapat dipilih."])
        + "\n",
    )
    write_text(
        failure_output,
        "# Kasus Gagal atau Perlu Dicermati\n\n"
        + "\n".join(failure_cases or ["- Tidak ada kasus gagal eksplisit pada artefak saat ini."])
        + "\n",
    )
    print(f"[summarize_results] ragas_rows: {len(ragas_rows)}")
    print(f"[summarize_results] traceability_rows: {len(traceability_rows)}")
    print(f"[summarize_results] missing_metrics: {len(missing_metrics)}")
    print(f"[summarize_results] output: {resolve_repo_path(output_path)}")


if __name__ == "__main__":
    main()
