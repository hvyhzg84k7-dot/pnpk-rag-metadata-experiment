from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


RAGAS_METRICS = ("context_relevance", "faithfulness", "answer_relevance")
TRACEABILITY_METRICS = ("page_match", "chapter_match", "section_match", "label_match")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float):
        return f"{value:.10f}".rstrip("0").rstrip(".")
    return str(value)


def score_complete(row: dict[str, Any]) -> bool:
    return all(row.get(metric) not in (None, "") for metric in RAGAS_METRICS) and not row.get("error")


def build_public_ragas_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public_rows: list[dict[str, Any]] = []
    for row in rows:
        public_rows.append(
            {
                "question_id": row.get("question_id"),
                "pipeline": row.get("pipeline"),
                "context_relevance": fmt(row.get("context_relevance")),
                "faithfulness": fmt(row.get("faithfulness")),
                "answer_relevance": fmt(row.get("answer_relevance")),
                "score_complete": int(score_complete(row)),
                "has_error": int(bool(row.get("error"))),
            }
        )
    return sorted(public_rows, key=lambda item: (item["question_id"], item["pipeline"]))


def build_public_questions(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    public_rows: list[dict[str, Any]] = []
    for row in rows:
        public_rows.append(
            {
                "question_id": row.get("question_id"),
                "question": row.get("question"),
                "source_page": row.get("source_page"),
                "chapter": row.get("chapter"),
                "section": row.get("section"),
                "labels": row.get("labels"),
            }
        )
    return sorted(public_rows, key=lambda item: item["question_id"])


def build_ragas_delta(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_question: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_question[str(row.get("question_id"))][str(row.get("pipeline"))] = row

    delta_rows: list[dict[str, Any]] = []
    for question_id in sorted(by_question):
        baseline = by_question[question_id].get("baseline", {})
        metadata = by_question[question_id].get("metadata", {})
        output: dict[str, Any] = {"question_id": question_id}
        for metric in RAGAS_METRICS:
            baseline_value = baseline.get(metric)
            metadata_value = metadata.get(metric)
            output[f"baseline_{metric}"] = fmt(baseline_value)
            output[f"metadata_{metric}"] = fmt(metadata_value)
            if baseline_value in (None, "") or metadata_value in (None, ""):
                output[f"delta_{metric}"] = ""
                output[f"comparison_{metric}"] = ""
                continue
            delta = float(metadata_value) - float(baseline_value)
            output[f"delta_{metric}"] = fmt(delta)
            if delta > 0:
                output[f"comparison_{metric}"] = "metadata_higher"
            elif delta < 0:
                output[f"comparison_{metric}"] = "metadata_lower"
            else:
                output[f"comparison_{metric}"] = "equal"
        delta_rows.append(output)
    return delta_rows


def build_public_retrieval(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public_rows: list[dict[str, Any]] = []
    for row in rows:
        for context in row.get("contexts", []):
            metadata = context.get("metadata") or {}
            labels = metadata.get("labels") or []
            public_rows.append(
                {
                    "question_id": row.get("question_id"),
                    "pipeline": row.get("pipeline"),
                    "rank": context.get("rank"),
                    "chunk_id": context.get("chunk_id"),
                    "score": fmt(context.get("score")),
                    "distance": fmt(context.get("distance")),
                    "page": metadata.get("page", ""),
                    "chapter": metadata.get("chapter", ""),
                    "section": metadata.get("section", ""),
                    "labels": "|".join(labels) if isinstance(labels, list) else str(labels),
                    "metadata_filter_used": int(bool(row.get("metadata_filter_used"))),
                    "metadata_filter_fallback_used": int(bool(row.get("metadata_filter_fallback_used"))),
                }
            )
    return sorted(public_rows, key=lambda item: (item["question_id"], item["pipeline"], int(item["rank"])))


def build_public_traceability_scores(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    public_rows: list[dict[str, Any]] = []
    for row in rows:
        public_rows.append(
            {
                "question_id": row.get("question_id"),
                "pipeline": row.get("pipeline"),
                "top_k": row.get("top_k"),
                "page_match": row.get("page_match"),
                "chapter_match": row.get("chapter_match"),
                "section_match": row.get("section_match"),
                "label_match": row.get("label_match"),
                "matched_rank": row.get("matched_rank"),
                "metadata_filter_used": row.get("metadata_filter_used"),
                "metadata_filter_fallback_used": row.get("metadata_filter_fallback_used"),
            }
        )
    return public_rows


def build_traceability_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_pipeline: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_pipeline[str(row.get("pipeline"))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for pipeline in sorted(by_pipeline):
        items = by_pipeline[pipeline]
        output: dict[str, Any] = {"pipeline": pipeline, "n": len(items)}
        for metric in TRACEABILITY_METRICS:
            values = [float(item.get(metric) or 0) for item in items]
            output[f"{metric}_mean"] = fmt(sum(values) / len(values) if values else 0)
            output[f"{metric}_matched_count"] = int(sum(values))
        output["metadata_filter_used_count"] = sum(1 for item in items if item.get("metadata_filter_used") == "True")
        output["metadata_filter_fallback_used_count"] = sum(
            1 for item in items if item.get("metadata_filter_fallback_used") == "True"
        )
        summary_rows.append(output)
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build public-safe calculation artifacts without answers, contexts, or source text."
    )
    parser.add_argument("--evaluation-dataset", type=Path)
    parser.add_argument("--ragas-scores", required=True, type=Path)
    parser.add_argument("--retrieval-outputs", required=True, type=Path)
    parser.add_argument("--traceability-scores", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("."), type=Path)
    args = parser.parse_args()

    ragas_rows = read_jsonl(args.ragas_scores)
    retrieval_rows = read_jsonl(args.retrieval_outputs)
    traceability_rows = read_csv(args.traceability_scores)
    output_dir = args.output_dir

    if args.evaluation_dataset:
        write_csv(
            output_dir / "data/evaluation/evaluation_questions_public.csv",
            build_public_questions(read_csv(args.evaluation_dataset)),
            [
                "question_id",
                "question",
                "source_page",
                "chapter",
                "section",
                "labels",
            ],
        )

    write_csv(
        output_dir / "results/ragas/ragas_scores_public.csv",
        build_public_ragas_scores(ragas_rows),
        [
            "question_id",
            "pipeline",
            "context_relevance",
            "faithfulness",
            "answer_relevance",
            "score_complete",
            "has_error",
        ],
    )
    write_csv(
        output_dir / "results/ragas/ragas_delta_by_question.csv",
        build_ragas_delta(ragas_rows),
        [
            "question_id",
            "baseline_context_relevance",
            "metadata_context_relevance",
            "delta_context_relevance",
            "comparison_context_relevance",
            "baseline_faithfulness",
            "metadata_faithfulness",
            "delta_faithfulness",
            "comparison_faithfulness",
            "baseline_answer_relevance",
            "metadata_answer_relevance",
            "delta_answer_relevance",
            "comparison_answer_relevance",
        ],
    )
    write_csv(
        output_dir / "results/retrieval/retrieval_contexts_public.csv",
        build_public_retrieval(retrieval_rows),
        [
            "question_id",
            "pipeline",
            "rank",
            "chunk_id",
            "score",
            "distance",
            "page",
            "chapter",
            "section",
            "labels",
            "metadata_filter_used",
            "metadata_filter_fallback_used",
        ],
    )
    write_csv(
        output_dir / "results/traceability/traceability_scores_public.csv",
        build_public_traceability_scores(traceability_rows),
        [
            "question_id",
            "pipeline",
            "top_k",
            "page_match",
            "chapter_match",
            "section_match",
            "label_match",
            "matched_rank",
            "metadata_filter_used",
            "metadata_filter_fallback_used",
        ],
    )
    write_csv(
        output_dir / "results/traceability/traceability_summary_public.csv",
        build_traceability_summary(traceability_rows),
        [
            "pipeline",
            "n",
            "page_match_mean",
            "page_match_matched_count",
            "chapter_match_mean",
            "chapter_match_matched_count",
            "section_match_mean",
            "section_match_matched_count",
            "label_match_mean",
            "label_match_matched_count",
            "metadata_filter_used_count",
            "metadata_filter_fallback_used_count",
        ],
    )


if __name__ == "__main__":
    main()
