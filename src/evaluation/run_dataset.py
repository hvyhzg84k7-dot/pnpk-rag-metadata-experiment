from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (
    add_common_arguments,
    add_question_filter_arguments,
    filter_by_question_id,
    load_config,
    load_local_env,
    resolve_repo_path,
    write_jsonl,
)
from retrieval.retrieve import create_openai_client, retrieve_contexts


REQUIRED_COLUMNS = {
    "question_id",
    "question",
    "reference_answer",
    "reference_context",
    "source_page",
    "chapter",
    "section",
    "labels",
}


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = resolve_repo_path(path)
    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Kolom dataset tidak lengkap: {missing}")

        rows: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            question_id = (row.get("question_id") or "").strip()
            if not question_id:
                raise ValueError(f"question_id kosong pada baris {line_number}")
            if question_id in seen_ids:
                raise ValueError(f"question_id duplikat: {question_id}")
            seen_ids.add(question_id)

            labels_raw = row.get("labels") or ""
            try:
                labels = json.loads(labels_raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"labels bukan JSON valid pada {question_id}: {exc}") from exc
            if not isinstance(labels, list) or not labels:
                raise ValueError(f"labels harus berupa JSON array non-kosong pada {question_id}")
            labels = [str(label).strip() for label in labels if str(label).strip()]
            if not labels:
                raise ValueError(f"labels kosong setelah normalisasi pada {question_id}")

            normalized = {key: (value or "").strip() for key, value in row.items()}
            normalized["labels"] = labels
            rows.append(normalized)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Jalankan semua pertanyaan dataset evaluasi pada dua pipeline."
    )
    add_common_arguments(parser)
    parser.add_argument("--dataset", help="Path evaluation_dataset.csv.")
    parser.add_argument("--output", help="Path retrieval_outputs.jsonl.")
    parser.add_argument("--top-k", type=int, help="Jumlah konteks teratas.")
    add_question_filter_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    load_local_env()
    config = load_config(args.config)
    dataset_path = args.dataset or config["data"]["evaluation_dataset"]
    output_path = args.output or config["results"]["retrieval_outputs"]
    top_k = args.top_k or int(config["retrieval"]["top_k"])
    dataset = load_dataset(dataset_path)
    dataset = filter_by_question_id(
        dataset,
        question_id_min=args.question_id_min,
        question_id_max=args.question_id_max,
        question_ids=args.question_ids,
    )

    if args.dry_run:
        print(f"[run_dataset] dry_run dataset: {resolve_repo_path(dataset_path)}")
        print(f"[run_dataset] dry_run questions: {len(dataset)}")
        print(f"[run_dataset] dry_run output_rows: {len(dataset) * 2}")
        print(f"[run_dataset] dry_run top_k: {top_k}")
        print(f"[run_dataset] dry_run output: {resolve_repo_path(output_path)}")
        return

    client = create_openai_client(config)
    rows: list[dict[str, Any]] = []
    for item in dataset:
        question_id = item["question_id"]
        question = item["question"]
        reference = {
            "reference_answer": item["reference_answer"],
            "reference_context": item["reference_context"],
            "source_page": item["source_page"],
            "chapter": item["chapter"],
            "section": item["section"],
            "labels": item["labels"],
        }
        for pipeline in ["baseline", "metadata"]:
            metadata_filter = {"labels": item["labels"]} if pipeline == "metadata" else None
            query_info: dict[str, Any] = {}
            contexts = retrieve_contexts(
                config=config,
                question=question,
                pipeline=pipeline,
                top_k=top_k,
                client=client,
                metadata_filter=metadata_filter,
                query_info=query_info,
            )
            rows.append(
                {
                    "question_id": question_id,
                    "pipeline": pipeline,
                    "question": question,
                    "top_k": top_k,
                    **reference,
                    **query_info,
                    "contexts": contexts,
                }
            )
            top_context = contexts[0] if contexts else {}
            top_metadata = top_context.get("metadata", {})
            print(
                "[run_dataset] "
                f"{question_id} {pipeline} top={top_context.get('chunk_id', '')} "
                f"page={top_metadata.get('page', '')} "
                f"labels={top_metadata.get('labels', [])} "
                f"filter_used={query_info.get('metadata_filter_used', False)} "
                f"fallback={query_info.get('metadata_filter_fallback_used', False)}"
            )

    count = write_jsonl(output_path, rows)
    print(f"[run_dataset] questions: {len(dataset)}")
    print(f"[run_dataset] output_rows: {count}")
    print(f"[run_dataset] output: {resolve_repo_path(output_path)}")


if __name__ == "__main__":
    main()
