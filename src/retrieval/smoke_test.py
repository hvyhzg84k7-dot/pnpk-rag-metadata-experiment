from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_common_arguments, load_config, load_local_env, resolve_repo_path, write_jsonl
from retrieval.retrieve import create_openai_client, retrieve_contexts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Jalankan smoke test retrieval untuk beberapa pertanyaan awal."
    )
    add_common_arguments(parser)
    parser.add_argument("--output", help="Path smoke_test_outputs.jsonl.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    load_local_env()
    config = load_config(args.config)
    output_path = args.output or config["results"]["smoke_test_outputs"]
    top_k = int(config["retrieval"]["top_k"])
    questions = config["retrieval"].get("smoke_questions", [])
    if not questions:
        raise ValueError("retrieval.smoke_questions belum tersedia di konfigurasi.")

    if args.dry_run:
        print(f"[smoke_test] dry_run questions: {len(questions)}")
        print(f"[smoke_test] dry_run output_rows: {len(questions) * 2}")
        print(f"[smoke_test] dry_run top_k: {top_k}")
        return

    client = create_openai_client(config)
    rows = []
    for item in questions:
        question_id = item["question_id"]
        question = item["question"]
        for pipeline in ["baseline", "metadata"]:
            metadata_filter = item.get("metadata_filter") if pipeline == "metadata" else None
            query_info: dict[str, object] = {}
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
                    "expected_labels": item.get("labels", []),
                    "pipeline": pipeline,
                    "question": question,
                    "top_k": top_k,
                    **query_info,
                    "contexts": contexts,
                }
            )
            top_context = contexts[0] if contexts else {}
            top_metadata = top_context.get("metadata", {})
            top_labels = top_metadata.get("labels", [])
            print(
                "[smoke_test] "
                f"{question_id} {pipeline} top={top_context.get('chunk_id', '')} "
                f"labels={top_labels} page={top_metadata.get('page', '')} "
                f"filter_used={query_info.get('metadata_filter_used', False)} "
                f"fallback={query_info.get('metadata_filter_fallback_used', False)}"
            )

    count = write_jsonl(output_path, rows)
    print(f"[smoke_test] questions: {len(questions)}")
    print(f"[smoke_test] output_rows: {count}")
    print(f"[smoke_test] output: {resolve_repo_path(output_path)}")


if __name__ == "__main__":
    main()
