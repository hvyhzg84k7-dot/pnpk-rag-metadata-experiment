from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (
    add_common_arguments,
    add_question_filter_arguments,
    filter_by_question_id,
    load_config,
    load_local_env,
    read_jsonl,
    resolve_repo_path,
    write_jsonl,
)
from openai_client import OpenAIClient, is_local_base_url


def openai_api_settings(config: dict[str, Any]) -> tuple[str, str | None]:
    api_config = config.get("model_api", {})
    base_url = api_config.get("base_url") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    api_key = api_config.get("api_key")
    if not api_key and base_url and is_local_base_url(str(base_url)):
        api_key = "ollama"
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and base_url:
        api_key = "ollama"
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY belum tersedia di environment.")
    return str(api_key), str(base_url) if base_url else None


def should_use_no_auth_embeddings(config: dict[str, Any]) -> bool:
    api_config = config.get("model_api", {})
    evaluation_config = config.get("evaluation", {})
    return not api_config.get("api_key") and not evaluation_config.get("embedding_api_key")


def metric_value(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(number):
            return number
    return None


def parse_rate_limit_reset_seconds(message: str) -> float | None:
    match = re.search(r"reset after (\d+(?:\.\d+)?)s", message)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def infer_retry_delay_seconds(message: str, *, default_seconds: float, buffer_seconds: float) -> float:
    reset_seconds = parse_rate_limit_reset_seconds(message)
    if reset_seconds is None:
        return default_seconds
    return max(default_seconds, reset_seconds + buffer_seconds)


def source_input_hash(
    row: dict[str, Any],
    *,
    context_top_k: int | None = None,
    evaluation_fingerprint: str = "",
) -> str:
    contexts = [
        str(context.get("chunk_text", ""))
        for context in row.get("contexts", [])
        if str(context.get("chunk_text", "")).strip()
    ]
    if context_top_k and context_top_k > 0:
        contexts = contexts[:context_top_k]
    payload = {
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "reference_answer": row.get("reference_answer", ""),
        "contexts": contexts,
        "evaluation_fingerprint": evaluation_fingerprint,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def score_row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("question_id", "")), str(row.get("pipeline", ""))


def score_row_complete(row: dict[str, Any]) -> bool:
    return all(metric_value(row, metric) is not None for metric in [
        "context_relevance",
        "faithfulness",
        "answer_relevance",
    ])


def missing_metric_names(row: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for metric in ["context_relevance", "faithfulness", "answer_relevance"]:
        if metric_value(row, metric) is None:
            missing.append(metric)
    return missing


def score_row_status(
    source: dict[str, Any],
    existing: dict[str, Any] | None,
    *,
    context_top_k: int | None = None,
    evaluation_fingerprint: str = "",
) -> tuple[str, str, str]:
    key = score_row_key(source)
    input_hash = source_input_hash(
        source,
        context_top_k=context_top_k,
        evaluation_fingerprint=evaluation_fingerprint,
    )
    if existing is None:
        return key[0], key[1], "missing"
    if existing.get("input_hash") != input_hash:
        return key[0], key[1], "input_changed"
    if not score_row_complete(existing):
        missing_metrics = ",".join(missing_metric_names(existing))
        suffix = f":{missing_metrics}" if missing_metrics else ""
        return key[0], key[1], f"incomplete{suffix}"
    return key[0], key[1], "complete"


def summarize_pending_rows(
    rows: list[dict[str, Any]],
    existing_rows_by_key: dict[tuple[str, str], dict[str, Any]],
    *,
    context_top_k: int | None = None,
    evaluation_fingerprint: str = "",
) -> tuple[list[dict[str, Any]], list[str]]:
    pending_rows: list[dict[str, Any]] = []
    status_lines: list[str] = []
    for row in rows:
        key = score_row_key(row)
        question_id, pipeline, status = score_row_status(
            row,
            existing_rows_by_key.get(key),
            context_top_k=context_top_k,
            evaluation_fingerprint=evaluation_fingerprint,
        )
        if status != "complete":
            pending_rows.append(row)
            status_lines.append(f"{question_id} {pipeline} {status}")
    return pending_rows, status_lines


def build_ragas_dataset(
    rows: list[dict[str, Any]],
    *,
    context_top_k: int | None = None,
) -> list[dict[str, Any]]:
    dataset_rows: list[dict[str, Any]] = []
    for row in rows:
        contexts = [
            str(context.get("chunk_text", ""))
            for context in row.get("contexts", [])
            if str(context.get("chunk_text", "")).strip()
        ]
        if context_top_k and context_top_k > 0:
            contexts = contexts[:context_top_k]
        dataset_rows.append(
            {
                "user_input": row["question"],
                "response": row["answer"],
                "retrieved_contexts": contexts,
                "reference": row.get("reference_answer", ""),
            }
        )
    return dataset_rows


def raw_to_score_row(source: dict[str, Any], raw: dict[str, Any], *, input_hash: str) -> dict[str, Any]:
    context_relevance = metric_value(raw, "nv_context_relevance", "context_relevance")
    faithfulness_value = metric_value(raw, "faithfulness")
    answer_relevance = metric_value(raw, "answer_relevancy", "answer_relevance")
    return {
        "question_id": source.get("question_id", ""),
        "pipeline": source.get("pipeline", ""),
        "question": source.get("question", ""),
        "input_hash": input_hash,
        "context_relevance": context_relevance if context_relevance is not None else "",
        "faithfulness": faithfulness_value if faithfulness_value is not None else "",
        "answer_relevance": answer_relevance if answer_relevance is not None else "",
        "raw_ragas": raw,
        "error": source.get("error", ""),
    }


def write_summary(path: str | Path, score_rows: list[dict[str, Any]]) -> None:
    output_path = resolve_repo_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["context_relevance", "faithfulness", "answer_relevance"]
    pipelines = sorted({row["pipeline"] for row in score_rows})
    summary_rows: list[dict[str, Any]] = []
    for metric in metrics:
        values_by_pipeline: dict[str, list[float]] = {}
        for pipeline in pipelines:
            values = [
                float(row[metric])
                for row in score_rows
                if row["pipeline"] == pipeline and row.get(metric) not in ("", None)
            ]
            values_by_pipeline[pipeline] = values
            summary_rows.append(
                {
                    "metric": metric,
                    "pipeline": pipeline,
                    "n": len(values),
                    "mean": sum(values) / len(values) if values else "",
                    "min": min(values) if values else "",
                    "max": max(values) if values else "",
                    "metadata_minus_baseline": "",
                    "metadata_higher_count": "",
                    "metadata_equal_count": "",
                    "metadata_lower_count": "",
                }
            )

        baseline_values = {
            row["question_id"]: row.get(metric)
            for row in score_rows
            if row["pipeline"] == "baseline" and row.get(metric) not in ("", None)
        }
        metadata_values = {
            row["question_id"]: row.get(metric)
            for row in score_rows
            if row["pipeline"] == "metadata" and row.get(metric) not in ("", None)
        }
        shared_ids = sorted(set(baseline_values) & set(metadata_values))
        higher = equal = lower = 0
        for question_id in shared_ids:
            baseline = float(baseline_values[question_id])
            metadata = float(metadata_values[question_id])
            if metadata > baseline:
                higher += 1
            elif metadata < baseline:
                lower += 1
            else:
                equal += 1
        baseline_mean = (
            sum(values_by_pipeline.get("baseline", [])) / len(values_by_pipeline["baseline"])
            if values_by_pipeline.get("baseline")
            else None
        )
        metadata_mean = (
            sum(values_by_pipeline.get("metadata", [])) / len(values_by_pipeline["metadata"])
            if values_by_pipeline.get("metadata")
            else None
        )
        summary_rows.append(
            {
                "metric": metric,
                "pipeline": "comparison",
                "n": len(shared_ids),
                "mean": "",
                "min": "",
                "max": "",
                "metadata_minus_baseline": (
                    metadata_mean - baseline_mean
                    if metadata_mean is not None and baseline_mean is not None
                    else ""
                ),
                "metadata_higher_count": higher,
                "metadata_equal_count": equal,
                "metadata_lower_count": lower,
            }
        )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "metric",
                "pipeline",
                "n",
                "mean",
                "min",
                "max",
                "metadata_minus_baseline",
                "metadata_higher_count",
                "metadata_equal_count",
                "metadata_lower_count",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hitung metrik RAGAS untuk jawaban baseline dan metadata."
    )
    add_common_arguments(parser)
    parser.add_argument("--answers", help="Path answers.jsonl.")
    parser.add_argument("--output", help="Path ragas_scores.jsonl.")
    parser.add_argument("--summary-output", help="Path ragas_summary.csv.")
    parser.add_argument("--limit", type=int, help="Batasi jumlah jawaban untuk smoke test.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size evaluasi RAGAS.")
    parser.add_argument(
        "--include-complete",
        action="store_true",
        help="Tetap iterasi semua row terpilih, termasuk yang sudah lengkap di cache.",
    )
    add_question_filter_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    load_local_env()
    config = load_config(args.config)
    answers_path = args.answers or config["results"]["answers"]
    output_path = args.output or config["results"]["ragas_scores"]
    summary_path = args.summary_output or config["results"]["ragas_summary"]
    evaluation_config = config.get("evaluation", {})
    llm_model = evaluation_config.get("llm_model") or config["models"]["generation"]
    embedding_model = evaluation_config.get("embedding_model") or config["models"]["embedding"]
    ragas_context_top_k = evaluation_config.get("ragas_context_top_k")
    context_top_k = int(ragas_context_top_k) if ragas_context_top_k else None
    evaluation_fingerprint = json.dumps(
        {
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "ragas_context_top_k": context_top_k,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    existing_rows = read_jsonl(output_path) if resolve_repo_path(output_path).exists() else []
    existing_rows_by_key = {score_row_key(row): row for row in existing_rows}

    if args.dry_run:
        print(f"[run_ragas] dry_run answers: {resolve_repo_path(answers_path)}")
        if resolve_repo_path(answers_path).exists():
            answer_rows = read_jsonl(answers_path)
            answer_rows = filter_by_question_id(
                answer_rows,
                question_id_min=args.question_id_min,
                question_id_max=args.question_id_max,
                question_ids=args.question_ids,
            )
            if args.limit is not None:
                answer_rows = answer_rows[: args.limit]
            print(f"[run_ragas] dry_run rows: {len(answer_rows)}")
            pending_rows, pending_status_lines = summarize_pending_rows(
                answer_rows,
                existing_rows_by_key,
                context_top_k=context_top_k,
                evaluation_fingerprint=evaluation_fingerprint,
            )
            print(f"[run_ragas] dry_run pending_rows: {len(pending_rows)}")
            for line in pending_status_lines:
                print(f"[run_ragas] dry_run pending {line}")
        else:
            print("[run_ragas] dry_run rows: answers file belum tersedia")
        print(f"[run_ragas] dry_run output: {resolve_repo_path(output_path)}")
        print(f"[run_ragas] dry_run summary: {resolve_repo_path(summary_path)}")
        return

    all_answer_rows = read_jsonl(answers_path)
    answer_rows = filter_by_question_id(
        all_answer_rows,
        question_id_min=args.question_id_min,
        question_id_max=args.question_id_max,
        question_ids=args.question_ids,
    )
    if args.limit is not None:
        answer_rows = answer_rows[: args.limit]

    pending_rows, pending_status_lines = summarize_pending_rows(
        answer_rows,
        existing_rows_by_key,
        context_top_k=context_top_k,
        evaluation_fingerprint=evaluation_fingerprint,
    )
    target_rows = answer_rows if args.include_complete else pending_rows

    from datasets import Dataset
    from langchain_core.embeddings import Embeddings
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.metrics import ContextRelevance, answer_relevancy, faithfulness
    from ragas.run_config import RunConfig

    class NoAuthOpenAIEmbeddings(Embeddings):
        def __init__(self, *, base_url: str, model: str, timeout: float, max_retries: int) -> None:
            self.model = model
            self.max_retries = max_retries
            self.client = OpenAIClient(base_url=base_url, provider="openai_compatible", timeout=timeout)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.client.embed_texts(
                model=self.model,
                texts=texts,
                task_type="RETRIEVAL_DOCUMENT",
                max_retries=self.max_retries,
            )

        def embed_query(self, text: str) -> list[float]:
            return self.client.embed_texts(
                model=self.model,
                texts=[text],
                task_type="RETRIEVAL_QUERY",
                max_retries=self.max_retries,
            )[0]

    api_key, base_url = openai_api_settings(config)
    timeout_seconds = float(evaluation_config.get("timeout_seconds", 180))
    max_retries = int(evaluation_config.get("max_retries", 1))
    max_workers = int(evaluation_config.get("max_workers", 1))
    max_incomplete_streak = int(evaluation_config.get("max_incomplete_streak", 3))
    streaming = bool(evaluation_config.get("streaming", False))
    evaluation_think = evaluation_config.get("think")
    sleep_seconds_between_rows = float(evaluation_config.get("sleep_seconds_between_rows", 0))
    row_max_attempts = int(evaluation_config.get("row_max_attempts", 3))
    retry_cooldown_seconds = float(evaluation_config.get("retry_cooldown_seconds", 4))
    retry_buffer_seconds = float(evaluation_config.get("retry_buffer_seconds", 1))
    max_output_tokens = int(
        evaluation_config.get(
            "max_output_tokens",
            config.get("generation", {}).get("max_output_tokens", 512),
        )
    )
    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=llm_model,
        temperature=0,
        max_completion_tokens=max_output_tokens,
        timeout=timeout_seconds,
        max_retries=max_retries,
        streaming=streaming,
        tiktoken_model_name=evaluation_config.get("tiktoken_model_name"),
        extra_body={"think": evaluation_think} if evaluation_think is not None else None,
    )
    if should_use_no_auth_embeddings(config):
        embeddings = NoAuthOpenAIEmbeddings(
            base_url=base_url or "",
            model=embedding_model,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
    else:
        embeddings = OpenAIEmbeddings(
            api_key=evaluation_config.get("embedding_api_key") or api_key,
            base_url=base_url,
            model=embedding_model,
            timeout=timeout_seconds,
            max_retries=max_retries,
            tiktoken_enabled=bool(evaluation_config.get("tiktoken_enabled", False)),
        )
    stored_rows_by_key: dict[tuple[str, str], dict[str, Any]] = dict(existing_rows_by_key)
    score_rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for source in all_answer_rows:
        key = score_row_key(source)
        input_hash = source_input_hash(
            source,
            context_top_k=context_top_k,
            evaluation_fingerprint=evaluation_fingerprint,
        )
        existing = existing_rows_by_key.get(key)
        if existing and existing.get("input_hash") == input_hash:
            score_rows_by_key[key] = existing

    cached_complete_count = sum(
        1
        for row in score_rows_by_key.values()
        if score_row_complete(row)
    )
    print(
        f"[run_ragas] cached_complete_rows: {cached_complete_count}/{len(all_answer_rows)}",
        flush=True,
    )
    print(
        f"[run_ragas] selected_rows: {len(answer_rows)} pending_rows: {len(pending_rows)} "
        f"mode={'all' if args.include_complete else 'pending_only'}",
        flush=True,
    )
    for line in pending_status_lines:
        print(f"[run_ragas] pending {line}", flush=True)
    print(
        "[run_ragas] evaluator "
        f"model={llm_model} streaming={streaming} timeout={timeout_seconds} "
        f"max_retries={max_retries} max_workers={max_workers} "
        f"row_max_attempts={row_max_attempts}",
        flush=True,
    )
    if not target_rows:
        stored_rows = [
            stored_rows_by_key[score_row_key(row)]
            for row in all_answer_rows
            if score_row_key(row) in stored_rows_by_key
        ]
        valid_rows = [
            score_rows_by_key[score_row_key(row)]
            for row in all_answer_rows
            if score_row_key(row) in score_rows_by_key
        ]
        count = write_jsonl(output_path, stored_rows)
        write_summary(summary_path, valid_rows)
        print("[run_ragas] nothing_to_do: semua row terpilih sudah lengkap.", flush=True)
        print(f"[run_ragas] valid_summary_rows: {len(valid_rows)}", flush=True)
        print(f"[run_ragas] output_rows: {count}", flush=True)
        print(f"[run_ragas] output: {resolve_repo_path(output_path)}", flush=True)
        print(f"[run_ragas] summary: {resolve_repo_path(summary_path)}", flush=True)
        return

    run_config = RunConfig(
        timeout=timeout_seconds,
        max_retries=max_retries,
        max_workers=max_workers,
    )
    metrics = [ContextRelevance(), faithfulness, answer_relevancy]
    incomplete_streak = 0
    for index, source in enumerate(target_rows, start=1):
        key = score_row_key(source)
        if key in score_rows_by_key and score_row_complete(score_rows_by_key[key]):
            print(f"[run_ragas] {index}/{len(target_rows)} {key[0]} {key[1]} cached=True", flush=True)
            continue

        input_hash = source_input_hash(
            source,
            context_top_k=context_top_k,
            evaluation_fingerprint=evaluation_fingerprint,
        )
        score_row: dict[str, Any] | None = None
        for attempt in range(1, row_max_attempts + 1):
            result = evaluate(
                Dataset.from_list(build_ragas_dataset([source], context_top_k=context_top_k)),
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
                raise_exceptions=False,
                show_progress=False,
                batch_size=1,
                run_config=run_config,
            )
            frame = result.to_pandas()
            raw_rows = json.loads(frame.to_json(orient="records", force_ascii=False))
            raw = raw_rows[0] if raw_rows else {}
            score_row = raw_to_score_row(source, raw, input_hash=input_hash)
            if score_row_complete(score_row):
                break
            if attempt >= row_max_attempts:
                break
            missing_metrics = ",".join(missing_metric_names(score_row)) or "-"
            raw_text = json.dumps(raw, ensure_ascii=False)
            delay_seconds = infer_retry_delay_seconds(
                raw_text,
                default_seconds=retry_cooldown_seconds,
                buffer_seconds=retry_buffer_seconds,
            )
            print(
                "[run_ragas] retry "
                f"{key[0]} {key[1]} attempt={attempt + 1}/{row_max_attempts} "
                f"missing={missing_metrics} sleep={delay_seconds:.1f}s",
                flush=True,
            )
            time.sleep(delay_seconds)
        if score_row is None:
            raise RuntimeError(f"Gagal membangun score_row untuk {key[0]} {key[1]}.")
        stored_rows_by_key[key] = score_row
        score_rows_by_key[key] = score_row
        is_complete = score_row_complete(score_row)
        incomplete_streak = 0 if is_complete else incomplete_streak + 1

        ordered_stored_rows = [
            stored_rows_by_key[score_row_key(row)]
            for row in all_answer_rows
            if score_row_key(row) in stored_rows_by_key
        ]
        ordered_valid_rows = [
            score_rows_by_key[score_row_key(row)]
            for row in all_answer_rows
            if score_row_key(row) in score_rows_by_key
        ]
        write_jsonl(output_path, ordered_stored_rows)
        write_summary(summary_path, ordered_valid_rows)
        print(
            "[run_ragas] "
            f"{index}/{len(target_rows)} {key[0]} {key[1]} "
            f"complete={is_complete} "
            f"context={score_row.get('context_relevance')} "
            f"faithfulness={score_row.get('faithfulness')} "
            f"answer_relevance={score_row.get('answer_relevance')}",
            flush=True,
        )
        if sleep_seconds_between_rows > 0 and index < len(target_rows):
            print(
                f"[run_ragas] cooldown after {key[0]} {key[1]} sleep={sleep_seconds_between_rows:.1f}s",
                flush=True,
            )
            time.sleep(sleep_seconds_between_rows)
        if max_incomplete_streak > 0 and incomplete_streak >= max_incomplete_streak:
            print(
                "[run_ragas] stopping: "
                f"{incomplete_streak} incomplete rows in a row; rerun later to continue from cache.",
                flush=True,
            )
            break

    stored_rows = [
        stored_rows_by_key[score_row_key(row)]
        for row in all_answer_rows
        if score_row_key(row) in stored_rows_by_key
    ]
    valid_rows = [
        score_rows_by_key[score_row_key(row)]
        for row in all_answer_rows
        if score_row_key(row) in score_rows_by_key
    ]
    count = write_jsonl(output_path, stored_rows)
    write_summary(summary_path, valid_rows)
    print(f"[run_ragas] valid_summary_rows: {len(valid_rows)}", flush=True)
    print(f"[run_ragas] output_rows: {count}", flush=True)
    print(f"[run_ragas] output: {resolve_repo_path(output_path)}", flush=True)
    print(f"[run_ragas] summary: {resolve_repo_path(summary_path)}", flush=True)


if __name__ == "__main__":
    main()
