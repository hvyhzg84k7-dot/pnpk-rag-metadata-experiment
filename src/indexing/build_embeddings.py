from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_common_arguments, load_config, load_local_env, read_jsonl, resolve_repo_path, write_jsonl
from openai_client import OpenAIClient


def text_cache_key(text: str, model: str) -> str:
    payload = {"model": model, "text": text}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def load_embedding_cache(path: str | None, model: str) -> dict[str, list[float]]:
    if not path:
        return {}
    cache_path = resolve_repo_path(path)
    if not cache_path.exists():
        return {}

    cache: dict[str, list[float]] = {}
    for row in read_jsonl(cache_path):
        key = row.get("cache_key")
        embedding = row.get("embedding")
        if row.get("embedding_model") == model and isinstance(key, str) and isinstance(embedding, list):
            cache[key] = embedding
    return cache


def write_embedding_cache(path: str | None, cache: dict[str, list[float]], model: str) -> None:
    if not path:
        return
    rows = [
        {
            "cache_key": key,
            "embedding_model": model,
            "embedding_dim": len(embedding),
            "embedding": embedding,
        }
        for key, embedding in sorted(cache.items())
    ]
    write_jsonl(path, rows)


def create_client(config: dict[str, Any]) -> OpenAIClient:
    timeout = float(config.get("embedding", {}).get("timeout_seconds", 60))
    api_config = config.get("model_api", {})
    return OpenAIClient(
        api_key=api_config.get("api_key"),
        base_url=api_config.get("base_url"),
        provider=api_config.get("provider"),
        timeout=timeout,
    )


def request_embeddings(
    client: OpenAIClient,
    *,
    model: str,
    texts: list[str],
    max_retries: int,
) -> list[list[float]]:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return client.embed_texts(
                model=model,
                texts=texts,
                task_type="RETRIEVAL_DOCUMENT",
                max_retries=0,
            )
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Gagal membuat embedding: {last_error}") from last_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Buat embedding untuk chunk baseline dan metadata."
    )
    add_common_arguments(parser)
    parser.add_argument("--chunks", help="Path chunk input.")
    parser.add_argument("--pipeline", choices=["baseline", "metadata"], help="Nama pipeline.")
    parser.add_argument("--output", help="Path output embedding JSONL.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    load_local_env()
    config = load_config(args.config)
    pipeline = args.pipeline or "baseline"
    if pipeline == "baseline":
        chunks_path = args.chunks or config["data"]["chunks_baseline"]
        output_path = args.output or config["data"]["embeddings_baseline"]
    else:
        chunks_path = args.chunks or config["data"]["chunks_metadata"]
        output_path = args.output or config["data"]["embeddings_metadata"]

    chunks = read_jsonl(chunks_path)
    if not chunks:
        raise ValueError(f"Tidak ada chunk untuk dibuat embedding: {chunks_path}")

    model = config["models"]["embedding"]
    embedding_config = config.get("embedding", {})
    batch_size = int(embedding_config.get("batch_size", 64))
    max_retries = int(embedding_config.get("max_retries", 2))
    throttle_seconds = float(embedding_config.get("throttle_seconds_between_batches", 0))
    cache_path = embedding_config.get("cache")
    cache = load_embedding_cache(cache_path, model)

    missing = []
    for chunk in chunks:
        key = text_cache_key(chunk["chunk_text"], model)
        if key not in cache:
            missing.append((key, chunk["chunk_text"]))

    unique_missing: dict[str, str] = {}
    for key, text in missing:
        unique_missing.setdefault(key, text)

    if args.dry_run:
        print(f"[build_embeddings] dry_run pipeline: {pipeline}")
        print(f"[build_embeddings] dry_run chunks: {len(chunks)}")
        print(f"[build_embeddings] dry_run cached_texts: {len(chunks) - len(missing)}")
        print(f"[build_embeddings] dry_run unique_missing_texts: {len(unique_missing)}")
        return

    client = create_client(config)
    keys = list(unique_missing.keys())
    for completed, key_batch in enumerate(batched(keys, batch_size), start=1):
        texts = [unique_missing[key] for key in key_batch]
        embeddings = request_embeddings(client, model=model, texts=texts, max_retries=max_retries)
        if len(embeddings) != len(key_batch):
            raise RuntimeError("Jumlah embedding dari API tidak sesuai jumlah input.")
        for key, embedding in zip(key_batch, embeddings, strict=True):
            cache[key] = embedding
        print(
            f"[build_embeddings] embedded_batches: {completed}/{max(1, (len(keys) + batch_size - 1) // batch_size)}",
            flush=True,
        )
        write_embedding_cache(cache_path, cache, model)
        if throttle_seconds > 0 and completed < max(1, (len(keys) + batch_size - 1) // batch_size):
            time.sleep(throttle_seconds)

    rows = []
    for chunk in chunks:
        key = text_cache_key(chunk["chunk_text"], model)
        embedding = cache[key]
        rows.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text_hash": key,
                "embedding_model": model,
                "embedding_dim": len(embedding),
                "embedding": embedding,
            }
        )

    count = write_jsonl(output_path, rows)
    write_embedding_cache(cache_path, cache, model)
    print(f"[build_embeddings] pipeline: {pipeline}")
    print(f"[build_embeddings] chunks: {len(chunks)}")
    print(f"[build_embeddings] embedding_model: {model}")
    print(f"[build_embeddings] embedding_dim: {rows[0]['embedding_dim'] if rows else 0}")
    print(f"[build_embeddings] api_texts_embedded: {len(unique_missing)}")
    print(f"[build_embeddings] output_embeddings: {count} -> {resolve_repo_path(output_path)}")


if __name__ == "__main__":
    main()
