from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_common_arguments, load_config, read_jsonl, resolve_repo_path


def load_embeddings_by_chunk_id(path: str) -> dict[str, list[float]]:
    rows = read_jsonl(path)
    embeddings: dict[str, list[float]] = {}
    for row in rows:
        chunk_id = row.get("chunk_id")
        embedding = row.get("embedding")
        if isinstance(chunk_id, str) and isinstance(embedding, list):
            embeddings[chunk_id] = embedding
    return embeddings


def metadata_for_pipeline(chunk: dict[str, Any], pipeline: str) -> dict[str, Any]:
    if pipeline == "baseline":
        return {"pipeline": "baseline"}
    labels = chunk.get("labels", [])
    if not isinstance(labels, list):
        labels = []
    return {
        "pipeline": "metadata",
        "source_document": chunk.get("source_document", ""),
        "page": int(chunk.get("page", 0)),
        "chapter": chunk.get("chapter", ""),
        "section": chunk.get("section", ""),
        "labels": [str(label) for label in labels],
        "page_chunk_index": int(chunk.get("page_chunk_index", 0)),
        "token_count": int(chunk.get("token_count", 0)),
    }


def collection_name(config: dict[str, Any], pipeline: str) -> str:
    if pipeline == "baseline":
        return config["chroma"]["baseline_collection"]
    return config["chroma"]["metadata_collection"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest chunk dan embedding ke collection ChromaDB."
    )
    add_common_arguments(parser)
    parser.add_argument("--pipeline", choices=["baseline", "metadata"], help="Nama pipeline.")
    parser.add_argument("--chunks", help="Path chunk input.")
    parser.add_argument("--embeddings", help="Path embedding JSONL.")
    parser.add_argument("--reset", action="store_true", help="Bangun ulang collection target.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    pipeline = args.pipeline or "baseline"
    if pipeline == "baseline":
        chunks_path = args.chunks or config["data"]["chunks_baseline"]
        embeddings_path = args.embeddings or config["data"]["embeddings_baseline"]
    else:
        chunks_path = args.chunks or config["data"]["chunks_metadata"]
        embeddings_path = args.embeddings or config["data"]["embeddings_metadata"]

    chunks = read_jsonl(chunks_path)
    embeddings = load_embeddings_by_chunk_id(embeddings_path)
    if not chunks:
        raise ValueError(f"Tidak ada chunk untuk ingestion: {chunks_path}")
    missing = [chunk["chunk_id"] for chunk in chunks if chunk["chunk_id"] not in embeddings]
    if missing:
        raise ValueError(f"Embedding tidak ditemukan untuk {len(missing)} chunk. Contoh: {missing[:5]}")

    persist_directory = resolve_repo_path(config["chroma"]["persist_directory"])
    persist_directory.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_directory))
    name = collection_name(config, pipeline)

    if args.reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=name,
        metadata={
            "experiment": config["experiment"]["name"],
            "pipeline": pipeline,
            "embedding_model": config["models"]["embedding"],
        },
    )

    if args.dry_run:
        print(f"[ingest_chroma] dry_run pipeline: {pipeline}")
        print(f"[ingest_chroma] dry_run collection: {name}")
        print(f"[ingest_chroma] dry_run chunks: {len(chunks)}")
        return

    ids = [chunk["chunk_id"] for chunk in chunks]
    documents = [chunk["chunk_text"] for chunk in chunks]
    vectors = [embeddings[chunk["chunk_id"]] for chunk in chunks]
    metadatas = [metadata_for_pipeline(chunk, pipeline) for chunk in chunks]

    collection.upsert(ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas)
    print(f"[ingest_chroma] pipeline: {pipeline}")
    print(f"[ingest_chroma] collection: {name}")
    print(f"[ingest_chroma] persist_directory: {persist_directory}")
    print(f"[ingest_chroma] chunks_ingested: {len(chunks)}")
    print(f"[ingest_chroma] collection_count: {collection.count()}")


if __name__ == "__main__":
    main()
