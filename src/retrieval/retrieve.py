from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_common_arguments, load_config, load_local_env, resolve_repo_path
from openai_client import OpenAIClient

METADATA_FILTER_FIELDS = {"source_document", "page", "chapter", "section", "labels"}


def create_openai_client(config: dict[str, Any]) -> OpenAIClient:
    timeout = float(config.get("embedding", {}).get("timeout_seconds", 60))
    api_config = config.get("model_api", {})
    return OpenAIClient(
        api_key=api_config.get("api_key"),
        base_url=api_config.get("base_url"),
        provider=api_config.get("provider"),
        timeout=timeout,
    )


def collection_name(config: dict[str, Any], pipeline: str) -> str:
    if pipeline == "baseline":
        return config["chroma"]["baseline_collection"]
    return config["chroma"]["metadata_collection"]


def embed_query(client: OpenAIClient, config: dict[str, Any], question: str) -> list[float]:
    vectors = client.embed_texts(
        model=config["models"]["embedding"],
        texts=[question],
        task_type="RETRIEVAL_QUERY",
        max_retries=int(config.get("embedding", {}).get("max_retries", 2)),
    )
    return vectors[0]


def normalize_metadata(pipeline: str, metadata: dict[str, Any] | None) -> dict[str, Any]:
    if pipeline == "baseline":
        return {}
    metadata = metadata or {}
    return {
        "source_document": metadata.get("source_document", ""),
        "page": metadata.get("page", ""),
        "chapter": metadata.get("chapter", ""),
        "section": metadata.get("section", ""),
        "labels": metadata.get("labels", []),
    }


def normalize_metadata_filter(metadata_filter: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metadata_filter:
        return None

    scalar_conditions: dict[str, Any] = {}
    labels_condition: dict[str, Any] | None = None
    for key, value in metadata_filter.items():
        if key not in METADATA_FILTER_FIELDS:
            raise ValueError(f"Field metadata filter tidak didukung: {key}")
        if value is None or value == "":
            continue
        if key == "labels":
            values = value if isinstance(value, list) else [value]
            labels = [str(item).strip() for item in values if str(item).strip()]
            if not labels:
                continue
            label_conditions = [{"labels": {"$contains": label}} for label in labels]
            labels_condition = label_conditions[0] if len(label_conditions) == 1 else {"$or": label_conditions}
            continue
        if isinstance(value, list):
            values = [item for item in value if item not in (None, "")]
            if not values:
                continue
            scalar_conditions[key] = values[0] if len(values) == 1 else {"$in": values}
        else:
            scalar_conditions[key] = value

    if labels_condition and scalar_conditions:
        return {"$and": [labels_condition, *[{key: value} for key, value in scalar_conditions.items()]]}
    if labels_condition:
        return labels_condition
    return scalar_conditions or None


def metadata_filter_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    metadata_filter: dict[str, Any] = {}
    if args.metadata_filter_json:
        loaded = json.loads(args.metadata_filter_json)
        if not isinstance(loaded, dict):
            raise ValueError("--metadata-filter-json harus berupa objek JSON.")
        metadata_filter.update(loaded)
    if args.metadata_label:
        metadata_filter["labels"] = args.metadata_label
    if args.metadata_chapter:
        metadata_filter["chapter"] = args.metadata_chapter
    if args.metadata_section:
        metadata_filter["section"] = args.metadata_section
    if args.metadata_page is not None:
        metadata_filter["page"] = args.metadata_page
    return metadata_filter or None


def query_chroma(
    collection: Any,
    *,
    query_embedding: list[float],
    top_k: int,
    where: dict[str, Any] | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    return collection.query(**kwargs)


def retrieve_contexts(
    *,
    config: dict[str, Any],
    question: str,
    pipeline: str,
    top_k: int | None = None,
    client: OpenAIClient | None = None,
    metadata_filter: dict[str, Any] | None = None,
    query_info: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    top_k = top_k or int(config["retrieval"]["top_k"])
    client = client or create_openai_client(config)
    query_embedding = embed_query(client, config, question)

    chroma_client = chromadb.PersistentClient(path=str(resolve_repo_path(config["chroma"]["persist_directory"])))
    collection = chroma_client.get_collection(collection_name(config, pipeline))
    retrieval_config = config.get("retrieval", {})
    filter_config = retrieval_config.get("metadata_filter", {})
    filter_enabled = bool(filter_config.get("enabled", True))
    fallback_without_filter = bool(filter_config.get("fallback_without_filter", True))
    where = normalize_metadata_filter(metadata_filter) if pipeline == "metadata" and filter_enabled else None
    result = query_chroma(
        collection,
        query_embedding=query_embedding,
        top_k=top_k,
        where=where,
    )
    fallback_used = False
    if where and fallback_without_filter and not result.get("ids", [[]])[0]:
        result = query_chroma(collection, query_embedding=query_embedding, top_k=top_k, where=None)
        fallback_used = True

    if query_info is not None:
        query_info.clear()
        query_info.update(
            {
                "metadata_filter": where or {},
                "metadata_filter_used": bool(where),
                "metadata_filter_fallback_used": fallback_used,
            }
        )

    contexts: list[dict[str, Any]] = []
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for index, chunk_id in enumerate(ids):
        distance = float(distances[index])
        contexts.append(
            {
                "rank": index + 1,
                "chunk_id": chunk_id,
                "score": max(0.0, 1.0 - distance),
                "distance": distance,
                "chunk_text": documents[index],
                "metadata": normalize_metadata(pipeline, metadatas[index]),
            }
        )
    return contexts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Jalankan retrieval top-k untuk satu pertanyaan."
    )
    add_common_arguments(parser)
    parser.add_argument("--pipeline", choices=["baseline", "metadata"], required=False)
    parser.add_argument("--question", help="Pertanyaan pengguna.")
    parser.add_argument("--top-k", type=int, help="Jumlah konteks teratas.")
    parser.add_argument("--question-id", default="CLI", help="ID pertanyaan.")
    parser.add_argument("--metadata-label", action="append", help="Filter labels untuk Pipeline B. Bisa diulang.")
    parser.add_argument("--metadata-chapter", help="Filter chapter untuk Pipeline B.")
    parser.add_argument("--metadata-section", help="Filter section untuk Pipeline B.")
    parser.add_argument("--metadata-page", type=int, help="Filter page untuk Pipeline B.")
    parser.add_argument("--metadata-filter-json", help="Filter metadata ChromaDB dalam format JSON object.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.question:
        raise ValueError("--question wajib diisi.")
    load_local_env()
    config = load_config(args.config)
    pipeline = args.pipeline or "metadata"
    metadata_filter = metadata_filter_from_args(args)
    query_info: dict[str, Any] = {}
    contexts = retrieve_contexts(
        config=config,
        question=args.question,
        pipeline=pipeline,
        top_k=args.top_k,
        metadata_filter=metadata_filter,
        query_info=query_info,
    )
    output = {
        "question_id": args.question_id,
        "pipeline": pipeline,
        "question": args.question,
        "top_k": args.top_k or int(config["retrieval"]["top_k"]),
        **query_info,
        "contexts": contexts,
    }
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
