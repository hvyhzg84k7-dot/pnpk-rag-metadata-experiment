from __future__ import annotations

import argparse
import json
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
from openai_client import OpenAIClient


def create_generation_client(config: dict[str, Any]) -> OpenAIClient:
    api_config = config.get("model_api", {})
    generation_config = config.get("generation", {})
    return OpenAIClient(
        api_key=api_config.get("api_key"),
        base_url=api_config.get("base_url"),
        provider=api_config.get("provider"),
        timeout=float(generation_config.get("timeout_seconds", 120)),
        hard_timeout=float(generation_config.get("hard_timeout_seconds", 90)),
    )


def format_contexts(contexts: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for context in contexts:
        metadata = context.get("metadata", {}) or {}
        labels = metadata.get("labels", [])
        source = (
            f"rank={context.get('rank')} chunk_id={context.get('chunk_id')} "
            f"page={metadata.get('page', '')} chapter={metadata.get('chapter', '')} "
            f"section={metadata.get('section', '')} labels={labels}"
        )
        blocks.append(f"[{source}]\n{context.get('chunk_text', '')}")
    return "\n\n".join(blocks)


def build_prompt(row: dict[str, Any], config: dict[str, Any]) -> tuple[str, str]:
    generation_config = config["generation"]
    unsupported = generation_config.get(
        "unsupported_answer",
        "Informasi tidak tersedia secara memadai dalam konteks yang diberikan.",
    )
    quote_limit = int(generation_config.get("supporting_quote_limit", 3))
    system_instruction = (
        "Anda adalah asisten evaluasi RAG untuk dokumen PNPK.\n\n"
        "TUGAS:\n"
        "- Jawab pertanyaan HANYA dari konteks yang diberikan.\n"
        "- Susun jawaban yang ringkas tetapi cukup lengkap untuk menjawab inti pertanyaan.\n"
        f'- Jika konteks tidak cukup, jawab persis: "{unsupported}"\n\n'
        "ATURAN:\n"
        "1. Jawaban harus fokus pada inti pertanyaan.\n"
        "2. Boleh menggabungkan beberapa poin hanya jika memang diperlukan untuk menjawab pertanyaan secara utuh.\n"
        "3. Jangan menambahkan informasi yang tidak didukung konteks.\n"
        "4. Jika ada beberapa detail relevan, pilih poin yang paling sentral dan paling langsung menjawab pertanyaan.\n"
        "5. Hindari detail sampingan yang tidak dibutuhkan untuk jawaban inti, walaupun detail itu masih ada di konteks.\n"
        "6. Gunakan istilah medis yang sama seperti di konteks.\n"
        "7. Jangan memberi diagnosis atau rekomendasi klinis final.\n"
        "8. Kembalikan HANYA JSON valid.\n"
        "9. Field \"answer\" sebaiknya 1 sampai 3 kalimat yang padat.\n"
        f"10. supporting_quotes maksimal {quote_limit} kutipan pendek yang benar-benar mendukung jawaban.\n"
        "11. generation_notes harus singkat, satu kalimat saja."
    )
    prompt = f"""
Pertanyaan:
{row["question"]}

Konteks PNPK:
{format_contexts(row.get("contexts", []))}

Kerjakan seperti ini:
1. Cari kalimat konteks yang paling langsung menjawab pertanyaan.
2. Tentukan 2 sampai 4 poin inti saja jika pertanyaan membutuhkan beberapa hal.
3. Tulis jawaban langsung, ringkas, dan cukup lengkap terhadap pertanyaan.
4. Jika pertanyaan meminta definisi, alasan, tujuan, fokus, atau perbedaan, jawab sesuai bentuk yang diminta pertanyaan.
5. Jika pertanyaan meminta hal yang perlu diperhatikan, utamakan poin utama dan jangan memasukkan rincian tambahan yang bukan inti.
6. Jika pertanyaan menanyakan fokus edukasi, sebutkan fokus utama terlebih dahulu, lalu pendukung yang benar-benar disebutkan konteks.
7. Jangan menambahkan konteks lain yang tidak membantu menjawab inti pertanyaan.
8. Jika tidak ada jawaban langsung di konteks, gunakan fallback persis.

FORMAT OUTPUT (JSON):
{{
  "answer": "<jawaban ringkas, langsung, dan cukup lengkap>",
  "supporting_quotes": ["<kutipan 1>", "<kutipan 2>"],
  "generation_notes": "<contoh: didukung rank=1 chunk_id=B-0007>"
}}
""".strip()
    return system_instruction, prompt


def normalize_supporting_quotes(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    quotes: list[str] = []
    for item in value:
        quote = str(item).strip()
        if quote:
            quotes.append(quote)
        if len(quotes) >= limit:
            break
    return quotes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Buat jawaban berbasis konteks PNPK dari output retrieval."
    )
    add_common_arguments(parser)
    parser.add_argument("--input", help="Path retrieval output JSONL.")
    parser.add_argument("--output", help="Path answers.jsonl.")
    parser.add_argument("--limit", type=int, help="Batasi jumlah baris retrieval untuk smoke test.")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Tampilkan token respons model saat generation untuk debugging endpoint.",
    )
    add_question_filter_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    load_local_env()
    config = load_config(args.config)
    input_path = args.input or config["results"]["retrieval_outputs"]
    output_path = args.output or config["results"]["answers"]
    rows = read_jsonl(input_path)
    rows = filter_by_question_id(
        rows,
        question_id_min=args.question_id_min,
        question_id_max=args.question_id_max,
        question_ids=args.question_ids,
    )
    if args.limit is not None:
        rows = rows[: args.limit]

    if args.dry_run:
        print(f"[answer] dry_run input: {resolve_repo_path(input_path)}")
        print(f"[answer] dry_run rows: {len(rows)}")
        print(f"[answer] dry_run output: {resolve_repo_path(output_path)}")
        return

    client = create_generation_client(config)
    quote_limit = int(config.get("generation", {}).get("supporting_quote_limit", 3))
    sleep_seconds_between_rows = float(config.get("generation", {}).get("sleep_seconds_between_rows", 0))
    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        system_instruction, prompt = build_prompt(row, config)
        error = ""
        answer = ""
        supporting_quotes: list[str] = []
        generation_notes = ""
        try:
            if args.stream:
                print(
                    "\n[answer:stream] "
                    f"{index}/{len(rows)} {row.get('question_id')} {row.get('pipeline')}\n",
                    file=sys.stderr,
                    flush=True,
                )
            response = client.generate_json(
                model=config["models"]["generation"],
                system_instruction=system_instruction,
                prompt=prompt,
                temperature=float(config.get("generation", {}).get("temperature", 0)),
                max_output_tokens=int(config.get("generation", {}).get("max_output_tokens", 512)),
                max_retries=int(config.get("generation", {}).get("max_retries", 1)),
                think=config.get("generation", {}).get("think"),
                stream=args.stream,
                stream_callback=(
                    lambda text: print(text, end="", file=sys.stderr, flush=True)
                    if args.stream
                    else None
                ),
            )
            if args.stream:
                print("\n[answer:stream:end]\n", file=sys.stderr, flush=True)
            answer = str(response.get("answer", "")).strip()
            supporting_quotes = normalize_supporting_quotes(
                response.get("supporting_quotes"),
                limit=quote_limit,
            )
            generation_notes = str(response.get("generation_notes", "")).strip()
            if not answer:
                raise RuntimeError(f"Respons tidak berisi answer: {json.dumps(response, ensure_ascii=False)}")
        except Exception as exc:
            error = repr(exc)
            answer = config["generation"].get(
                "unsupported_answer",
                "Informasi tidak tersedia secara memadai dalam konteks yang diberikan.",
            )
            generation_notes = "generation_failed"

        output_rows.append(
            {
                "question_id": row.get("question_id", ""),
                "pipeline": row.get("pipeline", ""),
                "question": row.get("question", ""),
                "answer": answer,
                "supporting_quotes": supporting_quotes,
                "reference_answer": row.get("reference_answer", ""),
                "reference_context": row.get("reference_context", ""),
                "source_page": row.get("source_page", ""),
                "chapter": row.get("chapter", ""),
                "section": row.get("section", ""),
                "labels": row.get("labels", []),
                "top_k": row.get("top_k", ""),
                "metadata_filter": row.get("metadata_filter", {}),
                "metadata_filter_used": row.get("metadata_filter_used", False),
                "metadata_filter_fallback_used": row.get("metadata_filter_fallback_used", False),
                "contexts": row.get("contexts", []),
                "generation_notes": generation_notes,
                "error": error,
            }
        )
        print(
            "[answer] "
            f"{index}/{len(rows)} {row.get('question_id')} {row.get('pipeline')} "
            f"answer_chars={len(answer)} error={bool(error)}"
        )
        if sleep_seconds_between_rows > 0 and index < len(rows):
            print(
                f"[answer] cooldown after {row.get('question_id')} {row.get('pipeline')} "
                f"sleep={sleep_seconds_between_rows:.1f}s"
            )
            time.sleep(sleep_seconds_between_rows)

    count = write_jsonl(output_path, output_rows)
    print(f"[answer] output_rows: {count}")
    print(f"[answer] output: {resolve_repo_path(output_path)}")


if __name__ == "__main__":
    main()
