from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_common_arguments, load_config, load_local_env, read_jsonl, resolve_repo_path, write_jsonl
from openai_client import OpenAIClient


def token_encoder(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception as exc:
            raise RuntimeError(
                "Tokenizer cl100k_base belum tersedia. Jalankan `make exp-tokenizer-cache` "
                "atau pastikan TIKTOKEN_CACHE_DIR mengarah ke cache tiktoken yang valid."
            ) from exc


def chunk_text(text: str, *, chunk_size: int, overlap: int, encoder) -> list[tuple[str, int]]:
    tokens = encoder.encode(text)
    if not tokens:
        return []
    if len(tokens) <= chunk_size:
        return [(text, len(tokens))]

    chunks: list[tuple[str, int]] = []
    step = max(chunk_size - overlap, 1)
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk = encoder.decode(chunk_tokens).strip()
        if chunk:
            chunks.append((chunk, len(chunk_tokens)))
        if end == len(tokens):
            break
        start += step
    return chunks


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def choose_primary_label(row: dict, chunk: str) -> str:
    text = chunk.lower()
    section = row.get("section", "").lower()
    page = int(row.get("page", 0))

    if page >= 116:
        return "rekomendasi dm tipe-2"
    if 111 <= page <= 115:
        if contains_any(text, ["ketoasidosis", "kad", "asidosis", "rehidrasi", "edema serebri"]):
            return "rekomendasi kad"
        if "hipoglikemia" in text:
            return "rekomendasi hipoglikemia"
        if contains_any(text, ["nefropati", "retinopati", "neuropati", "dislipidemia", "tiroid", "celiac"]):
            return "rekomendasi skrining komplikasi dm tipe-1"
        return "rekomendasi dm tipe-1"

    if "masa transisi" in section or "transisi" in section:
        return "masa transisi"
    if "tenaga medis" in section or "tenaga kesehatan" in section:
        return "peran tim kesehatan diabetes anak"

    if "diabetes melitus tipe-2" in section or "dm tipe-2" in text or "tipe-2" in text or "tipe 2" in text:
        if contains_any(text, ["pendekatan diagnosis", "diagnosis dm tipe-2", "menentukan tipe dm"]):
            return "diagnosis dm tipe-2"
        if "skrining diabetes melitus tipe-2" in text or "skrining dm tipe-2" in text:
            return "skrining dm tipe-2"
        if "edukasi" in text:
            return "edukasi dm tipe-2"
        if contains_any(text, ["gaya hidup", "modifikasi gaya hidup", "menurunkan berat badan", "aktivitas fisik rutin"]):
            return "gaya hidup dm tipe-2"
        if contains_any(text, ["pengaturan makan", "asupan", "diet", "makanan", "karbohidrat", "kalori"]):
            return "pengaturan makan dm tipe-2"
        if "aktivitas fisik" in text:
            return "aktivitas fisik dm tipe-2"
        if contains_any(text, ["terapi medikamentosa", "metformin", "glp-1", "sulfoniluria"]):
            return "terapi medikamentosa dm tipe-2"
        if contains_any(text, ["monitoring diabetes melitus tipe-2", "monitor glukosa", "hba1c", "pemeriksaan postprandial"]):
            return "monitoring dm tipe-2"
        if contains_any(text, ["komplikasi", "komorbiditas", "dislipidemia", "osa", "spok", "nafld", "nefropati", "retinopati", "hipertensi", "aterosklerosis"]):
            return "komplikasi dm tipe-2"
        if page == 96:
            return "diagnosis dm tipe-2"
        if page == 97:
            return "edukasi dm tipe-2"
        if 98 <= page <= 100:
            return "gaya hidup dm tipe-2"
        if 100 <= page <= 102:
            return "terapi medikamentosa dm tipe-2"
        if 103 <= page <= 104:
            return "monitoring dm tipe-2"
        if 104 <= page <= 108:
            return "komplikasi dm tipe-2"
        return "gaya hidup dm tipe-2"

    if contains_any(text, ["definisi dan klasifikasi", "diabetes adalah kelainan metabolik", "dibagi menjadi 3 tipe"]):
        return "definisi diabetes"
    if "karakteristik dm tipe-1" in text or "diabetes monogenik" in text or "stadium pada dm" in text:
        return "klasifikasi diabetes"
    if "c-peptida" in text or "c-peptide" in text:
        return "pemeriksaan c-peptida"
    if contains_any(text, ["penggunaan klinis pemeriksaan insulin", "pengukuran kadar insulin", "pemeriksaan insulin"]):
        return "pemeriksaan insulin"

    if "diagnosis" in section and "diabetes melitus tipe-1" not in section and "diabetes melitus tipe-2" not in section:
        return "diagnosis umum diabetes"

    if contains_any(text, ["pendekatan diagnosis diabetes melitus tipe-1", "gambaran klinis", "karakteristik klinis saat diagnosis", "keterlambatan diagnosis"]):
        return "diagnosis dm tipe-1"
    if contains_any(text, ["penyakit tiroid", "penyakit celiac", "penyakit addison"]):
        return "skrining komorbiditas dm tipe-1"
    if contains_any(text, ["rekomendasi untuk skrining", "penapisan komplikasi", "nefropati", "retinopati", "neuropati", "mikroalbuminuria"]):
        return "skrining komplikasi dm tipe-1"
    if "pengelolaan diabetes melitus tipe-1" in text or "sasaran dan tujuan khusus pengelolaan" in text:
        return "pengelolaan dm tipe-1"
    if contains_any(text, ["kerja insulin", "insulin merupakan elemen utama", "terapi insulin pertama kali"]):
        return "terapi insulin"
    if contains_any(text, ["insulin kerja cepat", "ultra rapid acting", "insulin kerja pendek", "insulin kerja menengah", "insulin kerja panjang", "insulin basal", "insulin kerja campuran", "profil farmakokinetik"]):
        return "jenis insulin"
    if contains_any(text, ["penyimpanan insulin", "simpanlah jarum", "masa kadaluarsa", "suhu ruangan"]):
        return "penyimpanan insulin"
    if contains_any(text, ["regimen insulin", "regimen basal bolus", "pompa insulin", "regimen dua kali", "regimen tiga"]):
        return "regimen insulin"
    if contains_any(text, ["penyesuaian dosis", "rasio insulin-karbohidrat", "koreksi hiperglikemia", "disesuaikan", "bulan puasa"]):
        return "penyesuaian insulin"
    if contains_any(text, ["dosis insulin", "total dosis harian", "kebutuhan insulin"]):
        return "dosis insulin"
    if contains_any(text, ["penyerapan insulin", "tempat suntikan", "penyuntikan", "rotasi penyuntikan", "reaksi lokal", "lipohipertrofi"]):
        return "teknik penyuntikan insulin"
    if contains_any(text, ["pengaturan makan", "rekomendasi diet", "karbohidrat", "indeks glikemik", "asupan energi"]):
        return "pengaturan makan dm tipe-1"
    if contains_any(text, ["aktivitas fisik dan olahraga", "olahraga", "latihan fisik"]):
        return "aktivitas fisik dm tipe-1"
    if contains_any(text, ["hba1c", "target hba1c", "kadar hba1c"]):
        return "pemantauan hba1c"
    if contains_any(text, ["glukosa darah mandiri", "pemantauan glukosa", "monitoring kadar glukosa", "kadar glukosa darah"]):
        return "pemantauan glukosa"
    if contains_any(text, ["keton", "keton darah", "keton urin", "ketonuria", "ketonemia", "pemeriksaan keton"]):
        return "pemeriksaan keton"
    if contains_any(text, ["keadaan sakit", "saat sakit", "sedang sakit", "manajemen saat sakit"]):
        return "tata laksana saat sakit"
    if contains_any(text, ["operasi", "anestesi", "tindakan mayor", "tindakan minor"]):
        return "pengelolaan operasi"
    if contains_any(text, ["ramadhan", "puasa", "iftar", "sahur"]):
        return "puasa ramadhan"
    if contains_any(text, ["edukasi", "perkemahan diabetes", "penyuluh diabetes"]):
        return "edukasi diabetes dm tipe-1"
    if contains_any(text, ["sekolah", "guru", "teman sebaya", "absensi sekolah"]):
        return "manajemen di sekolah"
    if contains_any(text, ["perjalanan", "bepergian", "travelling", "perjalanan jauh"]):
        return "manajemen saat perjalanan"
    if contains_any(text, ["aspek psikososial", "psikososial", "depresi", "kecemasan", "kesehatan mental"]):
        return "aspek psikososial"
    if "hipoglikemia" in text and 85 <= page <= 90:
        return "hipoglikemia"
    if contains_any(text, ["ketoasidosis", "kad", "edema serebri", "asidosis", "rehidrasi"]):
        return "komplikasi akut kad"
    if "hipoglikemia" in text:
        return "hipoglikemia"
    if contains_any(text, ["komplikasi kronik", "mikrovaskular", "makrovaskular", "nefropati", "retinopati", "neuropati", "dislipidemia", "hipertensi"]):
        return "komplikasi kronik dm tipe-1"
    if contains_any(text, ["gangguan tiroid", "penyakit addison", "penyakit celiac"]):
        return "penyakit penyerta dm tipe-1"
    if contains_any(text, ["pertumbuhan dan diabetes", "gangguan pertumbuhan", "kurva pertumbuhan", "pubertas terlambat"]):
        return "pertumbuhan dm tipe-1"

    if "diabetes melitus tipe-1" in section or "tipe-1" in section or "tipe 1" in section:
        return "pengelolaan dm tipe-1"
    return "diagnosis umum diabetes"


def append_label(labels: list[str], label: str, max_labels: int) -> None:
    if label not in labels and len(labels) < max_labels:
        labels.append(label)


LABEL_ALIASES = {
    "monitoring dm tipe-1": "pemantauan glukosa",
    "monitoring diabetes melitus tipe-1": "pemantauan glukosa",
    "pemantauan dm tipe-1": "pemantauan glukosa",
    "monitoring diabetes": "pemantauan glukosa",
    "monitoring hba1c": "pemantauan hba1c",
    "diagnosis diabetes": "diagnosis umum diabetes",
    "dm tipe-1": "diagnosis dm tipe-1",
    "dm tipe-2": "diagnosis dm tipe-2",
}


def normalize_label(label: str) -> str:
    normalized = " ".join(label.strip().lower().split())
    return LABEL_ALIASES.get(normalized, normalized)


def choose_labels(row: dict, chunk: str, *, max_labels: int = 3) -> list[str]:
    text = chunk.lower()
    labels = [choose_primary_label(row, chunk)]

    if contains_any(text, ["diagnosis", "kriteria diagnosis", "gambaran klinis"]):
        append_label(labels, "diagnosis umum diabetes", max_labels)
    if "c-peptida" in text or "c-peptide" in text:
        append_label(labels, "pemeriksaan c-peptida", max_labels)
    if contains_any(text, ["pemeriksaan insulin", "pengukuran kadar insulin", "insulin perifer"]):
        append_label(labels, "pemeriksaan insulin", max_labels)
    if contains_any(text, ["hba1c", "target hba1c", "kadar hba1c"]):
        append_label(labels, "pemantauan hba1c", max_labels)
    if contains_any(text, ["glukosa darah mandiri", "pemantauan glukosa", "monitoring kadar glukosa"]):
        append_label(labels, "pemantauan glukosa", max_labels)
    if contains_any(text, ["ketoasidosis", "kad", "asidosis", "keton"]):
        append_label(labels, "komplikasi akut kad", max_labels)
    if contains_any(text, ["keton darah", "keton urin", "ketonuria", "ketonemia", "pemeriksaan keton"]):
        append_label(labels, "pemeriksaan keton", max_labels)
    if "hipoglikemia" in text:
        append_label(labels, "hipoglikemia", max_labels)
    if contains_any(text, ["edukasi", "penyuluh diabetes"]) and ("tipe-2" not in text and "diabetes melitus tipe-2" not in row.get("section", "").lower()):
        append_label(labels, "edukasi diabetes dm tipe-1", max_labels)
    if "edukasi" in text and ("tipe-2" in text or "diabetes melitus tipe-2" in row.get("section", "").lower()):
        append_label(labels, "edukasi dm tipe-2", max_labels)
    if contains_any(text, ["gaya hidup", "modifikasi gaya hidup", "menurunkan berat badan"]):
        append_label(labels, "gaya hidup dm tipe-2", max_labels)
    if contains_any(text, ["metformin", "glp-1", "terapi medikamentosa"]):
        append_label(labels, "terapi medikamentosa dm tipe-2", max_labels)
    if contains_any(text, ["keadaan sakit", "saat sakit", "sedang sakit", "manajemen saat sakit"]):
        append_label(labels, "tata laksana saat sakit", max_labels)

    return labels[:max_labels]


def chunk_cache_key(row: dict[str, Any], chunk: str) -> str:
    payload = {
        "page": row.get("page"),
        "chapter": row.get("chapter", ""),
        "section": row.get("section", ""),
        "chunk_text": chunk,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def load_label_cache(path: str | None, model: str) -> dict[str, list[str]]:
    if not path:
        return {}
    cache_path = resolve_repo_path(path)
    if not cache_path.exists():
        return {}
    cache: dict[str, list[str]] = {}
    for row in read_jsonl(cache_path):
        key = row.get("cache_key")
        labels = row.get("labels")
        if row.get("label_model") == model and isinstance(key, str) and isinstance(labels, list):
            cache[key] = [str(label).strip().lower() for label in labels if str(label).strip()]
    return cache


def write_label_cache(path: str | None, cache: dict[str, list[str]], model: str) -> None:
    if not path:
        return
    rows = [
        {"cache_key": key, "label_model": model, "labels": labels}
        for key, labels in sorted(cache.items())
    ]
    write_jsonl(path, rows)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(stripped[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Respons model tidak berisi objek JSON valid.")


class AiLabelAnnotator:
    def __init__(self, config: dict[str, Any]) -> None:
        label_config = config.get("metadata", {}).get("label_annotation", {})
        self.model = label_config.get("model") or config.get("models", {}).get("generation")
        self.temperature = float(label_config.get("temperature", 0))
        self.max_retries = int(label_config.get("max_retries", 2))
        self.timeout = float(label_config.get("timeout_seconds", 30))
        self.hard_timeout = float(label_config.get("hard_timeout_seconds", 0))
        self.concurrency = int(label_config.get("concurrency", 4))
        self.max_output_tokens = int(label_config.get("max_output_tokens", 256))
        self.max_labels = int(label_config.get("max_labels_per_chunk", 3))
        self.labels = list(config["metadata"]["label_vocabulary"])
        self.valid_labels = set(self.labels)

        api_config = config.get("model_api", {})
        self.client = OpenAIClient(
            api_key=api_config.get("api_key"),
            base_url=api_config.get("base_url"),
            provider=api_config.get("provider"),
            timeout=self.timeout,
            hard_timeout=self.hard_timeout,
        )

    def annotate(self, row: dict[str, Any], chunk: str) -> list[str]:
        label_list = json.dumps(self.labels, ensure_ascii=False)
        messages = [
            {
                "role": "system",
                "content": (
                    "Anda memberi label semantik untuk satu chunk PNPK. "
                    "Tugas Anda adalah memilih 1 sampai 3 label PALING RELEVAN dari daftar label yang diberikan.\n\n"
                    "Aturan wajib:\n"
                    "1. Pilih label berdasarkan TOPIK UTAMA chunk, bukan sekadar kata yang disebut sekali.\n"
                    "2. Utamakan label yang paling spesifik. Jika ada label spesifik yang cocok, jangan memilih label yang lebih umum sebagai penggantinya.\n"
                    "3. Tambahkan label kedua atau ketiga hanya jika topik itu benar-benar dibahas secara substansial dalam chunk.\n"
                    "4. Jangan memilih label hanya karena istilah seperti insulin, HbA1c, keton, edukasi, atau diagnosis muncul sekilas.\n"
                    "5. Jika chunk bercampur karena batas chunk, pilih topik yang paling dominan berdasarkan jumlah isi yang dibahas.\n"
                    "6. Gunakan HANYA label dari daftar. Jangan membuat label baru, jangan mengubah ejaan label.\n"
                    "7. Urutkan label dari yang paling dominan ke yang paling sekunder.\n\n"
                    "Panduan disambiguasi penting:\n"
                    "- Bedakan `pemeriksaan insulin` dan `pemeriksaan c-peptida`. Jangan menyatukannya kecuali keduanya memang dibahas substansial dan perlu masuk dua label terpisah.\n"
                    "- Gunakan `komplikasi akut kad` hanya jika fokus chunk membahas KAD, ketosis, asidosis, rehidrasi, atau tata laksana akut terkait.\n"
                    "- Bedakan `pemantauan glukosa` dan `pemantauan hba1c`. Jika chunk membahas keduanya secara nyata, keduanya boleh muncul sebagai dua label terpisah.\n"
                    "- Bedakan `edukasi dm tipe-2`, `gaya hidup dm tipe-2`, `pengaturan makan dm tipe-2`, dan `aktivitas fisik dm tipe-2`. Pilih yang paling langsung mewakili isi chunk.\n"
                    "- Bedakan `tata laksana saat sakit` dari `pemeriksaan keton`. Penyebutan keton tidak otomatis berarti tata laksana saat sakit, dan sebaliknya.\n"
                    "- Untuk topik DM tipe-2, pilih label DM tipe-2 yang paling spesifik daripada label umum diabetes.\n\n"
                    "Jawab satu baris JSON valid, tanpa markdown, tanpa penjelasan tambahan. "
                    'Format wajib: {"labels":["..."]}.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Daftar labels yang boleh dipilih: {label_list}\n\n"
                    f"Halaman: {row.get('page', '')}\n"
                    f"Bab: {row.get('chapter', '')}\n"
                    f"Bagian: {row.get('section', '')}\n\n"
                    "Instruksi pemilihan:\n"
                    "- Gunakan informasi `Bagian` sebagai petunjuk, tetapi isi chunk tetap menjadi dasar utama.\n"
                    "- Jika isi chunk terutama berupa daftar faktor, kriteria, komplikasi, atau rekomendasi, pilih label yang langsung mewakili daftar itu.\n"
                    "- Jika chunk hanya menyentuh topik kedua secara singkat, jangan masukkan sebagai label tambahan.\n"
                    f"- Maksimal label: {self.max_labels}\n\n"
                    f"Chunk:\n{chunk[:1800]}"
                ),
            },
        ]

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                parsed = self.client.generate_json(
                    model=self.model,
                    system_instruction=messages[0]["content"],
                    prompt=messages[1]["content"],
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                    max_retries=0,
                )
                raw_labels = parsed.get("labels", [])
                if not raw_labels and parsed.get("label"):
                    raw_labels = [parsed["label"]]
                if isinstance(raw_labels, str):
                    raw_labels = [raw_labels]
                if not isinstance(raw_labels, list):
                    raise ValueError("Field labels dari model bukan list.")
                labels: list[str] = []
                invalid_labels: list[str] = []
                for raw_label in raw_labels:
                    label = normalize_label(str(raw_label))
                    if not label:
                        continue
                    if label not in self.valid_labels:
                        invalid_labels.append(label)
                        continue
                    append_label(labels, label, self.max_labels)
                if labels:
                    return labels
                raise ValueError(f"Model tidak mengembalikan labels valid. Invalid labels: {invalid_labels!r}")
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Anotasi labels AI gagal: {last_error}") from last_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bangun chunk baseline dan chunk metadata dari korpus klinis PNPK."
    )
    add_common_arguments(parser)
    parser.add_argument("--input", help="Path pnpk_clinical_corpus.jsonl.")
    parser.add_argument("--baseline-output", help="Path chunks_baseline.jsonl.")
    parser.add_argument("--metadata-output", help="Path chunks_metadata.jsonl.")
    parser.add_argument(
        "--label-mode",
        choices=["ai", "heuristic"],
        help="Mode anotasi labels. Default mengikuti konfigurasi metadata.label_annotation.method.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    load_local_env()
    config = load_config(args.config)
    input_path = args.input or config["data"]["clinical_corpus"]
    baseline_output = args.baseline_output or config["data"]["chunks_baseline"]
    metadata_output = args.metadata_output or config["data"]["chunks_metadata"]

    chunk_size = int(config["chunking"]["chunk_size_tokens"])
    overlap = int(config["chunking"]["chunk_overlap_tokens"])
    model = config["models"]["embedding"]
    source_document = config["data"]["source_document_name"]
    encoder = token_encoder(model)
    label_config = config.get("metadata", {}).get("label_annotation", {})
    label_mode = args.label_mode or label_config.get("method", "heuristic")
    label_model = label_config.get("model") or config.get("models", {}).get("generation", "")
    fallback_mode = label_config.get("fallback", "heuristic")
    max_labels = int(label_config.get("max_labels_per_chunk", 3))
    label_cache_path = label_config.get("cache")
    label_cache = load_label_cache(label_cache_path, label_model) if label_mode == "ai" else {}
    label_annotator = AiLabelAnnotator(config) if label_mode == "ai" and not args.dry_run else None

    baseline_rows = []
    metadata_rows = []
    source_rows = read_jsonl(input_path)
    expected_chunks = sum(
        len(
            chunk_text(
                row["clean_text"],
                chunk_size=chunk_size,
                overlap=overlap,
                encoder=encoder,
            )
        )
        for row in source_rows
    )
    chunk_number = 1
    ai_label_count = 0
    cached_label_count = 0
    fallback_label_count = 0
    pending_label_jobs: list[dict[str, Any]] = []

    for row in source_rows:
        page_chunks = chunk_text(
            row["clean_text"],
            chunk_size=chunk_size,
            overlap=overlap,
            encoder=encoder,
        )
        for page_chunk_index, (text, token_count) in enumerate(page_chunks, start=1):
            chunk_suffix = f"{chunk_number:04d}"
            labels = choose_labels(row, text, max_labels=max_labels)
            pending_cache_key = None
            if label_mode == "ai":
                cache_key = chunk_cache_key(row, text)
                if cache_key in label_cache:
                    labels = label_cache[cache_key]
                    cached_label_count += 1
                else:
                    pending_cache_key = cache_key
            baseline_rows.append(
                {
                    "chunk_id": f"A-{chunk_suffix}",
                    "chunk_text": text,
                    "token_count": token_count,
                }
            )
            metadata_rows.append(
                {
                    "chunk_id": f"B-{chunk_suffix}",
                    "source_chunk_id": f"C-{chunk_suffix}",
                    "chunk_text": text,
                    "source_document": source_document,
                    "page": row["page"],
                    "page_chunk_index": page_chunk_index,
                    "chapter": row.get("chapter", ""),
                    "section": row.get("section", ""),
                    "labels": labels,
                    "token_count": token_count,
                }
            )
            if pending_cache_key is not None:
                pending_label_jobs.append(
                    {
                        "metadata_row_index": len(metadata_rows) - 1,
                        "cache_key": pending_cache_key,
                        "row": row,
                        "text": text,
                    }
                )
            chunk_number += 1

    if args.dry_run:
        print(f"[build_chunks] dry_run chunks: {len(metadata_rows)}")
        print(f"[build_chunks] dry_run label_mode: {label_mode}")
        if label_mode == "ai":
            print(f"[build_chunks] dry_run cached_labels: {cached_label_count}")
            print(f"[build_chunks] dry_run pending_ai_labels: {len(pending_label_jobs)}")
        return

    if label_annotator is not None and pending_label_jobs:
        total_pending = len(pending_label_jobs)
        max_workers = max(1, label_annotator.concurrency)
        print(
            f"[build_chunks] annotate_labels_ai_pending: {total_pending} chunks, concurrency={max_workers}",
            flush=True,
        )
        if max_workers == 1:
            for completed, job in enumerate(pending_label_jobs, start=1):
                try:
                    labels = label_annotator.annotate(job["row"], job["text"])
                    metadata_rows[job["metadata_row_index"]]["labels"] = labels
                    label_cache[job["cache_key"]] = labels
                    ai_label_count += 1
                except Exception:
                    if fallback_mode != "heuristic":
                        raise
                    fallback_label_count += 1

                if completed == 1 or completed % 10 == 0 or completed == total_pending:
                    print(f"[build_chunks] annotate_labels_ai_done: {completed}/{total_pending}", flush=True)
                if completed % 10 == 0:
                    write_label_cache(label_cache_path, label_cache, label_model)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(label_annotator.annotate, job["row"], job["text"]): job
                    for job in pending_label_jobs
                }
                for completed, future in enumerate(as_completed(futures), start=1):
                    job = futures[future]
                    try:
                        labels = future.result()
                        metadata_rows[job["metadata_row_index"]]["labels"] = labels
                        label_cache[job["cache_key"]] = labels
                        ai_label_count += 1
                    except Exception:
                        if fallback_mode != "heuristic":
                            raise
                        fallback_label_count += 1

                    if completed == 1 or completed % 10 == 0 or completed == total_pending:
                        print(f"[build_chunks] annotate_labels_ai_done: {completed}/{total_pending}", flush=True)
                    if completed % 10 == 0:
                        write_label_cache(label_cache_path, label_cache, label_model)

    baseline_count = write_jsonl(baseline_output, baseline_rows)
    metadata_count = write_jsonl(metadata_output, metadata_rows)
    if label_mode == "ai":
        write_label_cache(label_cache_path, label_cache, label_model)
    print(f"[build_chunks] input_pages: {len(source_rows)}")
    print(f"[build_chunks] chunk_size_tokens: {chunk_size}")
    print(f"[build_chunks] chunk_overlap_tokens: {overlap}")
    print(f"[build_chunks] label_mode: {label_mode}")
    if label_mode == "ai":
        print(f"[build_chunks] label_model: {label_model}")
        print(f"[build_chunks] label_ai_calls: {ai_label_count}")
        print(f"[build_chunks] label_cache_hits: {cached_label_count}")
        print(f"[build_chunks] label_heuristic_fallbacks: {fallback_label_count}")
    print(f"[build_chunks] baseline_chunks: {baseline_count} -> {resolve_repo_path(baseline_output)}")
    print(f"[build_chunks] metadata_chunks: {metadata_count} -> {resolve_repo_path(metadata_output)}")


if __name__ == "__main__":
    main()
