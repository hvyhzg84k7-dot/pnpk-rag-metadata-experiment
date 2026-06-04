from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment.yaml"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path konfigurasi eksperimen YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validasi argumen dan rencana eksekusi tanpa menulis artefak.",
    )


def add_question_filter_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--question-id-min", help="Mulai dari question_id ini, misalnya Q31.")
    parser.add_argument("--question-id-max", help="Berhenti sampai question_id ini, misalnya Q50.")
    parser.add_argument(
        "--question-ids",
        help="Daftar question_id spesifik dipisahkan koma, misalnya Q31,Q34,Q39.",
    )


def filter_by_question_id(
    rows: list[dict[str, Any]],
    *,
    question_id_min: str | None = None,
    question_id_max: str | None = None,
    question_ids: str | None = None,
) -> list[dict[str, Any]]:
    selected_ids = {
        item.strip()
        for item in (question_ids or "").split(",")
        if item.strip()
    }
    if not question_id_min and not question_id_max and not selected_ids:
        return rows
    return [
        row
        for row in rows
        if (not selected_ids or str(row.get("question_id", "")) in selected_ids)
        and (not question_id_min or str(row.get("question_id", "")) >= question_id_min)
        and (not question_id_max or str(row.get("question_id", "")) <= question_id_max)
    ]


def print_stage_banner(stage: str, config: str) -> None:
    print(f"[experiments] {stage}")
    print(f"[experiments] config: {config}")
    print("[experiments] status: scaffold siap; implementasi tahap ini belum dijalankan.")


def require_openai_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY belum tersedia di environment.")


def load_local_env(path: str | Path = DEFAULT_ENV_FILE) -> None:
    env_path = Path(path)
    if not env_path.is_absolute():
        env_path = REPO_ROOT / env_path
    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Konfigurasi tidak valid: {config_path}")
    return data


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    output_path = resolve_repo_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = resolve_repo_path(path)
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
