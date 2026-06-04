# PNPK RAG Metadata Experiment

Repositori ini berisi kode eksperimen dan artefak lampiran publik untuk penelitian evaluasi dua varian Retrieval-Augmented Generation (RAG):

- baseline tanpa metadata terstruktur;
- pipeline metadata untuk keterlacakan sumber.

Eksperimen ini mengevaluasi proses teknis RAG, retrieval, jawaban, traceability, dan RAGAS. Repositori ini bukan sistem diagnosis, bukan rekomendasi klinis, dan bukan pengganti keputusan tenaga medis.

## Isi Utama

```text
configs/experiment.example.yaml
data/evaluation/evaluation_questions_public.csv
results/calculation/README.md
results/ragas/ragas_scores_public.csv
results/ragas/ragas_delta_by_question.csv
results/ragas/ragas_summary_final.csv
results/retrieval/retrieval_contexts_public.csv
results/traceability/traceability_scores_public.csv
results/traceability/traceability_summary_public.csv
scripts/build_public_artifacts.py
src/
```

## Batas Data Publik

Yang disertakan hanya kode, template, daftar pertanyaan, skor numerik, metadata retrieval, dan ringkasan perhitungan.

Yang tidak disertakan:

- `.env`, API key, endpoint privat;
- PDF sumber;
- hasil ekstraksi teks;
- chunk, embedding, dan ChromaDB;
- jawaban acuan dan konteks acuan;
- jawaban sistem dan teks konteks hasil retrieval.

## Setup Singkat

```bash
uv sync
cp .env.example .env
cp configs/experiment.example.yaml configs/experiment.yaml
```

Isi `.env` dan `configs/experiment.yaml` dengan endpoint/model lokal masing-masing. Jangan commit `.env`.

## Artefak Lampiran

Penjelasan lampiran tersedia di:

```text
LAMPIRAN.md
```

Rumus dan proses pembentukan artefak publik tersedia di:

```text
results/calculation/README.md
```

Artefak publik dapat dibuat ulang dari repo privat dengan:

```bash
python scripts/build_public_artifacts.py \
  --evaluation-dataset ../experiments/data/evaluation/evaluation_dataset.csv \
  --ragas-scores ../experiments/results/ragas/ragas_scores_final.jsonl \
  --retrieval-outputs ../experiments/results/retrieval/retrieval_outputs.jsonl \
  --traceability-scores ../experiments/results/traceability/traceability_scores.csv \
  --output-dir .
```
