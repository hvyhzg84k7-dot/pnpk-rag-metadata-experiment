# Lampiran Artefak Eksperimen

Dokumen ini menjelaskan artefak publik yang digunakan sebagai lampiran penelitian. Artefak disusun agar perhitungan pada Bab 4 dapat diaudit tanpa menyebarkan teks dokumen sumber, jawaban acuan, konteks acuan, jawaban sistem, atau potongan konteks hasil retrieval.

## Lampiran 3.1 Daftar Pertanyaan Evaluasi

Daftar pertanyaan evaluasi tersedia pada:

```text
data/evaluation/evaluation_questions_public.csv
```

Berkas tersebut memuat 50 pertanyaan evaluasi dengan kolom:

```text
question_id, question, source_page, chapter, section, labels
```

Kolom `reference_answer` dan `reference_context` tidak disertakan pada artefak publik karena berisi jawaban acuan dan konteks rujukan yang diturunkan dari dokumen sumber. Dengan demikian, daftar pertanyaan tetap dapat diperiksa sebagai cakupan evaluasi tanpa mendistribusikan ulang isi dokumen PNPK.

## Lampiran 3.2 Rancangan Metadata Chunk dan Dataset Evaluasi

Metadata chunk yang digunakan pada pipeline metadata meliputi:

```text
source_document, page, chapter, section, labels
```

Struktur dataset evaluasi privat meliputi:

```text
question, reference_answer, reference_context, source_page, chapter, section, labels
```

Artefak publik hanya menampilkan bagian yang aman untuk audit, yaitu daftar pertanyaan, halaman sumber, bab, bagian, dan label. Jawaban acuan dan konteks acuan tidak dipublikasikan.

## Lampiran 4.1 Artefak Skor RAGAS dan Delta

Artefak RAGAS publik:

- `results/ragas/ragas_scores_public.csv`
- `results/ragas/ragas_delta_by_question.csv`
- `results/ragas/ragas_summary_final.csv`
- `results/calculation/README.md`

`ragas_scores_public.csv` memuat skor `context_relevance`, `faithfulness`, dan `answer_relevance` untuk setiap `question_id` dan pipeline.

`ragas_delta_by_question.csv` memuat perbandingan skor baseline dan metadata pada pertanyaan yang sama.

`ragas_summary_final.csv` memuat rata-rata, nilai minimum, nilai maksimum, dan ringkasan selisih metadata terhadap baseline.

Rumus perhitungan ringkas:

```text
mean(m, p) = sum(score_i for question i in pipeline p) / n
delta(m) = mean(m, metadata) - mean(m, baseline)
delta_i(m) = score_i(m, metadata) - score_i(m, baseline)
```

## Lampiran 4.2 Artefak Retrieval dan Keterlacakan Sumber

Artefak retrieval dan traceability publik:

- `results/retrieval/retrieval_contexts_public.csv`
- `results/traceability/traceability_scores_public.csv`
- `results/traceability/traceability_summary_public.csv`

`retrieval_contexts_public.csv` memuat `question_id`, pipeline, rank, `chunk_id`, skor retrieval, distance, page, chapter, section, labels, dan status filter metadata. Berkas ini tidak memuat `chunk_text`.

`traceability_scores_public.csv` memuat indikator biner:

```text
page_match, chapter_match, section_match, label_match
```

`traceability_summary_public.csv` memuat ringkasan rata-rata dan jumlah kecocokan untuk setiap indikator.

## Lampiran 4.3 Repositori Lampiran Publik

Repositori lampiran publik berisi:

- kode eksperimen;
- konfigurasi contoh;
- skrip pembentukan artefak publik;
- daftar pertanyaan evaluasi;
- skor numerik RAGAS;
- metadata retrieval tanpa teks konteks;
- ringkasan traceability.

Artefak privat yang tidak dimasukkan meliputi `.env`, API key, endpoint privat, PDF sumber, hasil ekstraksi teks, chunk, embedding, ChromaDB, jawaban acuan, konteks acuan, jawaban sistem, dan teks konteks hasil retrieval.

## Regenerasi Artefak Publik

Artefak publik dapat dibuat ulang dari repo privat dengan:

```bash
python scripts/build_public_artifacts.py \
  --evaluation-dataset ../experiments/data/evaluation/evaluation_dataset.csv \
  --ragas-scores ../experiments/results/ragas/ragas_scores_final.jsonl \
  --retrieval-outputs ../experiments/results/retrieval/retrieval_outputs.jsonl \
  --traceability-scores ../experiments/results/traceability/traceability_scores.csv \
  --output-dir .
```
