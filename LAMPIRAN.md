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

## Dasar Perhitungan Skor

Perhitungan pada lampiran ini mengikuti metode analisis yang dijelaskan pada Bab 3 dan digunakan untuk menyusun tabel hasil pada Bab 4. Setiap pertanyaan evaluasi dijalankan pada dua pipeline, yaitu baseline dan metadata. Dengan 50 pertanyaan evaluasi, terdapat 50 skor untuk setiap metrik pada masing-masing pipeline.

Skor mentah RAGAS per pertanyaan disimpan dalam bentuk publik pada `results/ragas/ragas_scores_public.csv`. Kolom `context_relevance`, `faithfulness`, dan `answer_relevance` merupakan skor per `question_id` dan pipeline. Rata-rata skor untuk metrik `m` pada pipeline `p` dihitung sebagai berikut:

```text
s_bar(m, p) = sum(s(i, m, p) for i = 1..N) / N
```

Keterangan:

```text
s_bar(m, p)  = rata-rata skor metrik m pada pipeline p
s(i, m, p)  = skor pertanyaan ke-i untuk metrik m pada pipeline p
N           = jumlah pertanyaan evaluasi, yaitu 50
m           = context_relevance, faithfulness, atau answer_relevance
p           = baseline atau metadata
```

Nilai delta pada Bab 4 dan pada `results/ragas/ragas_summary_final.csv` dihitung sebagai selisih rata-rata metadata terhadap baseline:

```text
delta(m) = s_bar(m, metadata) - s_bar(m, baseline)
```

Selain delta rata-rata, lampiran juga menyertakan perbandingan per pertanyaan pada `results/ragas/ragas_delta_by_question.csv`. Perhitungan per pertanyaan menggunakan rumus berikut:

```text
delta_i(m) = s(i, m, metadata) - s(i, m, baseline)
```

Kolom `comparison_*` ditentukan dari nilai `delta_i(m)`:

```text
metadata_higher = delta_i(m) > 0
equal           = delta_i(m) = 0
metadata_lower  = delta_i(m) < 0
```

Indikator keterlacakan sumber tidak dihitung oleh RAGAS, tetapi dihitung terpisah sesuai metode Bab 3. Setiap indikator bernilai 1 jika minimal satu dari lima konteks teratas memiliki metadata yang cocok dengan rujukan dataset, dan bernilai 0 jika tidak cocok atau metadata tidak tersedia.

```text
traceability_mean(indicator, p) =
  sum(match(i, indicator, p) for i = 1..N) / N
```

Keterangan:

```text
indicator = page_match, chapter_match, section_match, atau label_match
match     = 1 jika minimal satu konteks top-5 cocok, 0 jika tidak
p         = baseline atau metadata
N         = 50
```

Pada baseline, indikator keterlacakan bernilai 0 karena pipeline baseline tidak menyimpan metadata halaman, bab, bagian, dan label. Nilai 0 tersebut berarti field metadata tidak tersedia untuk dihitung, bukan berarti semua konteks baseline pasti tidak relevan secara semantik.

## Lampiran 4.1 Artefak Skor RAGAS dan Delta

Artefak RAGAS publik:

- `results/ragas/ragas_scores_public.csv`
- `results/ragas/ragas_delta_by_question.csv`
- `results/ragas/ragas_summary_final.csv`
- `results/calculation/README.md`

`ragas_scores_public.csv` memuat skor `context_relevance`, `faithfulness`, dan `answer_relevance` untuk setiap `question_id` dan pipeline.

`ragas_delta_by_question.csv` memuat perbandingan skor baseline dan metadata pada pertanyaan yang sama.

`ragas_summary_final.csv` memuat rata-rata, nilai minimum, nilai maksimum, dan ringkasan selisih metadata terhadap baseline.

Nilai pada ketiga artefak tersebut dihitung menggunakan rumus pada bagian "Dasar Perhitungan Skor". Dengan demikian, angka yang muncul pada Bab 4 dapat ditelusuri dari skor per pertanyaan, rata-rata per pipeline, delta rata-rata, dan delta per pertanyaan.

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

Rata-rata pada `traceability_summary_public.csv` dihitung sebagai proporsi jumlah pertanyaan yang memiliki kecocokan metadata pada lima konteks teratas. Perhitungan ini mengikuti indikator keterlacakan sumber yang dijelaskan pada bagian "Dasar Perhitungan Skor".

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
