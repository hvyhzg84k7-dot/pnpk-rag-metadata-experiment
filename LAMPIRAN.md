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

Ringkasan numerik utama yang diturunkan dari artefak publik adalah sebagai berikut:

| Item | Jumlah |
|------|-------:|
| jumlah pertanyaan evaluasi | 50 |
| jumlah keluaran RAGAS | 100 |
| jumlah baris `retrieval_contexts_public` | 488 |

Skor mentah RAGAS per pertanyaan disimpan dalam bentuk publik pada `results/ragas/ragas_scores_public.csv`. Kolom `context_relevance`, `faithfulness`, dan `answer_relevance` merupakan skor per `question_id` dan pipeline. Rata-rata skor untuk metrik $m$ pada pipeline $p$ dihitung sebagai berikut:

$$
\bar{s}(m,p) = \frac{\sum_{i=1}^{N} s(i,m,p)}{N}
$$

Keterangan:

| Simbol | Keterangan |
|--------|------------|
| $\bar{s}(m,p)$ | rata-rata skor metrik $m$ pada pipeline $p$ |
| $s(i,m,p)$ | skor pertanyaan ke-$i$ untuk metrik $m$ pada pipeline $p$ |
| $N$ | jumlah pertanyaan evaluasi, yaitu 50 |
| $m$ | `context_relevance`, `faithfulness`, atau `answer_relevance` |
| $p$ | `baseline` atau `metadata` |

Nilai delta pada Bab 4 dan pada `results/ragas/ragas_summary_final.csv` dihitung sebagai selisih rata-rata metadata terhadap baseline:

$$
\Delta(m) = \bar{s}(m, \text{metadata}) - \bar{s}(m, \text{baseline})
$$

Selain delta rata-rata, lampiran juga menyertakan perbandingan per pertanyaan pada `results/ragas/ragas_delta_by_question.csv`. Perhitungan per pertanyaan menggunakan rumus berikut:

$$
\Delta_i(m) = s(i,m,\text{metadata}) - s(i,m,\text{baseline})
$$

Kolom `comparison_*` ditentukan dari nilai $\Delta_i(m)$:

| Kondisi | Nilai `comparison_*` |
|---------|---------------------|
| $\Delta_i(m) > 0$ | `metadata_higher` |
| $\Delta_i(m) = 0$ | `equal` |
| $\Delta_i(m) < 0$ | `metadata_lower` |

Indikator keterlacakan sumber tidak dihitung oleh RAGAS, tetapi dihitung terpisah sesuai metode Bab 3. Setiap indikator bernilai 1 jika minimal satu dari lima konteks teratas memiliki metadata yang cocok dengan rujukan dataset, dan bernilai 0 jika tidak cocok atau metadata tidak tersedia.

$$
\mathrm{traceability\_mean}(\text{indicator}, p) =
\frac{\sum_{i=1}^{N} \text{match}(i, \text{indicator}, p)}{N}
$$

Keterangan:

| Simbol | Keterangan |
|--------|------------|
| `indicator` | `page_match`, `chapter_match`, `section_match`, atau `label_match` |
| `match` | 1 jika minimal satu konteks top-5 cocok, 0 jika tidak |
| $p$ | `baseline` atau `metadata` |
| $N$ | 50 |

Pada baseline, indikator keterlacakan bernilai 0 karena pipeline baseline tidak menyimpan metadata halaman, bab, bagian, dan label. Nilai 0 tersebut berarti field metadata tidak tersedia untuk dihitung, bukan berarti semua konteks baseline pasti tidak relevan secara semantik.

Ringkasan hasil terhitung dari artefak publik:

| Metrik | Baseline | Metadata | $\Delta$ |
|--------|---------:|---------:|---------:|
| `context_relevance` | 0.9200 | 0.9800 | 0.0600 |
| `faithfulness` | 0.8861 | 0.9662 | 0.0800 |
| `answer_relevance` | 0.8354 | 0.8553 | 0.0199 |

Distribusi per pertanyaan pada `results/ragas/ragas_delta_by_question.csv` juga sudah diringkas sebagai berikut:

| Metrik | `metadata_higher` | `equal` | `metadata_lower` |
|--------|------------------:|-------:|-----------------:|
| `context_relevance` | 5 | 45 | 0 |
| `faithfulness` | 11 | 38 | 1 |
| `answer_relevance` | 23 | 4 | 23 |

Ringkasan keterlacakan sumber pada lima konteks teratas:

| Indikator | Baseline | Metadata |
|-----------|---------:|---------:|
| `page_match` | 0/50 | 48/50 |
| `chapter_match` | 0/50 | 50/50 |
| `section_match` | 0/50 | 50/50 |
| `label_match` | 0/50 | 50/50 |

Pada metrik keterlacakan, `metadata_filter_used` muncul pada 50 dari 50 pertanyaan, sedangkan `metadata_filter_fallback_used` bernilai 0 dari 50 pertanyaan. Ini menunjukkan filter metadata aktif pada seluruh evaluasi dan tidak perlu fallback.

## Contoh Substitusi Angka ke Rumus

Bagian ini menunjukkan bagaimana angka pada CSV masuk ke rumus pada bagian sebelumnya.

### RAGAS

Untuk `context_relevance` pada baseline:

$$
\bar{s}(\mathrm{context\_relevance}, \text{baseline}) = \frac{46.00}{50} = 0.9200
$$

Untuk `context_relevance` pada metadata:

$$
\bar{s}(\mathrm{context\_relevance}, \text{metadata}) = \frac{49.00}{50} = 0.9800
$$

Sehingga delta metrik tersebut adalah:

$$
\Delta(\mathrm{context\_relevance}) = 0.9800 - 0.9200 = 0.0600
$$

Untuk `faithfulness`:

$$
\begin{aligned}
\bar{s}(\text{faithfulness}, \text{baseline}) &= \frac{44.3073870573}{50} = 0.8861477411 \\
\bar{s}(\text{faithfulness}, \text{metadata}) &= \frac{48.3095238094}{50} = 0.9661904762 \\
\Delta(\text{faithfulness}) &= 0.9661904762 - 0.8861477411 = 0.0800427351
\end{aligned}
$$

Untuk `answer_relevance`:

$$
\begin{aligned}
\bar{s}(\mathrm{answer\_relevance}, \text{baseline}) &= \frac{41.7709470526}{50} = 0.8354189411 \\
\bar{s}(\mathrm{answer\_relevance}, \text{metadata}) &= \frac{42.7672762990}{50} = 0.8553455260 \\
\Delta(\mathrm{answer\_relevance}) &= 0.8553455260 - 0.8354189411 = 0.0199265849
\end{aligned}
$$

### Per Pertanyaan

Untuk Q03 pada `answer_relevance`:

$$
\Delta_3(\mathrm{answer\_relevance}) = 0.7994447183 - 0.8947299409 = -0.0952852226
$$

Nilai negatif ini menyebabkan `comparison_answer_relevance` bernilai `metadata_lower`.

### Traceability

Untuk `page_match` pada metadata:

$$
\mathrm{traceability\_mean}(\mathrm{page\_match}, \text{metadata}) = \frac{48}{50} = 0.9600
$$

Untuk indikator lain:

$$
\begin{aligned}
\mathrm{traceability\_mean}(\mathrm{chapter\_match}, \text{metadata}) &= \frac{50}{50} = 1.0000 \\
\mathrm{traceability\_mean}(\mathrm{section\_match}, \text{metadata}) &= \frac{50}{50} = 1.0000 \\
\mathrm{traceability\_mean}(\mathrm{label\_match}, \text{metadata}) &= \frac{50}{50} = 1.0000
\end{aligned}
$$

Baseline tetap bernilai $0 / 50 = 0.0000$ pada keempat indikator karena field metadata tidak tersedia.

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

## Lampiran 4.3 Detail Kasus Representatif Evaluasi RAG

Bagian ini menampilkan pertanyaan, jawaban acuan, jawaban baseline, jawaban
metadata, skor RAGAS, sumber konteks, dan interpretasi untuk empat kasus
representatif yang dibahas pada Bab 4. Kasus ini digunakan untuk menjelaskan
perilaku teknis sistem RAG, bukan sebagai validasi klinis atau rekomendasi
medis.

### Q26 - Berhasil

**Pertanyaan:** Kapan anak dengan diabetes perlu melakukan pemeriksaan keton
mandiri?

**Jawaban acuan:** Pemeriksaan keton perlu tersedia dan dilakukan saat anak
sakit disertai demam atau muntah, saat glukosa darah di atas 250 mg/dL pada
anak yang tidak sehat atau meningkat persisten, saat poliuria persisten
disertai peningkatan glukosa darah terutama dengan nyeri abdomen atau napas
cepat, serta pada anak lebih muda atau pengguna pompa insulin bila diperlukan.

**Jawaban baseline:** Anak dengan diabetes perlu memeriksa keton mandiri saat
sakit disertai demam dan/atau muntah, saat glukosa darah >250 mg/dL pada anak
yang tidak sehat atau meningkat >250 mg/dL secara persisten, serta saat
poliuria persisten disertai peningkatan glukosa darah terutama bila ada nyeri
abdomen atau napas cepat. Pemeriksaan keton darah sebaiknya tersedia pada anak
yang lebih muda atau pengguna pompa insulin.

**Jawaban metadata:** Anak dengan diabetes perlu melakukan pemeriksaan keton
mandiri saat sakit yang disertai demam dan/atau muntah, saat glukosa darah di
atas 250 mg/dL pada anak yang tidak sehat atau meningkat >250 mg/dL secara
persisten, serta ketika ada poliuria persisten disertai peningkatan glukosa
darah terutama bila disertai nyeri abdomen atau napas cepat. Pemeriksaan keton
darah juga sebaiknya tersedia pada anak yang lebih muda atau yang menggunakan
pompa insulin.

| Pipeline | context relevance | faithfulness | answer relevance |
| --- | ---: | ---: | ---: |
| Baseline | 1.0000 | 1.0000 | 0.9581 |
| Metadata | 1.0000 | 1.0000 | 1.0000 |

**Sumber metadata:** halaman 50, BAB III, bagian C. Diabetes Melitus Tipe-1,
label `tata laksana saat sakit`, `pemeriksaan keton`, dan `komplikasi akut kad`.

**Interpretasi:** Kasus ini menunjukkan kondisi ketika retrieval, grounding
terhadap konteks, dan jawaban metadata berjalan selaras. Interpretasi ini hanya
digunakan untuk menilai perilaku teknis sistem RAG.

### Q22 - Campuran

**Pertanyaan:** Menjelang masa transisi layanan diabetes, apa saja yang perlu
disiapkan?

**Jawaban acuan:** PNPK menjelaskan masa transisi dapat dimulai pada usia 17
hingga 20 tahun. Persiapan transisi mencakup kebijakan klinik transisi,
pendidikan dan konseling tentang manajemen diri diabetes, kontrol dan
komplikasi, perbedaan sistem pediatrik dan dewasa, navigasi layanan kesehatan,
penilaian kesiapan, serta diskusi waktu transfer dengan pasien muda dan
keluarga.

**Jawaban baseline:** Menjelang masa transisi layanan diabetes, perlu
disiapkan persiapan transisi dan edukasi, termasuk konseling tentang manajemen
diri diabetes, kontrol dan komplikasi diabetes, perbedaan sistem pediatrik dan
dewasa, serta navigasi perawatan kesehatan. Selain itu, perlu penilaian
kesiapan, diskusi dengan pasien usia muda dan keluarga tentang waktu terbaik
untuk transfer, identifikasi layanan orang dewasa, dokumentasi ringkasan
diabetes klinis tertulis dan rencana transisi, serta komunikasi langsung dan
serah terima antar tim perawatan.

**Jawaban metadata:** Menjelang masa transisi layanan diabetes, yang perlu
disiapkan adalah persiapan transisi dan pendidikan, termasuk konseling tentang
manajemen diri diabetes, kontrol dan komplikasi diabetes, perbedaan antara
sistem pediatrik dan dewasa, serta navigasi perawatan kesehatan. Selain itu
perlu penilaian kesiapan oleh pediatrik, diskusi dengan pasien usia muda dan
keluarga tentang waktu terbaik untuk transfer, identifikasi layanan orang
dewasa yang mampu memenuhi kebutuhan, serta dokumentasi ringkasan diabetes
klinis tertulis dan rencana transisi.

| Pipeline | context relevance | faithfulness | answer relevance |
| --- | ---: | ---: | ---: |
| Baseline | 1.0000 | 1.0000 | 0.9561 |
| Metadata | 1.0000 | 1.0000 | 0.9561 |

**Sumber metadata:** halaman 109, BAB III, bagian E. Pengelolaan Pada Masa
Transisi, label `masa transisi`, `edukasi diabetes dm tipe-1`, dan `aspek
psikososial`.

**Interpretasi:** Kasus ini tidak menunjukkan kegagalan jawaban. Skor kedua
pipeline setara, sedangkan metadata tetap memberi nilai teknis karena sumber
konteks dapat ditelusuri sampai halaman, bagian, dan label.

### Q10 - Gagal pada Tabel dan Perbandingan

**Pertanyaan:** Apa perbedaan cara kerja beberapa jenis insulin yang disebut di
PNPK?

**Jawaban acuan:** PNPK membedakan insulin berdasarkan awitan, puncak kerja,
dan lama kerja. Contohnya, rapid-acting analog memiliki awitan sekitar 0,1
sampai 0,2 jam, puncak 1 sampai 3 jam, dan lama kerja 3 sampai 5 jam; regular
insulin memiliki awitan 0,5 sampai 1 jam, puncak 2 sampai 4 jam, dan lama kerja
5 sampai 8 jam; NPH memiliki awitan 2 sampai 4 jam, puncak 4 sampai 12 jam, dan
lama kerja 12 sampai 24 jam.

**Jawaban baseline:** Perbedaan utamanya ada pada awitan kerja, puncak kerja,
dan lama kerja. Insulin kerja cepat memiliki awitan paling cepat dan durasi
paling singkat, insulin regular lebih lambat dan lebih lama, NPH lebih lambat
lagi dengan puncak kerja yang lebih jelas, sedangkan insulin kerja panjang
seperti glargine, detemir, dan degludec bekerja lebih lama dengan profil lebih
basal dan glargine/detemir relatif tidak mempunyai kadar puncak.

**Jawaban metadata:** Beberapa jenis insulin berbeda pada awitan kerja, puncak
kerja, dan lama kerja. Insulin kerja cepat seperti aspart/glulisine/lispro mulai
0,1-0,3 jam, puncak 1-3 jam, lama 3-5 jam; regular/kerja pendek mulai 0,5-1
jam, puncak 2-4 jam, lama 5-8 jam; NPH kerja menengah mulai 2-4 jam, puncak
4-12 jam, lama 12-24 jam; sedangkan glargine, detemir, dan degludec sebagai
kerja panjang bekerja lebih lama dengan puncak minimal atau tidak ada.

| Pipeline | context relevance | faithfulness | answer relevance |
| --- | ---: | ---: | ---: |
| Baseline | 1.0000 | 0.6923 | 0.7224 |
| Metadata | 1.0000 | 0.9333 | 0.7096 |

**Sumber metadata:** halaman 23, BAB III, bagian C. Diabetes Melitus Tipe-1,
label `jenis insulin` dan `regimen insulin`.

**Interpretasi:** Kasus ini menunjukkan bahwa konteks tabel atau perbandingan
dapat berhasil diambil, tetapi jawaban akhir masih dapat menjadi terlalu umum
atau tidak sepenuhnya sesuai dengan bentuk jawaban acuan. Masalah utama dibaca
sebagai keterbatasan generasi jawaban, bukan kegagalan retrieval.

### Q27 - Gagal pada Intensi Pertanyaan

**Pertanyaan:** Kalau anak dengan diabetes sedang sakit, kenapa pengelolaannya
perlu perhatian khusus?

**Jawaban acuan:** Saat anak dengan diabetes sakit, perlu diwaspadai
hiperglikemia dengan ketoasidosis atau hipoglikemia. PNPK menjelaskan bahwa
hormon stres dapat meningkatkan glukoneogenesis dan resistensi insulin,
sedangkan sakit berat dapat meningkatkan produksi keton, menyebabkan asidosis,
mual, muntah, dehidrasi, dan gangguan keseimbangan asam basa.

**Jawaban baseline:** Karena saat sakit kebutuhan insulin dapat meningkat,
sementara produksi keton juga dapat bertambah sehingga memicu asidosis,
dehidrasi, ketoasidosis, koma, dan kematian. Pada anak yang muntah atau diare,
kadar glukosa darah juga bisa turun, sehingga pengelolaannya perlu lebih
cermat.

**Jawaban metadata:** Karena saat sakit kebutuhan insulin dan risiko gangguan
metabolik dapat berubah, sehingga insulin tidak boleh dihentikan dan dosisnya
mungkin perlu disesuaikan. Selain itu, perlu pemantauan glukosa darah dan keton
lebih sering karena muntah bisa menandakan kekurangan insulin dan berisiko
menuju dekompensasi metabolik.

| Pipeline | context relevance | faithfulness | answer relevance |
| --- | ---: | ---: | ---: |
| Baseline | 1.0000 | 1.0000 | 0.8427 |
| Metadata | 1.0000 | 1.0000 | 0.6100 |

**Sumber metadata:** halaman 52, BAB III, bagian C. Diabetes Melitus Tipe-1,
label `tata laksana saat sakit`, `pengelolaan dm tipe-1`, dan `edukasi diabetes
dm tipe-1`.

**Interpretasi:** Kasus ini menunjukkan bahwa jawaban dapat tetap didukung
konteks, tetapi bergeser dari pertanyaan sebab-akibat menuju langkah
pengelolaan. Karena itu, faithfulness tinggi tidak otomatis berarti jawaban
paling sesuai dengan intensi pertanyaan.

## Lampiran 4.4 Repositori Lampiran Publik

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
