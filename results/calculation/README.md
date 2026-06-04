# Calculation Artifacts

This folder documents how public-safe appendix data is calculated without redistributing source-document text.

## Metrics

For each metric \(m\) and pipeline \(p\), the thesis uses the arithmetic mean:

```text
mean(m, p) = sum(score_i for question i in pipeline p) / n
```

The difference between the metadata pipeline and the baseline pipeline is:

```text
delta(m) = mean(m, metadata) - mean(m, baseline)
```

For row-level comparison, the same subtraction is applied per `question_id`:

```text
delta_i(m) = score_i(m, metadata) - score_i(m, baseline)
```

Traceability indicators are binary per `question_id` and pipeline:

```text
indicator_mean = matched_count / n
```

## Included Safe Artifacts

- `results/ragas/ragas_scores_public.csv`: row-level RAGAS scores without questions, answers, references, retrieved contexts, or raw RAGAS payloads.
- `data/evaluation/evaluation_questions_public.csv`: evaluation question list with source page, section, and labels, without reference answers or reference contexts.
- `results/ragas/ragas_delta_by_question.csv`: per-question baseline score, metadata score, and delta for each RAGAS metric.
- `results/ragas/ragas_summary_final.csv`: aggregate RAGAS mean, min, max, and metadata-minus-baseline comparison.
- `results/retrieval/retrieval_contexts_public.csv`: retrieved rank, score, distance, chunk ID, and metadata fields without `chunk_text`.
- `results/traceability/traceability_scores_public.csv`: row-level binary traceability indicators without notes or source text.
- `results/traceability/traceability_summary_public.csv`: aggregate traceability means and matched counts.

## Excluded Private Columns

The public artifacts intentionally exclude:

- generated answers;
- reference answers;
- reference contexts;
- supporting quotes;
- retrieved context text;
- raw RAGAS payloads;
- source-document chunks;
- embeddings;
- API endpoint configuration.

## Regeneration

From the private thesis repository, regenerate the safe artifacts with:

```bash
python scripts/build_public_artifacts.py \
  --evaluation-dataset ../experiments/data/evaluation/evaluation_dataset.csv \
  --ragas-scores ../experiments/results/ragas/ragas_scores_final.jsonl \
  --retrieval-outputs ../experiments/results/retrieval/retrieval_outputs.jsonl \
  --traceability-scores ../experiments/results/traceability/traceability_scores.csv \
  --output-dir .
```
