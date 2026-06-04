# Publication Checklist

Before publishing this folder as a GitHub repository, verify the following:

- `configs/experiment.yaml` is not committed.
- `.env` is not committed.
- `data/raw/` is not committed.
- `data/processed/` is not committed.
- `chroma/` is not committed.
- generated answer files and retrieval output files are not committed.
- RAGAS score JSONL files with row-level answers or contexts are not committed.
- no private IP address, local filesystem path, API key, or internal endpoint appears in tracked files.
- included result files contain only numeric scores, question IDs, labels, or aggregate metadata, not source-document text.
- `LAMPIRAN.md` may include the four representative Bab 4 cases only; do not add the full answer dataset or full retrieved contexts.
- public calculation artifacts are generated from `scripts/build_public_artifacts.py`, not by manually copying private JSONL files.

Suggested verification commands:

```bash
find . -maxdepth 4 -type f | sort
rg -n "10\\.0\\.0\\.|/Users/|OPENAI_API_KEY=.+|BEGIN .*KEY|api_key: [^$]" .
rg -n "reference_context|chunk_text|contexts" data results
```

The last command may find CSV headers or metadata-only files. Do not publish files where those fields contain source-document text.
