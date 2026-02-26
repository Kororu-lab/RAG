# Eval Input Formats

This folder contains example inputs for large harness runs (for example, 50 x 3 query types).

## 1) Minimal input for current ablation runner

Use `queries_only_example.jsonl` when running `scripts/run_ablation.py`.

Required per line:
- `query_id` (recommended, string)
- one of: `query`, `question`, or `text`

Extra fields are allowed but ignored by `run_ablation.py`.

Example run:

```bash
uv run python scripts/run_ablation.py \
  --queries src/eval/queries_only_example.jsonl \
  --profiles E0,E1,E2,E3,E4,E5
```

## 2) Rich QA set for retrieval/eval metrics

Use `qa_set_example.json` or `qa_set_example.jsonl` for harness evaluation.

Recommended fields:
- `query_id`
- `query_type`: `single_lang_single_topic`, `multi_lang_single_topic`, `single_lang_multi_topic`
- `query`
- `expected_languages`: list of allowed lang IDs
- `expected_topics`: free tags you define for analysis
- `gold_docs`: doc-level relevance (`source_file`)
- `gold_chunks`: chunk-level relevance (`source_file`, `chunk_id`)

The rich format supports:
- retrieval recall at doc/chunk level,
- per-type slicing (3 groups),
- language-detection and filter diagnostics analysis.

