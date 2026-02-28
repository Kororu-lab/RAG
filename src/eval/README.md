# Eval Input Formats

This folder contains example inputs for large harness runs (for example, 50 x 3 query types).

## 1) Minimal input for ablation runner

Use `queries_only_example.jsonl` for smoke runs with no gold labels.

Required per line:
- `query_id` (recommended, string)
- one of: `query`, `question`, or `text`

Extra fields are allowed.

Example run:

```bash
uv run python src/eval/run_ablation.py \
  --queries src/eval/queries_only_example.jsonl \
  --profiles B0,B1,B2,B3,B4,B5,B6,B7
```

Default output directory is `eval/runs/<run_id>/` (override with `--outdir`).

## 2) Rich QA set for retrieval/eval metrics

Use `qa_set_example.json` or `qa_set_example.jsonl` for full metrics.

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
- profile ablation (`B0..B6`) and separate E2E track (`B7`).

Metrics are reported at `K={5,10,20,30}`:
- `chunk_recall_at_K`
- `doc_recall_at_K`
- `mrr_at_K`

Output layout (per run id):
- `B0..B6/`: retrieval traces + summaries
- `B7/`: E2E traces + summaries
- `retrieval_macro_micro.csv`: retrieval-only aggregates
- `e2e_macro_micro.csv`: E2E aggregates (separate table)
