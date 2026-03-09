#!/usr/bin/env bash
set -euo pipefail

RUN_ID="final_clean_b0_b7_final"
RUN_DIR="eval/runs/${RUN_ID}"
EMBED_MODEL_PATH="${EMBED_MODEL_PATH:-$HOME/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181}"
RERANK_MODEL_PATH="${RERANK_MODEL_PATH:-$HOME/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e}"

if [[ ! -d "$EMBED_MODEL_PATH" ]]; then
  echo "Missing local embedding model snapshot: $EMBED_MODEL_PATH" >&2
  exit 1
fi
if [[ ! -d "$RERANK_MODEL_PATH" ]]; then
  echo "Missing local reranker snapshot: $RERANK_MODEL_PATH" >&2
  exit 1
fi

mkdir -p eval/finalize eval/runs
rm -rf "$RUN_DIR"
rm -f \
  eval/finalize/main_table_metrics.csv \
  eval/finalize/groupwise_metrics.csv \
  eval/finalize/profile_name_map.json \
  eval/finalize/run_manifest.json

UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/eval/test_ablation_profiles_b0_b1.py \
  tests/eval/test_metadata_store.py \
  tests/eval/test_split_materialization.py \
  tests/eval/test_b7_split_refinements.py \
  tests/retrieve/test_split_budget.py \
  tests/ui/test_runtime_config.py

UV_CACHE_DIR=/tmp/uv-cache uv run python -m py_compile \
  src/ui/graph_web.py \
  src/agent/service.py \
  src/agent/graph.py \
  src/agent/nodes.py \
  src/eval/run_ablation.py \
  src/eval/materialized_language_split.py \
  src/eval/metadata_store.py \
  src/eval/profiles.py \
  src/retrieve/rag_retrieve.py \
  src/retrieve/split_budget.py \
  src/ui/runtime_config.py

RUN_ID="${RUN_ID}" \
EMBED_MODEL_PATH="${EMBED_MODEL_PATH}" \
RERANK_MODEL_PATH="${RERANK_MODEL_PATH}" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
UV_CACHE_DIR=/tmp/uv-cache uv run python - <<'PY'
from copy import deepcopy
from pathlib import Path
import os
import traceback

from src.eval.profiles import deep_merge, load_profile_file, resolve_profile_config, select_profiles
from src.eval.run_ablation import _load_qaset, _utc_now, _write_csv, _write_json, run_profile
from src.utils import load_config

run_id = os.environ["RUN_ID"]
query_file = "test_revised.jsonl"
profile_config = "config/ablation_profiles.yaml"
selected_profile_arg = "B0,B1,B2,B3,B4,B6,B7"
out_dir = Path("eval/runs") / run_id
out_dir.mkdir(parents=True, exist_ok=True)
run_meta_path = out_dir / "run_meta.json"

base_config = load_config()
base_config.setdefault("embedding", {})
base_config.setdefault("reranker", {})
base_config["embedding"]["model_name"] = os.environ["EMBED_MODEL_PATH"]
base_config["reranker"]["model_name"] = os.environ["RERANK_MODEL_PATH"]

defaults, profiles_map = load_profile_file(Path(profile_config))
selected_profiles = select_profiles(profiles_map, selected_profile_arg)
queries = _load_qaset(Path(query_file))

run_meta = {
    "run_id": run_id,
    "started_at": _utc_now(),
    "query_file": query_file,
    "query_count": len(queries),
    "profile_config": profile_config,
    "profiles_planned": selected_profiles,
    "profiles_results": [],
    "embedding_model_path": os.environ["EMBED_MODEL_PATH"],
    "reranker_model_path": os.environ["RERANK_MODEL_PATH"],
}
_write_json(run_meta_path, run_meta)

retrieval_macro_rows = []

for profile_name in selected_profiles:
    profile_overrides = deepcopy(profiles_map.get(profile_name) or {})
    description = str(profile_overrides.pop("description", ""))
    ablation_block = (profiles_map.get(profile_name, {}) or {}).get("ablation", {}) or {}
    base_profile_name = str(ablation_block.get("base_profile") or "").strip()
    if base_profile_name and base_profile_name in profiles_map:
        base_profile_overrides = deepcopy(profiles_map.get(base_profile_name) or {})
        base_profile_overrides.pop("description", None)
        profile_overrides = deep_merge(base_profile_overrides, profile_overrides)
        description = f"{description} (base={base_profile_name})".strip()

    profile_cfg = resolve_profile_config(
        base_config=base_config,
        defaults=defaults,
        profile_name=profile_name,
        profile_overrides=profile_overrides,
        fixed_k=None,
        fixed_top_n=None,
    )

    status_row = {
        "profile": profile_name,
        "effective_profile": profile_name,
        "description": description,
        "db_name": str((profile_cfg.get("database", {}) or {}).get("dbname", "")),
        "db_table": str((profile_cfg.get("database", {}) or {}).get("table_name", "")),
        "status": "running",
        "started_at": _utc_now(),
    }
    run_meta["profiles_results"].append(status_row)
    _write_json(run_meta_path, run_meta)

    try:
        result = run_profile(
            run_id=run_id,
            profile_name=profile_name,
            profile_cfg=profile_cfg,
            queries=queries,
            out_dir=out_dir,
        )
        status_row.update(
            {
                "status": "completed",
                "completed_at": _utc_now(),
                "records": result.get("records", 0),
                "track": result.get("track", "retrieval"),
                "backend_mode_effective": result.get("backend_mode_effective", "unknown"),
                "backend_mode_config": result.get("backend_mode_config", "unknown"),
                "backend_mode_required": result.get("backend_mode_required", "any"),
                "config_hash": result.get("config_hash", ""),
                "query_count_used": len(queries),
            }
        )
        retrieval_macro_rows.extend(result.get("macro_micro_rows", []))
    except Exception as exc:
        status_row.update(
            {
                "status": "failed",
                "completed_at": _utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    _write_json(run_meta_path, run_meta)

_write_csv(out_dir / "retrieval_macro_micro.csv", retrieval_macro_rows)
_write_csv(out_dir / "e2e_macro_micro.csv", [])
completed = sum(1 for row in run_meta["profiles_results"] if row.get("status") == "completed")
failed = sum(1 for row in run_meta["profiles_results"] if row.get("status") == "failed")
run_meta["completed_at"] = _utc_now()
run_meta["status"] = "completed" if failed == 0 else "completed_with_issues"
run_meta["counts"] = {"planned": len(selected_profiles), "completed": completed, "failed": failed}
_write_json(run_meta_path, run_meta)
print(f"[FinalHarness] Done. Output: {out_dir}")
PY

UV_CACHE_DIR=/tmp/uv-cache uv run python src/eval/build_finalize_summary.py \
  --run-dir "${RUN_DIR}" \
  --query-file test_revised.jsonl \
  --out-dir eval/finalize \
  --repo-root . \
  --metadata-path metadata.enriched.csv
