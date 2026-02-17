"""Retriever-core ablation harness.

Runs RAGRetriever.retrieve_documents() under controlled condition profiles,
writing audit.jsonl + metrics.json for reproducibility. No UI/graph imports.

Usage:
    python -m src.eval.run_retrieval_ablation --queries queries.jsonl
    python -m src.eval.run_retrieval_ablation --queries queries.jsonl --conditions baseline,hybrid,viking_soft
    python -m src.eval.run_retrieval_ablation --queries queries.jsonl --out-dir results/ablation_001

Query file format (JSONL):
    {"query": "bashkir negation", "metadata": {"lang": "bashkir"}}
    {"query": "vowel harmony in turkic languages"}
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Ensure project root is on path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import src.utils

# ---------------------------------------------------------------------------
# Condition profiles — each overrides specific config keys.
# "baseline" is always vector-only, no reranker, no viking, no recursive.
# ---------------------------------------------------------------------------
CONDITION_PROFILES: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "retrieval.hybrid_search.enabled": False,
        "reranker.enabled": False,
        "retrieval.recursive_retrieval": False,
        "retrieval.viking.enabled": False,
        "retrieval.vision_search": False,
    },
    "hybrid": {
        "retrieval.hybrid_search.enabled": True,
        "reranker.enabled": False,
        "retrieval.recursive_retrieval": False,
        "retrieval.viking.enabled": False,
        "retrieval.vision_search": False,
    },
    "hybrid_rerank": {
        "retrieval.hybrid_search.enabled": True,
        "reranker.enabled": True,
        "retrieval.recursive_retrieval": False,
        "retrieval.viking.enabled": False,
        "retrieval.vision_search": False,
    },
    "viking_soft": {
        "retrieval.hybrid_search.enabled": True,
        "reranker.enabled": True,
        "retrieval.recursive_retrieval": False,
        "retrieval.viking.enabled": True,
        "retrieval.viking.mode": "soft",
    },
    "viking_strict": {
        "retrieval.hybrid_search.enabled": True,
        "reranker.enabled": True,
        "retrieval.recursive_retrieval": False,
        "retrieval.viking.enabled": True,
        "retrieval.viking.mode": "strict",
    },
    "raptor_on": {
        "retrieval.hybrid_search.enabled": True,
        "reranker.enabled": True,
        "retrieval.recursive_retrieval": True,
        "retrieval.viking.enabled": False,
    },
    "full": {
        "retrieval.hybrid_search.enabled": True,
        "reranker.enabled": True,
        "retrieval.recursive_retrieval": True,
        "retrieval.viking.enabled": True,
        "retrieval.viking.mode": "soft",
    },
}


def _apply_overrides(config: dict, overrides: Dict[str, Any]) -> dict:
    """Apply dotted-key overrides to a nested config dict."""
    config = deepcopy(config)
    for dotted_key, value in overrides.items():
        keys = dotted_key.split(".")
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return config


def _sha256_short(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_project_root, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def load_queries(path: str) -> List[Dict[str, Any]]:
    """Load queries from a JSONL file."""
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            queries.append(json.loads(line))
    return queries


def run_condition(
    condition_name: str,
    overrides: Dict[str, Any],
    queries: List[Dict[str, Any]],
    base_config: dict,
    audit_fp,
) -> Dict[str, Any]:
    """Run all queries under a single condition. Returns condition-level metrics."""
    patched_config = _apply_overrides(base_config, overrides)

    # Import here to keep this harness retrieval-core only.
    from src.retrieve.rag_retrieve import RAGRetriever

    condition_metrics = {
        "condition": condition_name,
        "overrides": overrides,
        "queries": [],
    }

    with src.utils.use_config_override(patched_config):
        with RAGRetriever() as retriever:
            for qi, q_entry in enumerate(queries):
                query_text = q_entry["query"]
                metadata = q_entry.get("metadata") or {}
                query_hash = _sha256_short(query_text)

                print(f"\n{'='*60}")
                print(f"[{condition_name}] Query {qi+1}/{len(queries)}: {query_text}")
                print(f"{'='*60}")

                t0 = time.monotonic()
                try:
                    docs = retriever.retrieve_documents(query_text, metadata=metadata)
                except Exception as e:
                    print(f"ERROR: {e}")
                    docs = []
                elapsed = time.monotonic() - t0

                # Build audit record
                doc_records = []
                for d in docs:
                    doc_records.append({
                        "source_file": d.metadata.get("source_file"),
                        "ref_id": d.metadata.get("ref_id"),
                        "level": d.metadata.get("level"),
                        "score": d.metadata.get("score"),
                        "chunk_id": d.metadata.get("chunk_id"),
                        "lang": d.metadata.get("lang"),
                        "_viking_scope": d.metadata.get("_viking_scope"),
                    })

                audit_entry = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "condition": condition_name,
                    "query": query_text,
                    "query_hash": query_hash,
                    "metadata": metadata,
                    "n_docs": len(docs),
                    "elapsed_sec": round(elapsed, 3),
                    "docs": doc_records,
                }
                audit_fp.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")
                audit_fp.flush()

                # Condition-level metrics placeholder
                condition_metrics["queries"].append({
                    "query": query_text,
                    "query_hash": query_hash,
                    "n_docs": len(docs),
                    "elapsed_sec": round(elapsed, 3),
                    "source_files": [d["source_file"] for d in doc_records],
                })

    return condition_metrics


def main():
    parser = argparse.ArgumentParser(description="Retriever ablation harness")
    parser.add_argument("--queries", required=True, help="Path to queries JSONL file")
    parser.add_argument(
        "--conditions",
        default=None,
        help="Comma-separated condition names (default: all). "
             f"Available: {','.join(CONDITION_PROFILES.keys())}",
    )
    parser.add_argument("--out-dir", default=None, help="Output directory (default: auto-timestamped)")
    args = parser.parse_args()

    # Resolve conditions
    if args.conditions:
        condition_names = [c.strip() for c in args.conditions.split(",")]
        for c in condition_names:
            if c not in CONDITION_PROFILES:
                parser.error(f"Unknown condition '{c}'. Available: {list(CONDITION_PROFILES.keys())}")
    else:
        condition_names = list(CONDITION_PROFILES.keys())

    # Load queries
    queries = load_queries(args.queries)
    if not queries:
        parser.error(f"No queries found in {args.queries}")
    print(f"Loaded {len(queries)} queries from {args.queries}")

    # Output directory
    if args.out_dir:
        out_dir = args.out_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(_project_root, "results", f"ablation_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # Reproducibility metadata
    git_hash = _get_git_hash()
    config_hash = _sha256_short(json.dumps(src.utils.load_config(), sort_keys=True, default=str))
    run_meta = {
        "start_time": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash,
        "config_hash": config_hash,
        "query_file": os.path.abspath(args.queries),
        "query_count": len(queries),
        "conditions": condition_names,
    }

    # Save base config for this run
    base_config = src.utils.load_config()

    audit_path = os.path.join(out_dir, "audit.jsonl")
    metrics_path = os.path.join(out_dir, "metrics.json")

    all_metrics = {"run": run_meta, "conditions": []}

    with open(audit_path, "w", encoding="utf-8") as audit_fp:
        for ci, cond_name in enumerate(condition_names):
            print(f"\n{'#'*70}")
            print(f"# Condition {ci+1}/{len(condition_names)}: {cond_name}")
            print(f"{'#'*70}")

            overrides = CONDITION_PROFILES[cond_name]
            cond_metrics = run_condition(
                cond_name, overrides, queries, base_config, audit_fp
            )
            all_metrics["conditions"].append(cond_metrics)

    all_metrics["run"]["end_time"] = datetime.now(timezone.utc).isoformat()

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    print(f"\nAblation complete.")
    print(f"  Audit log:  {audit_path}")
    print(f"  Metrics:    {metrics_path}")
    print(f"  Conditions: {len(condition_names)}, Queries: {len(queries)}")


if __name__ == "__main__":
    main()
