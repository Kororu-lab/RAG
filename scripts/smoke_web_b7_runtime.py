#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.utils
from src.agent.service import run_query_stream
from src.ui.runtime_config import apply_runtime_overrides


def _pick_t2_query(path: Path) -> tuple[str, str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id", "")).strip()
            query_type = str(row.get("query_type", "")).strip()
            if query_id.startswith("T2_") or query_type == "multi_lang_single_topic":
                return query_id or "T2_smoke", str(row["query"])
    raise RuntimeError("No T2 query found in test_revised.jsonl")


def main() -> None:
    query_id, query_text = _pick_t2_query(Path("test_revised.jsonl"))
    base_config = src.utils.load_config()
    selected_model = str((base_config.get("llm_retrieval") or {}).get("model_name", "gpt-oss:20b"))
    runtime_cfg = apply_runtime_overrides(
        base_config,
        selected_model=selected_model,
        language_detection_enabled=True,
        language_filter_enabled=True,
        viking_use_detected_language_enabled=True,
        bm25_enabled=False,
        reranker_enabled=True,
        recursive_enabled=False,
        vision_enabled=False,
        sibling_expansion_enabled=False,
        query_split_enabled=True,
        query_split_max_languages=2,
        query_split_branch_k_mode="full_per_branch",
        viking_enabled=False,
        viking_soft=True,
        viking_max_exp=2,
        retrieval_llm_timeout_sec=int((base_config.get("llm_retrieval") or {}).get("request_timeout_sec", 60)),
        generation_timeout_sec=int((base_config.get("rag") or {}).get("generation_timeout_sec", 600)),
        grading_enabled=False,
        hallucination_check_enabled=False,
    )

    retrieve_payload = None
    with src.utils.use_config_override(runtime_cfg):
        for event in run_query_stream(query_text, search_count=0):
            if "retrieve" in event:
                retrieve_payload = event["retrieve"]
                break

    if not isinstance(retrieve_payload, dict):
        raise RuntimeError("retrieve payload missing from web runtime smoke")

    diagnostics = retrieve_payload.get("retrieval_diagnostics", {}) or {}
    branch_rows = diagnostics.get("split_branch_diagnostics", []) or []
    no_filter_count = sum(1 for row in branch_rows if row.get("filter_stage_used") == "no_filter")
    local_reranker_all = all(bool(row.get("local_reranker_active", False)) for row in branch_rows) if branch_rows else False
    branch_queries = diagnostics.get("split_branch_queries", []) or []

    summary = {
        "query_id": query_id,
        "query": query_text,
        "b7_controller_mode": diagnostics.get("split_controller_mode"),
        "split_applied": bool(diagnostics.get("split_applied", False)),
        "split_dimension": diagnostics.get("split_dimension"),
        "split_reason": diagnostics.get("split_reason"),
        "branch_count": len(branch_queries),
        "branch_queries": branch_queries,
        "accepted_branch_no_filter_count": no_filter_count,
        "local_reranker_active_all": local_reranker_all,
        "post_merge_rerank_mode": diagnostics.get("split_post_merge_rerank_mode"),
        "post_merge_rerank_applied": bool(diagnostics.get("split_global_rerank_applied", False)),
        "final_top10_origin": diagnostics.get("split_branch_origin_final_top10", {}),
        "merge_trace": diagnostics.get("split_merge_trace", {}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if summary["b7_controller_mode"] != "t71_materialized_language_local_b6":
        raise SystemExit("B7 controller mode mismatch")
    if not summary["split_applied"]:
        raise SystemExit("Finalized B7 split was not applied")
    if summary["branch_count"] != 2:
        raise SystemExit("Expected exactly two branch queries")
    if no_filter_count != 0:
        raise SystemExit("Accepted B7 branches used no_filter")
    if not local_reranker_all:
        raise SystemExit("Local reranker was not active for all accepted branches")
    if not summary["post_merge_rerank_applied"]:
        raise SystemExit("Post-merge rerank did not run")


if __name__ == "__main__":
    main()
