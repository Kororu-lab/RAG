from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


def apply_runtime_overrides(
    base_config: Dict[str, Any],
    *,
    selected_model: str,
    language_detection_enabled: bool,
    language_filter_enabled: bool,
    viking_use_detected_language_enabled: bool,
    bm25_enabled: bool,
    reranker_enabled: bool,
    recursive_enabled: bool,
    vision_enabled: bool,
    sibling_expansion_enabled: bool,
    query_split_enabled: bool,
    query_split_max_languages: int,
    query_split_branch_k_mode: str,
    viking_enabled: bool,
    viking_soft: bool,
    viking_max_exp: int,
    retrieval_llm_timeout_sec: int,
    generation_timeout_sec: int,
    original_evidence_generation_enabled: bool,
    whole_doc_on_multi_relevant_enabled: bool,
    grading_enabled: bool,
    hallucination_check_enabled: bool,
) -> Dict[str, Any]:
    """Return runtime config with UI-selected overrides applied."""
    config = deepcopy(base_config)

    # Model override
    config.setdefault("llm_retrieval", {})["model_name"] = selected_model
    config.setdefault("llm_retrieval", {})["request_timeout_sec"] = int(retrieval_llm_timeout_sec)
    if "llm" in config:
        config["llm"]["model_name"] = selected_model

    # Retrieval settings override
    retrieval = config.setdefault("retrieval", {})
    retrieval["enable_llm_language_detection"] = bool(language_detection_enabled)
    retrieval["enable_language_filter"] = bool(language_filter_enabled)
    retrieval["viking_use_detected_language"] = bool(viking_use_detected_language_enabled)
    hybrid_search = retrieval.setdefault("hybrid_search", {})
    hybrid_search["enabled"] = bool(bm25_enabled)
    hybrid_search["mode"] = "dense_bm25_union"
    hybrid_search["union_bm25_overfetch"] = int(hybrid_search.get("union_bm25_overfetch", 10) or 10)
    retrieval["recursive_retrieval"] = bool(recursive_enabled)
    retrieval["vision_search"] = bool(vision_enabled)
    retrieval["sibling_expansion"] = bool(sibling_expansion_enabled)

    query_split = retrieval.setdefault("query_split", {})
    # Split requires language metadata to be meaningful; auto-disable when metadata detection is off.
    query_split["enabled"] = bool(query_split_enabled and language_detection_enabled)
    query_split["max_languages"] = int(query_split_max_languages)
    query_split["branch_k_mode"] = str(query_split_branch_k_mode)
    if query_split["enabled"]:
        retrieval["strict_l0_only"] = True
        retrieval["enable_parent_topic_filter"] = True
        retrieval["enable_child_topic_filter"] = True
        query_split["controller_mode"] = "t71_materialized_language_local_b6"
        query_split["axis_mode"] = "language_only"
        query_split["enable_language_axis"] = True
        query_split["enable_topic_axis"] = False
        query_split["strict_single_axis"] = True
        query_split["max_branches"] = 2
        query_split["branch_query_mode"] = "surface_preserving_materialized"
        query_split["feedback_gating"] = False
        query_split["post_merge_rerank_mode"] = "global"
        query_split["language_diversity_guard_enabled"] = True
        query_split["language_diversity_min_per_branch"] = 1

    # Viking routing override
    viking = retrieval.setdefault("viking", {})
    viking["enabled"] = bool(viking_enabled)
    viking["mode"] = "soft" if bool(viking_soft) else "strict"
    viking["max_expansions"] = int(viking_max_exp)

    # Reranker override
    config.setdefault("reranker", {})["enabled"] = bool(reranker_enabled or bm25_enabled)

    # Pipeline controls
    rag = config.setdefault("rag", {})
    rag["generation_timeout_sec"] = int(generation_timeout_sec)
    rag["original_evidence_generation_enabled"] = bool(original_evidence_generation_enabled)
    rag["whole_doc_on_multi_relevant_enabled"] = bool(
        whole_doc_on_multi_relevant_enabled and original_evidence_generation_enabled
    )
    rag["skip_grading"] = not bool(grading_enabled)
    rag["skip_hallucination"] = not bool(hallucination_check_enabled)

    return config
