from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


def apply_runtime_overrides(
    base_config: Dict[str, Any],
    *,
    selected_model: str,
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
    retrieval.setdefault("hybrid_search", {})["enabled"] = bool(bm25_enabled)
    retrieval["recursive_retrieval"] = bool(recursive_enabled)
    retrieval["vision_search"] = bool(vision_enabled)
    retrieval["sibling_expansion"] = bool(sibling_expansion_enabled)

    query_split = retrieval.setdefault("query_split", {})
    query_split["enabled"] = bool(query_split_enabled)
    query_split["max_languages"] = int(query_split_max_languages)
    query_split["branch_k_mode"] = str(query_split_branch_k_mode)

    # Viking routing override
    viking = retrieval.setdefault("viking", {})
    viking["enabled"] = bool(viking_enabled)
    viking["mode"] = "soft" if bool(viking_soft) else "strict"
    viking["max_expansions"] = int(viking_max_exp)

    # Reranker override
    config.setdefault("reranker", {})["enabled"] = bool(reranker_enabled)

    # Pipeline controls
    rag = config.setdefault("rag", {})
    rag["generation_timeout_sec"] = int(generation_timeout_sec)
    rag["skip_grading"] = not bool(grading_enabled)
    rag["skip_hallucination"] = not bool(hallucination_check_enabled)

    return config
