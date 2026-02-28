from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


RERANK_MIN_TOP_N_FOR_K30 = 30
RERANK_K30_PROFILES = {"B6", "B7"}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_profile_file(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    defaults = payload.get("defaults", {}) or {}
    profiles = payload.get("profiles", {}) or {}
    if not profiles:
        raise ValueError(f"No profiles found in {path}")
    return defaults, profiles


def select_profiles(profiles_map: Dict[str, Any], profiles_arg: str) -> List[str]:
    if profiles_arg.strip():
        selected = [p.strip() for p in profiles_arg.split(",") if p.strip()]
    else:
        selected = list(profiles_map.keys())

    missing = [p for p in selected if p not in profiles_map]
    if missing:
        raise ValueError(f"Unknown profiles: {missing}")
    return selected


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def resolve_profile_config(
    *,
    base_config: Dict[str, Any],
    defaults: Dict[str, Any],
    profile_name: str,
    profile_overrides: Dict[str, Any],
    fixed_k: int | None,
    fixed_top_n: int | None,
) -> Dict[str, Any]:
    resolved = deep_merge(base_config, defaults)
    resolved = deep_merge(resolved, profile_overrides)

    retrieval = resolved.setdefault("retrieval", {})
    reranker = resolved.setdefault("reranker", {})
    ablation = resolved.setdefault("ablation", {})

    if ablation.get("force_deterministic", True):
        retrieval["dynamic_scaling"] = False
        if "fixed_k" not in retrieval:
            retrieval["fixed_k"] = retrieval.get("k", 30)
        if "fixed_top_n" not in retrieval:
            retrieval["fixed_top_n"] = reranker.get("top_n", 30)

    if fixed_k is not None:
        retrieval["fixed_k"] = int(fixed_k)
    if fixed_top_n is not None:
        retrieval["fixed_top_n"] = int(fixed_top_n)

    if profile_name in RERANK_K30_PROFILES and bool(reranker.get("enabled", False)):
        reranker_top_n = _as_int(reranker.get("top_n", 30), 30)
        retrieval_top_n = _as_int(retrieval.get("fixed_top_n", reranker_top_n), reranker_top_n)
        enforced = max(RERANK_MIN_TOP_N_FOR_K30, reranker_top_n, retrieval_top_n)
        reranker["top_n"] = enforced
        retrieval["fixed_top_n"] = enforced

    return resolved
