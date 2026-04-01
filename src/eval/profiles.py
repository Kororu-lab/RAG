from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


RERANK_MIN_TOP_N_FOR_K30 = 30
RERANK_K30_PROFILES = {"B2", "B6", "B7"}
FILTERING_MODE_DEFAULT = "viking"
FILTERING_MODE_CHOICES = {"viking", "metadata"}
FILTERING_SWITCHABLE_PROFILES = {"B4", "B6", "B7"}


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


def profile_base_chain(
    profiles_map: Dict[str, Any],
    profile_name: str,
    *,
    _seen: Tuple[str, ...] = (),
) -> List[str]:
    if profile_name in _seen:
        cycle = " -> ".join(list(_seen) + [profile_name])
        raise ValueError(f"Cyclic profile inheritance detected: {cycle}")

    profile = profiles_map.get(profile_name) or {}
    ablation_block = profile.get("ablation", {}) or {}
    base_profile = str(ablation_block.get("base_profile") or "").strip()
    if not base_profile or base_profile not in profiles_map:
        return []

    return profile_base_chain(
        profiles_map,
        base_profile,
        _seen=tuple(list(_seen) + [profile_name]),
    ) + [base_profile]


def resolve_profile_overrides(
    profiles_map: Dict[str, Any],
    profile_name: str,
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for inherited_name in profile_base_chain(profiles_map, profile_name) + [profile_name]:
        overrides = deepcopy(profiles_map.get(inherited_name) or {})
        overrides.pop("description", None)
        merged = deep_merge(merged, overrides)
    return merged


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
    filtering_mode: str = FILTERING_MODE_DEFAULT,
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

    normalized_filtering_mode = str(filtering_mode or FILTERING_MODE_DEFAULT).strip().lower()
    if normalized_filtering_mode not in FILTERING_MODE_CHOICES:
        raise ValueError(
            f"Unknown filtering mode: {filtering_mode!r}. "
            f"Expected one of {sorted(FILTERING_MODE_CHOICES)}."
        )

    if profile_name in FILTERING_SWITCHABLE_PROFILES:
        viking_cfg = retrieval.setdefault("viking", {})
        viking_cfg["enabled"] = normalized_filtering_mode == "viking"

    if profile_name in RERANK_K30_PROFILES and bool(reranker.get("enabled", False)):
        reranker_top_n = _as_int(reranker.get("top_n", 30), 30)
        retrieval_top_n = _as_int(retrieval.get("fixed_top_n", reranker_top_n), reranker_top_n)
        enforced = max(RERANK_MIN_TOP_N_FOR_K30, reranker_top_n, retrieval_top_n)
        reranker["top_n"] = enforced
        retrieval["fixed_top_n"] = enforced

    return resolved
