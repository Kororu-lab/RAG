from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SplitBudgets:
    k_allocs: List[int]
    top_n_allocs: List[int]
    merge_cap: int


def balanced_allocations(total_budget: int, branch_count: int) -> List[int]:
    if branch_count <= 0:
        return []
    total = max(0, int(total_budget))
    base = total // branch_count
    remainder = total % branch_count
    allocations = [base] * branch_count
    for idx in range(remainder):
        allocations[idx] += 1
    return allocations


def compute_split_budgets(
    *,
    total_k: int,
    total_top_n: int,
    branch_count: int,
    mode: str,
) -> SplitBudgets:
    """
    Compute per-branch retrieval/rerank budgets and merged cap.

    Modes:
    - balanced: split total budget across branches; merge cap is total_k
    - full_per_branch: each branch gets full budget; merge cap scales by branch_count
    """
    count = max(0, int(branch_count))
    normalized_mode = str(mode or "balanced").strip().lower()

    if normalized_mode == "balanced":
        k_allocs = balanced_allocations(total_k, count)
        top_n_allocs = balanced_allocations(total_top_n, count)
        merge_cap = max(0, int(total_k))
        return SplitBudgets(k_allocs=k_allocs, top_n_allocs=top_n_allocs, merge_cap=merge_cap)

    if normalized_mode == "full_per_branch":
        k_each = max(0, int(total_k))
        top_n_each = max(0, int(total_top_n))
        k_allocs = [k_each] * count
        top_n_allocs = [top_n_each] * count
        merge_cap = k_each * count
        return SplitBudgets(k_allocs=k_allocs, top_n_allocs=top_n_allocs, merge_cap=merge_cap)

    raise ValueError(f"unsupported_split_branch_k_mode:{mode}")
