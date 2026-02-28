from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _safe_chunk_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def build_gold_sets(item: Dict[str, Any]) -> Tuple[set[str], set[Tuple[str, str]]]:
    gold_docs = set(str(v) for v in (item.get("gold_docs") or []) if str(v).strip())

    gold_chunks: set[Tuple[str, str]] = set()
    for chunk in item.get("gold_chunks") or []:
        if not isinstance(chunk, dict):
            continue
        source_file = str(chunk.get("source_file", "")).strip()
        if not source_file:
            continue
        chunk_id = _safe_chunk_id(chunk.get("chunk_id"))
        gold_chunks.add((source_file, chunk_id))

    return gold_docs, gold_chunks


def compute_query_metrics(
    *,
    ranked_docs: Sequence[Dict[str, Any]],
    gold_docs: set[str],
    gold_chunks: set[Tuple[str, str]],
    ks: Sequence[int],
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    for k in ks:
        top_docs = ranked_docs[:k]

        top_doc_set = set()
        top_chunk_set = set()
        for row in top_docs:
            source_file = str(row.get("source_file", "")).strip()
            if source_file:
                top_doc_set.add(source_file)
                top_chunk_set.add((source_file, _safe_chunk_id(row.get("chunk_id"))))

        if gold_docs:
            doc_hits = len(gold_docs.intersection(top_doc_set))
            out[f"doc_recall_at_{k}"] = doc_hits / float(len(gold_docs))
        else:
            out[f"doc_recall_at_{k}"] = 0.0

        if gold_chunks:
            chunk_hits = len(gold_chunks.intersection(top_chunk_set))
            out[f"chunk_recall_at_{k}"] = chunk_hits / float(len(gold_chunks))
        else:
            out[f"chunk_recall_at_{k}"] = 0.0

        reciprocal_rank = 0.0
        for rank, row in enumerate(top_docs, start=1):
            source_file = str(row.get("source_file", "")).strip()
            chunk_key = (source_file, _safe_chunk_id(row.get("chunk_id")))
            if gold_chunks:
                if chunk_key in gold_chunks:
                    reciprocal_rank = 1.0 / rank
                    break
            elif gold_docs and source_file in gold_docs:
                reciprocal_rank = 1.0 / rank
                break
        out[f"mrr_at_{k}"] = reciprocal_rank

    return out


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(mean(values))


def aggregate_macro_micro(
    *,
    rows: Sequence[Dict[str, Any]],
    ks: Sequence[int],
    profile: str,
    track: str,
) -> List[Dict[str, Any]]:
    metrics = ["chunk_recall", "doc_recall", "mrr"]
    out: List[Dict[str, Any]] = []

    for metric in metrics:
        for k in ks:
            key = f"{metric}_at_{k}"
            micro = _mean(float(r.get(key, 0.0)) for r in rows)
            out.append(
                {
                    "profile": profile,
                    "track": track,
                    "aggregation": "micro",
                    "metric": metric,
                    "k": k,
                    "value": round(micro, 6),
                }
            )

            per_type_values = defaultdict(list)
            for row in rows:
                q_type = str(row.get("query_type", "unknown"))
                per_type_values[q_type].append(float(row.get(key, 0.0)))

            macro_components = [_mean(vals) for vals in per_type_values.values() if vals]
            macro = _mean(macro_components)
            out.append(
                {
                    "profile": profile,
                    "track": track,
                    "aggregation": "macro",
                    "metric": metric,
                    "k": k,
                    "value": round(macro, 6),
                }
            )

    return out
