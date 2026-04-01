from __future__ import annotations

import re
from typing import Any, Dict, List

from src.eval.metadata_store import MetadataStore

_NUMERAL_RANGE_RE = re.compile(r"1\s*(?:부터\s*10(?:까지)?|[-~]\s*10)")
_NUMERAL_FORM_RE = re.compile(
    r"(기수사|서수사|기수|서수|숫자|수\s*체계|numeral|cardinal|ordinal)"
)
_NUMERAL_BASE_RE = re.compile(r"(진법|10진법|20진법|십진법|이십진법|decimal|vigesimal)")


def _contains_term(query_lower: str, term_lower: str) -> bool:
    if not term_lower:
        return False
    if re.fullmatch(r"[a-z0-9_-]+", term_lower):
        if len(term_lower) < 3:
            return False
        return re.search(rf"(?<![a-z0-9]){re.escape(term_lower)}(?![a-z0-9])", query_lower) is not None
    return term_lower in query_lower


def _bump(store: Dict[str, float], key: str, value: float) -> None:
    if not key:
        return
    store[key] = store.get(key, 0.0) + float(value)


def _apply_numeral_semantic_boost(query_lower: str, child_scores: Dict[str, float]) -> None:
    boost = 0.0
    if _NUMERAL_RANGE_RE.search(query_lower):
        boost += 1.0
    if _NUMERAL_FORM_RE.search(query_lower):
        boost += 1.0
    if _NUMERAL_BASE_RE.search(query_lower):
        boost += 1.0
    if boost > 0.0:
        _bump(child_scores, "27_num", boost)


def detect_topics_from_query_with_scores(
    query_text: str,
    phenomenon_lexicon: Dict[str, List[str]],
    category_lexicon: Dict[str, str],
    metadata_store: MetadataStore | None,
) -> Dict[str, Any]:
    q = str(query_text).strip().lower()
    if not q:
        return {
            "parent_topics": [],
            "child_topics": [],
            "parent_topic_scores": {},
            "child_topic_scores": {},
            "parent_topics_high": [],
            "child_topics_high": [],
        }

    parent_scores: Dict[str, float] = {}
    child_scores: Dict[str, float] = {}

    for term, ids in phenomenon_lexicon.items():
        term_lower = str(term).strip().lower()
        if _contains_term(q, term_lower):
            for pid in ids:
                _bump(child_scores, str(pid), 1.0)

    for term, cat_id in category_lexicon.items():
        term_lower = str(term).strip().lower()
        if _contains_term(q, term_lower):
            _bump(parent_scores, str(cat_id), 1.0)

    for topic_id in re.findall(r"\b\d{2}_[a-z0-9_]+\b", q):
        _bump(child_scores, topic_id, 3.0)
    for topic_id in re.findall(r"\b\d_[a-z0-9_]+\b", q):
        _bump(parent_scores, topic_id, 3.0)

    if metadata_store:
        known_parent = {t for arr in metadata_store.parent_topics_by_lang.values() for t in arr}
        known_child = {t for arr in metadata_store.child_topics_by_lang.values() for t in arr}
        for topic_id in sorted(known_parent):
            if _contains_term(q, topic_id.lower()):
                _bump(parent_scores, topic_id, 1.5)
        for topic_id in sorted(known_child):
            if _contains_term(q, topic_id.lower()):
                _bump(child_scores, topic_id, 1.5)

    _apply_numeral_semantic_boost(q, child_scores)

    ordered_parent = sorted(parent_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    ordered_child = sorted(child_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    parent_topics = [topic for topic, _score in ordered_parent]
    child_topics = [topic for topic, _score in ordered_child]
    parent_topics_high = [topic for topic, score in ordered_parent if score >= 2.0]
    child_topics_high = [topic for topic, score in ordered_child if score >= 2.0]

    return {
        "parent_topics": parent_topics,
        "child_topics": child_topics,
        "parent_topic_scores": {topic: round(score, 4) for topic, score in ordered_parent},
        "child_topic_scores": {topic: round(score, 4) for topic, score in ordered_child},
        "parent_topics_high": parent_topics_high,
        "child_topics_high": child_topics_high,
    }
