from __future__ import annotations

import re
from copy import deepcopy
from time import perf_counter
from typing import Any, Dict, List, Sequence, Tuple

from langchain_core.documents import Document

from src.eval.materialized_language_split import materialize_language_split_plan
from src.eval.metadata_store import MetadataStore
from src.eval.query_topic_detection import (
    detect_topics_from_query_with_scores as _shared_detect_topics_from_query_with_scores,
)
from src.retrieve.rag_retrieve import RAGRetriever


FINALIZED_B7_CONTROLLER_MODES = {
    "t71_materialized_language_local_b6",
    "finalized_materialized_language_split",
}


def _doc_identity(doc: Document) -> Tuple[str, str]:
    source_file = doc.metadata.get("source_file")
    chunk_id = doc.metadata.get("chunk_id")
    if source_file is not None and chunk_id is not None:
        return (str(source_file), str(chunk_id))
    ref_id = doc.metadata.get("ref_id")
    if ref_id:
        return ("ref", str(ref_id))
    db_id = doc.metadata.get("_db_id")
    if db_id is not None:
        return ("db", str(db_id))
    return ("obj", str(id(doc)))


def _doc_score(doc: Document) -> float:
    try:
        score = doc.metadata.get("score")
        if score is None:
            return float("-inf")
        return float(score)
    except Exception:
        return float("-inf")


def _merge_split_docs(
    *,
    branch_docs: Dict[str, List[Document]],
    branch_order: Sequence[str],
    merge_cap: int,
) -> List[Document]:
    sorted_docs: Dict[str, List[Document]] = {}
    for branch_id in branch_order:
        docs = list(branch_docs.get(branch_id, []))
        docs.sort(key=_doc_score, reverse=True)
        sorted_docs[branch_id] = docs

    merged: List[Document] = []
    seen: set[Tuple[str, str]] = set()
    cursor = {branch_id: 0 for branch_id in branch_order}
    cap = max(0, int(merge_cap))
    if cap <= 0:
        return []

    while len(merged) < cap:
        progressed = False
        for branch_id in branch_order:
            docs = sorted_docs.get(branch_id, [])
            idx = cursor.get(branch_id, 0)
            while idx < len(docs):
                doc = docs[idx]
                idx += 1
                key = _doc_identity(doc)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(doc)
                progressed = True
                break
            cursor[branch_id] = idx
            if len(merged) >= cap:
                break
        if not progressed:
            break
    return merged


def _merge_with_branch_minimums(
    *,
    ranked_docs: Sequence[Document],
    branch_order: Sequence[str],
    branch_origin_by_doc: Dict[Tuple[str, str], str],
    final_top_n: int,
    min_per_branch: int,
) -> List[Document]:
    target_n = max(0, int(final_top_n))
    branch_min = max(0, int(min_per_branch))
    if target_n <= 0 or branch_min <= 0:
        return list(ranked_docs)[:target_n]

    selected: List[Document] = []
    seen: set[Tuple[str, str]] = set()
    for branch_id in branch_order:
        kept = 0
        for doc in ranked_docs:
            key = _doc_identity(doc)
            if key in seen:
                continue
            if branch_origin_by_doc.get(key) != branch_id:
                continue
            selected.append(doc)
            seen.add(key)
            kept += 1
            if kept >= branch_min or len(selected) >= target_n:
                break
        if len(selected) >= target_n:
            break

    if len(selected) < target_n:
        for doc in ranked_docs:
            key = _doc_identity(doc)
            if key in seen:
                continue
            selected.append(doc)
            seen.add(key)
            if len(selected) >= target_n:
                break

    return selected[:target_n]


def _apply_language_diversity_guard(
    docs: Sequence[Document],
    *,
    branch_order: Sequence[str],
    branch_origin_by_doc: Dict[Tuple[str, str], str],
    final_top_n: int,
    min_per_branch: int,
) -> tuple[List[Document], Dict[str, Any]]:
    guarded = _merge_with_branch_minimums(
        ranked_docs=list(docs),
        branch_order=branch_order,
        branch_origin_by_doc=branch_origin_by_doc,
        final_top_n=final_top_n,
        min_per_branch=min_per_branch,
    )
    origin_counts: Dict[str, int] = {}
    for doc in guarded[: max(0, int(final_top_n))]:
        origin = branch_origin_by_doc.get(_doc_identity(doc), "unknown")
        origin_counts[origin] = origin_counts.get(origin, 0) + 1
    return guarded, {
        "applied": True,
        "min_per_branch": int(min_per_branch),
        "origin_distribution": origin_counts,
    }


def _to_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


def _contains_term(query_lower: str, term_lower: str) -> bool:
    if not term_lower:
        return False
    if re.fullmatch(r"[a-z0-9_-]+", term_lower):
        if len(term_lower) < 3:
            return False
        return re.search(rf"(?<![a-z0-9]){re.escape(term_lower)}(?![a-z0-9])", query_lower) is not None
    return term_lower in query_lower


def _normalize_query_text(text: str) -> str:
    q = str(text).strip().lower()
    if not q:
        return ""
    q = re.sub(r"[^0-9a-z가-힣_\-\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _compact_text(text: str) -> str:
    return re.sub(r"[\s\-_]+", "", text)


def _strip_korean_particles(token: str) -> str:
    suffixes = [
        "에서는",
        "으로는",
        "에게는",
        "에서의",
        "으로의",
        "와의",
        "과의",
        "에서",
        "으로",
        "에게",
        "한테",
        "부터",
        "까지",
        "처럼",
        "보다",
        "마다",
        "조차",
        "밖에",
        "만",
        "도",
        "의",
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "과",
        "와",
        "랑",
        "에",
    ]
    stem = token
    for _ in range(2):
        changed = False
        for sfx in suffixes:
            if stem.endswith(sfx) and len(stem) - len(sfx) >= 2:
                stem = stem[: -len(sfx)]
                changed = True
                break
        if not changed:
            break
    return stem


def _language_context_for_alias(query_norm: str, alias: str) -> bool:
    if not alias:
        return False
    if re.search(r"[가-힣]", alias):
        if alias.endswith("어") or alias.endswith("언어"):
            return True
        return (f"{alias}어" in query_norm) or (f"{alias} 언어" in query_norm)
    if re.fullmatch(r"[a-z0-9 ]+", alias):
        return bool(
            re.search(rf"(?<![a-z0-9]){re.escape(alias)}\s+(language|lang)(?![a-z0-9])", query_norm)
            or re.search(rf"(?<![a-z0-9])(language|lang)\s+{re.escape(alias)}(?![a-z0-9])", query_norm)
        )
    return False


def _detect_languages_from_query(
    query_text: str,
    metadata_store: MetadataStore | None,
) -> Dict[str, Any]:
    if not metadata_store:
        return {
            "selected_languages": [],
            "candidate_languages": [],
            "language_confidences": {},
            "high": [],
            "medium": [],
            "low": [],
            "scores": {},
            "risk_only_languages": [],
            "language_conflict": False,
        }

    q_norm = _normalize_query_text(query_text)
    if not q_norm:
        return {
            "selected_languages": [],
            "candidate_languages": [],
            "language_confidences": {},
            "high": [],
            "medium": [],
            "low": [],
            "scores": {},
            "risk_only_languages": [],
            "language_conflict": False,
        }

    q_compact = _compact_text(q_norm)
    tokens = re.findall(r"[0-9a-z가-힣_\-]+", q_norm)
    token_stems = set(tokens)
    for tok in tokens:
        if re.search(r"[가-힣]", tok):
            stem = _strip_korean_particles(tok)
            if stem:
                token_stems.add(stem)
                token_stems.add(_compact_text(stem))

    lang_scores: Dict[str, float] = {}
    lang_sources: Dict[str, set[str]] = {}
    lang_exact_nonrisk: Dict[str, int] = {}
    lang_risk_hits: Dict[str, int] = {}
    lang_context_hits: Dict[str, int] = {}
    risk_only_by_lang: Dict[str, bool] = {}

    for alias, entries in metadata_store.alias_index.items():
        if not alias or not entries:
            continue
        alias_compact = _compact_text(alias)
        match_type = ""
        if _contains_term(q_norm, alias):
            match_type = "exact"
        elif alias in token_stems:
            match_type = "stem"
        elif len(alias_compact) >= 4 and alias_compact in q_compact:
            match_type = "compact"
        if not match_type:
            continue

        alias_langs = {str(e.get("lang_id")) for e in entries if e.get("lang_id")}
        ambiguous_alias = len(alias_langs) > 1
        has_context = _language_context_for_alias(q_norm, alias)

        for entry in entries:
            lang_id = str(entry.get("lang_id", "")).strip()
            if not lang_id:
                continue
            source = str(entry.get("source", "")).strip() or "unknown"
            risk = bool(entry.get("risk", False))
            base = 3.0 if match_type == "exact" else (2.5 if match_type == "stem" else 2.0)
            if risk:
                base -= 1.5
            if ambiguous_alias:
                base -= 0.75
            if base < 0.25:
                base = 0.25

            lang_scores[lang_id] = lang_scores.get(lang_id, 0.0) + base
            lang_sources.setdefault(lang_id, set()).add(source)
            if (not risk) and (not ambiguous_alias) and match_type == "exact":
                lang_exact_nonrisk[lang_id] = lang_exact_nonrisk.get(lang_id, 0) + 1
            if risk:
                lang_risk_hits[lang_id] = lang_risk_hits.get(lang_id, 0) + 1
            if has_context:
                lang_context_hits[lang_id] = lang_context_hits.get(lang_id, 0) + 1

    if not lang_scores:
        return {
            "selected_languages": [],
            "candidate_languages": [],
            "language_confidences": {},
            "high": [],
            "medium": [],
            "low": [],
            "scores": {},
            "risk_only_languages": [],
            "language_conflict": False,
        }

    ordered = sorted(lang_scores.items(), key=lambda kv: kv[1], reverse=True)
    conf_map: Dict[str, str] = {}
    for idx, (lang_id, score) in enumerate(ordered):
        second = ordered[idx + 1][1] if idx + 1 < len(ordered) else 0.0
        source_count = len(lang_sources.get(lang_id, set()))
        exact_nonrisk = lang_exact_nonrisk.get(lang_id, 0)
        risk_hits = lang_risk_hits.get(lang_id, 0)
        context_hits = lang_context_hits.get(lang_id, 0)

        if score >= 3.0 and (exact_nonrisk >= 1 or source_count >= 2):
            conf = "high"
        elif score >= 1.75:
            conf = "medium"
        else:
            conf = "low"

        risk_only = risk_hits > 0 and exact_nonrisk == 0
        risk_only_by_lang[lang_id] = risk_only
        if risk_only:
            if context_hits <= 0:
                conf = "low"
            elif conf == "high":
                conf = "medium"

        margin = score - second
        if conf == "high" and margin < 0.5 and exact_nonrisk == 0 and source_count < 2:
            conf = "medium"
        if conf == "medium" and margin < 0.2 and risk_only:
            conf = "low"

        conf_map[lang_id] = conf

    high = [lang for lang, _ in ordered if conf_map.get(lang) == "high"]
    medium = [lang for lang, _ in ordered if conf_map.get(lang) == "medium"]
    low = [lang for lang, _ in ordered if conf_map.get(lang) == "low"]

    if high:
        selected = list(high)
    elif len(medium) == 1:
        selected = [medium[0]]
    else:
        selected = []

    candidates = [lang for lang in high + medium]
    top_score = ordered[0][1] if ordered else 0.0
    language_conflict = False
    if high:
        for lang in medium:
            score = lang_scores.get(lang, 0.0)
            if (top_score - score) < 0.35:
                language_conflict = True
                break
    return {
        "selected_languages": selected,
        "candidate_languages": candidates,
        "language_confidences": conf_map,
        "high": high,
        "medium": medium,
        "low": low,
        "scores": {lang: round(score, 4) for lang, score in ordered},
        "risk_only_languages": [lang for lang, is_risk_only in risk_only_by_lang.items() if is_risk_only],
        "language_conflict": language_conflict,
    }


def _detect_topics_from_query_with_scores(
    query_text: str,
    phenomenon_lexicon: Dict[str, List[str]],
    category_lexicon: Dict[str, str],
    metadata_store: MetadataStore | None,
) -> Dict[str, Any]:
    return _shared_detect_topics_from_query_with_scores(
        query_text=query_text,
        phenomenon_lexicon=phenomenon_lexicon,
        category_lexicon=category_lexicon,
        metadata_store=metadata_store,
    )


def build_query_only_metadata(
    *,
    query_text: str,
    metadata_store: MetadataStore | None,
    phenomenon_lexicon: Dict[str, List[str]],
    category_lexicon: Dict[str, str],
) -> Dict[str, Any]:
    lang_detect = _detect_languages_from_query(query_text, metadata_store)
    detected_languages = list(lang_detect.get("selected_languages", []))
    topic_detect = _detect_topics_from_query_with_scores(
        query_text,
        phenomenon_lexicon,
        category_lexicon,
        metadata_store,
    )
    detected_parent_topics = list(topic_detect.get("parent_topics", []))
    detected_child_topics = list(topic_detect.get("child_topics", []))
    detected_parent_topics_high = list(topic_detect.get("parent_topics_high", []))
    detected_child_topics_high = list(topic_detect.get("child_topics_high", []))

    lang_conf = "low"
    if any(str(v) == "high" for v in lang_detect.get("language_confidences", {}).values()):
        lang_conf = "high"
    elif detected_languages:
        lang_conf = "medium"
    child_conf = "high" if detected_child_topics_high else ("medium" if detected_child_topics else "low")
    parent_conf = "high" if detected_parent_topics_high else ("medium" if detected_parent_topics else "low")

    diagnostics = {
        "lang_detect_method": "query_only_alias",
        "lang_detect_stage": "query_text",
        "lang_json_retry_count": 0,
        "lang_json_valid_stage_a": True,
        "lang_json_valid_stage_b": True,
        "lang_json_fail_open": False,
        "detected_languages": detected_languages,
        "detected_languages_candidates": lang_detect.get("candidate_languages", []),
        "detected_languages_high": lang_detect.get("high", []),
        "detected_languages_medium": lang_detect.get("medium", []),
        "detected_languages_low": lang_detect.get("low", []),
        "language_confidences": lang_detect.get("language_confidences", {}),
        "lang_detect_scores": lang_detect.get("scores", {}),
        "risk_only_languages": lang_detect.get("risk_only_languages", []),
        "language_conflict": bool(lang_detect.get("language_conflict", False)),
        "detected_topics_categories": detected_parent_topics,
        "detected_topics_phenomena": detected_child_topics,
        "detected_topics_categories_high": detected_parent_topics_high,
        "detected_topics_phenomena_high": detected_child_topics_high,
        "topic_scores_parent": topic_detect.get("parent_topic_scores", {}),
        "topic_scores_phenomena": topic_detect.get("child_topic_scores", {}),
        "detector_confidence": {
            "language": lang_conf,
            "parent_topic": parent_conf,
            "child_topic": child_conf,
        },
        "split_applied": False,
        "split_branches": [],
        "split_json_retry_count": 0,
        "split_json_fail_open": False,
        "timeout_location": "none",
        "filter_applied": False,
        "metadata_mode": "query_only",
    }

    return {
        "lang": detected_languages if len(detected_languages) > 1 else (detected_languages[0] if detected_languages else None),
        "topics": {
            "categories": detected_parent_topics,
            "phenomena": detected_child_topics,
        },
        "families": [],
        "regions": [],
        "_diagnostics": diagnostics,
    }


def _stage_metadata(base_metadata: Dict[str, Any], stage: str) -> Dict[str, Any]:
    stage_meta = deepcopy(base_metadata)
    stage_meta["families"] = []
    stage_meta["regions"] = []

    topics = stage_meta.get("topics", {}) if isinstance(stage_meta.get("topics"), dict) else {}
    parent_topics = _to_list(topics.get("categories"))
    child_topics = _to_list(topics.get("phenomena"))
    languages = _to_list(stage_meta.get("lang"))

    if stage == "child_lang":
        stage_meta["lang"] = languages if len(languages) > 1 else (languages[0] if languages else None)
        stage_meta["topics"] = {"categories": [], "phenomena": child_topics}
    elif stage == "parent_lang":
        stage_meta["lang"] = languages if len(languages) > 1 else (languages[0] if languages else None)
        stage_meta["topics"] = {"categories": parent_topics, "phenomena": []}
    elif stage == "lang_only":
        stage_meta["lang"] = languages if len(languages) > 1 else (languages[0] if languages else None)
        stage_meta["topics"] = {"categories": [], "phenomena": []}
    else:
        stage_meta["lang"] = None
        stage_meta["topics"] = {"categories": [], "phenomena": []}

    return stage_meta


def retrieve_with_filter_fallback(
    *,
    retriever: RAGRetriever,
    query_text: str,
    base_metadata: Dict[str, Any],
    fixed_k: int,
    fixed_top_n: int,
    allow_no_filter: bool = True,
) -> tuple[List[Document], Dict[str, Any]]:
    diagnostics = base_metadata.get("_diagnostics", {}) if isinstance(base_metadata.get("_diagnostics"), dict) else {}
    parent_topics = _to_list((base_metadata.get("topics") or {}).get("categories"))
    child_topics = _to_list((base_metadata.get("topics") or {}).get("phenomena"))
    languages = _to_list(base_metadata.get("lang"))
    conf = diagnostics.get("detector_confidence", {}) if isinstance(diagnostics.get("detector_confidence"), dict) else {}

    stages: List[str] = []
    if retriever.enable_child_topic_filter and retriever.enable_language_filter and child_topics and languages and conf.get("child_topic") == "high":
        stages.append("child_lang")
    if retriever.enable_parent_topic_filter and retriever.enable_language_filter and parent_topics and languages and conf.get("parent_topic") in {"medium", "high"}:
        stages.append("parent_lang")
    if retriever.enable_language_filter and languages and conf.get("language") in {"medium", "high"}:
        stages.append("lang_only")
    if allow_no_filter:
        stages.append("no_filter")

    deduped_stages: List[str] = []
    for stage in stages:
        if stage not in deduped_stages:
            deduped_stages.append(stage)

    attempted: List[Dict[str, Any]] = []
    final_docs: List[Document] = []
    selected_stage = "no_filter"
    fallback_index = 0
    for idx, stage in enumerate(deduped_stages):
        stage_meta = _stage_metadata(base_metadata, stage)
        docs = retriever.retrieve_documents(
            query_text,
            metadata=stage_meta,
            k=fixed_k,
            top_n=fixed_top_n,
        )
        attempted.append({"stage": stage, "doc_count": len(docs), "zero_hit": len(docs) == 0})
        if docs:
            final_docs = docs
            selected_stage = stage
            fallback_index = idx
            break
    else:
        if deduped_stages:
            selected_stage = deduped_stages[-1]
            fallback_index = len(deduped_stages) - 1

    diagnostics["filter_stage_used"] = selected_stage
    diagnostics["fallback_stage"] = fallback_index
    diagnostics["filter_stage_attempts"] = attempted
    diagnostics["doc_count"] = len(final_docs)
    diagnostics["zero_hit"] = len(final_docs) == 0
    diagnostics["filter_applied"] = selected_stage != "no_filter"
    return final_docs, diagnostics


def _top_docs_snapshot(docs: Sequence[Document], limit: int = 5) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for doc in list(docs)[: max(0, int(limit))]:
        out.append(
            {
                "source_file": doc.metadata.get("source_file"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "lang": doc.metadata.get("lang"),
                "score": doc.metadata.get("score"),
                "rerank_score": doc.metadata.get("rerank_score"),
            }
        )
    return out


def _top_chunks_snapshot(docs: Sequence[Document], limit: int = 10) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for doc in list(docs)[: max(0, int(limit))]:
        content = re.sub(r"\s+", " ", str(getattr(doc, "page_content", "") or "")).strip()
        out.append(
            {
                "source_file": doc.metadata.get("source_file"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "lang": doc.metadata.get("lang"),
                "header": doc.metadata.get("original_header") or doc.metadata.get("header"),
                "score": doc.metadata.get("score"),
                "content_preview": content[:240],
            }
        )
    return out


def _build_split_branch_trace_row(
    *,
    branch_id: str,
    target_label: str,
    target_lang_id: str,
    branch_query_text: str,
    branch_query_mode: str,
    branch_meta: Dict[str, Any],
    branch_diag: Dict[str, Any],
    branch_result_docs: Sequence[Document],
    local_reranker_active: bool,
    branch_anchor_retained: bool,
    branch_topic_anchor: str,
) -> Dict[str, Any]:
    return {
        "branch_id": branch_id,
        "target_label": target_label,
        "target_lang_id": target_lang_id,
        "actual_query_sent": branch_query_text,
        "branch_query_mode": branch_query_mode,
        "branch_query_text": branch_query_text,
        "branch_effective_language": branch_meta.get("_diagnostics", {}).get("branch_effective_language", ""),
        "branch_language_source": branch_meta.get("_diagnostics", {}).get("branch_language_source", ""),
        "branch_language_fragment": branch_meta.get("_diagnostics", {}).get("branch_language_fragment", ""),
        "branch_language_anchor_span": branch_meta.get("_diagnostics", {}).get("branch_language_anchor_span", []),
        "filter_stage_used": branch_diag.get("filter_stage_used", "no_filter"),
        "fallback_stage": int(branch_diag.get("fallback_stage", 0) or 0),
        "doc_count": len(branch_result_docs),
        "zero_hit": len(branch_result_docs) == 0,
        "local_reranker_active": bool(local_reranker_active),
        "anchor_retained": bool(branch_anchor_retained),
        "topic_anchor": str(branch_topic_anchor or ""),
        "top5": _top_docs_snapshot(branch_result_docs, limit=5),
        "top_docs": _top_docs_snapshot(branch_result_docs, limit=5),
        "top_chunks": _top_chunks_snapshot(branch_result_docs, limit=min(10, len(branch_result_docs))),
    }


def _build_finalized_branch_metadata(
    *,
    branch_query_text: str,
    target_label: str,
    target_lang_id: str,
    metadata_store: MetadataStore | None,
    phenomenon_lexicon: Dict[str, List[str]],
    category_lexicon: Dict[str, str],
    branch_id: str,
    anchor_retained: bool,
    topic_anchor: str,
) -> Dict[str, Any]:
    branch_meta = build_query_only_metadata(
        query_text=branch_query_text,
        metadata_store=metadata_store,
        phenomenon_lexicon=phenomenon_lexicon,
        category_lexicon=category_lexicon,
    )
    branch_diag = branch_meta.setdefault("_diagnostics", {})
    branch_meta["lang"] = target_lang_id
    branch_diag["detected_languages"] = [target_lang_id]
    branch_diag["detected_languages_candidates"] = [target_lang_id]
    branch_diag["detected_languages_high"] = [target_lang_id]
    branch_diag["detected_languages_medium"] = []
    branch_diag["detected_languages_low"] = []
    branch_diag["language_confidences"] = {target_lang_id: "high"}
    branch_diag["lang_detect_scores"] = {target_lang_id: 1000.0}
    branch_diag["language_conflict"] = False
    detector_conf = branch_diag.get("detector_confidence")
    if not isinstance(detector_conf, dict):
        detector_conf = {}
    detector_conf["language"] = "high"
    branch_diag["detector_confidence"] = detector_conf
    branch_diag["split_branch_id"] = branch_id
    branch_diag["branch_effective_language"] = target_lang_id
    branch_diag["branch_resolved_target_lang_id"] = target_lang_id
    branch_diag["branch_language_source"] = "hard_seeded_target"
    branch_diag["branch_query_materialized"] = True
    branch_diag["branch_query_target_span"] = target_label
    branch_diag["anchor_retained"] = bool(anchor_retained)
    branch_diag["topic_anchor"] = str(topic_anchor or "")
    return branch_meta


def run_finalized_b7_retrieval(
    *,
    retriever: RAGRetriever,
    query_id: str,
    query_text: str,
    profile_cfg: Dict[str, Any],
    fixed_k: int,
    fixed_top_n: int,
    metadata_store: MetadataStore | None,
    phenomenon_lexicon: Dict[str, List[str]],
    category_lexicon: Dict[str, str],
) -> tuple[Dict[str, Any], List[Document], List[str], float, float]:
    t_meta = perf_counter()
    llm_retrieval_cfg = profile_cfg.get("llm_retrieval", {}) or {}
    request_timeout_sec = int(llm_retrieval_cfg.get("request_timeout_sec", 600) or 600)
    metadata = build_query_only_metadata(
        query_text=query_text,
        metadata_store=metadata_store,
        phenomenon_lexicon=phenomenon_lexicon,
        category_lexicon=category_lexicon,
    )
    metadata_elapsed_sec = perf_counter() - t_meta

    diagnostics = metadata.setdefault("_diagnostics", {})
    diagnostics.setdefault("metadata_mode", "query_only")
    diagnostics.setdefault("split_applied", False)
    diagnostics.setdefault("split_branches", [])
    diagnostics.setdefault("split_json_retry_count", 0)
    diagnostics.setdefault("split_json_fail_open", False)
    diagnostics.setdefault("filter_applied", False)
    diagnostics.setdefault("timeout_location", "none")
    diagnostics["split_controller_mode"] = "t71_materialized_language_local_b6"

    query_split_cfg = (
        profile_cfg.get("retrieval", {}).get("query_split", {})
        if isinstance(profile_cfg.get("retrieval", {}).get("query_split"), dict)
        else {}
    )

    warnings: List[str] = []
    plan = materialize_language_split_plan(
        query_id=query_id,
        query_text=query_text,
        request_timeout_sec=request_timeout_sec,
        metadata_store=metadata_store,
        split_mode="recovered",
        renderer_mode="surface_preserving",
    )

    diagnostics["split_materialized_plan"] = plan
    diagnostics["split_branch_query_mode"] = "surface_preserving_materialized"
    diagnostics["split_reason"] = "materialized_language_split" if bool(plan.get("should_split", False)) else str(plan.get("abstain_reason", "materialized_abstain"))

    t_retrieve = perf_counter()
    if not bool(plan.get("should_split", False)) or len(plan.get("branch_queries", [])) != 2:
        docs, diag_update = retrieve_with_filter_fallback(
            retriever=retriever,
            query_text=query_text,
            base_metadata=metadata,
            fixed_k=fixed_k,
            fixed_top_n=fixed_top_n,
        )
        diagnostics.update(diag_update)
        diagnostics["split_applied"] = False
        diagnostics["split_dimension"] = "none"
        diagnostics["split_branches"] = []
        diagnostics["split_branch_query_count"] = 0
        diagnostics["split_branch_filters"] = []
        diagnostics["split_branch_doc_counts"] = {}
        diagnostics["split_branch_diagnostics"] = []
        diagnostics["split_branch_queries"] = []
        retrieval_elapsed_sec = perf_counter() - t_retrieve
        return metadata, docs, warnings, metadata_elapsed_sec, retrieval_elapsed_sec

    diagnostics["split_applied"] = True
    diagnostics["split_dimension"] = "language"
    diagnostics["split_branches"] = [str(row.get("branch_id")) for row in plan.get("branch_queries", [])]
    diagnostics["split_branch_query_count"] = len(diagnostics["split_branches"])

    branch_docs: Dict[str, List[Document]] = {}
    branch_diag_rows: List[Dict[str, Any]] = []
    branch_doc_counts: Dict[str, int] = {}
    branch_origin_by_doc: Dict[Tuple[str, str], str] = {}
    branch_order: List[str] = []
    split_branch_queries: List[Dict[str, Any]] = []
    split_branch_filters: List[Dict[str, Any]] = []
    branch_retrieval_failed = False

    original_use_reranker = bool(getattr(retriever, "use_reranker", False))
    try:
        for branch in plan.get("branch_queries", []):
            branch_id = str(branch.get("branch_id", "")).strip()
            target_label = str(branch.get("target_label", "")).strip()
            target_lang_id = str(branch.get("target_lang_id", "")).strip()
            branch_query_text = str(branch.get("query", "")).strip()
            anchor_retained = bool(branch.get("anchor_retained", False))
            topic_anchor = str(branch.get("topic_anchor", "") or "")
            if not branch_id or not target_label or not target_lang_id or not branch_query_text:
                branch_retrieval_failed = True
                continue

            branch_meta = _build_finalized_branch_metadata(
                branch_query_text=branch_query_text,
                target_label=target_label,
                target_lang_id=target_lang_id,
                metadata_store=metadata_store,
                phenomenon_lexicon=phenomenon_lexicon,
                category_lexicon=category_lexicon,
                branch_id=branch_id,
                anchor_retained=anchor_retained,
                topic_anchor=topic_anchor,
            )
            branch_result_docs, branch_diag = retrieve_with_filter_fallback(
                retriever=retriever,
                query_text=branch_query_text,
                base_metadata=branch_meta,
                fixed_k=fixed_k,
                fixed_top_n=fixed_top_n,
                allow_no_filter=False,
            )
            local_reranker_active = bool(getattr(retriever, "use_reranker", False))

            branch_docs[branch_id] = list(branch_result_docs)
            branch_order.append(branch_id)
            branch_doc_counts[branch_id] = len(branch_result_docs)
            for doc in branch_result_docs:
                key = _doc_identity(doc)
                if key not in branch_origin_by_doc:
                    branch_origin_by_doc[key] = branch_id
            split_branch_queries.append(
                {
                    "branch_id": branch_id,
                    "branch_query_mode": "surface_preserving_materialized",
                    "branch_query_text": branch_query_text,
                    "target_label": target_label,
                    "target_lang_id": target_lang_id,
                }
            )
            split_branch_filters.append(
                {
                    "branch_id": branch_id,
                    "lang": target_lang_id,
                    "topics": branch_meta.get("topics", {}),
                    "target_label": target_label,
                    "target_lang_id": target_lang_id,
                }
            )
            branch_diag_rows.append(
                _build_split_branch_trace_row(
                    branch_id=branch_id,
                    target_label=target_label,
                    target_lang_id=target_lang_id,
                    branch_query_text=branch_query_text,
                    branch_query_mode="surface_preserving_materialized",
                    branch_meta=branch_meta,
                    branch_diag=branch_diag,
                    branch_result_docs=branch_result_docs,
                    local_reranker_active=local_reranker_active,
                    branch_anchor_retained=anchor_retained,
                    branch_topic_anchor=topic_anchor,
                )
            )
            if branch_diag.get("filter_stage_used") == "no_filter" or len(branch_result_docs) == 0:
                branch_retrieval_failed = True
    finally:
        retriever.use_reranker = original_use_reranker

    diagnostics["split_branch_doc_counts"] = branch_doc_counts
    diagnostics["split_branch_diagnostics"] = branch_diag_rows
    diagnostics["split_branch_queries"] = split_branch_queries
    diagnostics["split_branch_filters"] = split_branch_filters

    if branch_retrieval_failed or len(branch_order) != 2:
        diagnostics["split_applied"] = False
        diagnostics["split_dimension"] = "none"
        diagnostics["split_reason"] = "t71_branch_retrieval_failed"
        diagnostics["split_reverted_to_unsplit"] = True
        docs, diag_update = retrieve_with_filter_fallback(
            retriever=retriever,
            query_text=query_text,
            base_metadata=metadata,
            fixed_k=fixed_k,
            fixed_top_n=fixed_top_n,
        )
        diagnostics.update(diag_update)
        retrieval_elapsed_sec = perf_counter() - t_retrieve
        return metadata, docs, warnings, metadata_elapsed_sec, retrieval_elapsed_sec

    def _branch_origin_distribution(rows: Sequence[Document], limit: int = 10) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for doc in list(rows)[: max(0, int(limit))]:
            origin = branch_origin_by_doc.get(_doc_identity(doc), "unknown")
            counts[origin] = counts.get(origin, 0) + 1
        return counts

    merge_cap = max(
        fixed_top_n,
        min(
            int(fixed_top_n) * max(1, len(branch_order)),
            int(fixed_k) * max(1, len(branch_order)),
        ),
    )
    merged_docs = _merge_split_docs(
        branch_docs=branch_docs,
        branch_order=branch_order,
        merge_cap=merge_cap,
    )
    diagnostics["split_merge_cap"] = merge_cap
    diagnostics["split_merged_doc_count"] = len(merged_docs)
    diagnostics["split_merge_unique_docs"] = len({_doc_identity(doc) for doc in merged_docs})
    diagnostics["split_branch_candidate_k"] = int(fixed_k)
    diagnostics["split_branch_origin_pre_top10"] = _branch_origin_distribution(merged_docs, limit=10)
    diagnostics["split_merged_top5_pre_rerank"] = _top_docs_snapshot(merged_docs, limit=5)
    diagnostics["split_post_merge_candidate_pool_size"] = len(merged_docs)

    post_merge_rerank_mode = str(query_split_cfg.get("post_merge_rerank_mode", "global")).strip().lower()
    diagnostics["split_post_merge_rerank_mode"] = post_merge_rerank_mode
    if original_use_reranker and merged_docs and post_merge_rerank_mode != "none":
        docs = retriever.perform_rerank(query_text, merged_docs, top_n=fixed_top_n)
        diagnostics["split_global_rerank_applied"] = True
    else:
        docs = merged_docs[:fixed_top_n]
        diagnostics["split_global_rerank_applied"] = False

    diversity_guard_enabled = bool(query_split_cfg.get("language_diversity_guard_enabled", True))
    diagnostics["split_language_diversity_guard"] = {
        "applied": False,
        "min_per_branch": 0,
        "origin_distribution": {},
    }
    if diversity_guard_enabled and docs:
        docs, diversity_stats = _apply_language_diversity_guard(
            docs,
            branch_order=branch_order,
            branch_origin_by_doc=branch_origin_by_doc,
            final_top_n=fixed_top_n,
            min_per_branch=max(1, int(query_split_cfg.get("language_diversity_min_per_branch", 1) or 1)),
        )
        diagnostics["split_language_diversity_guard"] = diversity_stats

    final_origin = _branch_origin_distribution(docs, limit=10)
    diagnostics["split_branch_origin_final_top10"] = final_origin
    diagnostics["split_final_top5"] = _top_docs_snapshot(docs, limit=5)
    final_top10_origin_ranked = [
        branch_origin_by_doc.get(_doc_identity(doc), "unknown")
        for doc in list(docs)[:10]
    ]
    diagnostics["split_merge_trace"] = {
        "query_id": query_id,
        "branch_count": len(branch_order),
        "pre_merge_origin": branch_doc_counts,
        "final_top10_origin": final_top10_origin_ranked,
        "final_top10_docs": _top_docs_snapshot(docs, limit=10),
        "provenance_retained": len(set(final_top10_origin_ranked)) >= 2,
        "notes": str(diagnostics.get("split_reason", "")),
    }
    diagnostics["filter_stage_used"] = "split_branches"
    diagnostics["fallback_stage"] = 0
    diagnostics["doc_count"] = len(docs)
    diagnostics["zero_hit"] = len(docs) == 0
    diagnostics["filter_applied"] = True

    retrieval_elapsed_sec = perf_counter() - t_retrieve
    return metadata, docs, warnings, metadata_elapsed_sec, retrieval_elapsed_sec
