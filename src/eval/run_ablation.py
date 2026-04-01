#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import os
import random
import re
import sys
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from langchain_core.documents import Document


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agent.graph import app as rag_app
from src.eval.finalized_b7_runtime import (
    FINALIZED_B7_CONTROLLER_MODES,
    build_query_only_metadata,
    retrieve_with_filter_fallback,
    run_finalized_b7_retrieval,
)
from src.eval.materialized_language_split import (
    PROMPT_BUNDLE as MATERIALIZED_LANGUAGE_SPLIT_PROMPT_BUNDLE,
    materialize_language_split_plan,
    resolve_target_label_to_lang_id,
    validate_materialized_branch_queries,
)
from src.eval.metadata_store import MetadataStore, load_metadata_store
from src.eval.metrics import aggregate_macro_micro, build_gold_sets, compute_query_metrics
from src.eval.query_topic_detection import (
    detect_topics_from_query_with_scores as _shared_detect_topics_from_query_with_scores,
)
from src.eval.profiles import (
    FILTERING_MODE_CHOICES,
    FILTERING_MODE_DEFAULT,
    RERANK_K30_PROFILES,
    RERANK_MIN_TOP_N_FOR_K30,
    deep_merge,
    load_profile_file,
    profile_base_chain,
    resolve_profile_config,
    resolve_profile_overrides,
    select_profiles,
)
from src.retrieve.rag_retrieve import RAGRetriever, extract_query_metadata
from src.retrieve.viking_lexicon_loader import load_viking_lexicon
from src.utils import load_config, use_config_override


K_VALUES = (5, 10, 20, 30)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config_hash(cfg: Dict[str, Any]) -> str:
    payload = json.dumps(cfg, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_qaset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Query/QA file not found: {path}")

    def _as_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if "|" in text:
                return [part.strip() for part in text.split("|") if part.strip()]
            return [text]
        return []

    def _as_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
        return default

    def _normalize_query_item(obj: Dict[str, Any], fallback_query_id: str) -> Dict[str, Any]:
        query = obj.get("query") or obj.get("question") or obj.get("text")
        if not query:
            return {}

        expected_languages = _as_list(
            obj.get("expected_languages") or obj.get("languages") or obj.get("oracle_languages")
        )
        expected_parent_topics = _as_list(
            obj.get("expected_parent_topics") or obj.get("parent_topics")
        )
        expected_child_topics = _as_list(
            obj.get("expected_child_topics") or obj.get("child_topics")
        )
        expected_topics = _as_list(obj.get("expected_topics"))
        expected_families = _as_list(
            obj.get("expected_families") or obj.get("families")
        )
        expected_regions = _as_list(
            obj.get("expected_regions") or obj.get("regions")
        )
        split_axes = _as_list(obj.get("split_axes"))
        needs_split = _as_bool(obj.get("needs_split"), default=False)

        if not split_axes:
            query_type = str(obj.get("query_type") or "unknown")
            if query_type == "multi_lang_single_topic":
                split_axes = ["lang"]
                needs_split = True if obj.get("needs_split") is None else needs_split
            elif query_type == "single_lang_multi_topic":
                split_axes = ["topic"]
                needs_split = True if obj.get("needs_split") is None else needs_split
            elif query_type == "multi_lang_multi_topic":
                split_axes = ["lang_topic"]
                needs_split = True if obj.get("needs_split") is None else needs_split

        return {
            "query_id": str(obj.get("query_id") or obj.get("id") or fallback_query_id),
            "query_type": str(obj.get("query_type") or "unknown"),
            "query": str(query),
            "expected_languages": expected_languages,
            "expected_topics": expected_topics,
            "expected_parent_topics": expected_parent_topics,
            "expected_child_topics": expected_child_topics,
            "expected_families": expected_families,
            "expected_regions": expected_regions,
            "needs_split": needs_split,
            "split_axes": split_axes,
            "gold_docs": obj.get("gold_docs") or [],
            "gold_chunks": obj.get("gold_chunks") or [],
            "eval_split": str(obj.get("eval_split") or obj.get("split") or "").strip().lower(),
            "raw": obj,
        }

    rows: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, str):
                    rows.append(
                        {
                            "query_id": f"Q{line_no:04d}",
                            "query_type": "unknown",
                            "query": obj,
                            "expected_languages": [],
                            "expected_topics": [],
                            "expected_parent_topics": [],
                            "expected_child_topics": [],
                            "expected_families": [],
                            "expected_regions": [],
                            "needs_split": False,
                            "split_axes": [],
                            "gold_docs": [],
                            "gold_chunks": [],
                            "eval_split": "",
                            "raw": {},
                        }
                    )
                    continue

                normalized = _normalize_query_item(obj, fallback_query_id=f"Q{line_no:04d}")
                if normalized:
                    rows.append(normalized)
        return rows

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("JSON input must be a list of QA objects or {'items': [...]}.")

    for i, obj in enumerate(items, start=1):
        if not isinstance(obj, dict):
            continue
        normalized = _normalize_query_item(obj, fallback_query_id=f"Q{i:04d}")
        if normalized:
            rows.append(normalized)

    return rows


def _doc_key(meta: Dict[str, Any], rank: int) -> str:
    source_file = meta.get("source_file")
    chunk_id = meta.get("chunk_id")
    if source_file is not None and chunk_id is not None:
        return f"{source_file}:{chunk_id}"
    if source_file is not None:
        return f"{source_file}:?"
    ref_id = meta.get("ref_id")
    if ref_id:
        return str(ref_id)
    db_id = meta.get("_db_id")
    if db_id is not None:
        return f"db:{db_id}"
    return f"rank:{rank}"


def _doc_identity(doc: Document) -> Tuple[str, str]:
    source_file = str(doc.metadata.get("source_file", "")).strip()
    chunk_id = doc.metadata.get("chunk_id")
    if source_file:
        return source_file, str(chunk_id) if chunk_id is not None else ""
    ref_id = str(doc.metadata.get("ref_id", "")).strip()
    if ref_id:
        return "ref", ref_id
    db_id = doc.metadata.get("_db_id")
    if db_id is not None:
        return "db", str(db_id)
    return "obj", str(id(doc))


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
    branch_order: List[str],
    merge_cap: int,
) -> List[Document]:
    prepared: Dict[str, List[Document]] = {}
    for branch_id in branch_order:
        docs = list(branch_docs.get(branch_id, []))
        docs.sort(key=_doc_score, reverse=True)
        prepared[branch_id] = docs

    merged: List[Document] = []
    seen = set()
    cursor = {branch_id: 0 for branch_id in branch_order}
    cap = max(0, int(merge_cap))
    if cap <= 0:
        return []

    while len(merged) < cap:
        progressed = False
        for branch_id in branch_order:
            docs = prepared.get(branch_id, [])
            idx = cursor[branch_id]
            while idx < len(docs):
                candidate = docs[idx]
                idx += 1
                key = _doc_identity(candidate)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(candidate)
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


def _documents_to_ranked_rows(docs: Sequence[Document]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rank, doc in enumerate(docs, start=1):
        meta = deepcopy(doc.metadata or {})
        rows.append(
            {
                "rank": rank,
                "doc_key": _doc_key(meta, rank),
                "db_id": meta.get("_db_id"),
                "source_file": meta.get("source_file"),
                "chunk_id": meta.get("chunk_id"),
                "level": meta.get("level"),
                "lang": meta.get("lang"),
                "ref_id": meta.get("ref_id"),
                "score": meta.get("score"),
                "vector_rank": meta.get("_vector_rank"),
            }
        )
    return rows


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


def _normalize_alias(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    raw = raw.replace("/", " ").replace("\\", " ")
    raw = re.sub(r"[\"'`’“”·•]", "", raw)
    raw = re.sub(r"[()\[\]{}.,:;!?]", " ", raw)
    raw = re.sub(r"[\s\-_]+", " ", raw).strip()
    return raw


def _compact_alias(value: str) -> str:
    return re.sub(r"[\s\-_]+", "", str(value or ""))


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


def _strip_language_suffix(token: str) -> str:
    text = str(token or "").strip()
    if text.endswith("언어") and len(text) > 2:
        return text[:-2]
    if text.endswith("어") and len(text) > 1:
        return text[:-1]
    if text.endswith(" language") and len(text) > 9:
        return text[:-9].strip()
    return text


_HANGUL_ONSET = [
    "g",
    "kk",
    "n",
    "d",
    "tt",
    "r",
    "m",
    "b",
    "pp",
    "s",
    "ss",
    "",
    "j",
    "jj",
    "ch",
    "k",
    "t",
    "p",
    "h",
]
_HANGUL_VOWEL = [
    "a",
    "ae",
    "ya",
    "yae",
    "eo",
    "e",
    "yeo",
    "ye",
    "o",
    "wa",
    "wae",
    "oe",
    "yo",
    "u",
    "wo",
    "we",
    "wi",
    "yu",
    "eu",
    "ui",
    "i",
]
_HANGUL_CODA = [
    "",
    "k",
    "k",
    "ks",
    "n",
    "nj",
    "nh",
    "t",
    "l",
    "lk",
    "lm",
    "lb",
    "ls",
    "lt",
    "lp",
    "lh",
    "m",
    "p",
    "ps",
    "t",
    "t",
    "ng",
    "t",
    "t",
    "k",
    "t",
    "p",
    "h",
]


def _romanize_hangul(text: str) -> str:
    s = str(text or "")
    out: list[str] = []
    for ch in s:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            idx = code - 0xAC00
            onset = idx // 588
            vowel = (idx % 588) // 28
            coda = idx % 28
            out.append(_HANGUL_ONSET[onset] + _HANGUL_VOWEL[vowel] + _HANGUL_CODA[coda])
        elif ch.isalnum():
            out.append(ch.lower())
    return "".join(out)


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
) -> dict[str, Any]:
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
) -> dict[str, Any]:
    return _shared_detect_topics_from_query_with_scores(
        query_text=query_text,
        phenomenon_lexicon=phenomenon_lexicon,
        category_lexicon=category_lexicon,
        metadata_store=metadata_store,
    )


def _build_query_only_metadata(
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


def _retrieve_with_filter_fallback(
    *,
    retriever: RAGRetriever,
    query_text: str,
    base_metadata: Dict[str, Any],
    fixed_k: int,
    fixed_top_n: int,
    allow_no_filter: bool = True,
) -> tuple[list[Document], Dict[str, Any]]:
    diagnostics = base_metadata.get("_diagnostics", {}) if isinstance(base_metadata.get("_diagnostics"), dict) else {}
    parent_topics = _to_list((base_metadata.get("topics") or {}).get("categories"))
    child_topics = _to_list((base_metadata.get("topics") or {}).get("phenomena"))
    languages = _to_list(base_metadata.get("lang"))
    conf = diagnostics.get("detector_confidence", {}) if isinstance(diagnostics.get("detector_confidence"), dict) else {}

    stages: list[str] = []
    if retriever.enable_child_topic_filter and retriever.enable_language_filter and child_topics and languages and conf.get("child_topic") == "high":
        stages.append("child_lang")
    if retriever.enable_parent_topic_filter and retriever.enable_language_filter and parent_topics and languages and conf.get("parent_topic") in {"medium", "high"}:
        stages.append("parent_lang")
    if retriever.enable_language_filter and languages and conf.get("language") in {"medium", "high"}:
        stages.append("lang_only")
    if allow_no_filter:
        stages.append("no_filter")
    deduped_stages: list[str] = []
    for stage in stages:
        if stage not in deduped_stages:
            deduped_stages.append(stage)

    attempted: list[dict[str, Any]] = []
    final_docs: list[Document] = []
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
        selected_stage = deduped_stages[-1]
        fallback_index = len(deduped_stages) - 1

    diagnostics["filter_stage_used"] = selected_stage
    diagnostics["fallback_stage"] = fallback_index
    diagnostics["filter_stage_attempts"] = attempted
    diagnostics["doc_count"] = len(final_docs)
    diagnostics["zero_hit"] = len(final_docs) == 0
    diagnostics["filter_applied"] = selected_stage != "no_filter"

    return final_docs, diagnostics


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
    branch_soft_bias_stats: Dict[str, Any],
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
        "soft_child_topic_bias": branch_soft_bias_stats,
        "top5": _top_docs_snapshot(branch_result_docs, limit=5),
        "top_docs": _top_docs_snapshot(branch_result_docs, limit=5),
        "top_chunks": _top_chunks_snapshot(branch_result_docs, limit=min(10, len(branch_result_docs))),
    }


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


def _derive_group_label(query_id: str, query_type: str) -> str:
    prefix = str(query_id).split("_", 1)[0].strip().upper()
    if prefix in {"T1", "T2", "T3", "T4", "T5", "T6"}:
        return prefix
    mapping = {
        "single_lang_single_topic": "T1",
        "multi_lang_single_topic": "T2",
        "single_lang_multi_topic": "T3",
    }
    return mapping.get(str(query_type or ""), "UNKNOWN")


def _row_group_label(row: Dict[str, Any]) -> str:
    return _derive_group_label(
        str(row.get("query_id", "")),
        str(row.get("query_type", "unknown")),
    ) if not str(row.get("group_label", "") or "").strip() else str(row.get("group_label", "")).strip().upper()


def _build_split_branch_metadata(
    *,
    base_metadata: Dict[str, Any],
    axis: str,
    value: str,
    child_hard_filter: bool = True,
    parent_topic_for_child_soft: str = "",
    branch_language_source: str = "",
    branch_language_fragment: str = "",
    branch_language_anchor_span: tuple[int, int] | None = None,
) -> Dict[str, Any]:
    branch_meta = deepcopy(base_metadata)
    branch_diag = branch_meta.get("_diagnostics")
    if not isinstance(branch_diag, dict):
        branch_diag = {}
        branch_meta["_diagnostics"] = branch_diag

    topics = branch_meta.get("topics")
    if not isinstance(topics, dict):
        topics = {"categories": [], "phenomena": []}
        branch_meta["topics"] = topics
    topics.setdefault("categories", [])
    topics.setdefault("phenomena", [])

    if axis == "language":
        branch_meta["lang"] = value
        branch_diag["detected_languages"] = [value]
        branch_diag["detected_languages_high"] = [value]
        branch_diag["detected_languages_medium"] = []
        branch_diag["detected_languages_low"] = []
        branch_diag["language_confidences"] = {str(value): "high"}
        branch_diag["language_conflict"] = False
        detector_conf = branch_diag.get("detector_confidence")
        if not isinstance(detector_conf, dict):
            detector_conf = {}
        detector_conf["language"] = "high"
        branch_diag["detector_confidence"] = detector_conf
        branch_diag["branch_effective_language"] = value
        branch_diag["branch_language_source"] = branch_language_source or "high"
        if branch_language_fragment:
            branch_diag["branch_language_fragment"] = branch_language_fragment
        if branch_language_anchor_span is not None:
            branch_diag["branch_language_anchor_span"] = list(branch_language_anchor_span)
    elif axis == "child_topic":
        if child_hard_filter:
            topics["phenomena"] = [value]
            topics["categories"] = []
        else:
            topics["phenomena"] = []
            if parent_topic_for_child_soft:
                topics["categories"] = [parent_topic_for_child_soft]
            branch_diag["soft_child_topic_bias"] = [value]
    elif axis == "parent_topic":
        topics["categories"] = [value]
        topics["phenomena"] = []
    return branch_meta


def _lang_display_name(lang_id: str, metadata_store: MetadataStore | None) -> str:
    lang = str(lang_id or "").strip()
    if not lang:
        return ""
    if not metadata_store:
        return lang
    row = metadata_store.lang_rows.get(lang, {})
    ko = str(row.get("core_name_ko", "")).strip()
    en = str(row.get("core_name_en", "")).strip()
    if ko:
        return ko
    if en:
        return en
    return lang


def _topic_label_map(
    *,
    phenomenon_lexicon: Dict[str, List[str]],
    category_lexicon: Dict[str, str],
) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for term, cat_id in category_lexicon.items():
        cid = str(cat_id).strip()
        t = str(term).strip()
        if cid and t and cid not in labels:
            labels[cid] = t
    for term, child_ids in phenomenon_lexicon.items():
        t = str(term).strip()
        if not t:
            continue
        for child_id in child_ids:
            cid = str(child_id).strip()
            if cid and cid not in labels:
                labels[cid] = t
    return labels


def _topic_terms_map(
    *,
    phenomenon_lexicon: Dict[str, List[str]],
    category_lexicon: Dict[str, str],
) -> Dict[str, set[str]]:
    terms: Dict[str, set[str]] = {}
    for term, cat_id in category_lexicon.items():
        cid = str(cat_id).strip()
        t = _normalize_query_text(term)
        if cid and t:
            terms.setdefault(cid, set()).add(t)
    for term, child_ids in phenomenon_lexicon.items():
        t = _normalize_query_text(term)
        if not t:
            continue
        for child_id in child_ids:
            cid = str(child_id).strip()
            if cid:
                terms.setdefault(cid, set()).add(t)
    return terms


def _query_text_without_compare_markers(text: str) -> str:
    out = str(text or "").strip()
    if not out:
        return out
    replacements = [
        ("비교해 주세요", "설명해 주세요"),
        ("비교해주", "설명해주"),
        ("비교해", "설명해"),
        ("비교하여", "설명하여"),
        ("비교", "설명"),
        ("각각", ""),
    ]
    for src, dst in replacements:
        out = out.replace(src, dst)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _has_split_cue(text: str) -> bool:
    q = str(text or "").lower()
    if not q:
        return False
    cues = [
        "비교",
        "각각",
        "및",
        "그리고",
        "연관",
        "함께",
        "와 ",
        "과 ",
        " and ",
        ",",
    ]
    return any(cue in q for cue in cues)


def _has_coordination_cue(text: str) -> bool:
    q = _normalize_query_text(text)
    if not q:
        return False
    return bool(re.search(r"\b(and|및)\b|[가-힣a-z0-9_]+\s*(와|과|/|,)\s*[가-힣a-z0-9_]+", q))


def _topic_parent_from_child(child_topic: str, metadata_store: MetadataStore | None) -> str:
    child = str(child_topic or "").strip()
    if not child or not metadata_store:
        return ""
    parent_hits: Dict[str, int] = {}
    for pairs in metadata_store.topic_pairs_by_lang.values():
        for pair in pairs:
            text = str(pair or "").strip()
            if "/" not in text:
                continue
            parent, child_id = text.split("/", 1)
            parent = str(parent).strip()
            child_id = str(child_id).strip()
            if not parent or not child_id:
                continue
            if child_id == child:
                parent_hits[parent] = parent_hits.get(parent, 0) + 1
    if not parent_hits:
        return ""
    return sorted(parent_hits.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _topic_candidate_allowed_for_split(
    *,
    topic_id: str,
    topic_label: str,
    query_text: str,
) -> bool:
    label = _normalize_query_text(topic_label)
    q = _normalize_query_text(query_text)
    if not q:
        return False
    generic_terms = {
        "비교하다",
        "비교",
        "설명하다",
        "설명",
        "의미하다",
        "의미",
        "구조",
        "차이",
        "관련",
        "성격",
        "특징",
    }
    if label in generic_terms:
        return False
    # Prevent clause-topic false positives from syllable prompts ("음절" != "절 clause").
    if str(topic_id) == "41_cl" or label in {"절", "clause"}:
        clause_pat = re.compile(r"(?:^|\\s)(절\\s*구조|종속절|주절|절\\s*유형|문장\\s*절|clause)(?:\\s|$)")
        if "음절" in q and not clause_pat.search(q):
            return False
    return True


def _has_topic_pair_cue(
    *,
    query_text: str,
    topic_candidates: Sequence[Tuple[str, str]],
    topic_labels: Dict[str, str],
    topic_terms: Dict[str, set[str]] | None = None,
) -> bool:
    q = _normalize_query_text(query_text)
    if not q:
        return False
    if not _has_coordination_cue(q):
        return False

    topic_terms = topic_terms or {}

    def _topic_mentioned(topic_id: str) -> bool:
        tid = str(topic_id).strip()
        if not tid:
            return False
        if _contains_term(q, tid.lower()):
            return True
        label = _normalize_query_text(topic_labels.get(tid, ""))
        if label and _contains_term(q, label):
            return True
        for term in sorted(topic_terms.get(tid, set())):
            if not term:
                continue
            if _contains_term(q, term):
                return True
        return False

    mentions = 0
    seen: set[str] = set()
    for _axis, topic_id in topic_candidates:
        tid = str(topic_id).strip()
        if not tid or tid in seen:
            continue
        seen.add(tid)
        if _topic_mentioned(tid):
            mentions += 1
    return mentions >= 2


def _topic_pair_has_distinct_fragments(
    *,
    query_text: str,
    topic_candidates: Sequence[Tuple[str, str]],
    topic_labels: Dict[str, str],
    topic_terms: Dict[str, set[str]] | None = None,
) -> bool:
    fragments = _parallel_fragments_with_spans(query_text)
    if len(fragments) < 2:
        return False

    topic_terms = topic_terms or {}

    def _fragment_mentions(fragment_text: str, topic_id: str) -> bool:
        q = _normalize_query_text(fragment_text)
        tid = str(topic_id).strip()
        if not q or not tid:
            return False
        if _contains_term(q, tid.lower()):
            return True
        label = _normalize_query_text(topic_labels.get(tid, ""))
        if label and _contains_term(q, label):
            return True
        for term in sorted(topic_terms.get(tid, set())):
            if term and _contains_term(q, term):
                return True
        return False

    matched_fragments: set[int] = set()
    matched_topics: set[str] = set()
    for frag in fragments:
        frag_text = str(frag.get("text", "")).strip()
        if not frag_text:
            continue
        found_topic = ""
        for _axis, topic_id in topic_candidates:
            if _fragment_mentions(frag_text, str(topic_id)):
                found_topic = str(topic_id)
                break
        if found_topic:
            matched_fragments.add(int(frag.get("index", 0)))
            matched_topics.add(found_topic)
    return len(matched_fragments) >= 2 and len(matched_topics) >= 2


def _parallel_fragments_with_spans(text: str) -> list[dict[str, Any]]:
    q = _normalize_query_text(text)
    if not q:
        return []

    connectors = re.compile(r"\s*(와|과|및|and|/|,)\s*")
    fragments: list[dict[str, Any]] = []
    cursor = 0
    last_end = 0
    for match in connectors.finditer(q):
        left = q[last_end:match.start()].strip()
        if left:
            fragments.append(
                {
                    "text": left,
                    "connector_after": match.group(1),
                    "span": (last_end, match.start()),
                    "index": cursor,
                }
            )
            cursor += 1
        last_end = match.end()

    tail = q[last_end:].strip()
    if tail:
        fragments.append(
            {
                "text": tail,
                "connector_after": "",
                "span": (last_end, len(q)),
                "index": cursor,
            }
        )
    return fragments


def _coordination_fragments(text: str) -> list[str]:
    def _last_token(value: str) -> str:
        toks = re.findall(r"[0-9a-z가-힣_-]+", value)
        if not toks:
            return ""
        return str(toks[-1]).strip()

    def _first_token(value: str) -> str:
        toks = re.findall(r"[0-9a-z가-힣_-]+", value)
        if not toks:
            return ""
        return str(toks[0]).strip()

    def _clean_token(value: str) -> str:
        token = str(value).strip()
        if not token:
            return ""
        token = _strip_korean_particles(token)
        token = _strip_language_suffix(token)
        if token and re.search(r"[가-힣]", token):
            token = f"{token}어" if not token.endswith("어") else token
        return token.strip()

    raw_parts = _parallel_fragments_with_spans(text)
    out: list[str] = []
    for idx, part in enumerate(raw_parts):
        p = str(part.get("text", "")).strip()
        if not p:
            continue
        tok = _last_token(p) if idx == 0 else _first_token(p)
        cleaned = _clean_token(tok)
        if cleaned:
            out.append(cleaned)
    return out


def _language_confidence_from_fragment_score(score: float) -> str:
    if score >= 2.75:
        return "high"
    if score >= 1.4:
        return "medium"
    return "low"


def _is_contextual_language_fragment(
    *,
    query_text: str,
    fragment_text: str,
    anchor_span: tuple[int, int] | None,
) -> bool:
    q = _normalize_query_text(query_text)
    frag = _normalize_query_text(fragment_text)
    if not q or not frag:
        return False

    donor_markers = [
        "차용",
        "차용어",
        "외래",
        "외래어",
        "기원",
        "영향",
        "계통",
        "유래",
        "borrow",
        "borrowed",
        "loanword",
        "source language",
        "donor language",
        "from ",
        "origin",
    ]
    if not any(marker in q for marker in donor_markers):
        return False

    start = max(0, int(anchor_span[0]) if anchor_span else 0)
    end = min(len(q), int(anchor_span[1]) if anchor_span else len(q))
    left = q[max(0, start - 24):start]
    right = q[end:min(len(q), end + 24)]
    window = f"{left} {frag} {right}".strip()
    patterns = [
        rf"{re.escape(frag)}[^.]*?(차용|외래|영향|기원|유래|borrow|source language|donor language|origin)",
        rf"(차용|외래|영향|기원|유래|borrow|source language|donor language|origin)[^.]*?{re.escape(frag)}",
        rf"from\s+{re.escape(frag)}",
        rf"{re.escape(frag)}\s*(에서|로부터)\s*(차용|유래|영향)",
        rf"{re.escape(frag)}\s*의\s*(영향|기원|계통)",
    ]
    return any(re.search(pattern, window) for pattern in patterns)


def _coordination_language_evidence(
    *,
    query_text: str,
    metadata_store: MetadataStore | None,
    max_branches: int,
    allow_medium: bool,
    allow_risk: bool = False,
) -> dict[str, Any]:
    fragments = _parallel_fragments_with_spans(query_text)
    if len(fragments) < 2:
        return {
            "evidence": [],
            "suppressed_contextual": [],
            "suppressed_risk_only": [],
        }

    evidence_rows: list[dict[str, Any]] = []
    contextual_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    seen_fragment_lang: set[tuple[int, str]] = set()
    for frag in fragments[: max(2, max_branches * 2)]:
        frag_text = str(frag.get("text", "")).strip()
        if not frag_text:
            continue
        cleaned = ""
        toks = re.findall(r"[0-9a-z가-힣_-]+", frag_text)
        if toks:
            token = toks[-1] if frag.get("index", 0) == 0 else toks[0]
            cleaned = _strip_korean_particles(_strip_language_suffix(token))
            if cleaned and re.search(r"[가-힣]", cleaned) and not cleaned.endswith("어"):
                cleaned = f"{cleaned}어"
        candidate_text = cleaned or frag_text
        resolved = _resolve_language_from_fragment(candidate_text, metadata_store)
        if not resolved:
            continue
        lang, score = resolved
        confidence = _language_confidence_from_fragment_score(float(score))
        if confidence == "low":
            continue
        if confidence == "medium" and not allow_medium:
            continue
        if _is_contextual_language_fragment(
            query_text=query_text,
            fragment_text=candidate_text,
            anchor_span=frag.get("span"),
        ):
            contextual_rows.append(
                {
                    "lang": str(lang),
                    "score": float(score),
                    "confidence": confidence,
                    "fragment": candidate_text,
                    "fragment_raw": frag_text,
                    "fragment_index": int(frag.get("index", 0)),
                    "anchor_span": tuple(frag.get("span", (0, 0))),
                }
            )
            continue
        entries = (metadata_store.alias_index.get(_normalize_alias(candidate_text), []) if metadata_store else [])
        risk_only = bool(entries) and all(bool(e.get("risk", False)) for e in entries)
        if risk_only and not allow_risk:
            risk_rows.append(
                {
                    "lang": str(lang),
                    "score": float(score),
                    "confidence": confidence,
                    "fragment": candidate_text,
                    "fragment_raw": frag_text,
                    "fragment_index": int(frag.get("index", 0)),
                    "anchor_span": tuple(frag.get("span", (0, 0))),
                }
            )
            continue
        dedupe_key = (int(frag.get("index", 0)), str(lang))
        if dedupe_key in seen_fragment_lang:
            continue
        seen_fragment_lang.add(dedupe_key)
        evidence_rows.append(
            {
                "lang": str(lang),
                "score": float(score),
                "confidence": confidence,
                "fragment": candidate_text,
                "fragment_raw": frag_text,
                "fragment_index": int(frag.get("index", 0)),
                "anchor_span": tuple(frag.get("span", (0, 0))),
                "risk_only": risk_only,
            }
        )

    # Keep only the strongest evidence per language, but preserve fragment identity.
    best_by_lang: dict[str, dict[str, Any]] = {}
    for row in evidence_rows:
        lang = str(row.get("lang", "")).strip()
        if not lang:
            continue
        prev = best_by_lang.get(lang)
        if prev is None or float(row.get("score", 0.0)) > float(prev.get("score", 0.0)):
            best_by_lang[lang] = row
    ordered = sorted(
        best_by_lang.values(),
        key=lambda item: (-float(item.get("score", 0.0)), str(item.get("lang", ""))),
    )
    return {
        "evidence": ordered[:max_branches],
        "suppressed_contextual": contextual_rows,
        "suppressed_risk_only": risk_rows,
    }


def _resolve_language_from_fragment(
    fragment: str,
    metadata_store: MetadataStore | None,
) -> tuple[str, float] | None:
    if not metadata_store:
        return None
    frag = _normalize_alias(fragment)
    if not frag:
        return None
    frag_compact = _compact_alias(frag)
    best: tuple[str, float] | None = None
    scored_by_lang: Dict[str, float] = {}

    def _update(lang: str, score: float) -> None:
        if not lang:
            return
        prev = scored_by_lang.get(lang)
        if prev is None or score > prev:
            scored_by_lang[lang] = score

    for alias, entries in metadata_store.alias_index.items():
        alias_norm = _normalize_alias(alias)
        if not alias_norm:
            continue
        alias_compact = _compact_alias(alias_norm)
        matched = False
        if alias_norm == frag:
            matched = True
            base = 3.0
        elif alias_norm in frag or frag in alias_norm:
            matched = True
            base = 2.0
        elif len(alias_compact) >= 3 and (alias_compact == frag_compact or alias_compact in frag_compact):
            matched = True
            base = 1.5
        else:
            base = 0.0
        if not matched:
            continue
        for entry in entries:
            lang = str(entry.get("lang_id", "")).strip()
            if not lang:
                continue
            score = base
            if bool(entry.get("risk", False)):
                score -= 1.0
            if score <= 0:
                continue
            _update(lang, score)

    # Conservative fuzzy fallback for Korean transliteration drift.
    if not scored_by_lang and re.search(r"[가-힣]", frag):
        frag_stem = _strip_language_suffix(_strip_korean_particles(frag))
        frag_stem = _compact_alias(frag_stem)
        if len(frag_stem) >= 2:
            for alias, entries in metadata_store.alias_index.items():
                alias_norm = _normalize_alias(alias)
                if not alias_norm or not re.search(r"[가-힣]", alias_norm):
                    continue
                alias_stem = _strip_language_suffix(_strip_korean_particles(alias_norm))
                alias_stem = _compact_alias(alias_stem)
                if len(alias_stem) < 2:
                    continue
                ratio = difflib.SequenceMatcher(None, frag_stem, alias_stem).ratio()
                if ratio < 0.72:
                    continue
                base = 1.0 + (ratio - 0.72) * 2.0
                for entry in entries:
                    lang = str(entry.get("lang_id", "")).strip()
                    if not lang:
                        continue
                    score = base
                    if bool(entry.get("risk", False)):
                        score -= 0.8
                    if score <= 0:
                        continue
                    _update(lang, score)

    # Hangul -> latin fuzzy fallback against non-Hangul aliases.
    if not scored_by_lang and re.search(r"[가-힣]", frag):
        frag_stem = _strip_language_suffix(_strip_korean_particles(frag))
        frag_rom = _compact_alias(_romanize_hangul(frag_stem))
        if len(frag_rom) >= 3:
            for alias, entries in metadata_store.alias_index.items():
                alias_norm = _normalize_alias(alias)
                if not alias_norm or re.search(r"[가-힣]", alias_norm):
                    continue
                alias_stem = _strip_language_suffix(alias_norm)
                alias_rom = _compact_alias(alias_stem)
                if len(alias_rom) < 3:
                    continue
                ratio = difflib.SequenceMatcher(None, frag_rom, alias_rom).ratio()
                if ratio < 0.74:
                    continue
                base = 0.95 + (ratio - 0.74) * 1.8
                for entry in entries:
                    lang = str(entry.get("lang_id", "")).strip()
                    if not lang:
                        continue
                    score = base
                    if bool(entry.get("risk", False)):
                        score -= 0.8
                    if score <= 0:
                        continue
                    _update(lang, score)

    if not scored_by_lang:
        return best
    ordered = sorted(scored_by_lang.items(), key=lambda kv: (-kv[1], kv[0]))
    top_lang, top_score = ordered[0]
    if len(ordered) > 1:
        second_score = ordered[1][1]
        if (top_score - second_score) < 0.25:
            return None
    best = (top_lang, float(top_score))
    return best


def _coordination_language_candidates(
    *,
    query_text: str,
    metadata_store: MetadataStore | None,
    max_branches: int,
) -> list[str]:
    frags = _coordination_fragments(query_text)
    if len(frags) < 2:
        return []
    resolved: list[tuple[str, float]] = []
    for frag in frags[:6]:
        r = _resolve_language_from_fragment(frag, metadata_store)
        if r:
            resolved.append(r)
    if not resolved:
        return []
    uniq: dict[str, float] = {}
    for lang, score in resolved:
        uniq[lang] = max(score, uniq.get(lang, 0.0))
    ordered = sorted(uniq.items(), key=lambda kv: (-kv[1], kv[0]))
    langs = [lang for lang, _ in ordered]
    if len(langs) < 2:
        return []
    return langs[:max_branches]


def _build_branch_query_text(
    *,
    query_text: str,
    split_axis: str,
    branch_value: str,
    branch_mode: str,
    metadata_store: MetadataStore | None,
    topic_labels: Dict[str, str],
) -> str:
    original = str(query_text or "").strip()
    mode = str(branch_mode or "preserve_tag").strip().lower()
    if not original:
        return original

    if split_axis == "language":
        label = _lang_display_name(branch_value, metadata_store) or branch_value
        if mode == "focused":
            focused = _query_text_without_compare_markers(original)
            focused = re.sub(
                r"[A-Za-z0-9가-힣_-]+\s*(와|과|및|and|&)\s*[A-Za-z0-9가-힣_-]+",
                label,
                focused,
                count=1,
            )
            return f"{label} 관련 질문: {focused}"
        return f'{original} [branch_scope: lang={branch_value}; label="{label}"]'

    if split_axis == "topic":
        topic_id = str(branch_value).strip()
        topic_label = topic_labels.get(topic_id) or topic_id
        if mode == "focused":
            focused = _query_text_without_compare_markers(original)
            return f"{topic_label} 관련 질문: {focused}"
        return f'{original} [branch_scope: topic={topic_id}; label="{topic_label}"]'

    return original


def _plan_materialized_language_split(
    *,
    query_id: str,
    query_text: str,
    metadata: Dict[str, Any],
    metadata_store: MetadataStore | None,
    phenomenon_lexicon: Dict[str, List[str]],
    category_lexicon: Dict[str, str],
    request_timeout_sec: int,
    controller_mode: str,
) -> Dict[str, Any]:
    controller_mode_norm = str(controller_mode).strip().lower()
    recovered_frontdoor = controller_mode_norm == "t71_materialized_language_local_b6"
    plan = materialize_language_split_plan(
        query_id=query_id,
        query_text=query_text,
        request_timeout_sec=request_timeout_sec,
        metadata_store=metadata_store,
        split_mode="recovered" if recovered_frontdoor else "strict",
        renderer_mode="surface_preserving" if recovered_frontdoor else "template",
    )
    branches: List[Dict[str, Any]] = []
    split_branch_filters: List[Dict[str, Any]] = []
    if bool(plan.get("should_split", False)):
        for branch in plan.get("branch_queries", []):
            if not isinstance(branch, dict):
                continue
            branch_query_text = str(branch.get("query", "")).strip()
            target_label = str(branch.get("target_label", "")).strip()
            target_lang_id = str(branch.get("target_lang_id", "")).strip()
            branch_id = str(branch.get("branch_id", "")).strip() or f"lang={target_label}"
            if not branch_query_text or not target_label or not target_lang_id:
                continue
            branch_meta = _build_query_only_metadata(
                query_text=branch_query_text,
                metadata_store=metadata_store,
                phenomenon_lexicon=phenomenon_lexicon,
                category_lexicon=category_lexicon,
            )
            branch_diag = branch_meta.setdefault("_diagnostics", {})
            if not isinstance(branch_diag, dict):
                branch_diag = {}
                branch_meta["_diagnostics"] = branch_diag
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
            split_branch_filters.append(
                {
                    "branch_id": branch_id,
                    "lang": target_lang_id,
                    "topics": branch_meta.get("topics", {}),
                    "target_label": target_label,
                    "target_lang_id": target_lang_id,
                }
            )
            branches.append(
                {
                    "branch_id": branch_id,
                    "target_label": target_label,
                    "target_lang_id": target_lang_id,
                    "metadata": branch_meta,
                    "query_text": branch_query_text,
                    "branch_query_mode": "surface_preserving_materialized" if recovered_frontdoor else "deterministic_materialized",
                    "anchor_retained": bool(branch.get("anchor_retained", False)),
                    "topic_anchor": str(branch.get("topic_anchor", "") or ""),
                }
            )

    return {
        "split_applied": bool(plan.get("should_split", False)) and len(branches) == 2,
        "split_axis": "language" if bool(plan.get("should_split", False)) and len(branches) == 2 else "none",
        "split_reason": "materialized_language_split" if bool(plan.get("should_split", False)) and len(branches) == 2 else str(plan.get("abstain_reason", "materialized_abstain")),
        "branches": branches if len(branches) == 2 else [],
        "split_branch_filters": split_branch_filters if len(branches) == 2 else [],
        "branch_query_mode": "surface_preserving_materialized" if recovered_frontdoor else "deterministic_materialized",
        "controller_mode": controller_mode_norm,
        "split_materialized_plan": plan,
        "feedback_gating_enabled": False,
    }


def _doc_topic_ids(doc: Document) -> Tuple[str, str]:
    meta = doc.metadata or {}
    parent = str(meta.get("parent_topic") or meta.get("category_id") or "").strip()
    child = str(meta.get("child_topic") or meta.get("phenomenon_id") or "").strip()
    if parent and child:
        return parent, child
    source_file = str(meta.get("source_file", "")).strip()
    if source_file:
        parts = source_file.split("/")
        if len(parts) >= 5 and parts[0] == "doc":
            parent = parent or str(parts[2]).strip()
            child = child or str(parts[3]).strip()
    return parent, child


def _coverage_counts_for_languages(
    docs: Sequence[Document],
    candidates: Sequence[str],
) -> Dict[str, int]:
    counts = {str(lang): 0 for lang in candidates}
    for doc in docs:
        lang = str((doc.metadata or {}).get("lang", "")).strip()
        if lang in counts:
            counts[lang] += 1
    return counts


def _coverage_counts_for_topics(
    docs: Sequence[Document],
    candidates: Sequence[str],
) -> Dict[str, int]:
    counts = {str(topic): 0 for topic in candidates}
    for doc in docs:
        parent, child = _doc_topic_ids(doc)
        for topic in list({parent, child}):
            if topic in counts:
                counts[topic] += 1
    return counts


def _apply_soft_child_topic_bias(
    docs: Sequence[Document],
    *,
    target_child_topic: str,
    boost: float,
    stage: str,
) -> tuple[List[Document], Dict[str, Any]]:
    target = str(target_child_topic or "").strip()
    boost_value = float(boost or 0.0)
    if not target or boost_value <= 0.0:
        return list(docs), {
            "target_child_topic": target,
            "boost": boost_value,
            "matched_docs_count": 0,
            "applied": False,
            "stage": stage,
        }

    matched_docs_count = 0
    adjusted_docs: List[Document] = []
    for doc in docs:
        parent_topic, child_topic = _doc_topic_ids(doc)
        source_file = str((doc.metadata or {}).get("source_file", "")).strip()
        matched = child_topic == target or f"/{target}/" in source_file
        if matched:
            matched_docs_count += 1
            try:
                base_score = float(doc.metadata.get("score", 0.0) or 0.0)
            except Exception:
                base_score = 0.0
            doc.metadata["split_topic_bias_target"] = target
            doc.metadata["split_topic_bias_stage"] = stage
            doc.metadata["split_topic_bias_boost"] = boost_value
            doc.metadata["split_topic_bias_base_score"] = base_score
            doc.metadata["score"] = base_score + boost_value
            if "rerank_score" in doc.metadata:
                doc.metadata["split_topic_bias_base_rerank_score"] = doc.metadata.get("rerank_score")
                doc.metadata["split_topic_bias_adjusted_rerank_score"] = base_score + boost_value
        adjusted_docs.append(doc)

    adjusted_docs.sort(key=_doc_score, reverse=True)
    return adjusted_docs, {
        "target_child_topic": target,
        "boost": boost_value,
        "matched_docs_count": matched_docs_count,
        "applied": matched_docs_count > 0,
        "stage": stage,
    }


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


def _plan_query_only_split(
    *,
    query_id: str,
    query_text: str,
    metadata: Dict[str, Any],
    retrieval_cfg: Dict[str, Any],
    metadata_store: MetadataStore | None,
    phenomenon_lexicon: Dict[str, List[str]],
    category_lexicon: Dict[str, str],
    request_timeout_sec: int,
) -> Dict[str, Any]:
    query_split_cfg = retrieval_cfg.get("query_split", {}) if isinstance(retrieval_cfg.get("query_split"), dict) else {}
    if not bool(query_split_cfg.get("enabled", False)):
        return {
            "split_applied": False,
            "split_axis": "none",
            "split_reason": "disabled",
            "branches": [],
            "split_branch_filters": [],
            "branch_query_mode": "none",
        }

    controller_mode = str(query_split_cfg.get("controller_mode", "heuristic")).strip().lower()
    if controller_mode in {"materialized_language_only", "t7_materialized_language_local_b6", "t71_materialized_language_local_b6"}:
        return _plan_materialized_language_split(
            query_id=query_id,
            query_text=query_text,
            metadata=metadata,
            metadata_store=metadata_store,
            phenomenon_lexicon=phenomenon_lexicon,
            category_lexicon=category_lexicon,
            request_timeout_sec=request_timeout_sec,
            controller_mode=controller_mode,
        )

    diagnostics = metadata.get("_diagnostics", {}) if isinstance(metadata.get("_diagnostics"), dict) else {}
    require_high = bool(query_split_cfg.get("require_high_confidence", True))
    axis_mode = str(query_split_cfg.get("axis_mode", "auto")).strip().lower()
    enable_language_axis = bool(query_split_cfg.get("enable_language_axis", True))
    enable_topic_axis = bool(query_split_cfg.get("enable_topic_axis", True))
    if axis_mode == "language_only":
        enable_topic_axis = False
    elif axis_mode == "topic_only":
        enable_language_axis = False
    strict_single_axis = bool(query_split_cfg.get("strict_single_axis", True))
    max_branches = max(1, int(query_split_cfg.get("max_branches", 4)))
    branch_query_mode = str(query_split_cfg.get("branch_query_mode", "preserve_tag")).strip().lower()
    topic_child_hard_filter = bool(query_split_cfg.get("topic_child_hard_filter", False))
    topic_same_parent_only = bool(query_split_cfg.get("topic_same_parent_only", False))
    topic_require_pair_cue = bool(query_split_cfg.get("topic_require_pair_cue", False))
    topic_require_distinct_fragments = bool(query_split_cfg.get("topic_require_distinct_fragments", False))
    topic_apply_candidate_guard = bool(query_split_cfg.get("topic_apply_candidate_guard", True))
    topic_split_requires_language = bool(query_split_cfg.get("topic_split_requires_language", False))
    language_medium_requires_coordination = bool(query_split_cfg.get("language_medium_requires_coordination", True))
    topic_labels = _topic_label_map(
        phenomenon_lexicon=phenomenon_lexicon,
        category_lexicon=category_lexicon,
    )
    topic_terms = _topic_terms_map(
        phenomenon_lexicon=phenomenon_lexicon,
        category_lexicon=category_lexicon,
    )

    if require_high:
        high_languages = [str(v).strip() for v in diagnostics.get("detected_languages_high", []) if str(v).strip()]
        high_child_topics = [str(v).strip() for v in diagnostics.get("detected_topics_phenomena_high", []) if str(v).strip()]
        high_parent_topics = [str(v).strip() for v in diagnostics.get("detected_topics_categories_high", []) if str(v).strip()]
    else:
        high_languages = [str(v).strip() for v in diagnostics.get("detected_languages_candidates", []) if str(v).strip()]
        high_child_topics = [str(v).strip() for v in diagnostics.get("detected_topics_phenomena", []) if str(v).strip()]
        high_parent_topics = [str(v).strip() for v in diagnostics.get("detected_topics_categories", []) if str(v).strip()]
    risk_only_languages = {str(v).strip() for v in diagnostics.get("risk_only_languages", []) if str(v).strip()}
    language_conflict = bool(diagnostics.get("language_conflict", False))
    language_confidences = diagnostics.get("language_confidences", {})
    if not isinstance(language_confidences, dict):
        language_confidences = {}
    lang_detect_scores = diagnostics.get("lang_detect_scores", {})
    if not isinstance(lang_detect_scores, dict):
        lang_detect_scores = {}
    parent_topic_scores = diagnostics.get("topic_scores_parent", {})
    if not isinstance(parent_topic_scores, dict):
        parent_topic_scores = {}
    child_topic_scores = diagnostics.get("topic_scores_phenomena", {})
    if not isinstance(child_topic_scores, dict):
        child_topic_scores = {}

    branches: List[Dict[str, Any]] = []
    split_axis = "none"
    split_reason = "no_high_confidence_candidates"
    split_branch_filters: List[Dict[str, Any]] = []

    language_candidates = [lang for lang in high_languages if lang]
    language_branch_sources: Dict[str, str] = {lang: "high" for lang in language_candidates}
    language_branch_fragments: Dict[str, Dict[str, Any]] = {}
    language_split_allow_medium_with_cue = bool(query_split_cfg.get("language_split_allow_medium_with_cue", False))
    language_split_min_score = float(query_split_cfg.get("language_split_min_score", 2.0) or 2.0)
    has_coordination_cue = _has_coordination_cue(query_text)
    coordination_info = _coordination_language_evidence(
        query_text=query_text,
        metadata_store=metadata_store,
        max_branches=max_branches,
        allow_medium=language_split_allow_medium_with_cue,
    )
    coordination_evidence = [
        row for row in coordination_info.get("evidence", [])
        if float(row.get("score", 0.0)) >= language_split_min_score
    ]
    coordination_high = [
        str(row.get("lang"))
        for row in coordination_evidence
        if str(row.get("confidence")) == "high"
    ]
    coordination_candidates = [str(row.get("lang")) for row in coordination_evidence]
    coordination_contextual_mentions = coordination_info.get("suppressed_contextual", [])
    coordination_risk_only_mentions = coordination_info.get("suppressed_risk_only", [])
    language_has_query_cue = has_coordination_cue or _has_split_cue(query_text)
    language_split_ok = (
        enable_language_axis
        and len(language_candidates) >= 2
        and len(language_candidates) <= max_branches
        and not language_conflict
        and not all(lang in risk_only_languages for lang in language_candidates)
        and language_has_query_cue
    )
    if not language_split_ok and enable_language_axis and len(coordination_high) >= 2:
        candidates = [
            row for row in coordination_evidence
            if str(row.get("lang")) not in risk_only_languages and str(row.get("confidence")) == "high"
        ]
        if len(candidates) >= 2 and len(candidates) <= max_branches and not language_conflict:
            language_candidates = [str(row.get("lang")) for row in candidates]
            language_branch_sources = {
                str(row.get("lang")): "coordination"
                for row in candidates
            }
            language_branch_fragments = {
                str(row.get("lang")): row
                for row in candidates
            }
            language_split_ok = True
            split_reason = "coordination_parallel_language"
    if (
        not language_split_ok
        and enable_language_axis
        and language_split_allow_medium_with_cue
        and (has_coordination_cue if language_medium_requires_coordination else _has_split_cue(query_text))
    ):
        medium_rows = [
            row for row in coordination_evidence
            if str(row.get("confidence")) in {"medium", "high"}
            and str(row.get("lang")) not in risk_only_languages
        ]
        medium_candidates = [str(row.get("lang")) for row in medium_rows]
        has_high_anchor = any(str(row.get("confidence")) == "high" for row in medium_rows)
        if (
            len(medium_candidates) >= 2
            and len(medium_candidates) <= max_branches
            and not language_conflict
            and has_high_anchor
        ):
            language_candidates = medium_candidates
            language_branch_sources = {
                str(row.get("lang")): (
                    "high" if str(row.get("confidence")) == "high" else "medium_fragment"
                )
                for row in medium_rows
            }
            language_branch_fragments = {
                str(row.get("lang")): row
                for row in medium_rows
            }
            language_split_ok = True
            split_reason = "multi_lang_score_with_cue"
    if (
        not language_split_ok
        and enable_language_axis
        and len(language_candidates) >= 2
        and not language_has_query_cue
        and split_reason == "no_high_confidence_candidates"
    ):
        split_reason = "no_language_comparison_cue"
    if (
        not language_split_ok
        and coordination_contextual_mentions
        and len(coordination_contextual_mentions) >= 1
        and split_reason == "no_high_confidence_candidates"
    ):
        split_reason = "contextual_language_mentions_only"
    elif (
        not language_split_ok
        and coordination_risk_only_mentions
        and len(coordination_risk_only_mentions) >= 1
        and split_reason == "no_high_confidence_candidates"
    ):
        split_reason = "special_risk_alias_only"
    if language_split_ok:
        split_axis = "language"
        if split_reason == "no_high_confidence_candidates":
            split_reason = "multi_high_conf_language"
        for lang in language_candidates:
            branch_id = f"lang={lang}"
            branch_fragment = language_branch_fragments.get(lang, {})
            branch_meta = _build_split_branch_metadata(
                base_metadata=metadata,
                axis="language",
                value=lang,
                branch_language_source=language_branch_sources.get(lang, "high"),
                branch_language_fragment=str(branch_fragment.get("fragment", "")),
                branch_language_anchor_span=branch_fragment.get("anchor_span"),
            )
            branch_meta["_diagnostics"]["split_branch_id"] = branch_id
            branch_query_text = _build_branch_query_text(
                query_text=query_text,
                split_axis=split_axis,
                branch_value=lang,
                branch_mode=branch_query_mode,
                metadata_store=metadata_store,
                topic_labels=topic_labels,
            )
            split_branch_filters.append(
                {
                    "branch_id": branch_id,
                    "lang": lang,
                    "topics": branch_meta.get("topics", {}),
                    "branch_language_source": language_branch_sources.get(lang, "high"),
                    "branch_language_fragment": branch_fragment.get("fragment", ""),
                    "branch_language_anchor_span": list(branch_fragment.get("anchor_span", (0, 0))),
                }
            )
            branches.append(
                {
                    "branch_id": branch_id,
                    "metadata": branch_meta,
                    "query_text": branch_query_text,
                    "branch_query_mode": branch_query_mode,
                }
            )
    elif enable_topic_axis:
        topic_candidates: List[Tuple[str, str]] = []
        topic_split_min_score = float(query_split_cfg.get("topic_split_min_score", 2.0) or 2.0)
        topic_pair_cue_ok = has_coordination_cue
        topic_gate_blocked = topic_split_requires_language and len(language_candidates) == 0
        if topic_gate_blocked:
            split_reason = "topic_split_requires_language"
        elif len(high_child_topics) >= 2:
            topic_candidates = [("child_topic", topic) for topic in high_child_topics]
            split_axis = "topic"
            split_reason = "multi_high_conf_child_topic"
        elif len(high_parent_topics) >= 2:
            topic_candidates = [("parent_topic", topic) for topic in high_parent_topics]
            split_axis = "topic"
            split_reason = "multi_high_conf_parent_topic"
        elif _has_split_cue(query_text):
            scored_child = [
                (str(topic), float(score))
                for topic, score in child_topic_scores.items()
                if str(topic).strip()
            ]
            scored_parent = [
                (str(topic), float(score))
                for topic, score in parent_topic_scores.items()
                if str(topic).strip()
            ]
            child_candidates_by_score = [
                topic for topic, score in sorted(scored_child, key=lambda kv: (-kv[1], kv[0]))
                if score >= topic_split_min_score
            ]
            parent_candidates_by_score = [
                topic for topic, score in sorted(scored_parent, key=lambda kv: (-kv[1], kv[0]))
                if score >= topic_split_min_score
            ]
            if len(child_candidates_by_score) >= 2:
                topic_candidates = [("child_topic", topic) for topic in child_candidates_by_score]
                split_axis = "topic"
                split_reason = "multi_topic_score_child_with_cue"
            elif len(parent_candidates_by_score) >= 2:
                topic_candidates = [("parent_topic", topic) for topic in parent_candidates_by_score]
                split_axis = "topic"
                split_reason = "multi_topic_score_parent_with_cue"

        if topic_candidates and topic_apply_candidate_guard:
            filtered_candidates: List[Tuple[str, str]] = []
            for axis, topic in topic_candidates:
                label = topic_labels.get(topic, topic)
                if _topic_candidate_allowed_for_split(
                    topic_id=str(topic),
                    topic_label=str(label),
                    query_text=query_text,
                ):
                    filtered_candidates.append((axis, topic))
            if len(filtered_candidates) != len(topic_candidates):
                split_reason = "topic_candidates_filtered"
            topic_candidates = filtered_candidates

        if topic_candidates and topic_require_pair_cue:
            topic_pair_cue_ok = _has_topic_pair_cue(
                query_text=query_text,
                topic_candidates=topic_candidates,
                topic_labels=topic_labels,
                topic_terms=topic_terms,
            )
            if not topic_pair_cue_ok:
                split_reason = "topic_pair_cue_missing"
                topic_candidates = []
            elif topic_require_distinct_fragments and not _topic_pair_has_distinct_fragments(
                query_text=query_text,
                topic_candidates=topic_candidates,
                topic_labels=topic_labels,
                topic_terms=topic_terms,
            ):
                split_reason = "topic_pair_fragments_not_distinct"
                topic_candidates = []

        if topic_candidates and topic_same_parent_only:
            same_parent_ok = False
            if all(axis == "child_topic" for axis, _topic in topic_candidates):
                parent_set = {
                    _topic_parent_from_child(topic, metadata_store)
                    for _axis, topic in topic_candidates
                }
                parent_set = {p for p in parent_set if p}
                same_parent_ok = len(parent_set) == 1
            if not same_parent_ok:
                split_reason = "topic_parent_mismatch"
                topic_candidates = []

        # Keep branch count bounded deterministically instead of dropping split.
        if topic_candidates and len(topic_candidates) > max_branches:
            if all(axis == "child_topic" for axis, _topic in topic_candidates):
                topic_candidates = sorted(
                    topic_candidates,
                    key=lambda item: (
                        -float(child_topic_scores.get(item[1], 0.0)),
                        str(item[1]),
                    ),
                )[:max_branches]
            else:
                topic_candidates = sorted(
                    topic_candidates,
                    key=lambda item: (
                        -float(parent_topic_scores.get(item[1], 0.0)),
                        str(item[1]),
                    ),
                )[:max_branches]

        if topic_candidates and len(topic_candidates) <= max_branches:
            for axis, topic in topic_candidates:
                branch_id = f"topic={topic}"
                parent_for_soft = ""
                if axis == "child_topic":
                    # Child hard-filter can over-prune; keep parent hard filter when possible.
                    if high_parent_topics:
                        parent_for_soft = high_parent_topics[0]
                    elif parent_topic_scores:
                        parent_for_soft = sorted(
                            parent_topic_scores.items(),
                            key=lambda kv: (-float(kv[1]), kv[0]),
                        )[0][0]
                branch_meta = _build_split_branch_metadata(base_metadata=metadata, axis=axis, value=topic)
                if axis == "child_topic" and not topic_child_hard_filter:
                    branch_meta = _build_split_branch_metadata(
                        base_metadata=metadata,
                        axis=axis,
                        value=topic,
                        child_hard_filter=False,
                        parent_topic_for_child_soft=parent_for_soft,
                    )
                branch_meta["_diagnostics"]["split_branch_id"] = branch_id
                branch_query_text = _build_branch_query_text(
                    query_text=query_text,
                    split_axis=split_axis,
                    branch_value=topic,
                    branch_mode=branch_query_mode,
                    metadata_store=metadata_store,
                    topic_labels=topic_labels,
                )
                split_branch_filters.append(
                    {
                        "branch_id": branch_id,
                        "lang": branch_meta.get("lang"),
                        "topics": branch_meta.get("topics", {}),
                        "soft_child_topic_bias": (branch_meta.get("_diagnostics") or {}).get("soft_child_topic_bias", []),
                    }
                )
                branches.append(
                    {
                        "branch_id": branch_id,
                        "metadata": branch_meta,
                        "query_text": branch_query_text,
                        "branch_query_mode": branch_query_mode,
                    }
                )
        elif split_axis == "topic" and topic_candidates and len(topic_candidates) > max_branches:
            split_axis = "none"
            split_reason = "topic_branch_limit_exceeded"

    if strict_single_axis and split_axis not in {"language", "topic"}:
        branches = []

    if len(branches) < 2:
        return {
            "split_applied": False,
            "split_axis": "none",
            "split_reason": split_reason,
            "branches": [],
            "split_branch_filters": [],
            "branch_query_mode": branch_query_mode,
        }

    return {
        "split_applied": True,
        "split_axis": split_axis,
        "split_reason": split_reason,
        "branches": branches,
        "split_branch_filters": split_branch_filters,
        "require_high_confidence": require_high,
        "language_candidates_high": language_candidates,
        "language_candidates_coordination": coordination_candidates,
        "language_coordination_evidence": coordination_evidence,
        "language_contextual_mentions": coordination_contextual_mentions,
        "language_risk_only_mentions": coordination_risk_only_mentions,
        "topic_candidates_high": high_child_topics if high_child_topics else high_parent_topics,
        "branch_query_mode": branch_query_mode,
        "feedback_gating_enabled": bool(query_split_cfg.get("feedback_gating", False)),
        "feedback_min_docs_per_branch": max(1, int(query_split_cfg.get("feedback_min_docs_per_branch", 1))),
        "axis_mode": axis_mode,
        "topic_same_parent_only": topic_same_parent_only,
        "topic_require_pair_cue": topic_require_pair_cue,
        "topic_require_distinct_fragments": topic_require_distinct_fragments if enable_topic_axis else False,
        "language_medium_requires_coordination": language_medium_requires_coordination,
    }


def _retrieve_with_optional_split(
    *,
    retriever: RAGRetriever,
    query_item: Dict[str, Any],
    profile_cfg: Dict[str, Any],
    fixed_k: int,
    fixed_top_n: int,
    metadata_store: MetadataStore | None,
    phenomenon_lexicon: Dict[str, List[str]],
    category_lexicon: Dict[str, str],
) -> Tuple[Dict[str, Any], List[Document], List[str], float, float]:
    query_text = query_item["query"]
    t_meta = perf_counter()
    retrieval_cfg = profile_cfg.get("retrieval", {})
    llm_retrieval_cfg = profile_cfg.get("llm_retrieval", {}) or {}
    retrieval_timeout_sec_effective = _safe_int(llm_retrieval_cfg.get("request_timeout_sec", 600), 600)
    if retrieval_timeout_sec_effective <= 0:
        retrieval_timeout_sec_effective = 600
    query_split_cfg = retrieval_cfg.get("query_split", {}) if isinstance(retrieval_cfg.get("query_split"), dict) else {}
    metadata_mode = str(retrieval_cfg.get("metadata_mode", "query_only")).lower()
    split_controller_mode = str(query_split_cfg.get("controller_mode", "heuristic")).strip().lower()
    if split_controller_mode in FINALIZED_B7_CONTROLLER_MODES:
        return run_finalized_b7_retrieval(
            retriever=retriever,
            query_id=str(query_item.get("query_id", "")),
            query_text=query_text,
            profile_cfg=profile_cfg,
            fixed_k=fixed_k,
            fixed_top_n=fixed_top_n,
            metadata_store=metadata_store,
            phenomenon_lexicon=phenomenon_lexicon,
            category_lexicon=category_lexicon,
        )
    if metadata_mode == "runtime":
        metadata = extract_query_metadata(query_text, profile_cfg)
    else:
        metadata = build_query_only_metadata(
            query_text=query_text,
            metadata_store=metadata_store,
            phenomenon_lexicon=phenomenon_lexicon,
            category_lexicon=category_lexicon,
        )

    metadata_elapsed_sec = perf_counter() - t_meta
    diagnostics = metadata.setdefault("_diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
        metadata["_diagnostics"] = diagnostics
    diagnostics.setdefault("metadata_mode", metadata_mode)
    diagnostics.setdefault("split_applied", False)
    diagnostics.setdefault("split_branches", [])
    diagnostics.setdefault("split_json_retry_count", 0)
    diagnostics.setdefault("split_json_fail_open", False)
    diagnostics.setdefault("filter_applied", False)
    diagnostics.setdefault("timeout_location", "none")
    warnings: List[str] = []

    split_plan = _plan_query_only_split(
        query_id=str(query_item.get("query_id", "")),
        query_text=query_text,
        metadata=metadata,
        retrieval_cfg=retrieval_cfg,
        metadata_store=metadata_store,
        phenomenon_lexicon=phenomenon_lexicon,
        category_lexicon=category_lexicon,
        request_timeout_sec=retrieval_timeout_sec_effective,
    )
    diagnostics["split_applied"] = bool(split_plan.get("split_applied", False))
    diagnostics["split_dimension"] = str(split_plan.get("split_axis", "none"))
    diagnostics["split_reason"] = str(split_plan.get("split_reason", "none"))
    diagnostics["split_branches"] = [
        str(branch.get("branch_id"))
        for branch in split_plan.get("branches", [])
        if isinstance(branch, dict) and str(branch.get("branch_id", "")).strip()
    ]
    diagnostics["split_branch_query_count"] = len(diagnostics["split_branches"])
    diagnostics["split_branch_filters"] = split_plan.get("split_branch_filters", [])
    diagnostics["split_candidate_languages_high"] = split_plan.get("language_candidates_high", [])
    diagnostics["split_candidate_topics_high"] = split_plan.get("topic_candidates_high", [])
    diagnostics["split_language_coordination_evidence"] = split_plan.get("language_coordination_evidence", [])
    diagnostics["split_language_contextual_mentions"] = split_plan.get("language_contextual_mentions", [])
    diagnostics["split_language_risk_only_mentions"] = split_plan.get("language_risk_only_mentions", [])
    diagnostics["split_branch_query_mode"] = split_plan.get("branch_query_mode", "none")
    diagnostics["split_controller_mode"] = split_plan.get("controller_mode", "heuristic")
    diagnostics["split_materialized_plan"] = split_plan.get("split_materialized_plan", {})
    diagnostics["split_feedback_gating_enabled"] = bool(split_plan.get("feedback_gating_enabled", False))
    diagnostics["split_topic_same_parent_only"] = bool(split_plan.get("topic_same_parent_only", False))
    diagnostics["split_topic_require_pair_cue"] = bool(split_plan.get("topic_require_pair_cue", False))
    diagnostics["split_topic_require_distinct_fragments"] = bool(
        split_plan.get("topic_require_distinct_fragments", False)
    )
    diagnostics["split_language_medium_requires_coordination"] = bool(
        split_plan.get("language_medium_requires_coordination", False)
    )

    t_retrieve = perf_counter()
    filters_enabled = any(
        (
            bool(retriever.enable_language_filter),
            bool(retriever.enable_parent_topic_filter),
            bool(retriever.enable_child_topic_filter),
            bool(retriever.enable_family_filter),
            bool(retriever.enable_region_filter),
        )
    )

    preflight_docs: List[Document] = []
    preflight_diag: Dict[str, Any] = {}
    if diagnostics.get("split_applied", False) and bool(split_plan.get("feedback_gating_enabled", False)):
        min_docs_per_branch = max(1, int(split_plan.get("feedback_min_docs_per_branch", 1)))
        if filters_enabled:
            preflight_docs, preflight_diag = retrieve_with_filter_fallback(
                retriever=retriever,
                query_text=query_text,
                base_metadata=metadata,
                fixed_k=fixed_k,
                fixed_top_n=fixed_top_n,
            )
        else:
            preflight_docs = retriever.retrieve_documents(
                query_text,
                metadata=metadata,
                k=fixed_k,
                top_n=fixed_top_n,
            )
            preflight_diag = {
                "filter_stage_used": "no_filter",
                "fallback_stage": 0,
                "doc_count": len(preflight_docs),
                "zero_hit": len(preflight_docs) == 0,
                "filter_applied": False,
            }

        split_axis = str(split_plan.get("split_axis", "none"))
        coverage_counts: Dict[str, int] = {}
        coverage_sufficient = False
        if split_axis == "language":
            candidates = [str(v) for v in split_plan.get("language_candidates_high", []) if str(v).strip()]
            coverage_counts = _coverage_counts_for_languages(preflight_docs, candidates)
            coverage_sufficient = bool(candidates) and all(coverage_counts.get(c, 0) >= min_docs_per_branch for c in candidates)
        elif split_axis == "topic":
            candidates = [str(v) for v in split_plan.get("topic_candidates_high", []) if str(v).strip()]
            coverage_counts = _coverage_counts_for_topics(preflight_docs, candidates)
            coverage_sufficient = bool(candidates) and all(coverage_counts.get(c, 0) >= min_docs_per_branch for c in candidates)

        diagnostics["split_feedback_gate_applied"] = True
        diagnostics["split_feedback_min_docs_per_branch"] = min_docs_per_branch
        diagnostics["split_feedback_preflight_doc_count"] = len(preflight_docs)
        diagnostics["split_feedback_coverage"] = coverage_counts
        diagnostics["split_feedback_coverage_sufficient"] = bool(coverage_sufficient)

        if coverage_sufficient:
            diagnostics["split_applied"] = False
            diagnostics["split_dimension"] = "none"
            diagnostics["split_reason"] = "feedback_sufficient_coverage"
            diagnostics["split_branches"] = []
            diagnostics["split_branch_query_count"] = 0
            diagnostics["split_branch_filters"] = []
            diagnostics["split_branch_query_mode"] = "none"
            docs = list(preflight_docs)
            diagnostics["filter_stage_used"] = preflight_diag.get("filter_stage_used", "no_filter")
            diagnostics["fallback_stage"] = int(preflight_diag.get("fallback_stage", 0) or 0)
            diagnostics["doc_count"] = len(docs)
            diagnostics["zero_hit"] = len(docs) == 0
            diagnostics["filter_applied"] = bool(preflight_diag.get("filter_applied", False))
            retrieval_elapsed_sec = perf_counter() - t_retrieve
            return metadata, docs, warnings, metadata_elapsed_sec, retrieval_elapsed_sec

    if diagnostics.get("split_applied", False):
        branch_docs: Dict[str, List[Document]] = {}
        branch_diag_rows: List[Dict[str, Any]] = []
        branch_order: List[str] = []
        branch_doc_counts: Dict[str, int] = {}
        branch_origin_by_doc: Dict[Tuple[str, str], str] = {}
        split_branch_queries: List[Dict[str, Any]] = []
        split_controller_mode = str(split_plan.get("controller_mode", diagnostics.get("split_controller_mode", "heuristic"))).strip().lower()
        t7_local_b6_mode = split_controller_mode in {"t7_materialized_language_local_b6", "t71_materialized_language_local_b6"}
        original_use_reranker = bool(getattr(retriever, "use_reranker", False))
        branch_retrieval_failed = False
        try:
            if original_use_reranker and not t7_local_b6_mode:
                retriever.use_reranker = False
            for branch in split_plan.get("branches", []):
                if not isinstance(branch, dict):
                    continue
                branch_id = str(branch.get("branch_id", "")).strip()
                branch_meta = branch.get("metadata")
                branch_query_text = str(branch.get("query_text", query_text))
                branch_query_mode = str(branch.get("branch_query_mode", diagnostics.get("split_branch_query_mode", "none")))
                branch_target_label = str(branch.get("target_label", "")).strip()
                branch_target_lang_id = str(branch.get("target_lang_id", "")).strip()
                if not branch_id or not isinstance(branch_meta, dict):
                    continue

                if filters_enabled:
                    branch_result_docs, branch_diag = _retrieve_with_filter_fallback(
                        retriever=retriever,
                        query_text=branch_query_text,
                        base_metadata=branch_meta,
                        fixed_k=fixed_k,
                        fixed_top_n=fixed_top_n,
                        allow_no_filter=not t7_local_b6_mode,
                    )
                else:
                    branch_result_docs = retriever.retrieve_documents(
                        branch_query_text,
                        metadata=branch_meta,
                        k=fixed_k,
                        top_n=fixed_top_n,
                    )
                    branch_diag = branch_meta.get("_diagnostics", {}) if isinstance(branch_meta.get("_diagnostics"), dict) else {}
                    branch_diag["filter_stage_used"] = "no_filter"
                    branch_diag["fallback_stage"] = 0
                    branch_diag["doc_count"] = len(branch_result_docs)
                    branch_diag["zero_hit"] = len(branch_result_docs) == 0

                branch_soft_bias_stats: Dict[str, Any] = {
                    "target_child_topic": "",
                    "boost": 0.0,
                    "matched_docs_count": 0,
                    "applied": False,
                    "stage": "pre_merge",
                }
                soft_bias_targets = [
                    str(v).strip()
                    for v in (branch_meta.get("_diagnostics", {}) or {}).get("soft_child_topic_bias", [])
                    if str(v).strip()
                ]
                soft_bias_enabled = bool(query_split_cfg.get("soft_child_topic_bias_enabled", False))
                if soft_bias_enabled and soft_bias_targets:
                    branch_result_docs, branch_soft_bias_stats = _apply_soft_child_topic_bias(
                        branch_result_docs,
                        target_child_topic=soft_bias_targets[0],
                        boost=float(query_split_cfg.get("soft_child_topic_bias_boost", 0.2) or 0.2),
                        stage="pre_merge",
                    )

                branch_docs[branch_id] = list(branch_result_docs)
                branch_order.append(branch_id)
                branch_doc_counts[branch_id] = len(branch_result_docs)
                for d in branch_result_docs:
                    key = _doc_identity(d)
                    if key not in branch_origin_by_doc:
                        branch_origin_by_doc[key] = branch_id
                split_branch_queries.append(
                    {
                        "branch_id": branch_id,
                        "branch_query_mode": branch_query_mode,
                        "branch_query_text": branch_query_text,
                        "target_label": branch_target_label,
                        "target_lang_id": branch_target_lang_id,
                    }
                )
                local_reranker_active = bool(getattr(retriever, "use_reranker", False))
                branch_diag_rows.append(
                    _build_split_branch_trace_row(
                        branch_id=branch_id,
                        target_label=branch_target_label,
                        target_lang_id=branch_target_lang_id,
                        branch_query_text=branch_query_text,
                        branch_query_mode=branch_query_mode,
                        branch_meta=branch_meta,
                        branch_diag=branch_diag,
                        branch_result_docs=branch_result_docs,
                        branch_soft_bias_stats=branch_soft_bias_stats,
                        local_reranker_active=local_reranker_active,
                        branch_anchor_retained=bool(branch.get("anchor_retained", False)),
                        branch_topic_anchor=str(branch.get("topic_anchor", "") or ""),
                    )
                )
                if t7_local_b6_mode:
                    if branch_diag.get("filter_stage_used") == "no_filter" or len(branch_result_docs) == 0:
                        branch_retrieval_failed = True
        finally:
            retriever.use_reranker = original_use_reranker

        if t7_local_b6_mode and branch_retrieval_failed:
            diagnostics["split_branch_doc_counts"] = branch_doc_counts
            diagnostics["split_branch_diagnostics"] = branch_diag_rows
            diagnostics["split_branch_queries"] = split_branch_queries
            diagnostics["split_applied"] = False
            diagnostics["split_dimension"] = "none"
            diagnostics["split_reason"] = "t7_branch_retrieval_failed"
            diagnostics["split_reverted_to_unsplit"] = True
            if filters_enabled:
                docs, revert_diag = retrieve_with_filter_fallback(
                    retriever=retriever,
                    query_text=query_text,
                    base_metadata=metadata,
                    fixed_k=fixed_k,
                    fixed_top_n=fixed_top_n,
                )
            else:
                docs = retriever.retrieve_documents(
                    query_text,
                    metadata=metadata,
                    k=fixed_k,
                    top_n=fixed_top_n,
                )
                revert_diag = {
                    "filter_stage_used": "no_filter",
                    "fallback_stage": 0,
                    "doc_count": len(docs),
                    "zero_hit": len(docs) == 0,
                    "filter_applied": False,
                }
            diagnostics["filter_stage_used"] = revert_diag.get("filter_stage_used", "no_filter")
            diagnostics["fallback_stage"] = int(revert_diag.get("fallback_stage", 0) or 0)
            diagnostics["doc_count"] = len(docs)
            diagnostics["zero_hit"] = len(docs) == 0
            diagnostics["filter_applied"] = bool(revert_diag.get("filter_applied", False))
            retrieval_elapsed_sec = perf_counter() - t_retrieve
            return metadata, docs, warnings, metadata_elapsed_sec, retrieval_elapsed_sec

        def _branch_origin_distribution(rows: Sequence[Document], limit: int = 10) -> Dict[str, int]:
            counts: Dict[str, int] = {}
            for d in list(rows)[: max(0, int(limit))]:
                origin = branch_origin_by_doc.get(_doc_identity(d), "unknown")
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
        diagnostics["split_branch_doc_counts"] = branch_doc_counts
        diagnostics["split_branch_diagnostics"] = branch_diag_rows
        diagnostics["split_branch_queries"] = split_branch_queries
        diagnostics["split_merge_cap"] = merge_cap
        diagnostics["split_merged_doc_count"] = len(merged_docs)
        diagnostics["split_merge_unique_docs"] = len({_doc_identity(d) for d in merged_docs})
        diagnostics["split_branch_candidate_k"] = int(fixed_k)
        starvation_threshold = max(3, int(max(1, fixed_top_n) * 0.15))
        starved = [branch_id for branch_id, count in branch_doc_counts.items() if int(count) < starvation_threshold]
        diagnostics["split_branch_starvation_threshold"] = starvation_threshold
        diagnostics["split_branch_starvation_branches"] = starved
        diagnostics["split_branch_starvation_count"] = len(starved)
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

        split_axis = str(diagnostics.get("split_dimension", "none"))
        soft_bias_enabled = bool(query_split_cfg.get("soft_child_topic_bias_enabled", False))
        soft_bias_post_rerank = bool(query_split_cfg.get("soft_child_topic_bias_post_rerank", False))
        diagnostics["split_soft_child_topic_bias_post_rerank"] = {
            "applied": False,
            "matched_docs_count": 0,
            "target_child_topic": "",
            "boost": 0.0,
            "stage": "post_rerank",
        }
        if split_axis == "topic" and soft_bias_enabled and soft_bias_post_rerank:
            matched_docs = 0
            post_boost = float(
                query_split_cfg.get(
                    "soft_child_topic_bias_post_rerank_boost",
                    query_split_cfg.get("soft_child_topic_bias_boost", 0.2),
                ) or 0.2
            )
            for doc in docs:
                target = str(doc.metadata.get("split_topic_bias_target", "")).strip()
                if not target:
                    continue
                matched_docs += 1
                try:
                    base_score = float(doc.metadata.get("score", 0.0) or 0.0)
                except Exception:
                    base_score = 0.0
                doc.metadata["split_topic_bias_post_rerank_boost"] = post_boost
                doc.metadata["split_topic_bias_stage"] = "post_rerank"
                doc.metadata["score"] = base_score + post_boost
            if matched_docs > 0:
                docs = sorted(docs, key=_doc_score, reverse=True)
                diagnostics["split_soft_child_topic_bias_post_rerank"] = {
                    "applied": True,
                    "matched_docs_count": matched_docs,
                    "target_child_topic": "mixed_branch_targets",
                    "boost": post_boost,
                    "stage": "post_rerank",
                }

        diversity_guard_enabled = bool(query_split_cfg.get("language_diversity_guard_enabled", False))
        diagnostics["split_language_diversity_guard"] = {
            "applied": False,
            "min_per_branch": 0,
            "origin_distribution": {},
        }
        if split_axis == "language" and diversity_guard_enabled and docs:
            docs, diversity_stats = _apply_language_diversity_guard(
                docs,
                branch_order=branch_order,
                branch_origin_by_doc=branch_origin_by_doc,
                final_top_n=fixed_top_n,
                min_per_branch=max(1, int(query_split_cfg.get("language_diversity_min_per_branch", 2) or 2)),
            )
            diagnostics["split_language_diversity_guard"] = diversity_stats
        final_origin = _branch_origin_distribution(docs, limit=10)
        diagnostics["split_branch_origin_final_top10"] = final_origin
        pre_origin = diagnostics.get("split_branch_origin_pre_top10", {})
        pre_dominance = 0.0
        final_dominance = 0.0
        pre_total = sum(int(v) for v in pre_origin.values()) if isinstance(pre_origin, dict) else 0
        final_total = sum(int(v) for v in final_origin.values())
        if pre_total > 0 and isinstance(pre_origin, dict):
            pre_dominance = max(int(v) for v in pre_origin.values()) / float(pre_total)
        if final_total > 0:
            final_dominance = max(int(v) for v in final_origin.values()) / float(final_total)
        diagnostics["split_branch_dominance_pre_top10"] = round(pre_dominance, 6)
        diagnostics["split_branch_dominance_final_top10"] = round(final_dominance, 6)
        diagnostics["split_branch_dominance_flag_pre_top10"] = pre_dominance >= 0.8
        diagnostics["split_branch_dominance_flag_final_top10"] = final_dominance >= 0.8
        diagnostics["split_final_top5"] = _top_docs_snapshot(docs, limit=5)
        final_top10_origin_ranked = [
            branch_origin_by_doc.get(_doc_identity(doc), "unknown")
            for doc in list(docs)[:10]
        ]
        diagnostics["split_merge_trace"] = {
            "query_id": query_item.get("query_id"),
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
        diagnostics["filter_applied"] = any(
            row.get("filter_stage_used", "no_filter") != "no_filter" for row in branch_diag_rows
        )
    else:
        if filters_enabled:
            docs, diag_update = retrieve_with_filter_fallback(
                retriever=retriever,
                query_text=query_text,
                base_metadata=metadata,
                fixed_k=fixed_k,
                fixed_top_n=fixed_top_n,
            )
            diagnostics.update(diag_update)
        else:
            docs = retriever.retrieve_documents(
                query_text,
                metadata=metadata,
                k=fixed_k,
                top_n=fixed_top_n,
            )
            diagnostics["filter_stage_used"] = "no_filter"
            diagnostics["fallback_stage"] = 0
            diagnostics["doc_count"] = len(docs)
            diagnostics["zero_hit"] = len(docs) == 0

    retrieval_elapsed_sec = perf_counter() - t_retrieve
    return metadata, docs, warnings, metadata_elapsed_sec, retrieval_elapsed_sec


def _build_query_row_base(
    *,
    run_id: str,
    profile: str,
    query_item: Dict[str, Any],
    db_name: str,
    db_table: str,
    backend_mode_effective: str,
    backend_mode_config: str,
    backend_mode_required: str,
    fixed_k: int,
    fixed_top_n: int,
    retrieval_timeout_sec_effective: int,
    metadata_elapsed_sec: float,
    retrieval_elapsed_sec: float,
    total_elapsed_sec: float,
    diagnostics: Dict[str, Any],
    warnings: List[str],
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "profile": profile,
        "query_id": query_item["query_id"],
        "query_type": query_item.get("query_type", "unknown"),
        "group_label": _derive_group_label(
            str(query_item.get("query_id", "")),
            str(query_item.get("query_type", "unknown")),
        ),
        "query_text": query_item["query"],
        "timestamp": _utc_now(),
        "db_name": db_name,
        "db_table": db_table,
        "backend_mode": backend_mode_effective,
        "backend_mode_effective": backend_mode_effective,
        "backend_mode_config": backend_mode_config,
        "backend_mode_required": backend_mode_required,
        "k": fixed_k,
        "top_n": fixed_top_n,
        "retrieval_timeout_sec_effective": retrieval_timeout_sec_effective,
        "metadata_elapsed_sec": round(metadata_elapsed_sec, 6),
        "retrieval_elapsed_sec": round(retrieval_elapsed_sec, 6),
        "total_elapsed_sec": round(total_elapsed_sec, 6),
        "lang_detect_method": diagnostics.get("lang_detect_method", "none"),
        "detected_languages": diagnostics.get("detected_languages", []),
        "lang_json_retry_count": int(diagnostics.get("lang_json_retry_count", 0)),
        "lang_json_fail_open": bool(diagnostics.get("lang_json_fail_open", False)),
        "lang_json_valid_stage_a": bool(diagnostics.get("lang_json_valid_stage_a", False)),
        "lang_json_valid_stage_b": bool(diagnostics.get("lang_json_valid_stage_b", False)),
        "filter_applied": bool(diagnostics.get("filter_applied", False)),
        "split_applied": bool(diagnostics.get("split_applied", False)),
        "split_dimension": diagnostics.get("split_dimension", "none"),
        "split_branches": diagnostics.get("split_branches", []),
        "split_json_retry_count": int(diagnostics.get("split_json_retry_count", 0)),
        "split_json_fail_open": bool(diagnostics.get("split_json_fail_open", False)),
        "metadata_mode": diagnostics.get("metadata_mode", "query_only"),
        "detected_topics_categories": diagnostics.get("detected_topics_categories", []),
        "detected_topics_phenomena": diagnostics.get("detected_topics_phenomena", []),
        "detector_confidence": diagnostics.get("detector_confidence", {}),
        "filter_stage_used": diagnostics.get("filter_stage_used", "no_filter"),
        "fallback_stage": int(diagnostics.get("fallback_stage", 0) or 0),
        "zero_hit": bool(diagnostics.get("zero_hit", False)),
        "doc_count_diag": int(diagnostics.get("doc_count", 0) or 0),
        "timeout_location": diagnostics.get("timeout_location", "none"),
        "warnings": warnings,
    }


def _run_retrieval_profile(
    *,
    run_id: str,
    profile_name: str,
    profile_cfg: Dict[str, Any],
    queries: List[Dict[str, Any]],
    profile_dir: Path,
) -> Dict[str, Any]:
    retrieval_cfg = profile_cfg.get("retrieval", {})
    reranker_cfg = profile_cfg.get("reranker", {})
    ablation_cfg = profile_cfg.get("ablation", {})

    fixed_k = _safe_int(retrieval_cfg.get("fixed_k", retrieval_cfg.get("k", 30)), 30)
    fixed_top_n = _safe_int(retrieval_cfg.get("fixed_top_n", reranker_cfg.get("top_n", 30)), 30)
    llm_retrieval_cfg = profile_cfg.get("llm_retrieval", {}) or {}
    retrieval_timeout_sec_effective = _safe_int(llm_retrieval_cfg.get("request_timeout_sec", 600), 600)
    if retrieval_timeout_sec_effective <= 0:
        retrieval_timeout_sec_effective = 600

    if profile_name in RERANK_K30_PROFILES and bool(reranker_cfg.get("enabled", False)) and fixed_top_n < RERANK_MIN_TOP_N_FOR_K30:
        raise RuntimeError(
            f"{profile_name} invalid: reranker enabled but fixed_top_n={fixed_top_n} < {RERANK_MIN_TOP_N_FOR_K30}."
        )

    required_backend_mode = str(ablation_cfg.get("required_backend_mode", "any")).lower()
    db_cfg = profile_cfg.get("database", {}) or {}
    db_name = str(db_cfg.get("dbname", ""))
    db_table = str(db_cfg.get("table_name", ""))
    backend_mode_config = str((profile_cfg.get("database", {}) or {}).get("type", "postgres")).lower()

    metadata_csv_path = str(retrieval_cfg.get("metadata_csv", "metadata.enriched.csv"))
    metadata_store = load_metadata_store(metadata_csv_path)
    lexicon_path = str(
        (retrieval_cfg.get("viking", {}) or {}).get("lexicon_path", "config/viking_lexicon.yaml")
    )
    try:
        phenomenon_lexicon, category_lexicon = load_viking_lexicon(lexicon_path)
    except Exception:
        phenomenon_lexicon, category_lexicon = {}, {}

    trace_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    with use_config_override(profile_cfg):
        with RAGRetriever() as retriever:
            backend_mode_effective = str(getattr(retriever, "backend_mode", "unknown")).lower()
            if required_backend_mode != "any" and backend_mode_effective != required_backend_mode:
                raise RuntimeError(
                    f"Backend mode mismatch: required={required_backend_mode}, actual={backend_mode_effective}"
                )

            for query_item in queries:
                query_text = query_item["query"]
                t_start = perf_counter()
                metadata, docs, split_warnings, metadata_elapsed_sec, retrieval_elapsed_sec = _retrieve_with_optional_split(
                    retriever=retriever,
                    query_item=query_item,
                    profile_cfg=profile_cfg,
                    fixed_k=fixed_k,
                    fixed_top_n=fixed_top_n,
                    metadata_store=metadata_store,
                    phenomenon_lexicon=phenomenon_lexicon,
                    category_lexicon=category_lexicon,
                )
                total_elapsed_sec = perf_counter() - t_start

                diagnostics = metadata.get("_diagnostics", {}) if isinstance(metadata, dict) else {}
                ranked_rows = _documents_to_ranked_rows(docs)
                gold_docs, gold_chunks = build_gold_sets(query_item)
                metric_values = compute_query_metrics(
                    ranked_docs=ranked_rows,
                    gold_docs=gold_docs,
                    gold_chunks=gold_chunks,
                    ks=K_VALUES,
                )

                warnings = list(split_warnings)
                if diagnostics.get("timeout_location", "none") != "none":
                    warnings.append(
                        f"Language/split timeout at {diagnostics.get('timeout_location')}; fail-open retrieval continued."
                    )

                row_base = _build_query_row_base(
                    run_id=run_id,
                    profile=profile_name,
                    query_item=query_item,
                    db_name=db_name,
                    db_table=db_table,
                    backend_mode_effective=backend_mode_effective,
                    backend_mode_config=backend_mode_config,
                    backend_mode_required=required_backend_mode,
                    fixed_k=fixed_k,
                    fixed_top_n=fixed_top_n,
                    retrieval_timeout_sec_effective=retrieval_timeout_sec_effective,
                    metadata_elapsed_sec=metadata_elapsed_sec,
                    retrieval_elapsed_sec=retrieval_elapsed_sec,
                    total_elapsed_sec=total_elapsed_sec,
                    diagnostics=diagnostics,
                    warnings=warnings,
                )

                trace_row = dict(row_base)
                trace_row["documents"] = ranked_rows
                trace_row["doc_count"] = len(ranked_rows)
                trace_row["retrieval_diagnostics"] = diagnostics
                trace_row.update(metric_values)
                trace_rows.append(trace_row)

                summary_row = dict(row_base)
                summary_row["doc_count"] = len(ranked_rows)
                summary_row.update(metric_values)
                summary_rows.append(summary_row)

    _write_jsonl(profile_dir / "retrieval_traces.jsonl", trace_rows)
    _write_csv(profile_dir / "summary.csv", summary_rows)
    _write_json(profile_dir / "resolved_config.json", profile_cfg)
    _export_split_profile_artifacts(
        profile_dir=profile_dir,
        profile_name=profile_name,
        profile_cfg=profile_cfg,
        trace_rows=trace_rows,
    )

    macro_micro_rows = aggregate_macro_micro(
        rows=summary_rows,
        ks=K_VALUES,
        profile=profile_name,
        track="retrieval",
    )
    _write_csv(profile_dir / "macro_micro.csv", macro_micro_rows)

    return {
        "profile": profile_name,
        "track": "retrieval",
        "records": len(summary_rows),
        "backend_mode_effective": backend_mode_effective,
        "backend_mode_config": backend_mode_config,
        "backend_mode_required": required_backend_mode,
        "config_hash": _config_hash(profile_cfg),
        "macro_micro_rows": macro_micro_rows,
        "summary_rows": summary_rows,
    }


def _run_e2e_profile(
    *,
    run_id: str,
    profile_name: str,
    profile_cfg: Dict[str, Any],
    queries: List[Dict[str, Any]],
    profile_dir: Path,
) -> Dict[str, Any]:
    retrieval_cfg = profile_cfg.get("retrieval", {})
    reranker_cfg = profile_cfg.get("reranker", {})
    ablation_cfg = profile_cfg.get("ablation", {})

    fixed_k = _safe_int(retrieval_cfg.get("fixed_k", retrieval_cfg.get("k", 30)), 30)
    fixed_top_n = _safe_int(retrieval_cfg.get("fixed_top_n", reranker_cfg.get("top_n", 30)), 30)
    llm_retrieval_cfg = profile_cfg.get("llm_retrieval", {}) or {}
    retrieval_timeout_sec_effective = _safe_int(llm_retrieval_cfg.get("request_timeout_sec", 600), 600)
    if retrieval_timeout_sec_effective <= 0:
        retrieval_timeout_sec_effective = 600

    if bool(reranker_cfg.get("enabled", False)) and fixed_top_n < RERANK_MIN_TOP_N_FOR_K30:
        raise RuntimeError(
            f"{profile_name} invalid: reranker enabled but fixed_top_n={fixed_top_n} < {RERANK_MIN_TOP_N_FOR_K30}."
        )

    required_backend_mode = str(ablation_cfg.get("required_backend_mode", "any")).lower()
    db_cfg = profile_cfg.get("database", {}) or {}
    db_name = str(db_cfg.get("dbname", ""))
    db_table = str(db_cfg.get("table_name", ""))
    backend_mode_config = str((profile_cfg.get("database", {}) or {}).get("type", "postgres")).lower()

    trace_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    first_metric_rows: List[Dict[str, Any]] = []
    final_metric_rows: List[Dict[str, Any]] = []

    with use_config_override(profile_cfg):
        for query_item in queries:
            query_text = query_item["query"]
            t_start = perf_counter()

            first_retrieve_docs: List[Document] = []
            first_diagnostics: Dict[str, Any] = {}
            final_docs: List[Document] = []
            warnings: List[str] = []
            timeout_locations: List[str] = []
            generation = ""
            generation_diagnostics: Dict[str, Any] = {}

            for event in rag_app.stream({"question": query_text, "search_count": 0}):
                for node_name, node_value in event.items():
                    if not isinstance(node_value, dict):
                        continue
                    if node_name == "retrieve" and not first_retrieve_docs:
                        first_retrieve_docs = list(node_value.get("documents") or [])
                        first_diagnostics = node_value.get("retrieval_diagnostics", {}) or {}
                    docs = node_value.get("documents")
                    if isinstance(docs, list):
                        final_docs = docs
                    if isinstance(node_value.get("warnings"), list):
                        warnings = list(node_value.get("warnings") or [])
                    if isinstance(node_value.get("timeout_locations"), list):
                        timeout_locations = list(node_value.get("timeout_locations") or [])
                    if isinstance(node_value.get("generation"), str):
                        generation = node_value.get("generation")
                    if isinstance(node_value.get("generation_diagnostics"), dict):
                        generation_diagnostics = dict(node_value.get("generation_diagnostics") or {})

            total_elapsed_sec = perf_counter() - t_start
            if required_backend_mode != "any" and backend_mode_config != required_backend_mode:
                raise RuntimeError(
                    f"Backend mode mismatch (E2E config): required={required_backend_mode}, config={backend_mode_config}"
                )

            first_ranked = _documents_to_ranked_rows(first_retrieve_docs)
            final_ranked = _documents_to_ranked_rows(final_docs)
            gold_docs, gold_chunks = build_gold_sets(query_item)
            first_metrics = compute_query_metrics(
                ranked_docs=first_ranked,
                gold_docs=gold_docs,
                gold_chunks=gold_chunks,
                ks=K_VALUES,
            )
            final_metrics = compute_query_metrics(
                ranked_docs=final_ranked,
                gold_docs=gold_docs,
                gold_chunks=gold_chunks,
                ks=K_VALUES,
            )

            diagnostics = first_diagnostics if isinstance(first_diagnostics, dict) else {}
            row_base = {
                "run_id": run_id,
                "profile": profile_name,
                "query_id": query_item["query_id"],
                "query_type": query_item.get("query_type", "unknown"),
                "group_label": _derive_group_label(
                    str(query_item.get("query_id", "")),
                    str(query_item.get("query_type", "unknown")),
                ),
                "query_text": query_text,
                "timestamp": _utc_now(),
                "db_name": db_name,
                "db_table": db_table,
                "backend_mode": backend_mode_config,
                "backend_mode_effective": backend_mode_config,
                "backend_mode_config": backend_mode_config,
                "backend_mode_required": required_backend_mode,
                "k": fixed_k,
                "top_n": fixed_top_n,
                "retrieval_timeout_sec_effective": retrieval_timeout_sec_effective,
                "total_elapsed_sec": round(total_elapsed_sec, 6),
                "lang_detect_method": diagnostics.get("lang_detect_method", "none"),
                "detected_languages": diagnostics.get("detected_languages", []),
                "lang_json_retry_count": int(diagnostics.get("lang_json_retry_count", 0)),
                "lang_json_fail_open": bool(diagnostics.get("lang_json_fail_open", False)),
                "lang_json_valid_stage_a": bool(diagnostics.get("lang_json_valid_stage_a", False)),
                "lang_json_valid_stage_b": bool(diagnostics.get("lang_json_valid_stage_b", False)),
                "filter_applied": bool(diagnostics.get("filter_applied", False)),
                "split_applied": bool(diagnostics.get("split_applied", False)),
                "split_branches": diagnostics.get("split_branches", []),
                "split_json_retry_count": int(diagnostics.get("split_json_retry_count", 0)),
                "split_json_fail_open": bool(diagnostics.get("split_json_fail_open", False)),
                "timeout_location": diagnostics.get("timeout_location", "none"),
                "warnings": warnings,
                "timeout_locations": timeout_locations,
                "first_doc_count": len(first_ranked),
                "final_doc_count": len(final_ranked),
                "generation": generation,
                "generation_diagnostics": generation_diagnostics,
                "generation_grounding_mode": generation_diagnostics.get("grounding_mode", "unknown"),
                "generation_inline_citation_coverage": float(generation_diagnostics.get("inline_citation_coverage", 0.0) or 0.0),
                "generation_guard_triggered": bool(generation_diagnostics.get("unsupported_claim_guard_triggered", False)),
                "generation_guard_reason": generation_diagnostics.get("fail_closed_reason", "none"),
            }

            trace_row = dict(row_base)
            trace_row["first_retrieve_documents"] = first_ranked
            trace_row["final_documents"] = final_ranked
            trace_row.update({f"first_{k}": v for k, v in first_metrics.items()})
            trace_row.update({f"final_{k}": v for k, v in final_metrics.items()})
            trace_rows.append(trace_row)

            summary_row = dict(row_base)
            summary_row.update({f"first_{k}": v for k, v in first_metrics.items()})
            summary_row.update({f"final_{k}": v for k, v in final_metrics.items()})
            summary_rows.append(summary_row)

            first_row = {
                "profile": profile_name,
                "query_id": query_item["query_id"],
                "query_type": query_item.get("query_type", "unknown"),
                "group_label": _derive_group_label(
                    str(query_item.get("query_id", "")),
                    str(query_item.get("query_type", "unknown")),
                ),
            }
            first_row.update(first_metrics)
            first_metric_rows.append(first_row)

            final_row = {
                "profile": profile_name,
                "query_id": query_item["query_id"],
                "query_type": query_item.get("query_type", "unknown"),
                "group_label": _derive_group_label(
                    str(query_item.get("query_id", "")),
                    str(query_item.get("query_type", "unknown")),
                ),
            }
            final_row.update(final_metrics)
            final_metric_rows.append(final_row)

    _write_jsonl(profile_dir / "e2e_traces.jsonl", trace_rows)
    _write_csv(profile_dir / "e2e_summary.csv", summary_rows)
    _write_json(profile_dir / "resolved_config.json", profile_cfg)

    first_macro = aggregate_macro_micro(
        rows=first_metric_rows,
        ks=K_VALUES,
        profile=profile_name,
        track="e2e_first_retrieve",
    )
    final_macro = aggregate_macro_micro(
        rows=final_metric_rows,
        ks=K_VALUES,
        profile=profile_name,
        track="e2e_final",
    )
    macro_rows = first_macro + final_macro
    _write_csv(profile_dir / "e2e_macro_micro.csv", macro_rows)

    return {
        "profile": profile_name,
        "track": "e2e",
        "records": len(summary_rows),
        "backend_mode_effective": backend_mode_config,
        "backend_mode_config": backend_mode_config,
        "backend_mode_required": required_backend_mode,
        "config_hash": _config_hash(profile_cfg),
        "macro_micro_rows": macro_rows,
    }


def _split_dev_test_queries(
    queries: List[Dict[str, Any]],
    *,
    seed: int = 20260304,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, List[str]]]:
    target_groups = {
        "T1": 15,
        "T2": 15,
        "T3": 15,
    }
    rng = random.Random(seed)
    by_group: Dict[str, List[Dict[str, Any]]] = {k: [] for k in target_groups}
    for q in queries:
        group_label = _derive_group_label(
            str(q.get("query_id", "")),
            str(q.get("query_type", "unknown")),
        )
        if group_label in {"T4", "T5", "T6"}:
            by_group.setdefault(group_label, []).append(q)
        elif group_label in by_group:
            by_group[group_label].append(q)

    # Preserve the legacy 45-query dev slice for T1-T3, but include all extra
    # groups when they exist so downstream selection and reporting do not erase
    # T4-T6 entirely.
    for group_label in ("T4", "T5", "T6"):
        pool = by_group.get(group_label, [])
        if pool:
            target_groups[group_label] = len(pool)

    selected: List[Dict[str, Any]] = []
    selected_ids_by_group: Dict[str, List[str]] = {}
    for group_label, n in target_groups.items():
        pool = sorted(by_group.get(group_label, []), key=lambda x: str(x.get("query_id", "")))
        if len(pool) < n:
            raise ValueError(
                f"Balanced dev build failed: group={group_label} needs {n}, found {len(pool)}"
            )
        chosen = rng.sample(pool, n)
        chosen = sorted(chosen, key=lambda x: str(x.get("query_id", "")))
        selected.extend(chosen)
        selected_ids_by_group[group_label] = [str(x.get("query_id", "")) for x in chosen]

    selected_ids = {str(q.get("query_id", "")) for q in selected}
    final_queries = [q for q in queries]
    dev_queries = [q for q in final_queries if str(q.get("query_id", "")) in selected_ids]
    return dev_queries, final_queries, selected_ids_by_group


def _macro_metric(summary_rows: List[Dict[str, Any]], metric_key: str) -> float:
    by_type: Dict[str, List[float]] = {}
    for row in summary_rows:
        group_label = _row_group_label(row)
        by_type.setdefault(group_label, []).append(float(row.get(metric_key, 0.0)))
    if not by_type:
        return 0.0
    per_type_values = []
    for values in by_type.values():
        if not values:
            continue
        per_type_values.append(sum(values) / float(len(values)))
    if not per_type_values:
        return 0.0
    return sum(per_type_values) / float(len(per_type_values))


def _group_metric(summary_rows: List[Dict[str, Any]], metric_key: str) -> Dict[str, float]:
    by_type: Dict[str, List[float]] = {}
    for row in summary_rows:
        group_label = _row_group_label(row)
        by_type.setdefault(group_label, []).append(float(row.get(metric_key, 0.0)))
    out: Dict[str, float] = {}
    for group_label, values in by_type.items():
        out[group_label] = (sum(values) / float(len(values))) if values else 0.0
    return out


def _operationally_stable(summary_rows: List[Dict[str, Any]]) -> tuple[bool, Dict[str, float]]:
    if not summary_rows:
        return False, {"empty_result_rate": 1.0, "timeout_rate": 1.0}
    total = float(len(summary_rows))
    empty_count = sum(1 for row in summary_rows if int(row.get("doc_count", 0)) <= 0)
    timeout_count = sum(1 for row in summary_rows if str(row.get("timeout_location", "none")) != "none")
    split_over_count = 0
    for row in summary_rows:
        branches = row.get("split_branches", [])
        if isinstance(branches, list) and len(branches) > 12:
            split_over_count += 1
    rates = {
        "empty_result_rate": empty_count / total,
        "timeout_rate": timeout_count / total,
        "branch_explosion_rate": split_over_count / total,
    }
    stable = (
        rates["empty_result_rate"] <= 0.05
        and rates["timeout_rate"] <= 0.05
        and rates["branch_explosion_rate"] <= 0.05
    )
    return stable, rates


def _profile_metric_lookup(
    macro_rows: List[Dict[str, Any]],
    profile: str,
    *,
    track: str = "retrieval",
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    wanted = {
        ("doc_recall", "10"): "doc_recall@10",
        ("chunk_recall", "10"): "chunk_recall@10",
        ("mrr", "10"): "mrr@10",
        ("doc_recall", "20"): "doc_recall@20",
        ("chunk_recall", "20"): "chunk_recall@20",
        ("mrr", "20"): "mrr@20",
    }
    for row in macro_rows:
        if str(row.get("profile")) != profile:
            continue
        if str(row.get("track")) != track:
            continue
        if str(row.get("aggregation")) != "macro":
            continue
        key = (str(row.get("metric")), str(row.get("k")))
        if key in wanted:
            try:
                out[wanted[key]] = float(row.get("value", 0.0))
            except Exception:
                out[wanted[key]] = 0.0
    return out


def _groupwise_metrics(summary_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    groups = sorted({_row_group_label(r) for r in summary_rows})
    out: Dict[str, Dict[str, float]] = {}
    for g in groups:
        rows = [r for r in summary_rows if _row_group_label(r) == g]
        if not rows:
            continue
        out[g] = {
            "doc_recall@10": sum(float(r.get("doc_recall_at_10", 0.0)) for r in rows) / len(rows),
            "chunk_recall@10": sum(float(r.get("chunk_recall_at_10", 0.0)) for r in rows) / len(rows),
            "mrr@10": sum(float(r.get("mrr_at_10", 0.0)) for r in rows) / len(rows),
            "doc_recall@20": sum(float(r.get("doc_recall_at_20", 0.0)) for r in rows) / len(rows),
            "chunk_recall@20": sum(float(r.get("chunk_recall_at_20", 0.0)) for r in rows) / len(rows),
            "mrr@20": sum(float(r.get("mrr_at_20", 0.0)) for r in rows) / len(rows),
        }
    return out


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _doc_top5_from_trace_row(trace_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs = trace_row.get("documents", [])
    if not isinstance(docs, list):
        return []
    out: List[Dict[str, Any]] = []
    for d in docs[:5]:
        out.append(
            {
                "source_file": d.get("source_file"),
                "chunk_id": d.get("chunk_id"),
                "doc_key": d.get("doc_key"),
                "score": d.get("score"),
            }
        )
    return out


def _trace_samples(path: Path, *, success_n: int = 5, fail_n: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    rows = _load_jsonl(path)
    successes: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for row in rows:
        zero_hit = bool(row.get("zero_hit", False))
        doc_count = int(row.get("doc_count", 0) or 0)
        item = {
            "query_id": row.get("query_id"),
            "query_text": row.get("query_text"),
            "detected_languages": row.get("detected_languages", []),
            "detected_parent_topics": row.get("detected_topics_categories", []),
            "detected_child_topics": row.get("detected_topics_phenomena", []),
            "detector_confidence": row.get("detector_confidence", {}),
            "filter_stage_used": row.get("filter_stage_used", "no_filter"),
            "fallback_stage": row.get("fallback_stage", 0),
            "doc_count": doc_count,
            "zero_hit": zero_hit,
            "top5": _doc_top5_from_trace_row(row),
        }
        if not zero_hit and doc_count > 0:
            successes.append(item)
        else:
            failures.append(item)
    return {"success": successes[:success_n], "failure": failures[:fail_n]}


def _export_split_profile_artifacts(
    *,
    profile_dir: Path,
    profile_name: str,
    profile_cfg: Dict[str, Any],
    trace_rows: Sequence[Dict[str, Any]],
) -> None:
    query_split_cfg = (
        (profile_cfg.get("retrieval", {}) or {}).get("query_split", {})
        if isinstance((profile_cfg.get("retrieval", {}) or {}).get("query_split"), dict)
        else {}
    )
    if not bool(query_split_cfg.get("enabled", False)):
        return

    split_rows: List[Dict[str, Any]] = []
    branch_rows: List[Dict[str, Any]] = []
    merge_rows: List[Dict[str, Any]] = []
    controller_mode = str(query_split_cfg.get("controller_mode", "heuristic")).strip().lower()

    for row in trace_rows:
        diagnostics = row.get("retrieval_diagnostics", {}) if isinstance(row.get("retrieval_diagnostics"), dict) else {}
        query_id = str(row.get("query_id", ""))
        query_type = str(row.get("query_type", "unknown"))
        group_label = _derive_group_label(query_id, query_type)
        should_split = bool(diagnostics.get("split_applied", False))
        branch_query_rows = [
            item for item in diagnostics.get("split_branch_queries", [])
            if isinstance(item, dict)
        ]
        materialized_plan = diagnostics.get("split_materialized_plan", {})
        if not isinstance(materialized_plan, dict):
            materialized_plan = {}

        if materialized_plan:
            extracted_targets = materialized_plan.get("extracted_targets", [])
            extracted_target_spans = materialized_plan.get("extracted_target_spans", [])
            resolved_target_lang_ids = materialized_plan.get("resolved_target_lang_ids", [])
            shared_ask = materialized_plan.get("shared_ask", "")
            branch_queries = [
                str(item.get("query", "")).strip()
                for item in materialized_plan.get("branch_queries", [])
                if isinstance(item, dict) and str(item.get("query", "")).strip()
            ]
            cleanliness_flags = materialized_plan.get("cleanliness_flags", {})
            policy_flags = materialized_plan.get("policy_flags", {})
            abstain_reason = "" if should_split else str(materialized_plan.get("abstain_reason", ""))
            notes = str(materialized_plan.get("notes", ""))
        else:
            extracted_targets = [
                str(item.get("target_label") or item.get("branch_effective_language") or "")
                for item in diagnostics.get("split_branch_diagnostics", [])
                if isinstance(item, dict) and str(item.get("target_label") or item.get("branch_effective_language") or "").strip()
            ]
            extracted_target_spans = []
            resolved_target_lang_ids = [
                str(item.get("target_lang_id") or item.get("branch_effective_language") or "").strip()
                for item in diagnostics.get("split_branch_diagnostics", [])
                if isinstance(item, dict) and str(item.get("target_lang_id") or item.get("branch_effective_language") or "").strip()
            ]
            shared_ask = ""
            branch_queries = [
                str(item.get("branch_query_text", "")).strip()
                for item in branch_query_rows
                if str(item.get("branch_query_text", "")).strip()
            ]
            branch_contract_rows = [
                {
                    "target_label": str(item.get("target_label") or item.get("branch_effective_language") or "").strip(),
                    "query": str(item.get("branch_query_text", "")).strip(),
                }
                for item in branch_query_rows
                if str(item.get("branch_query_text", "")).strip()
            ]
            cleanliness_flags = (
                validate_materialized_branch_queries(
                    str(row.get("query_text", "")),
                    branch_contract_rows,
                    extracted_targets,
                )
                if branch_contract_rows
                else {}
            )
            policy_flags = {"controller_mode": diagnostics.get("split_controller_mode", "heuristic")}
            abstain_reason = "" if should_split else str(diagnostics.get("split_reason", ""))
            notes = str(diagnostics.get("split_reason", ""))

        split_rows.append(
            {
                "query_id": query_id,
                "original_query": row.get("query_text", ""),
                "group_label": group_label,
                "should_split": should_split,
                "split_count": len(branch_queries) if should_split else 0,
                "axis": diagnostics.get("split_dimension", "none") if should_split else "none",
                "extracted_targets": extracted_targets,
                "extracted_target_spans": extracted_target_spans,
                "resolved_target_lang_ids": resolved_target_lang_ids,
                "shared_ask": shared_ask,
                "branch_queries": branch_queries,
                "abstain_reason": abstain_reason,
                "cleanliness_flags": cleanliness_flags,
                "policy_flags": policy_flags,
                "notes": notes,
            }
        )

        for item in diagnostics.get("split_branch_diagnostics", []):
            if not isinstance(item, dict):
                continue
            branch_rows.append(
                {
                    "query_id": query_id,
                    "branch_id": item.get("branch_id", ""),
                    "target_label": item.get("target_label") or item.get("branch_effective_language") or "",
                    "target_lang_id": item.get("target_lang_id") or item.get("branch_effective_language") or "",
                    "actual_query_sent": item.get("actual_query_sent") or item.get("branch_query_text") or "",
                    "retrieval_mode": (
                        "t71_materialized_language_split"
                        if controller_mode == "t71_materialized_language_local_b6"
                        else (
                            "t7_materialized_language_split"
                            if controller_mode == "t7_materialized_language_local_b6"
                            else ("materialized_language_split" if controller_mode == "materialized_language_only" else "frozen_b7_heuristic_split")
                        )
                    ),
                    "filter_stage_used": item.get("filter_stage_used", "no_filter"),
                    "local_reranker_active": bool(item.get("local_reranker_active", False)),
                    "anchor_retained": bool(item.get("anchor_retained", False)),
                    "topic_anchor": item.get("topic_anchor", ""),
                    "top_docs": item.get("top_docs", []),
                    "top_chunks": item.get("top_chunks", []),
                    "topk": len(item.get("top_chunks", [])),
                    "notes": item.get("branch_query_mode", ""),
                }
            )

        merge_trace = diagnostics.get("split_merge_trace", {})
        if isinstance(merge_trace, dict) and merge_trace:
            merge_rows.append(dict(merge_trace))

    _write_jsonl(profile_dir / "split_materialized.jsonl", split_rows)
    _write_jsonl(profile_dir / "branch_retrieval_traces.jsonl", branch_rows)
    _write_jsonl(profile_dir / "merge_trace.jsonl", merge_rows)
    if controller_mode in {"materialized_language_only", "t7_materialized_language_local_b6", "t71_materialized_language_local_b6"}:
        (profile_dir / "prompt_bundle.md").write_text(
            MATERIALIZED_LANGUAGE_SPLIT_PROMPT_BUNDLE.rstrip() + "\n",
            encoding="utf-8",
        )


def _select_b5_candidate_from_dev(
    *,
    candidate_results: Dict[str, Dict[str, Any]],
    parent_results: Dict[str, Dict[str, Any]],
) -> tuple[str | None, Dict[str, Any]]:
    parent_map = {
        "C1": ["B2", "B4"],
        "C2": ["B3", "B4"],
        "C3": ["B2", "B3", "B4"],
    }
    audits: Dict[str, Any] = {}
    winners: List[tuple[float, str]] = []

    for candidate_name, result in candidate_results.items():
        summary_rows = result.get("summary_rows") or []
        macro_mrr10 = _macro_metric(summary_rows, "mrr_at_10")
        candidate_group_chunk = _group_metric(summary_rows, "chunk_recall_at_10")

        drop_ok = True
        drop_detail: Dict[str, float] = {}
        for q_type, cand_value in candidate_group_chunk.items():
            parent_best = 0.0
            for parent_name in parent_map.get(candidate_name, []):
                parent_summary = (parent_results.get(parent_name) or {}).get("summary_rows") or []
                parent_group = _group_metric(parent_summary, "chunk_recall_at_10")
                parent_best = max(parent_best, float(parent_group.get(q_type, 0.0)))
            drop = parent_best - cand_value
            drop_detail[q_type] = round(drop, 6)
            if drop > 0.03:
                drop_ok = False

        stable, stability_rates = _operationally_stable(summary_rows)
        passes = drop_ok and stable
        audits[candidate_name] = {
            "macro_mrr10": round(macro_mrr10, 6),
            "drop_ok": drop_ok,
            "drop_detail": drop_detail,
            "stable": stable,
            "stability_rates": stability_rates,
            "passes": passes,
        }
        if passes:
            winners.append((macro_mrr10, candidate_name))

    if winners:
        winners.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return winners[0][1], {"candidates": audits, "selection_mode": "strict"}

    fallback: List[tuple[float, str]] = []
    for candidate_name, result in candidate_results.items():
        summary_rows = result.get("summary_rows") or []
        fallback.append((_macro_metric(summary_rows, "mrr_at_10"), candidate_name))
    if not fallback:
        return None, {"candidates": audits, "selection_mode": "none"}
    fallback.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return fallback[0][1], {"candidates": audits, "selection_mode": "fallback_best_mrr"}


def run_profile(
    *,
    run_id: str,
    profile_name: str,
    profile_cfg: Dict[str, Any],
    queries: List[Dict[str, Any]],
    out_dir: Path,
) -> Dict[str, Any]:
    profile_dir = out_dir / profile_name
    profile_dir.mkdir(parents=True, exist_ok=True)

    run_mode = str((profile_cfg.get("ablation", {}) or {}).get("run_mode", "retrieval")).lower()
    if run_mode == "e2e":
        return _run_e2e_profile(
            run_id=run_id,
            profile_name=profile_name,
            profile_cfg=profile_cfg,
            queries=queries,
            profile_dir=profile_dir,
        )

    return _run_retrieval_profile(
        run_id=run_id,
        profile_name=profile_name,
        profile_cfg=profile_cfg,
        queries=queries,
        profile_dir=profile_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase A/B retrieval ablation harness.")
    parser.add_argument("--queries", required=True, help="Path to QA JSON/JSONL.")
    parser.add_argument(
        "--profile-config",
        default="config/ablation_profiles.yaml",
        help="Path to ablation profile config YAML.",
    )
    parser.add_argument(
        "--profiles",
        default="",
        help="Comma-separated profile names (default: all from config).",
    )
    parser.add_argument(
        "--outdir",
        default="eval/runs",
        help="Output base directory.",
    )
    parser.add_argument("--run-id", default="", help="Optional run id. Default UTC timestamp.")
    parser.add_argument("--fixed-k", type=int, default=None, help="Override fixed_k for all profiles.")
    parser.add_argument(
        "--fixed-top-n",
        type=int,
        default=None,
        help="Override fixed_top_n for all profiles.",
    )
    parser.add_argument(
        "--filtering",
        choices=sorted(FILTERING_MODE_CHOICES),
        default=FILTERING_MODE_DEFAULT,
        help=(
            "Filtering mode for B4/B6/B7. "
            "Default 'viking' enables Viking-scoped filtering; "
            "'metadata' restores the previous metadata-only behavior."
        ),
    )
    parser.add_argument(
        "--b5-frozen-candidate",
        default="",
        help="Optional fixed B5 candidate profile (C1/C2/C3). Skips dev selection when set.",
    )
    args = parser.parse_args()

    base_config = load_config()
    profile_path = Path(args.profile_config)
    defaults, profiles_map = load_profile_file(profile_path)
    selected_profiles = select_profiles(profiles_map, args.profiles)
    if not args.profiles.strip():
        selected_profiles = [p for p in selected_profiles if not p.startswith("C")]

    queries = _load_qaset(Path(args.queries))
    if not queries:
        raise ValueError("No queries loaded from input file.")

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.outdir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "run_id": run_id,
        "started_at": _utc_now(),
        "query_file": str(Path(args.queries)),
        "query_count": len(queries),
        "profile_config": str(profile_path),
        "filtering_mode": args.filtering,
        "profiles_planned": selected_profiles,
        "profiles_results": [],
    }
    run_meta_path = out_dir / "run_meta.json"
    _write_json(run_meta_path, run_meta)

    explicit_split_used = False
    try:
        dev_queries, final_queries, dev_ids_by_group = _split_dev_test_queries(queries)
    except ValueError:
        # Stage-specific subset files may not contain all 3 canonical groups.
        # In that case, run both dev/final on the provided subset as-is.
        dev_queries = list(queries)
        final_queries = list(queries)
        by_group: Dict[str, List[str]] = {}
        for q in queries:
            group_label = _derive_group_label(
                str(q.get("query_id", "")),
                str(q.get("query_type", "unknown")),
            )
            by_group.setdefault(group_label, []).append(str(q.get("query_id", "")))
        dev_ids_by_group = by_group
        explicit_split_used = True
    run_meta["dev_query_count"] = len(dev_queries)
    run_meta["final_query_count"] = len(final_queries)
    run_meta["explicit_eval_split"] = explicit_split_used
    run_meta["balanced_dev_seed"] = 20260304
    run_meta["balanced_dev_ids_by_group"] = dev_ids_by_group
    _write_json(run_meta_path, run_meta)
    _write_json(out_dir / "balanced_dev45_query_ids.json", dev_ids_by_group)
    _write_csv(
        out_dir / "balanced_dev45_query_ids.csv",
        [
            {"group_label": group_label, "query_id": qid}
            for group_label, qids in dev_ids_by_group.items()
            for qid in qids
        ],
    )

    def _resolve_cfg(profile_name: str, profile_overrides: Dict[str, Any]) -> Dict[str, Any]:
        cfg = resolve_profile_config(
            base_config=base_config,
            defaults=defaults,
            profile_name=profile_name,
            profile_overrides=profile_overrides,
            fixed_k=args.fixed_k,
            fixed_top_n=args.fixed_top_n,
            filtering_mode=args.filtering,
        )
        llm_retrieval_cfg = cfg.setdefault("llm_retrieval", {})
        timeout_cfg_value = _safe_int(llm_retrieval_cfg.get("request_timeout_sec", 600), 600)
        llm_retrieval_cfg["request_timeout_sec"] = timeout_cfg_value if timeout_cfg_value > 0 else 600
        return cfg

    b5_selection_cfg = (profiles_map.get("B5", {}) or {}).get("ablation", {}) or {}
    default_ablation_cfg = defaults.get("ablation", {}) or {}
    candidate_profiles = b5_selection_cfg.get("b5_candidates") or default_ablation_cfg.get("b5_candidates") or ["C1", "C2", "C3"]
    candidate_profiles = [str(v).strip() for v in candidate_profiles if str(v).strip()]
    frozen_b5_candidate = args.b5_frozen_candidate.strip() or str(
        b5_selection_cfg.get("b5_frozen_candidate") or default_ablation_cfg.get("b5_frozen_candidate") or ""
    ).strip()

    def _base_profile_for(profile_name: str) -> str:
        ablation_block = (profiles_map.get(profile_name, {}) or {}).get("ablation", {}) or {}
        base_profile = str(ablation_block.get("base_profile") or "").strip()
        if base_profile and base_profile in profiles_map:
            return base_profile
        return ""

    needs_b5_selection = any(
        p in {"B5", "B6", "B7"}
        and not _base_profile_for(p)
        for p in selected_profiles
    )

    if needs_b5_selection and not frozen_b5_candidate:
        parent_profiles = {"B2", "B3", "B4"}
        dev_profiles = [p for p in set(candidate_profiles).union(parent_profiles) if p in profiles_map]
        dev_results: Dict[str, Dict[str, Any]] = {}
        dev_dir = out_dir / "_b5_dev_select"
        dev_dir.mkdir(parents=True, exist_ok=True)
        for profile_name in sorted(dev_profiles):
            profile_overrides = deepcopy(profiles_map.get(profile_name) or {})
            profile_overrides.pop("description", None)
            profile_cfg = _resolve_cfg(profile_name, profile_overrides)
            result = run_profile(
                run_id=f"{run_id}-dev",
                profile_name=profile_name,
                profile_cfg=profile_cfg,
                queries=dev_queries,
                out_dir=dev_dir,
            )
            dev_results[profile_name] = result

        candidate_results = {k: v for k, v in dev_results.items() if k.startswith("C")}
        parent_results = {k: v for k, v in dev_results.items() if k in {"B2", "B3", "B4"}}
        selected_candidate, selection_audit = _select_b5_candidate_from_dev(
            candidate_results=candidate_results,
            parent_results=parent_results,
        )
        frozen_b5_candidate = selected_candidate or ""
        run_meta["b5_selection"] = {
            "selected_candidate": frozen_b5_candidate,
            "audit": selection_audit,
            "dev_profiles_run": sorted(dev_profiles),
            "dev_query_count": len(dev_queries),
        }
        _write_json(run_meta_path, run_meta)

    retrieval_macro_rows: List[Dict[str, Any]] = []
    e2e_macro_rows: List[Dict[str, Any]] = []
    profile_summary_rows: Dict[str, List[Dict[str, Any]]] = {}

    for profile_name in selected_profiles:
        profile_entry = deepcopy(profiles_map.get(profile_name) or {})
        description = str(profile_entry.get("description", ""))
        effective_profile_name = profile_name
        base_chain = []
        base_profile_name = _base_profile_for(profile_name)
        if base_profile_name:
            base_chain = [*profile_base_chain(profiles_map, profile_name)]
            profile_overrides = resolve_profile_overrides(profiles_map, profile_name)
            description = f"{description} (base={'->'.join(base_chain)})".strip()
        elif frozen_b5_candidate and frozen_b5_candidate in profiles_map and profile_name in {"B5", "B6", "B7", "B8"}:
            profile_entry.pop("description", None)
            profile_overrides = resolve_profile_overrides(profiles_map, frozen_b5_candidate)
            profile_overrides = deep_merge(profile_overrides, profile_entry)
            description = f"{description} (base={frozen_b5_candidate})".strip()
        else:
            profile_entry.pop("description", None)
            profile_overrides = profile_entry

        profile_cfg = _resolve_cfg(effective_profile_name, profile_overrides)

        status_row = {
            "profile": profile_name,
            "effective_profile": effective_profile_name,
            "description": description,
            "db_name": str((profile_cfg.get("database", {}) or {}).get("dbname", "")),
            "db_table": str((profile_cfg.get("database", {}) or {}).get("table_name", "")),
            "status": "running",
            "started_at": _utc_now(),
        }
        run_meta["profiles_results"].append(status_row)
        _write_json(run_meta_path, run_meta)

        try:
            profile_queries = dev_queries if profile_name.startswith("C") else final_queries
            result = run_profile(
                run_id=run_id,
                profile_name=profile_name,
                profile_cfg=profile_cfg,
                queries=profile_queries,
                out_dir=out_dir,
            )
            status_row.update(
                {
                    "status": "completed",
                    "completed_at": _utc_now(),
                    "records": result.get("records", 0),
                    "track": result.get("track", "retrieval"),
                    "backend_mode_effective": result.get("backend_mode_effective", "unknown"),
                    "backend_mode_config": result.get("backend_mode_config", "unknown"),
                    "backend_mode_required": result.get("backend_mode_required", "any"),
                    "config_hash": result.get("config_hash", ""),
                    "query_count_used": len(profile_queries),
                }
            )
            if result.get("track") == "e2e":
                e2e_macro_rows.extend(result.get("macro_micro_rows", []))
            else:
                retrieval_macro_rows.extend(result.get("macro_micro_rows", []))
                profile_summary_rows[profile_name] = list(result.get("summary_rows", []))
        except Exception as exc:
            status_row.update(
                {
                    "status": "failed",
                    "completed_at": _utc_now(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )

        _write_json(run_meta_path, run_meta)

    _write_csv(out_dir / "retrieval_macro_micro.csv", retrieval_macro_rows)
    _write_csv(out_dir / "e2e_macro_micro.csv", e2e_macro_rows)

    # Phase-A query-only report package.
    phase_profiles = [p for p in selected_profiles if p in {"B0", "B1", "B2", "B3", "B4", "C1", "C2", "C3"}]
    profile_reports: List[Dict[str, Any]] = []
    for profile_name in phase_profiles:
        summary_rows = profile_summary_rows.get(profile_name, [])
        metrics = _profile_metric_lookup(retrieval_macro_rows, profile_name, track="retrieval")
        empty_result_count = sum(1 for r in summary_rows if int(r.get("doc_count", 0) or 0) <= 0)
        timeout_count = sum(1 for r in summary_rows if str(r.get("timeout_location", "none")) != "none")
        warning_count = sum(
            1
            for r in summary_rows
            if isinstance(r.get("warnings"), list) and len(r.get("warnings")) > 0
        )
        profile_reports.append(
            {
                "profile": profile_name,
                "query_count_used": len(summary_rows),
                "macro": metrics,
                "group_wise": _groupwise_metrics(summary_rows),
                "empty_result_count": empty_result_count,
                "timeout_count": timeout_count,
                "warning_count": warning_count,
            }
        )

    trace_profiles = ["B4", "C1", "C2", "C3"]
    trace_samples = {}
    for p_name in trace_profiles:
        trace_path = out_dir / p_name / "retrieval_traces.jsonl"
        trace_samples[p_name] = _trace_samples(trace_path, success_n=5, fail_n=5)

    b4_zero_hit_10 = []
    for row in _load_jsonl(out_dir / "B4" / "retrieval_traces.jsonl"):
        if bool(row.get("zero_hit", False)) or int(row.get("doc_count", 0) or 0) <= 0:
            b4_zero_hit_10.append(
                {
                    "query_id": row.get("query_id"),
                    "query_text": row.get("query_text"),
                    "detected_languages": row.get("detected_languages", []),
                    "detected_parent_topics": row.get("detected_topics_categories", []),
                    "detected_child_topics": row.get("detected_topics_phenomena", []),
                    "filter_stage_used": row.get("filter_stage_used", "no_filter"),
                    "fallback_stage": row.get("fallback_stage", 0),
                    "top5": _doc_top5_from_trace_row(row),
                }
            )
            if len(b4_zero_hit_10) >= 10:
                break

    report_payload = {
        "run_id": run_id,
        "query_file": str(Path(args.queries)),
        "balanced_dev_seed": 20260304,
        "balanced_dev_ids_by_group": dev_ids_by_group,
        "profiles": profile_reports,
        "trace_samples": trace_samples,
        "b4_zero_hit_cases_10": b4_zero_hit_10,
    }
    _write_json(out_dir / "phasea_query_only_report.json", report_payload)

    run_meta["completed_at"] = _utc_now()
    run_meta["status"] = "completed"
    run_meta["counts"] = {
        "planned": len(selected_profiles),
        "completed": len([r for r in run_meta["profiles_results"] if r.get("status") == "completed"]),
        "failed": len([r for r in run_meta["profiles_results"] if r.get("status") == "failed"]),
    }
    if frozen_b5_candidate:
        run_meta["b5_frozen_candidate"] = frozen_b5_candidate
    if run_meta["counts"]["failed"] > 0:
        run_meta["status"] = "completed_with_issues"
    _write_json(run_meta_path, run_meta)

    print(f"[Ablation] Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
