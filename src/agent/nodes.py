import threading
import json
import time
import re
from collections import defaultdict
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

import src.utils
from src.agent.state import GraphState
from src.agent.chains import get_grade_chain, get_rewrite_chain, get_hallucination_chain
from src.eval.finalized_b7_runtime import (
    FINALIZED_B7_CONTROLLER_MODES,
    run_finalized_b7_retrieval,
)
from src.eval.metadata_store import load_metadata_store
from src.llm.factory import get_llm
from src.llm.json_retry import parse_json_strict, repair_json_text
from src.llm.rag_generate import generate_answer
from src.retrieve.rag_retrieve import RAGRetriever, extract_query_metadata
from src.retrieve.split_budget import compute_split_budgets
from src.retrieve.viking_lexicon_loader import load_viking_lexicon

def _build_structured_metadata(d) -> Dict[str, str]:
    """
    Build structured metadata from source_file split.
    Fallback is used only when split fields are missing.
    """
    source_file = str(d.metadata.get("source_file", "unknown"))
    parts = [p for p in source_file.split("/") if p]

    language = d.metadata.get("lang")
    if not language:
        if len(parts) >= 2 and parts[0] == "doc":
            language = parts[1]
        else:
            language = "unknown"

    filename = parts[-1] if parts else "unknown"

    topic = parts[-2] if len(parts) >= 2 else ""
    if not topic:
        # Error-only fallback
        fallback_header = str(d.metadata.get("original_header", "")).strip()
        topic = fallback_header or "unknown"

    return {
        "Language": str(language),
        "Filename": str(filename),
        "Topic": str(topic),
        "SourceFile": source_file,
    }


def _format_structured_block(meta: Dict[str, str]) -> str:
    return (
        "[Structured Metadata]\n"
        f"Language: {meta['Language']}\n"
        f"Filename: {meta['Filename']}\n"
        f"Topic: {meta['Topic']}\n"
        f"SourceFile: {meta['SourceFile']}"
    )

def _get_generation_timeout_sec(default_sec: int = 600) -> int:
    try:
        config = src.utils.load_config()
        sec = int(config.get("rag", {}).get("generation_timeout_sec", default_sec))
        return sec if sec > 0 else default_sec
    except Exception:
        return default_sec


def _is_timeout_exception(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "timeout" in msg
        or "read timed out" in msg
        or "timed out" in msg
    )


def _append_warning(
    warnings: List[str],
    timeout_locations: List[str],
    message: str,
    timeout_location: str | None = None,
) -> None:
    if message not in warnings:
        warnings.append(message)
    if timeout_location and timeout_location not in timeout_locations:
        timeout_locations.append(timeout_location)


def _get_hallucination_settings() -> Dict[str, Any]:
    try:
        rag_cfg = src.utils.load_config().get("rag", {}) or {}
    except Exception:
        rag_cfg = {}
    mode = str(rag_cfg.get("hallucination_mode", "two_stage")).strip().lower()
    if mode not in {"legacy", "two_stage"}:
        mode = "two_stage"
    ambiguous_policy = str(
        rag_cfg.get("hallucination_ambiguous_policy", "pass_with_warning")
    ).strip().lower()
    if ambiguous_policy not in {"pass_with_warning", "rewrite"}:
        ambiguous_policy = "pass_with_warning"
    return {
        "hallucination_mode": mode,
        "hallucination_ambiguous_policy": ambiguous_policy,
        "skip_checker_on_stage1_pass": bool(
            rag_cfg.get("skip_checker_on_stage1_pass", True)
        ),
        "skip_rewrite_on_stable_retrieval": bool(
            rag_cfg.get("skip_rewrite_on_stable_retrieval", True)
        ),
        "allow_stable_pass_on_catastrophic_guard_fail": bool(
            rag_cfg.get("allow_stable_pass_on_catastrophic_guard_fail", False)
        ),
        "stable_retrieval_jaccard_threshold": float(
            rag_cfg.get("stable_retrieval_jaccard_threshold", 0.9)
        ),
    }


def _get_generation_settings() -> Dict[str, Any]:
    try:
        rag_cfg = src.utils.load_config().get("rag", {}) or {}
    except Exception:
        rag_cfg = {}
    return {
        "grounding_mode": str(rag_cfg.get("grounding_mode", "strict_extractive")).strip().lower(),
        "require_inline_citations": bool(rag_cfg.get("require_inline_citations", True)),
        "max_claims_per_answer": max(1, int(rag_cfg.get("max_claims_per_answer", 12))),
        "missing_info_phrase": str(rag_cfg.get("missing_info_phrase", "정보가 부족합니다")).strip() or "정보가 부족합니다",
        "grounding_min_citation_coverage": float(rag_cfg.get("grounding_min_citation_coverage", 1.0)),
        "fail_closed_on_guard": bool(rag_cfg.get("fail_closed_on_guard", False)),
        "enforce_korean_output_for_korean_query": bool(
            rag_cfg.get("enforce_korean_output_for_korean_query", True)
        ),
        "minimum_hangul_ratio": float(rag_cfg.get("minimum_hangul_ratio", 0.10)),
        "claim_count_fail_closed_min_coverage": float(
            rag_cfg.get("claim_count_fail_closed_min_coverage", 0.80)
        ),
        "readable_fail_closed_response_enabled": bool(
            rag_cfg.get("readable_fail_closed_response_enabled", True)
        ),
    }


def _get_grading_settings() -> Dict[str, Any]:
    try:
        rag_cfg = src.utils.load_config().get("rag", {}) or {}
    except Exception:
        rag_cfg = {}
    return {
        "branch_local_grading_enabled": bool(
            rag_cfg.get("branch_local_grading_enabled", True)
        ),
        "keep_maybe_docs_in_grading": bool(
            rag_cfg.get("keep_maybe_docs_in_grading", True)
        ),
    }


def _question_prefers_korean(text: str | None) -> bool:
    if not text:
        return False
    return bool(re.search(r"[가-힣]{2,}", text))


def _query_focus_terms(text: str | None) -> List[str]:
    if not text:
        return []
    raw_terms = re.findall(r"[가-힣A-Za-z]+", str(text).lower())
    stop_terms = {
        "설명", "설명해", "주세요", "비교", "비교하여", "나열", "나열해", "특징",
        "그", "및", "와", "과", "에서", "대한", "대해", "방법", "형태", "the",
        "and", "for", "with", "about", "please", "show", "list", "explain",
    }
    terms: List[str] = []
    seen: set[str] = set()
    for term in raw_terms:
        if re.search(r"[가-힣]", term):
            term = re.sub(r"(으로|에서|에게|한테|보다|부터|까지|처럼|이나|나|도|만|의|에|을|를|은|는|이|가|와|과)$", "", term)
        if len(term) <= 1 or term in stop_terms or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def _question_targets_noun_number(text: str | None) -> bool:
    q = str(text or "").lower()
    number_markers = [
        "복수", "단수", "명사의 수", "수 표지", "복수 표지", "plural", "singular",
        "number marking", "number", "plural marker",
    ]
    return any(marker in q for marker in number_markers)


def _question_targets_information_structure(text: str | None) -> bool:
    q = str(text or "").lower()
    markers = [
        "정보구조", "화제", "주제", "주어", "목적어", "어순", "정렬", "정보역할",
        "topic", "focus", "subject", "object", "alignment", "word order",
    ]
    return any(marker in q for marker in markers)


def _question_requests_examples(text: str | None) -> bool:
    q = str(text or "").lower()
    example_markers = [
        "예문", "예시", "보기", "실례", "example", "examples",
    ]
    return any(marker in q for marker in example_markers)


def _question_requests_comparison(text: str | None) -> bool:
    q = str(text or "").lower()
    comparison_markers = [
        "비교", "차이", "대조", "각각", "compare", "comparison", "contrast",
    ]
    return any(marker in q for marker in comparison_markers)


def _question_targets_interrogative_sentence(text: str | None) -> bool:
    q = str(text or "").lower()
    if not q:
        return False
    if "의문사" in q or "interrogative pronoun" in q:
        return False
    markers = [
        "판정 의문문",
        "의문문",
        "평서 의문문",
        "반복 의문문",
        "interrogative sentence",
        "polar question",
        "yes-no question",
    ]
    return any(marker in q for marker in markers)


def _runtime_interrogative_target_langs(metadata: Dict[str, Any]) -> List[str]:
    diagnostics = metadata.get("_diagnostics", {}) if isinstance(metadata, dict) else {}
    langs: List[str] = []
    if isinstance(diagnostics, dict):
        for branch in diagnostics.get("split_branch_queries", []) or []:
            lang_id = str(branch.get("target_lang_id", "") or "").strip()
            if lang_id and lang_id not in langs:
                langs.append(lang_id)
    if langs:
        return langs

    lang_value = metadata.get("lang") if isinstance(metadata, dict) else None
    if isinstance(lang_value, str):
        normalized = lang_value.strip()
        return [normalized] if normalized else []
    if isinstance(lang_value, list):
        for item in lang_value:
            normalized = str(item or "").strip()
            if normalized and normalized not in langs:
                langs.append(normalized)
    return langs


def _doc_has_interrogative_sentence_source(doc: Document, lang_id: str) -> bool:
    source_file = str(doc.metadata.get("source_file", "") or "")
    if not source_file.startswith(f"doc/{lang_id}/4_ss/44_stype/"):
        return False
    return source_file.endswith("/stype_q.html") or source_file.endswith("/44_stype.html")


def _runtime_interrogative_source_rescue(
    retriever: RAGRetriever,
    docs: List[Document],
    metadata: Dict[str, Any],
    question: str,
) -> tuple[List[Document], Dict[str, Any]]:
    diagnostics = {
        "applied": False,
        "added": 0,
        "langs": [],
    }
    if not _question_targets_interrogative_sentence(question):
        return docs, diagnostics
    if not isinstance(metadata, dict):
        return docs, diagnostics
    if not getattr(retriever, "conn", None):
        return docs, diagnostics

    target_langs = _runtime_interrogative_target_langs(metadata)
    if not target_langs:
        return docs, diagnostics

    diagnostics["applied"] = True
    existing_keys = {
        (str(doc.metadata.get("source_file", "") or ""), str(doc.metadata.get("chunk_id", "")))
        for doc in docs
    }
    rescued: List[Document] = []
    rescued_langs: List[str] = []
    table_name = str(getattr(retriever, "table_name", "linguistics_raptor") or "linguistics_raptor")
    query_sql = (
        f"SELECT content, metadata FROM {table_name}\n"
        "WHERE metadata->>'level' = '0'\n"
        "AND metadata->>'type' = 'L0_base'\n"
        "AND (metadata->>'source_file' = %s OR metadata->>'source_file' = %s)\n"
        "ORDER BY CASE WHEN metadata->>'source_file' = %s THEN 0 ELSE 1 END,\n"
        "         (metadata->>'chunk_id')::int ASC LIMIT %s"
    )

    with retriever.conn.cursor() as cur:
        for lang_id in target_langs:
            if any(_doc_has_interrogative_sentence_source(doc, lang_id) for doc in docs + rescued):
                continue
            stype_q_path = f"doc/{lang_id}/4_ss/44_stype/stype_q.html"
            stype_path = f"doc/{lang_id}/4_ss/44_stype/44_stype.html"
            cur.execute(query_sql, (stype_q_path, stype_path, stype_q_path, 4))
            rows = cur.fetchall()
            added_for_lang = 0
            for content, meta in rows:
                normalized_meta = _ensure_ref_metadata(meta)
                source_file = str(normalized_meta.get("source_file", "") or "")
                chunk_id = str(normalized_meta.get("chunk_id", "") or "")
                key = (source_file, chunk_id)
                if key in existing_keys:
                    continue
                actual_content = str(normalized_meta.get("original_content", content) or content or "")
                rescued_doc = Document(page_content=actual_content, metadata=normalized_meta)
                rescued_doc.metadata["_runtime_interrogative_rescue"] = True
                rescued_doc.metadata["_vector_rank"] = 10**9
                rescued_doc.metadata["score"] = float(rescued_doc.metadata.get("score", 0.0) or 0.0)
                rescued.append(rescued_doc)
                existing_keys.add(key)
                added_for_lang += 1
            if added_for_lang > 0:
                rescued_langs.append(lang_id)

    if rescued:
        print(
            "Runtime Interrogative Rescue: "
            f"added {len(rescued)} docs for langs={rescued_langs}"
        )
    diagnostics["added"] = len(rescued)
    diagnostics["langs"] = rescued_langs
    return docs + rescued, diagnostics


def _hangul_ratio(text: str | None) -> float:
    if not text:
        return 0.0
    hangul_chars = len(re.findall(r"[가-힣]", text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    total = hangul_chars + latin_chars
    if total <= 0:
        return 1.0
    return float(hangul_chars) / float(total)


def _is_output_language_mismatch(
    *,
    question: str | None,
    generation: str,
    enforce_korean_output: bool,
    minimum_hangul_ratio: float,
) -> tuple[bool, float]:
    if not enforce_korean_output or not _question_prefers_korean(question):
        return False, _hangul_ratio(generation)

    ratio = _hangul_ratio(generation)
    return ratio < max(0.0, min(1.0, float(minimum_hangul_ratio))), ratio


def _is_catastrophic_stage1_failure(
    generation_diagnostics: Dict[str, Any],
) -> bool:
    reason = str(generation_diagnostics.get("fail_closed_reason", "none")).strip().lower()
    if reason == "output_language_mismatch":
        return True

    unknown_tags = list(generation_diagnostics.get("unknown_citation_tags", []) or [])
    if unknown_tags:
        return True

    try:
        coverage = float(generation_diagnostics.get("inline_citation_coverage", 0.0))
    except Exception:
        coverage = 0.0
    return coverage <= 0.0


def _doc_fingerprint_key(d: Any) -> str:
    ref = _doc_canonical_ref(d).strip()
    if ref:
        return ref
    return "|".join([str(v) for v in _doc_identity(d)])


def _compute_retrieval_stability(history: List[List[str]], threshold: float) -> Dict[str, Any]:
    if len(history) < 2:
        return {
            "stable": False,
            "jaccard": 0.0,
            "threshold": threshold,
            "prev_count": 0,
            "curr_count": len(history[-1]) if history else 0,
        }
    prev = set(history[-2])
    curr = set(history[-1])
    union = prev | curr
    jaccard = (len(prev & curr) / len(union)) if union else 1.0
    return {
        "stable": bool(jaccard >= threshold and len(curr) > 0),
        "jaccard": jaccard,
        "threshold": threshold,
        "prev_count": len(prev),
        "curr_count": len(curr),
    }


def _canonical_ref_candidates(text: str) -> List[str]:
    if not text:
        return []
    pattern = r"[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+\.html:\d+"
    refs = re.findall(pattern, text)
    return list(dict.fromkeys([r.strip() for r in refs if r and r.strip()]))


def _canonicalize_ref_for_matching(ref: str) -> str:
    normalized = (ref or "").split(" [", 1)[0].strip().replace("\\", "/").lstrip("/")
    if not normalized:
        return ""
    if normalized.startswith("doc/"):
        parts = [p for p in normalized.split("/") if p]
        if len(parts) >= 3 and ".html:" in parts[-1]:
            return f"{parts[1]}/{parts[-1]}"
    return normalized


def _resolve_citation_token_against_allowed(token: str, allowed_refs: set[str]) -> List[str]:
    normalized = (token or "").split(" [", 1)[0].strip()
    if not normalized:
        return []

    resolved: set[str] = set()
    canonical_normalized = _canonicalize_ref_for_matching(normalized)
    if normalized in allowed_refs:
        resolved.add(_canonicalize_ref_for_matching(normalized))
    if canonical_normalized in allowed_refs:
        resolved.add(canonical_normalized)

    for candidate in _canonical_ref_candidates(normalized):
        canonical_candidate = _canonicalize_ref_for_matching(candidate)
        if candidate in allowed_refs or canonical_candidate in allowed_refs:
            resolved.add(canonical_candidate)

    if resolved:
        return sorted(r for r in resolved if r)

    if ".html:" not in normalized:
        return []

    suffix_matches: List[str] = []
    for ref in sorted(allowed_refs):
        if ref == normalized or ref.endswith(normalized) or ref.endswith(f"/{normalized}"):
            suffix_matches.append(ref)

    canonical_matches: List[str] = []
    for ref in suffix_matches:
        canonical = _canonical_ref_candidates(ref)
        candidates = canonical or [ref]
        canonical_matches.extend(
            [canon for canon in (_canonicalize_ref_for_matching(item) for item in candidates) if canon]
        )
    canonical_matches = list(dict.fromkeys(canonical_matches))
    if len(canonical_matches) == 1:
        return canonical_matches
    return []


def _doc_canonical_ref(d: Any) -> str:
    ref_id = str(d.metadata.get("ref_id", "")).strip()
    if ref_id:
        canon = _canonical_ref_candidates(ref_id)
        if canon:
            return canon[0]
        return ref_id.split(" [", 1)[0].strip()

    source_file = str(d.metadata.get("source_file", "")).strip().replace("\\", "/").lstrip("/")
    chunk_id = d.metadata.get("chunk_id")
    if source_file and chunk_id is not None:
        parts = [p for p in source_file.split("/") if p]
        if len(parts) >= 3 and parts[0] == "doc":
            return f"{parts[1]}/{parts[-1]}:{chunk_id}"
        return f"{source_file}:{chunk_id}"
    return source_file


def _ensure_ref_metadata(row_meta: Dict[str, Any]) -> Dict[str, Any]:
    meta = dict(row_meta or {})
    if str(meta.get("ref_id", "")).strip():
        return meta

    source_file = str(meta.get("source_file", "")).strip().replace("\\", "/").lstrip("/")
    chunk_id = meta.get("chunk_id")
    if not source_file or chunk_id is None:
        return meta

    parts = [p for p in source_file.split("/") if p]
    if len(parts) >= 3 and parts[0] == "doc":
        meta["ref_id"] = f"{parts[1]}/{parts[-1]}:{chunk_id}"
    else:
        meta["ref_id"] = f"{source_file}:{chunk_id}"
    return meta


def _allowed_ref_tags(documents: List[Any]) -> set[str]:
    refs: set[str] = set()
    for d in documents:
        ref_id = str(d.metadata.get("ref_id", "")).strip()
        source_file = str(d.metadata.get("source_file", "")).strip()
        chunk_id = d.metadata.get("chunk_id")
        if ref_id:
            refs.add(ref_id)
            trimmed_ref = ref_id.split(" [", 1)[0].strip()
            if trimmed_ref:
                refs.add(trimmed_ref)
            for canonical_ref in _canonical_ref_candidates(ref_id):
                refs.add(canonical_ref)
        if source_file:
            refs.add(source_file)
            if chunk_id is not None:
                refs.add(f"{source_file}:{chunk_id}")
        canonical_from_doc = _doc_canonical_ref(d)
        if canonical_from_doc:
            refs.add(canonical_from_doc)
    return refs


def _doc_ref_candidates(d: Any) -> set[str]:
    refs: set[str] = set()
    canonical_ref = _doc_canonical_ref(d)
    if canonical_ref:
        refs.add(canonical_ref)

    ref_id = str(d.metadata.get("ref_id", "")).strip()
    source_file = str(d.metadata.get("source_file", "")).strip()
    chunk_id = d.metadata.get("chunk_id")
    if ref_id:
        refs.add(ref_id)
        trimmed_ref = ref_id.split(" [", 1)[0].strip()
        if trimmed_ref:
            refs.add(trimmed_ref)
        for candidate in _canonical_ref_candidates(ref_id):
            refs.add(candidate)
    if source_file:
        refs.add(source_file)
        if chunk_id is not None:
            refs.add(f"{source_file}:{chunk_id}")
    return {
        _canonicalize_ref_for_matching(ref)
        for ref in refs
        if _canonicalize_ref_for_matching(ref)
    }


def _extract_bracket_tags(text: str) -> List[str]:
    tags = re.findall(r"\[([^\[\]\n]{1,160})\]", text or "")
    return [t.strip() for t in tags if t and t.strip()]


def _extract_citation_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    seen: set[str] = set()

    def _push(token: str) -> None:
        normalized = (token or "").strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        tokens.append(normalized)

    for tag in _extract_bracket_tags(text):
        canonical = _canonical_ref_candidates(tag)
        if canonical:
            for c in canonical:
                _push(c)
            continue
        normalized = tag.split(" [", 1)[0].strip()
        _push(normalized)

    for canonical in _canonical_ref_candidates(text):
        _push(canonical)

    return tokens


def _is_claim_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if s.endswith(":") and len(s) <= 40:
        return False
    lower = s.lower()
    if lower.startswith("reference sources"):
        return False
    if s.startswith("> [!WARNING]"):
        return False
    if re.fullmatch(r"[-*#\d\.\)\(\s:]+", s):
        return False
    return any(ch.isalnum() for ch in s)


def _is_markdown_table_separator(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if "|" not in s:
        return False
    compact = s.replace("|", "").replace(":", "").replace("-", "").replace(" ", "")
    return compact == ""


def _parse_grade_label(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return "no"
    token = re.split(r"\s+", text, maxsplit=1)[0].strip(".,;:()[]{}\"'")
    if token in {"yes", "maybe", "no", "no_topic", "no_other"}:
        if token in {"yes", "maybe"}:
            return token
        return "no"
    if "yes" in text and "no" not in text:
        return "yes"
    if "maybe" in text and "no" not in text:
        return "maybe"
    return "no"


def _is_case_only_borderline_doc(question: str, doc: Document) -> bool:
    if not _question_targets_noun_number(question):
        return False

    header = str(doc.metadata.get("original_header", "") or "").lower()
    content = str(doc.page_content or "").lower()
    sample = f"{header}\n{content[:400]}"

    case_markers = [
        "명사의 격", "격", "case", "주격", "대격", "속격", "처격", "여격", "탈격",
        "비교격", "도구격", "공동격",
    ]
    number_markers = [
        "명사의 수", "복수", "단수", "plural", "singular", "number", "복수 표지", "-wlak",
        "-ŋgula", "-sge", "nəge",
    ]

    has_case = any(marker in sample for marker in case_markers)
    has_number = any(marker in sample for marker in number_markers)
    return has_case and not has_number


def _is_information_structure_side_topic_doc(question: str, doc: Document) -> bool:
    if not _question_targets_information_structure(question):
        return False

    header = str(doc.metadata.get("original_header", "") or "").lower()
    content = str(doc.page_content or "").lower()
    sample = f"{header}\n{content[:500]}"

    info_markers = [
        "정보구조", "정보역할", "화제", "주제", "주어", "목적어", "어순", "정렬",
        "핵심논항", "topic", "subject", "object", "alignment", "word order",
    ]
    side_markers = [
        "격조사", "어기조사", "상 ", "상(", "aspect", "시제", "보조사", "부정", "particle",
    ]

    has_info = any(marker in sample for marker in info_markers)
    has_side = any(marker in sample for marker in side_markers)
    return has_side and not has_info


def _score_query_overlap(text: str, question: str | None) -> int:
    lower = str(text or "").lower()
    return sum(1 for term in _query_focus_terms(question) if term and term in lower)


def _score_fail_closed_doc(d: Any, *, question: str | None = None) -> tuple[int, int, int]:
    header = str(d.metadata.get("original_header", "") or "")
    content = str(d.page_content or "")
    score = (_score_query_overlap(header, question) * 4) + _score_query_overlap(content[:600], question)

    if _question_targets_noun_number(question) and _is_case_only_borderline_doc(question or "", d):
        score -= 8

    return (score, int(bool(header)), -len(content))


def _score_branch_rescue_doc(d: Any, *, question: str | None = None) -> tuple[int, int, int, int]:
    header = str(d.metadata.get("original_header", "") or "")
    content = str(d.page_content or "")
    ref_text = _doc_canonical_ref(d)
    overlap_score = (
        (_score_query_overlap(header, question) * 5)
        + (_score_query_overlap(content[:700], question) * 2)
        + (_score_query_overlap(ref_text, question) * 3)
    )

    if _question_targets_interrogative_sentence(question):
        sample = f"{header}\n{content[:500]}".lower()
        if "판정 의문문" in sample or "의문문" in sample or "interrogative" in sample:
            overlap_score += 4
        if re.search(r"(na\?|-(?:k|ka|kä)\b|\bq\b)", sample):
            overlap_score += 2

    if _question_targets_noun_number(question) and _is_case_only_borderline_doc(question or "", d):
        overlap_score -= 8
    if _question_targets_information_structure(question) and _is_information_structure_side_topic_doc(question or "", d):
        overlap_score -= 8

    return (
        overlap_score,
        _score_fail_closed_doc(d, question=question)[0],
        int(bool(header)),
        -len(content),
    )


def _doc_source_group_key(d: Any) -> str:
    source_file = str(d.metadata.get("source_file", "") or "").strip()
    if source_file:
        return source_file
    return _doc_canonical_ref(d)


def _score_source_group(
    docs: List[Any],
    *,
    question: str | None = None,
) -> tuple[int, int, int]:
    if not docs:
        return (0, 0, 0)

    top_doc_score = max(_score_fail_closed_doc(d, question=question)[0] for d in docs)
    source_overlap = sum(
        _score_query_overlap(
            f"{d.metadata.get('original_header', '')}\n{str(d.page_content or '')[:400]}",
            question,
        )
        for d in docs
    )
    return (
        (top_doc_score * 10) + (source_overlap * 3) + min(6, len(docs)) * 2,
        len(docs),
        source_overlap,
    )


def _order_fail_closed_docs(
    documents: List[Any],
    *,
    question: str | None = None,
) -> List[Any]:
    grouped: Dict[str, List[Any]] = defaultdict(list)
    source_order: List[str] = []
    for d in documents:
        key = _doc_source_group_key(d)
        if key not in grouped:
            source_order.append(key)
        grouped[key].append(d)

    ordered: List[Any] = []
    ranked_sources = sorted(
        source_order,
        key=lambda key: _score_source_group(grouped.get(key, []), question=question),
        reverse=True,
    )
    for key in ranked_sources:
        ordered.extend(
            sorted(
                grouped.get(key, []),
                key=lambda item: _score_fail_closed_doc(item, question=question),
                reverse=True,
            )
        )
    return ordered

def _line_has_allowed_citation(line: str, allowed_refs: set[str]) -> bool:
    tags = _extract_citation_tokens(line)
    if not tags:
        return False

    has_allowed = False
    for tag in tags:
        resolved = _resolve_citation_token_against_allowed(tag, allowed_refs)
        if resolved:
            has_allowed = True
            continue
        if "/" in tag or ".html" in tag:
            return False
    return has_allowed


def _trimmed_generation_is_too_sparse(
    trimmed_generation: str,
    *,
    trimmed_cov_stats: Dict[str, Any],
    pre_guard_cov_stats: Dict[str, Any],
    documents: List[Any],
    question: str | None = None,
) -> bool:
    trimmed_claim_count = int(trimmed_cov_stats.get("claim_count", 0) or 0)
    if trimmed_claim_count <= 0:
        return True

    pre_guard_claim_count = int(pre_guard_cov_stats.get("claim_count", 0) or 0)
    source_groups = {
        _doc_source_group_key(doc)
        for doc in documents
        if _doc_source_group_key(doc)
    }
    rich_answer_expected = (
        len(source_groups) >= 2
        or _question_requests_examples(question)
        or pre_guard_claim_count >= 4
    )
    if rich_answer_expected and trimmed_claim_count <= 1:
        return True

    if pre_guard_claim_count > 0:
        retained_ratio = trimmed_claim_count / float(pre_guard_claim_count)
        if rich_answer_expected and retained_ratio < 0.2 and len(trimmed_generation.strip()) < 180:
            return True

    return False


def _low_coverage_answer_is_unusable(
    *,
    cov_stats: Dict[str, Any],
    documents: List[Any],
    question: str | None = None,
) -> bool:
    claim_count = int(cov_stats.get("claim_count", 0) or 0)
    if claim_count <= 0:
        return True

    coverage = float(cov_stats.get("inline_citation_coverage", 0.0) or 0.0)
    source_groups = {
        _doc_source_group_key(doc)
        for doc in documents
        if _doc_source_group_key(doc)
    }
    rich_answer_expected = (
        len(source_groups) >= 2
        or _question_requests_examples(question)
        or claim_count >= 6
    )
    if not rich_answer_expected:
        return False
    return coverage < 0.2


def _comparison_branch_coverage_is_incomplete(
    generation: str,
    *,
    documents: List[Any],
    question: str | None = None,
) -> bool:
    if not _question_requests_comparison(question):
        return False

    available_langs = {
        _doc_language_tag(doc)
        for doc in documents
        if _doc_language_tag(doc) != "unknown"
    }
    if len(available_langs) < 2:
        return False

    allowed_refs = _allowed_ref_tags(documents)
    ref_lang_map: Dict[str, set[str]] = defaultdict(set)
    for doc in documents:
        lang = _doc_language_tag(doc)
        if lang == "unknown":
            continue
        for ref in _doc_ref_candidates(doc):
            ref_lang_map[ref].add(lang)

    cited_langs: set[str] = set()
    for line in (generation or "").splitlines():
        for tag in _extract_citation_tokens(line):
            for resolved in _resolve_citation_token_against_allowed(tag, allowed_refs):
                cited_langs.update(ref_lang_map.get(resolved, set()))

    return bool(cited_langs) and len(cited_langs) < len(available_langs)


def _trim_generation_to_supported_claims(
    generation: str,
    *,
    allowed_refs: set[str],
    max_claims: int,
) -> str:
    lines = (generation or "").splitlines()
    if not lines:
        return generation

    kept_lines: List[str] = []
    kept_claims = 0
    seen_claim = False
    idx = 0

    while idx < len(lines):
        raw = lines[idx].rstrip()
        stripped = raw.strip()
        if not stripped:
            if kept_lines and kept_lines[-1] != "":
                kept_lines.append("")
            idx += 1
            continue

        next_line = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
        is_table_header = "|" in stripped and _is_markdown_table_separator(next_line)
        if is_table_header:
            data_rows: List[str] = []
            j = idx + 2
            while j < len(lines):
                row = lines[j].rstrip()
                if "|" not in row.strip() or _is_markdown_table_separator(row.strip()):
                    break
                if _line_has_allowed_citation(row, allowed_refs):
                    data_rows.append(row)
                j += 1
            if data_rows:
                kept_lines.append(raw)
                kept_lines.append(lines[idx + 1].rstrip())
                for row in data_rows:
                    if kept_claims >= max_claims:
                        break
                    kept_lines.append(row)
                    kept_claims += 1
                seen_claim = True
            idx = j
            continue

        is_claim = _is_claim_line(stripped)
        if not is_claim:
            if not seen_claim or stripped.endswith(":") or len(stripped) <= 80:
                kept_lines.append(raw)
            idx += 1
            continue

        if _line_has_allowed_citation(stripped, allowed_refs):
            if kept_claims < max_claims:
                kept_lines.append(raw)
                kept_claims += 1
                seen_claim = True
        idx += 1

    trimmed = "\n".join(kept_lines).strip()
    return trimmed or generation


def _select_grading_question(
    question: str,
    doc: Document,
    retrieval_diagnostics: Dict[str, Any],
    *,
    branch_local_grading_enabled: bool,
) -> tuple[str, str]:
    if not branch_local_grading_enabled:
        return question, "original_question"

    doc_lang = str(
        doc.metadata.get("lang")
        or doc.metadata.get("lang_id")
        or ""
    ).strip()
    if not doc_lang:
        return question, "original_question"

    split_applied = bool(retrieval_diagnostics.get("split_applied", False))
    split_queries = retrieval_diagnostics.get("split_branch_queries", [])
    if not split_applied or not isinstance(split_queries, list):
        return question, "original_question"

    for row in split_queries:
        if not isinstance(row, dict):
            continue
        target_lang_id = str(row.get("target_lang_id", "")).strip()
        branch_query_text = str(row.get("branch_query_text", "")).strip()
        if target_lang_id and branch_query_text and target_lang_id == doc_lang:
            return branch_query_text, "branch_question"

    return question, "original_question"


def _rescue_empty_split_branches(
    *,
    original_documents: List[Document],
    kept_documents: List[Document],
    question: str,
    retrieval_diagnostics: Dict[str, Any],
) -> tuple[List[Document], Dict[str, Any]]:
    diagnostics = {
        "applied": False,
        "added": 0,
        "langs": [],
    }
    branch_queries = list(retrieval_diagnostics.get("split_branch_queries", []) or [])
    if not retrieval_diagnostics.get("split_applied") or not branch_queries:
        return kept_documents, diagnostics

    diagnostics["applied"] = True
    kept_keys = {_doc_fingerprint_key(doc) for doc in kept_documents if _doc_fingerprint_key(doc)}
    kept_langs = {_doc_language_tag(doc) for doc in kept_documents}
    rescued: List[Document] = []
    rescued_langs: List[str] = []

    for branch in branch_queries:
        lang_id = str(branch.get("target_lang_id", "") or "").strip()
        if not lang_id or lang_id in kept_langs:
            continue

        branch_question = str(branch.get("branch_query_text", "") or question).strip() or question
        candidates = [doc for doc in original_documents if _doc_language_tag(doc) == lang_id]
        if not candidates:
            continue

        ranked = sorted(
            candidates,
            key=lambda doc: _score_branch_rescue_doc(doc, question=branch_question),
            reverse=True,
        )

        added_for_lang = 0
        for doc in ranked:
            score = _score_branch_rescue_doc(doc, question=branch_question)[0]
            if score < 3:
                continue
            key = _doc_fingerprint_key(doc)
            if key and key in kept_keys:
                continue
            rescued.append(doc)
            if key:
                kept_keys.add(key)
            added_for_lang += 1
            if added_for_lang >= 2:
                break

        if added_for_lang > 0:
            rescued_langs.append(lang_id)

    if rescued:
        print(
            "  - EMPTY BRANCH RESCUE: "
            f"added {len(rescued)} docs for langs={rescued_langs}"
        )
    diagnostics["added"] = len(rescued)
    diagnostics["langs"] = rescued_langs
    return kept_documents + rescued, diagnostics


def _compute_citation_coverage(generation: str, allowed_refs: set[str]) -> Dict[str, Any]:
    raw_lines = (generation or "").splitlines()
    claim_lines: List[str] = []
    for idx, raw in enumerate(raw_lines):
        ln = raw.strip()
        if not _is_claim_line(ln):
            continue
        if _is_markdown_table_separator(ln):
            continue
        if "|" in ln:
            j = idx + 1
            while j < len(raw_lines) and not raw_lines[j].strip():
                j += 1
            if j < len(raw_lines) and _is_markdown_table_separator(raw_lines[j].strip()):
                # Markdown table header row is not a factual claim.
                continue
        claim_lines.append(ln)
    tagged_claims = 0
    unknown_tags: List[str] = []
    for ln in claim_lines:
        tags = _extract_citation_tokens(ln)
        if not tags:
            continue
        resolved_tags: List[str] = []
        line_unknown_tags: List[str] = []
        for tag in tags:
            resolved = _resolve_citation_token_against_allowed(tag, allowed_refs)
            resolved_tags.extend(resolved)
            if not resolved and ("/" in tag or ".html" in tag):
                line_unknown_tags.append(tag)
        has_allowed = bool(resolved_tags) and not line_unknown_tags
        if has_allowed:
            tagged_claims += 1
        unknown_tags.extend(line_unknown_tags)
    claim_count = len(claim_lines)
    coverage = (tagged_claims / claim_count) if claim_count > 0 else 0.0
    return {
        "claim_count": claim_count,
        "tagged_claim_count": tagged_claims,
        "inline_citation_coverage": coverage,
        "unknown_citation_tags": sorted(set(unknown_tags)),
    }


def _build_fail_closed_answer(
    documents: List[Any],
    missing_info_phrase: str,
    max_claims: int,
    *,
    question: str | None = None,
) -> str:
    settings = _get_generation_settings()
    if not bool(settings.get("readable_fail_closed_response_enabled", True)):
        return _build_fail_closed_answer_legacy(documents, missing_info_phrase, max_claims, question=question)
    return _build_fail_closed_answer_readable(documents, missing_info_phrase, max_claims, question=question)


def _build_fail_closed_answer_legacy(
    documents: List[Any],
    missing_info_phrase: str,
    max_claims: int,
    *,
    question: str | None = None,
) -> str:
    lines: List[str] = ["근거 기반 제한 응답:"]

    unique_docs: List[Any] = []
    seen_refs: set[str] = set()
    for d in documents:
        ref_id = _doc_canonical_ref(d)
        if not ref_id or ref_id in seen_refs:
            continue
        seen_refs.add(ref_id)
        unique_docs.append(d)

    if not unique_docs:
        return "\n".join([*lines, f"- {missing_info_phrase}."])

    summary_budget = max(1, min(4, max_claims))

    lines.append("")
    lines.append("핵심 근거:")
    summary_added = 0
    for d in unique_docs:
        ref_id = _doc_canonical_ref(d)
        content = re.sub(r"\s+", " ", (d.page_content or "").strip()).replace("^", "").strip()
        header = str(d.metadata.get("original_header", "")).strip()
        if not content:
            continue
        summary = _compact_fail_closed_text(content, max_chars=170)
        if header:
            lines.append(f"- {header}: {summary} [{ref_id}]")
        else:
            lines.append(f"- {summary} [{ref_id}]")
        summary_added += 1
        if summary_added >= summary_budget:
            break

    lines.append("")
    if summary_added >= 1:
        lines.append("- 위 항목은 근거가 확인된 범위만 정리했습니다.")
    else:
        lines.append(f"- 위 근거 외 세부사항은 {missing_info_phrase}.")
    return "\n".join(lines)


def _compact_fail_closed_text(text: str, *, max_chars: int = 120) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").replace("^", " ")).strip()
    if not cleaned:
        return ""

    sentence_patterns = [
        r"^(.{1,160}?(?:다|이다|있다|없다|된다|나뉜다|구성된다)\.)",
        r"^(.{1,160}?[.!?])",
    ]
    for pattern in sentence_patterns:
        m = re.match(pattern, cleaned)
        if m:
            sentence = m.group(1).strip()
            if sentence:
                return sentence

    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + " ..."


def _doc_language_tag(d: Any) -> str:
    lang = str(d.metadata.get("lang") or d.metadata.get("lang_id") or "").strip()
    if lang:
        return lang
    source_file = str(d.metadata.get("source_file", "")).strip().replace("\\", "/")
    parts = [p for p in source_file.split("/") if p]
    if len(parts) >= 2 and parts[0] == "doc":
        return parts[1]
    return "unknown"


def _build_readable_fact(d: Any, *, question: str | None = None) -> str:
    header = str(d.metadata.get("original_header", "")).strip()
    content = _compact_fail_closed_text(
        str(d.page_content or ""),
        max_chars=220,
    )
    if header and content:
        return f"{header}: {content}"
    if content:
        return content
    if header:
        return ""
    return "관련 근거가 확인되었다"


def _build_fail_closed_answer_readable(
    documents: List[Any],
    missing_info_phrase: str,
    max_claims: int,
    *,
    question: str | None = None,
) -> str:
    unique_docs: List[Any] = []
    seen_refs: set[str] = set()
    for d in documents:
        ref_id = _doc_canonical_ref(d)
        if not ref_id or ref_id in seen_refs:
            continue
        seen_refs.add(ref_id)
        unique_docs.append(d)

    if not unique_docs:
        return "\n".join([*lines, f"- {missing_info_phrase}."])

    docs_by_lang: Dict[str, List[Any]] = {}
    lang_order: List[str] = []
    for d in unique_docs:
        lang = _doc_language_tag(d)
        if lang not in docs_by_lang:
            docs_by_lang[lang] = []
            lang_order.append(lang)
        docs_by_lang[lang].append(d)

    multiple_langs = len(lang_order) >= 2
    if _question_requests_examples(question):
        fact_budget = max(1, min(8, max_claims))
    else:
        fact_budget = max(2, min(8, max_claims))
    per_lang_budget = max(1, fact_budget // max(1, len(lang_order)))
    if multiple_langs and not _question_requests_examples(question):
        per_lang_budget = min(per_lang_budget, 2)

    fact_lines: List[str] = []
    facts_added = 0
    for lang in lang_order:
        lang_docs = _order_fail_closed_docs(
            docs_by_lang.get(lang, []),
            question=question,
        )
        if not lang_docs:
            continue
        if multiple_langs:
            fact_lines.append("")
            fact_lines.append(f"{lang}:")
        added_for_lang = 0
        for d in lang_docs:
            doc_score = _score_fail_closed_doc(d, question=question)[0]
            if question and added_for_lang >= 1 and doc_score <= 0:
                continue
            ref_id = _doc_canonical_ref(d)
            fact = _build_readable_fact(d, question=question)
            if not fact:
                continue
            fact_lines.append(f"- {fact} [{ref_id}]")
            facts_added += 1
            added_for_lang += 1
            if added_for_lang >= per_lang_budget or facts_added >= fact_budget:
                break
        if facts_added >= fact_budget:
            break

    substantial_evidence = facts_added >= 2

    if substantial_evidence:
        if multiple_langs and not _question_requests_examples(question):
            grouped_lines = [line for line in fact_lines if line.strip()]
            return "\n".join(grouped_lines)
        lines: List[str] = []
        first_fact = next((line[2:] for line in fact_lines if line.startswith("- ")), "")
        if first_fact:
            lines.append(first_fact)
        remaining_facts = [
            line for line in fact_lines
            if line.startswith("- ")
        ]
        if first_fact and remaining_facts:
            remaining_facts = remaining_facts[1:]
        if remaining_facts:
            lines.append("")
            lines.append("설명:")
            lines.extend(remaining_facts)
        return "\n".join(lines)

    lines = ["근거 기반 제한 응답:", "", "확인된 근거:"]
    lines.extend(fact_lines)
    lines.append("")
    if facts_added >= 1:
        lines.append("- 위 항목은 근거가 확인된 범위만 정리했습니다.")
    else:
        lines.append(f"- 위 근거 외 세부사항은 {missing_info_phrase}.")
    return "\n".join(lines)


def _compress_generation_to_claim_budget(generation: str, max_claims: int) -> str:
    lines = (generation or "").splitlines()
    if not lines:
        return generation

    out_lines: List[str] = []
    kept_claims = 0
    prelude_done = False

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
            continue

        if _is_markdown_table_separator(stripped):
            if out_lines and "|" in out_lines[-1]:
                out_lines.append(line)
            continue

        is_claim = _is_claim_line(stripped)
        if not is_claim and not prelude_done:
            out_lines.append(line)
            continue

        prelude_done = True
        if is_claim:
            kept_claims += 1
            if kept_claims > max_claims:
                continue
        out_lines.append(line)

    compressed = "\n".join(out_lines).strip()
    return compressed or generation


def _apply_grounding_guard(
    generation: str,
    documents: List[Any],
    question: str | None = None,
) -> tuple[str, Dict[str, Any]]:
    settings = _get_generation_settings()
    mode = settings["grounding_mode"]
    require_inline = bool(settings["require_inline_citations"])
    min_cov = max(0.0, min(1.0, float(settings["grounding_min_citation_coverage"])))
    max_claims = int(settings["max_claims_per_answer"])
    missing_phrase = settings["missing_info_phrase"]
    fail_closed_on_guard = bool(settings["fail_closed_on_guard"])
    enforce_korean_output = bool(settings["enforce_korean_output_for_korean_query"])
    minimum_hangul_ratio = max(0.0, min(1.0, float(settings["minimum_hangul_ratio"])))
    claim_count_fail_closed_min_coverage = max(
        0.0, min(1.0, float(settings["claim_count_fail_closed_min_coverage"]))
    )
    allowed_refs = _allowed_ref_tags(documents)
    cov_stats = _compute_citation_coverage(generation, allowed_refs)
    language_mismatch, hangul_ratio = _is_output_language_mismatch(
        question=question,
        generation=generation,
        enforce_korean_output=enforce_korean_output,
        minimum_hangul_ratio=minimum_hangul_ratio,
    )
    pre_guard_cov_stats = dict(cov_stats)
    pre_guard_language_mismatch = bool(language_mismatch)
    pre_guard_hangul_ratio = float(hangul_ratio)

    if cov_stats["claim_count"] > max_claims:
        compressed_generation = _compress_generation_to_claim_budget(generation, max_claims)
        if compressed_generation != generation:
            compressed_cov_stats = _compute_citation_coverage(compressed_generation, allowed_refs)
            compressed_language_mismatch, compressed_hangul_ratio = _is_output_language_mismatch(
                question=question,
                generation=compressed_generation,
                enforce_korean_output=enforce_korean_output,
                minimum_hangul_ratio=minimum_hangul_ratio,
            )
            if (
                compressed_cov_stats["claim_count"] <= max_claims
                and not compressed_cov_stats["unknown_citation_tags"]
                and not compressed_language_mismatch
                and (
                    not require_inline
                    or compressed_cov_stats["inline_citation_coverage"] >= min_cov
                )
            ):
                generation = compressed_generation
                cov_stats = compressed_cov_stats
                language_mismatch = compressed_language_mismatch
                hangul_ratio = compressed_hangul_ratio

    if (
        require_inline
        and 0.0 < cov_stats["inline_citation_coverage"] < min_cov
        and not cov_stats["unknown_citation_tags"]
        and not language_mismatch
    ):
        trimmed_generation = _trim_generation_to_supported_claims(
            generation,
            allowed_refs=allowed_refs,
            max_claims=max_claims,
        )
        if trimmed_generation != generation:
            trimmed_cov_stats = _compute_citation_coverage(trimmed_generation, allowed_refs)
            trimmed_language_mismatch, trimmed_hangul_ratio = _is_output_language_mismatch(
                question=question,
                generation=trimmed_generation,
                enforce_korean_output=enforce_korean_output,
                minimum_hangul_ratio=minimum_hangul_ratio,
            )
            if (
                trimmed_cov_stats["claim_count"] > 0
                and not trimmed_cov_stats["unknown_citation_tags"]
                and not trimmed_language_mismatch
                and not _trimmed_generation_is_too_sparse(
                    trimmed_generation,
                    trimmed_cov_stats=trimmed_cov_stats,
                    pre_guard_cov_stats=pre_guard_cov_stats,
                    documents=documents,
                    question=question,
                )
            ):
                generation = trimmed_generation
                cov_stats = trimmed_cov_stats
                language_mismatch = trimmed_language_mismatch
                hangul_ratio = trimmed_hangul_ratio

    guard_triggered = False
    reason = "none"
    fail_closed_applied = False

    if mode == "strict_extractive":
        if cov_stats["claim_count"] > max_claims:
            if (
                cov_stats["inline_citation_coverage"] < claim_count_fail_closed_min_coverage
                or cov_stats["unknown_citation_tags"]
                or language_mismatch
            ):
                guard_triggered = True
                reason = "claim_count_exceeded"
        elif require_inline and cov_stats["inline_citation_coverage"] < min_cov:
            if (
                cov_stats["inline_citation_coverage"] <= 0.0
                or _low_coverage_answer_is_unusable(
                    cov_stats=cov_stats,
                    documents=documents,
                    question=question,
                )
            ):
                guard_triggered = True
                reason = "citation_coverage_low"
        elif _comparison_branch_coverage_is_incomplete(
            generation,
            documents=documents,
            question=question,
        ):
            guard_triggered = True
            reason = "branch_coverage_incomplete"
        elif cov_stats["unknown_citation_tags"]:
            guard_triggered = True
            reason = "unknown_citation_tags"
        elif language_mismatch:
            guard_triggered = True
            reason = "output_language_mismatch"

    if guard_triggered and fail_closed_on_guard:
        generation = _build_fail_closed_answer(
            documents=documents,
            missing_info_phrase=missing_phrase,
            max_claims=max_claims,
            question=question,
        )
        cov_stats = _compute_citation_coverage(generation, allowed_refs)
        fail_closed_applied = True

    diagnostics = {
        "grounding_mode": mode,
        "require_inline_citations": require_inline,
        "max_claims_per_answer": max_claims,
        "fail_closed_on_guard": fail_closed_on_guard,
        "fail_closed_applied": fail_closed_applied,
        "inline_citation_coverage": cov_stats["inline_citation_coverage"],
        "pre_guard_inline_citation_coverage": pre_guard_cov_stats["inline_citation_coverage"],
        "post_guard_inline_citation_coverage": cov_stats["inline_citation_coverage"],
        "claim_count_fail_closed_min_coverage": claim_count_fail_closed_min_coverage,
        "claim_count": cov_stats["claim_count"],
        "pre_guard_claim_count": pre_guard_cov_stats["claim_count"],
        "post_guard_claim_count": cov_stats["claim_count"],
        "tagged_claim_count": cov_stats["tagged_claim_count"],
        "unknown_citation_tags": cov_stats["unknown_citation_tags"],
        "pre_guard_unknown_citation_tags": pre_guard_cov_stats["unknown_citation_tags"],
        "post_guard_unknown_citation_tags": cov_stats["unknown_citation_tags"],
        "enforce_korean_output_for_korean_query": enforce_korean_output,
        "question_prefers_korean": _question_prefers_korean(question),
        "output_hangul_ratio": hangul_ratio,
        "pre_guard_output_hangul_ratio": pre_guard_hangul_ratio,
        "pre_guard_output_language_mismatch": pre_guard_language_mismatch,
        "output_language_mismatch": language_mismatch,
        "unsupported_claim_guard_triggered": guard_triggered,
        "fail_closed_reason": reason,
    }
    return generation, diagnostics


def _generate_with_timeout(query: str, context: str, timeout_sec: int) -> str:
    result = {"value": None, "error": None}
    timeout_fallback = (
        f"정보가 부족합니다\n\n(답변 생성 시간 제한 {timeout_sec}초를 초과했습니다.)"
    )

    def _worker():
        try:
            result["value"] = generate_answer(
                query=query,
                context=context,
                request_timeout_sec=timeout_sec,
            )
        except Exception as e:
            result["error"] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)

    if t.is_alive():
        return timeout_fallback

    if result["error"] is not None:
        if "timeout" in str(result["error"]).lower():
            return timeout_fallback
        raise result["error"]

    return result["value"]


def _doc_identity(doc) -> tuple:
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


def _doc_score(doc) -> float:
    try:
        score = doc.metadata.get("score")
        if score is None:
            return float("-inf")
        return float(score)
    except Exception:
        return float("-inf")


def _generate_split_queries(
    question: str,
    languages: List[str],
    retries: int,
    backoff_sec: float,
    json_retries: int,
    json_backoff_sec: float,
) -> tuple[Dict[str, str], int]:
    llm = get_llm("retrieval")
    prompt = ChatPromptTemplate.from_template(
        """Rewrite the user query into one focused mini-query per target language ID.
Return JSON only in this exact schema:
{{"queries": {{"lang_id_1": "query text", "lang_id_2": "query text"}}}}

Rules:
1) Keep the original user intent/topic.
2) Each mini-query must focus on only the specified language ID.
3) Use only these target language IDs: {languages_json}
4) Every target language ID must appear exactly once in "queries".
5) No prose, no markdown, JSON only.

User query: {question}
JSON:"""
    )
    chain = prompt | llm | StrOutputParser()
    attempts = max(1, retries + 1)
    last_error: Exception | None = None

    def _parse_split_json(raw: str) -> Dict[str, Any]:
        parsed = parse_json_strict(raw)
        if not isinstance(parsed, dict):
            raise ValueError("split_query_invalid_schema")
        return parsed

    json_retry_count = 0
    for attempt in range(1, attempts + 1):
        try:
            raw = (chain.invoke(
                {
                    "question": question,
                    "languages_json": json.dumps(languages, ensure_ascii=False),
                }
            ) or "").strip()
            try:
                parsed = _parse_split_json(raw)
            except Exception:
                repaired_raw, retries_used, repaired_ok = repair_json_text(
                    llm=llm,
                    raw_text=raw,
                    schema_hint='{"queries": {"lang_id_1": "query text", "lang_id_2": "query text"}}',
                    retries=max(0, int(json_retries)),
                    backoff_sec=max(0.0, float(json_backoff_sec)),
                    timeout_checker=_is_timeout_exception,
                    on_timeout=None,
                    timeout_location="split_query_json_repair",
                )
                json_retry_count += retries_used
                if not repaired_ok:
                    raise ValueError("split_query_json_invalid")
                parsed = _parse_split_json(repaired_raw)
            queries = parsed.get("queries", {}) if isinstance(parsed, dict) else {}
            if not isinstance(queries, dict):
                raise ValueError("split_query_invalid_schema")

            out: Dict[str, str] = {}
            for lang in languages:
                q = queries.get(lang)
                if not isinstance(q, str) or not q.strip():
                    raise ValueError(f"split_query_missing_lang:{lang}")
                out[lang] = q.strip()
            return out, json_retry_count
        except Exception as e:
            last_error = e
            if attempt >= attempts:
                break
            sleep_sec = max(0.0, backoff_sec) * attempt
            if sleep_sec > 0:
                time.sleep(sleep_sec)

    raise RuntimeError(f"split_query_generation_failed: {last_error}")


def _merge_split_docs(
    branch_docs: Dict[str, List[Any]],
    language_order: List[str],
    merge_cap: int,
) -> List[Any]:
    sorted_docs: Dict[str, List[Any]] = {}
    for lang in language_order:
        docs = list(branch_docs.get(lang, []))
        docs.sort(key=lambda d: _doc_score(d), reverse=True)
        sorted_docs[lang] = docs

    merged: List[Any] = []
    seen = set()
    cursor = {lang: 0 for lang in language_order}
    cap = max(0, int(merge_cap))
    if cap == 0:
        return []

    while len(merged) < cap:
        progressed = False
        for lang in language_order:
            docs = sorted_docs.get(lang, [])
            idx = cursor.get(lang, 0)
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
            cursor[lang] = idx
            if len(merged) >= cap:
                break
        if not progressed:
            break
    return merged


def retrieve_node(state: GraphState):
    """
    Retrieve documents (L1 + L0) based on the question.
    Features: Dynamic Top-K, Accumulative Retrieval.
    """
    print("---RETRIEVE---")
    question = state["question"]
    search_count = state.get("search_count", 0)
    existing_docs = state.get("documents", []) # Accumulative
    retrieval_fingerprints = [
        list(v) for v in state.get("retrieval_fingerprints", []) if isinstance(v, list)
    ]
    warnings = list(state.get("warnings", []))
    timeout_locations = list(state.get("timeout_locations", []))

    # Unload LLM to free VRAM for embeddings
    try:
        from src.utils import LLMUtility
        LLMUtility.unload_model("retrieval")
    except Exception:
        pass

    config = src.utils.load_config()
    retrieval_cfg = config.get("retrieval", {})
    reranker_cfg = config.get("reranker", {})
    dynamic_scaling = bool(retrieval_cfg.get("dynamic_scaling", True))

    if dynamic_scaling:
        # Dynamic Top-K scaling (3-tier)
        # Base: 15 for normal queries
        # Mid-broad: 30 for topic-specific linguistics queries
        # Super-broad: 60 for exhaustive queries (all, compare, list)
        super_broad_keywords = [
            "모든", "목록", "전체", "list", "all", "overview", "언어들",
            "데이터베이스", "db", "비교", "compare", "차이", "difference"
        ]

        mid_broad_keywords = [
            # Topic-specific linguistics terms
            "aspect", "시상", "tense", "양태", "modality", "mood",
            "연동", "serial", "svc", "구문", "construction",
            "음운", "phonology", "phoneme", "자음", "모음", "vowel", "consonant",
            "어순", "word order", "화제", "topic", "주어", "subject",
            "격", "case", "표지", "marker", "접사", "affix",
            "종류", "types", "features", "특징", "구조", "structure"
        ]

        q_lower = question.lower()

        if any(kw in q_lower for kw in super_broad_keywords):
            k, top_n = 60, 25
            print("  - Dynamic Scaling: Super-broad query detected.")
            print(f"    -> k={k}, top_n={top_n}")
        elif any(kw in q_lower for kw in mid_broad_keywords):
            k, top_n = 30, 15
            print("  - Dynamic Scaling: Mid-broad (topic-specific) query detected.")
            print(f"    -> k={k}, top_n={top_n}")
        else:
            k, top_n = 15, 10
            print(f"  - Dynamic Scaling: Normal query. k={k}, top_n={top_n}")
    else:
        k = int(retrieval_cfg.get("fixed_k", retrieval_cfg.get("k", 15)))
        top_n = int(retrieval_cfg.get("fixed_top_n", reranker_cfg.get("top_n", 10)))
        print(f"  - Dynamic Scaling: disabled. Using fixed k={k}, top_n={top_n}")


    with RAGRetriever() as retriever:
        query_split_cfg = retrieval_cfg.get("query_split", {})
        split_enabled = bool(query_split_cfg.get("enabled", True))
        controller_mode = str(query_split_cfg.get("controller_mode", "heuristic")).strip().lower()

        if split_enabled and controller_mode in FINALIZED_B7_CONTROLLER_MODES:
            print("  - Running finalized B7 materialized split runtime path...")
            metadata_store = load_metadata_store("metadata.enriched.csv")
            phenomenon_lexicon, category_lexicon = load_viking_lexicon(
                config.get("retrieval", {}).get("viking", {}).get("lexicon_path", "config/viking_lexicon.yaml")
            )
            metadata, new_docs, split_warnings, _meta_sec, _ret_sec = run_finalized_b7_retrieval(
                retriever=retriever,
                query_id=str(state.get("query_id") or "web_ui_query"),
                query_text=question,
                profile_cfg=config,
                fixed_k=k,
                fixed_top_n=top_n,
                metadata_store=metadata_store,
                phenomenon_lexicon=phenomenon_lexicon,
                category_lexicon=category_lexicon,
            )
            retrieval_diagnostics = metadata.get("_diagnostics", {})
            if not isinstance(retrieval_diagnostics, dict):
                retrieval_diagnostics = {}
                metadata["_diagnostics"] = retrieval_diagnostics
            new_docs, interrogative_rescue = _runtime_interrogative_source_rescue(
                retriever,
                new_docs,
                metadata,
                question,
            )
            retrieval_diagnostics["runtime_interrogative_rescue_applied"] = bool(
                interrogative_rescue.get("applied", False)
            )
            retrieval_diagnostics["runtime_interrogative_rescue_added"] = int(
                interrogative_rescue.get("added", 0)
            )
            retrieval_diagnostics["runtime_interrogative_rescue_langs"] = list(
                interrogative_rescue.get("langs", []) or []
            )
            for warning in split_warnings:
                if warning not in warnings:
                    warnings.append(warning)
        else:
            # Metadata Extraction Logic (legacy one-shot retrieval path only)
            print("  - Analyzing Query Metadata...")
            metadata = extract_query_metadata(question, retriever.config)
            retrieval_diagnostics = metadata.get("_diagnostics", {})
            if not isinstance(retrieval_diagnostics, dict):
                retrieval_diagnostics = {}
                metadata["_diagnostics"] = retrieval_diagnostics
            retrieval_diagnostics.setdefault("split_applied", False)
            retrieval_diagnostics.setdefault("split_branches", [])
            retrieval_diagnostics.setdefault("split_json_retry_count", 0)
            retrieval_diagnostics.setdefault("split_json_fail_open", False)
            timeout_loc = retrieval_diagnostics.get("timeout_location", "none")
            if timeout_loc != "none":
                print(f"TIMEOUT_DIAGNOSTIC timeout_location={timeout_loc}")
                _append_warning(
                    warnings,
                    timeout_locations,
                    f"Language detection timeout at {timeout_loc}; proceeding without language filter.",
                    timeout_loc,
                )
            new_docs = retriever.retrieve_documents(question, k=k, top_n=top_n, metadata=metadata)
    
    # Merge and deduplicate
    combined_docs = []
    seen_ids = set()
    for d in existing_docs + new_docs:
        ref_id = d.metadata.get('ref_id')
        if ref_id and ref_id not in seen_ids:
            combined_docs.append(d)
            seen_ids.add(ref_id)
        elif not ref_id: # fallback
             combined_docs.append(d)

    print(f"  - Total Documents (Accumulated): {len(combined_docs)}")
    current_fingerprint_set = set()
    for d in combined_docs:
        key = _doc_fingerprint_key(d)
        if key:
            current_fingerprint_set.add(key)
    current_fingerprint = sorted(current_fingerprint_set)
    retrieval_fingerprints = (retrieval_fingerprints + [current_fingerprint])[-8:]
    retrieval_diagnostics["retrieval_fingerprint_size"] = len(current_fingerprint)
    
    return {
        "documents": combined_docs,
        "question": question,
        "search_count": search_count,
        "warnings": warnings,
        "timeout_locations": timeout_locations,
        "retrieval_diagnostics": retrieval_diagnostics,
        "retrieval_fingerprints": retrieval_fingerprints,
    }

def grade_documents_node(state: GraphState):
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    print("---CHECK DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    retrieval_diagnostics = dict(state.get("retrieval_diagnostics", {}) or {})
    retrieval_fingerprints = [
        list(v) for v in state.get("retrieval_fingerprints", []) if isinstance(v, list)
    ]
    warnings = list(state.get("warnings", []))
    timeout_locations = list(state.get("timeout_locations", []))

    # Skip grading if disabled via config
    config = src.utils.load_config()
    if config.get("rag", {}).get("skip_grading", False):
        print("---SKIP DOCUMENT GRADING (disabled)---")
        retrieval_diagnostics["grading_topic_gate_applied"] = False
        return {
            "documents": documents,
            "question": question,
            "retrieval_diagnostics": retrieval_diagnostics,
            "retrieval_fingerprints": retrieval_fingerprints,
            "warnings": warnings,
            "timeout_locations": timeout_locations,
        }

    retrieval_diagnostics["grading_topic_gate_applied"] = False
    retrieval_diagnostics["grading_rejected_by_topic"] = 0
    retrieval_diagnostics["grading_rejected_other"] = 0
    retrieval_diagnostics["grading_relevant_count"] = 0
    retrieval_diagnostics["grading_maybe_count"] = 0
    retrieval_diagnostics["grading_branch_question_count"] = 0
    grading_settings = _get_grading_settings()

    try:
        grader = get_grade_chain()
    except Exception as e:
        timeout_location = "grade_documents"
        print(f"  - Grader init failed. Fail-open mode: {e}")
        if _is_timeout_exception(e):
            print(f"TIMEOUT_DIAGNOSTIC timeout_location={timeout_location}")
        _append_warning(
            warnings,
            timeout_locations,
            f"Document grading failed at {timeout_location}; all documents retained.",
            timeout_location,
        )
        return {
            "documents": documents,
            "question": question,
            "retrieval_diagnostics": retrieval_diagnostics,
            "retrieval_fingerprints": retrieval_fingerprints,
            "warnings": warnings,
            "timeout_locations": timeout_locations,
        }

    filtered_docs = []
    
    relevance_count = 0
    
    for d in documents:
        structured_meta = _build_structured_metadata(d)
        source_info = f"[Source]: {structured_meta['SourceFile']}"
        grading_question, grading_question_mode = _select_grading_question(
            question,
            d,
            retrieval_diagnostics,
            branch_local_grading_enabled=grading_settings["branch_local_grading_enabled"],
        )
        rich_content = (
            "[Grading Context]\n"
            f"OriginalUserQuestion: {question}\n"
            f"EffectiveDocumentQuestion: {grading_question}\n"
            f"QuestionMode: {grading_question_mode}\n"
            f"{_format_structured_block(structured_meta)}\n"
            "[Document Content]\n"
            f"{d.page_content}"
        )
        if grading_question_mode == "branch_question":
            retrieval_diagnostics["grading_branch_question_count"] = (
                int(retrieval_diagnostics.get("grading_branch_question_count", 0)) + 1
            )
        
        try:
            score = grader.invoke({"question": grading_question, "document": rich_content})
        except Exception as e:
            timeout_location = "grade_documents"
            print(f"  - Grading failed for {d.metadata.get('ref_id')}. Fail-open keep doc: {e}")
            if _is_timeout_exception(e):
                print(f"TIMEOUT_DIAGNOSTIC timeout_location={timeout_location}")
            _append_warning(
                warnings,
                timeout_locations,
                f"Document grading timeout/error at {timeout_location}; fail-open applied.",
                timeout_location,
            )
            filtered_docs.append(d)
            continue
        
        # Robust parsing for label output
        grade_label = _parse_grade_label(score)
        if grade_label == "maybe" and _is_case_only_borderline_doc(grading_question, d):
            grade_label = "no"
        if grade_label in {"yes", "maybe"} and _is_information_structure_side_topic_doc(grading_question, d):
            grade_label = "no"

        keep_doc = grade_label == "yes" or (
            grade_label == "maybe" and grading_settings["keep_maybe_docs_in_grading"]
        )

        if keep_doc:
            label_text = "RELEVANT" if grade_label == "yes" else "BORDERLINE-KEEP"
            print(
                f"  - GRADED DOCUMENT: {label_text} "
                f"({d.metadata.get('ref_id')}) [{source_info}]"
            )
            filtered_docs.append(d)
            relevance_count += 1
            retrieval_diagnostics["grading_relevant_count"] = (
                int(retrieval_diagnostics.get("grading_relevant_count", 0)) + 1
            )
            if grade_label == "maybe":
                retrieval_diagnostics["grading_maybe_count"] = (
                    int(retrieval_diagnostics.get("grading_maybe_count", 0)) + 1
                )
        else:
            print(f"  - GRADED DOCUMENT: NOT RELEVANT ({d.metadata.get('ref_id')})")
            retrieval_diagnostics["grading_rejected_other"] = (
                int(retrieval_diagnostics.get("grading_rejected_other", 0)) + 1
            )
            continue

    filtered_docs, empty_branch_rescue = _rescue_empty_split_branches(
        original_documents=documents,
        kept_documents=filtered_docs,
        question=question,
        retrieval_diagnostics=retrieval_diagnostics,
    )
    retrieval_diagnostics["grading_empty_branch_rescue_applied"] = bool(
        empty_branch_rescue.get("applied", False)
    )
    retrieval_diagnostics["grading_empty_branch_rescue_added"] = int(
        empty_branch_rescue.get("added", 0)
    )
    retrieval_diagnostics["grading_empty_branch_rescue_langs"] = list(
        empty_branch_rescue.get("langs", []) or []
    )
    if empty_branch_rescue.get("added", 0):
        relevance_count += int(empty_branch_rescue.get("added", 0))
        retrieval_diagnostics["grading_relevant_count"] = (
            int(retrieval_diagnostics.get("grading_relevant_count", 0))
            + int(empty_branch_rescue.get("added", 0))
        )

    if relevance_count == 0 and documents:
        # Model-specific grader failures can mark every doc as "no".
        # Fail-open with a small head slice to avoid rewrite-only loops.
        fallback_keep = min(3, len(documents))
        filtered_docs = list(documents[:fallback_keep])
        retrieval_diagnostics["grading_all_no_fail_open"] = True
        retrieval_diagnostics["grading_all_no_fallback_kept"] = fallback_keep
        print(
            f"  - GRADING FAIL-OPEN: all docs graded non-relevant; "
            f"keeping top {fallback_keep} docs for downstream generation."
        )
    else:
        retrieval_diagnostics["grading_all_no_fail_open"] = False
        retrieval_diagnostics["grading_all_no_fallback_kept"] = 0
            
    return {
        "documents": filtered_docs,
        "question": question,
        "retrieval_diagnostics": retrieval_diagnostics,
        "retrieval_fingerprints": retrieval_fingerprints,
        "warnings": warnings,
        "timeout_locations": timeout_locations,
    }

def generate_node(state: GraphState):
    """
    Generate answer using the retrieved documents.
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    retrieval_diagnostics = dict(state.get("retrieval_diagnostics", {}) or {})
    retrieval_fingerprints = [
        list(v) for v in state.get("retrieval_fingerprints", []) if isinstance(v, list)
    ]
    warnings = list(state.get("warnings", []))
    timeout_locations = list(state.get("timeout_locations", []))
    
    # Format Context
    text_context_lines = []
    vision_context_lines = []
    
    for d in documents:
        if d.metadata.get('type') == 'vision':
            vision_context_lines.append(d.page_content)
        else:
            structured_meta = _build_structured_metadata(d)
            level = d.metadata.get('level', '0')
            prefix = "[Summary]" if str(level) == '1' else "[Detail]"
            source_info = f"[Source: {structured_meta['SourceFile']}]"
            text_context_lines.append(
                f"{prefix} Ref: {d.metadata.get('ref_id', 'Chunk')} {source_info}\n"
                f"{_format_structured_block(structured_meta)}\n"
                "[Document Content]\n"
                f"{d.page_content}"
            )

    context_str = "\n\n".join(text_context_lines)
    if vision_context_lines:
        context_str += "\n\n[Visual Evidence Found]\n" + "\n---\n".join(vision_context_lines)
        
    if not context_str.strip():
        context_str = "No relevant documents found."

    timeout_sec = _get_generation_timeout_sec(default_sec=600)
    generation = _generate_with_timeout(query=question, context=context_str, timeout_sec=timeout_sec)
    generation, generation_diagnostics = _apply_grounding_guard(
        generation=generation,
        documents=documents,
        question=question,
    )
    if generation_diagnostics.get("unsupported_claim_guard_triggered", False):
        guard_reason = generation_diagnostics.get("fail_closed_reason", "unknown")
        fail_closed_applied = bool(generation_diagnostics.get("fail_closed_applied", False))
        mode_tag = "fail-closed" if fail_closed_applied else "fail-open-warning"
        warning_msg = f"Generation grounding guard triggered ({mode_tag}, reason={guard_reason})."
        if warning_msg not in warnings:
            warnings.append(warning_msg)
        print(
            "GENERATION_GUARD_DIAGNOSTICS "
            f"triggered=true reason={guard_reason} "
            f"fail_closed_applied={str(fail_closed_applied).lower()} "
            f"pre_coverage={generation_diagnostics.get('pre_guard_inline_citation_coverage', 0.0):.3f} "
            f"coverage={generation_diagnostics.get('post_guard_inline_citation_coverage', generation_diagnostics.get('inline_citation_coverage', 0.0)):.3f} "
            f"pre_claim_count={generation_diagnostics.get('pre_guard_claim_count', generation_diagnostics.get('claim_count', 0))} "
            f"claim_count={generation_diagnostics.get('post_guard_claim_count', generation_diagnostics.get('claim_count', 0))}"
        )
    else:
        print(
            "GENERATION_GUARD_DIAGNOSTICS "
            f"triggered=false pre_coverage={generation_diagnostics.get('pre_guard_inline_citation_coverage', generation_diagnostics.get('inline_citation_coverage', 0.0)):.3f} "
            f"coverage={generation_diagnostics.get('post_guard_inline_citation_coverage', generation_diagnostics.get('inline_citation_coverage', 0.0)):.3f} "
            f"pre_claim_count={generation_diagnostics.get('pre_guard_claim_count', generation_diagnostics.get('claim_count', 0))} "
            f"claim_count={generation_diagnostics.get('post_guard_claim_count', generation_diagnostics.get('claim_count', 0))}"
        )
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "generation_diagnostics": generation_diagnostics,
        "retrieval_diagnostics": retrieval_diagnostics,
        "retrieval_fingerprints": retrieval_fingerprints,
        "warnings": warnings,
        "timeout_locations": timeout_locations,
    }

def check_hallucination_node(state: GraphState):
    """
    Checks if the generated answer is grounded in the documents.
    """
    print("---CHECK HALLUCINATION---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    retrieval_diagnostics = dict(state.get("retrieval_diagnostics", {}) or {})
    generation_diagnostics = dict(state.get("generation_diagnostics", {}) or {})
    retrieval_fingerprints = [
        list(v) for v in state.get("retrieval_fingerprints", []) if isinstance(v, list)
    ]
    warnings = list(state.get("warnings", []))
    timeout_locations = list(state.get("timeout_locations", []))
    hallu_settings = _get_hallucination_settings()
    generation_settings = _get_generation_settings()
    min_cov = max(
        0.0,
        min(1.0, float(generation_settings.get("grounding_min_citation_coverage", 1.0))),
    )
    try:
        max_claims = int(
            generation_diagnostics.get(
                "max_claims_per_answer",
                generation_settings.get("max_claims_per_answer", 12),
            )
        )
    except Exception:
        max_claims = int(generation_settings.get("max_claims_per_answer", 12))
    stable_threshold = max(
        0.0,
        min(1.0, float(hallu_settings.get("stable_retrieval_jaccard_threshold", 0.9))),
    )
    stable_info = _compute_retrieval_stability(retrieval_fingerprints, stable_threshold)
    generation_diagnostics["hallucination_mode"] = hallu_settings["hallucination_mode"]
    generation_diagnostics["retrieval_stable"] = stable_info["stable"]
    generation_diagnostics["retrieval_jaccard"] = stable_info["jaccard"]
    generation_diagnostics["rewrite_skipped_due_to_stability"] = False
    generation_diagnostics["stage1_failed_metric"] = "none"
    generation_diagnostics["catastrophic_stage1_fail"] = False

    print(
        "RETRIEVAL_STABILITY_DIAGNOSTICS "
        f"stable={str(stable_info['stable']).lower()} "
        f"jaccard={stable_info['jaccard']:.3f} "
        f"threshold={stable_info['threshold']:.3f} "
        f"prev_count={stable_info['prev_count']} curr_count={stable_info['curr_count']}"
    )

    def _pack(status: str, text: str) -> Dict[str, Any]:
        return {
            "documents": documents,
            "generation": text,
            "hallucination_status": status,
            "retrieval_diagnostics": retrieval_diagnostics,
            "generation_diagnostics": generation_diagnostics,
            "warnings": warnings,
            "timeout_locations": timeout_locations,
            "retrieval_fingerprints": retrieval_fingerprints,
        }

    # Skip hallucination check if disabled via config
    config = src.utils.load_config()
    if config.get("rag", {}).get("skip_hallucination", False):
        print("---SKIP HALLUCINATION CHECK (disabled)---")
        return _pack("pass", generation)

    try:
        checker = get_hallucination_chain()
    except Exception as e:
        timeout_location = "hallucination_check"
        print(f"  - Hallucination checker init failed. Fail-open pass-with-warning: {e}")
        if _is_timeout_exception(e):
            print(f"TIMEOUT_DIAGNOSTIC timeout_location={timeout_location}")
        _append_warning(
            warnings,
            timeout_locations,
            f"Hallucination checker init timeout/error at {timeout_location}; fail-open applied.",
            timeout_location,
        )
        warning_msg = (
            "\n\n> [!WARNING] (Runtime): Hallucination check initialization timed out/failed. "
            "Answer is returned in fail-open mode."
        )
        return _pack("pass", generation + warning_msg)
    
    # Prepare documents text for checker
    full_text = "\n".join([d.page_content for d in documents])
    
    if len(full_text) > 30000:
        docs_text = full_text[:30000] + "... (truncated)"
    else:
        docs_text = full_text
        
    # Check for valid refusal
    refusal_keywords = ["정보가 부족", "알 수 없습니다", "문맥에 나타나 있지 않으", "제공된 문맥", "information is missing"]
    if any(k in generation for k in refusal_keywords):
        print("  - DECISION: GROUNDED REFUSAL (Pass)")
        return _pack("pass", generation)

    if hallu_settings["hallucination_mode"] == "two_stage":
        try:
            coverage = float(generation_diagnostics.get("inline_citation_coverage", 0.0))
        except Exception:
            coverage = 0.0
        try:
            claim_count = int(generation_diagnostics.get("claim_count", 0))
        except Exception:
            claim_count = 0
        unknown_tags = list(generation_diagnostics.get("unknown_citation_tags", []) or [])
        stage1_pass = bool(
            coverage >= min_cov
            and claim_count <= max_claims
            and not unknown_tags
        )
        stage1_failed_metric = "none"
        if coverage < min_cov:
            stage1_failed_metric = "coverage"
        elif claim_count > max_claims:
            stage1_failed_metric = "claim_count"
        elif unknown_tags:
            stage1_failed_metric = "unknown_tags"
        generation_diagnostics["stage1_failed_metric"] = stage1_failed_metric
        print(
            "HALLUCINATION_STAGE1_DIAGNOSTICS "
            f"pass={str(stage1_pass).lower()} "
            f"coverage={coverage:.3f} min_cov={min_cov:.3f} "
            f"claim_count={claim_count} max_claims={max_claims} "
            f"unknown_tags={len(unknown_tags)}"
        )
        if stage1_pass and hallu_settings["skip_checker_on_stage1_pass"]:
            print("  - DECISION: STAGE1 PASS -> SKIP CHECKER -> PASS")
            generation_diagnostics["checker_skipped_on_stage1_pass"] = True
            return _pack("pass", generation)
        if not stage1_pass:
            reason = "stage1_guard_failed"
            generation_diagnostics["hallucination_fail_reason"] = reason
            catastrophic_stage1_fail = _is_catastrophic_stage1_failure(
                generation_diagnostics
            )
            generation_diagnostics["catastrophic_stage1_fail"] = catastrophic_stage1_fail
            if (
                hallu_settings["skip_rewrite_on_stable_retrieval"]
                and stable_info["stable"]
                and (
                    hallu_settings["allow_stable_pass_on_catastrophic_guard_fail"]
                    or not catastrophic_stage1_fail
                )
            ):
                print("  - DECISION: STABLE RETRIEVAL + STAGE1 FAIL -> PASS WITH WARNING")
                warning_msg = (
                    "\n\n> [!WARNING] (Stability): 근거 규칙 검증에 경고가 있으나, "
                    "재검색 결과가 반복되어 현재 답변을 종료합니다."
                )
                _append_warning(
                    warnings,
                    timeout_locations,
                    "Hallucination stage-1 failed with stable retrieval; rewrite skipped.",
                    None,
                )
                generation_diagnostics["rewrite_skipped_due_to_stability"] = True
                return _pack("pass", generation + warning_msg)
            if stable_info["stable"] and catastrophic_stage1_fail:
                print("  - DECISION: CATASTROPHIC STAGE1 FAIL -> REWRITE")
            print("  - DECISION: STAGE1 GUARD FAIL -> REWRITE")
            return _pack("fail", generation)

    try:
        score = checker.invoke({"documents": docs_text, "generation": generation})
    except Exception as e:
        timeout_location = "hallucination_check"
        print(f"  - Hallucination check failed. Fail-open pass-with-warning: {e}")
        if _is_timeout_exception(e):
            print(f"TIMEOUT_DIAGNOSTIC timeout_location={timeout_location}")
        _append_warning(
            warnings,
            timeout_locations,
            f"Hallucination check timeout/error at {timeout_location}; fail-open applied.",
            timeout_location,
        )
        warning_msg = (
            "\n\n> [!WARNING] (Runtime): Hallucination check timed out/failed. "
            "Answer is returned in fail-open mode."
        )
        return _pack("pass", generation + warning_msg)

    grade = str(score).lower().strip()
    
    # Tiered Logic
    if "yes" in grade:
        print("  - DECISION: GROUNDED (Pass)")
        return _pack("pass", generation)
        
    elif "ambiguous" in grade:
        if hallu_settings["hallucination_ambiguous_policy"] == "pass_with_warning":
            print("  - DECISION: AMBIGUOUS -> PASS WITH WARNING")
            warning_msg = (
                "\n\n> [!WARNING] (Uncertainty): This answer is based on inference or partial information. "
                "Please verify specific details."
            )
            new_generation = generation + warning_msg
            return _pack("pass", new_generation)
        print("  - DECISION: AMBIGUOUS -> REWRITE")
        return _pack("fail", generation)
        
    else:
        generation_diagnostics["hallucination_fail_reason"] = "checker_no"
        if (
            hallu_settings["skip_rewrite_on_stable_retrieval"]
            and stable_info["stable"]
        ):
            print("  - DECISION: CHECKER NO + STABLE RETRIEVAL -> PASS WITH WARNING")
            warning_msg = (
                "\n\n> [!WARNING] (Stability): Hallucination checker가 보수적으로 실패를 반환했지만, "
                "재검색 결과가 반복되어 현재 답변을 종료합니다."
            )
            _append_warning(
                warnings,
                timeout_locations,
                "Hallucination checker returned no with stable retrieval; rewrite skipped.",
                None,
            )
            generation_diagnostics["rewrite_skipped_due_to_stability"] = True
            return _pack("pass", generation + warning_msg)
        print("  - DECISION: HALLUCINATION DETECTED (Fail)")
        return _pack("fail", generation)


def rewrite_query_node(state: GraphState):
    """
    Transform the query to produce a better question.
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    retrieval_diagnostics = dict(state.get("retrieval_diagnostics", {}) or {})
    retrieval_fingerprints = [
        list(v) for v in state.get("retrieval_fingerprints", []) if isinstance(v, list)
    ]
    search_count = state.get("search_count", 0)
    warnings = list(state.get("warnings", []))
    timeout_locations = list(state.get("timeout_locations", []))

    better_question = question
    try:
        rewriter = get_rewrite_chain()
        better_question = rewriter.invoke({"question": question})
    except Exception as e:
        timeout_location = "rewrite_query"
        print(f"  - Rewrite failed. Fail-open keep original query: {e}")
        if _is_timeout_exception(e):
            print(f"TIMEOUT_DIAGNOSTIC timeout_location={timeout_location}")
        _append_warning(
            warnings,
            timeout_locations,
            f"Rewrite timeout/error at {timeout_location}; original query retained.",
            timeout_location,
        )
    
    print(f"  - Old Query: {question}")
    print(f"  - New Query: {better_question}")
    
    return {
        "documents": documents,
        "question": better_question,
        "search_count": search_count + 1,
        "retrieval_diagnostics": retrieval_diagnostics,
        "retrieval_fingerprints": retrieval_fingerprints,
        "warnings": warnings,
        "timeout_locations": timeout_locations,
    }
