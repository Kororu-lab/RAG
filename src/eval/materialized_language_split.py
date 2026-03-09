from __future__ import annotations

import html
import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.eval.metadata_store import MetadataStore
from src.llm.factory import get_llm
from src.llm.json_retry import parse_json_strict, repair_json_text

PROMPT_EXTRACTOR_SYSTEM = """You analyze a raw linguistic QA query and decide whether it should be decomposed into exactly two top-level language-wise retrieval branches.

Use only the raw query text and ordinary language understanding. Do not assume access to metadata tables, alias inventories, corpus structure, database schema, or gold labels.

You must decide only between:
- NO_SPLIT
- TWO_LANGUAGE_SPLIT

Return TWO_LANGUAGE_SPLIT only if the query directly asks for answers about exactly two top-level language targets. Comparison or contrast is shared intent, not a third target and not a blocker.

Reject split when the apparent pair is actually:
- dialects or subvarieties inside one language,
- donor/source/etymological/contextual mentions,
- a topic contrast inside one language,
- examples or background mentions,
- unclear or mixed targets.

Targets must come from explicit spans in the query. Do not invent language names.

Return JSON only."""

PROMPT_EXTRACTOR_USER = """Query ID: {query_id}
Original query: {query_text}

Return this schema:
{{
  "query_id": "...",
  "original_query": "...",
  "decision": "NO_SPLIT | TWO_LANGUAGE_SPLIT",
  "confidence": 0.0,
  "comparison_present": false,
  "dialect_like": false,
  "topic_like": false,
  "contextual_only": false,
  "uncertain": false,
  "shared_ask": "...",
  "targets": [
    {{
      "label": "...",
      "anchor_span": "...",
      "confidence": 0.0,
      "notes": "..."
    }}
  ],
  "abstain_reason": "...",
  "notes": "..."
}}

Rules:
- If decision is TWO_LANGUAGE_SPLIT, there must be exactly two top-level language targets.
- shared_ask must be a short Korean phrase that can be reused in a standalone query template.
- shared_ask must not contain comparative wording such as 비교/대조/차이/각각.
- shared_ask must not contain the other language target.
- If uncertain, use NO_SPLIT and explain why.
"""

EXTRACTOR_SCHEMA = {
    "query_id": "...",
    "original_query": "...",
    "decision": "NO_SPLIT | TWO_LANGUAGE_SPLIT",
    "confidence": 0.0,
    "comparison_present": False,
    "dialect_like": False,
    "topic_like": False,
    "contextual_only": False,
    "uncertain": False,
    "shared_ask": "...",
    "targets": [
        {
            "label": "...",
            "anchor_span": "...",
            "confidence": 0.0,
            "notes": "...",
        }
    ],
    "abstain_reason": "...",
    "notes": "...",
}

PROMPT_BUNDLE = "\n\n".join(
    [
        "# Materialized Language Split Extractor Prompt",
        "## Extractor System\n" + PROMPT_EXTRACTOR_SYSTEM,
        "## Extractor User\n" + PROMPT_EXTRACTOR_USER,
    ]
)

_PLAN_CACHE: Dict[Tuple[str, str, int, str, str], Dict[str, Any]] = {}

_LANGUAGE_SURFACE_RE = r"[A-Za-z가-힣0-9_-]+어(?:\s*\([A-Za-z0-9_\-]+\))?"
_LANGUAGE_PAIR_RE = re.compile(
    rf"(?P<a>{_LANGUAGE_SURFACE_RE})\s*(?:와|과|및|and|/|,)\s*(?P<b>{_LANGUAGE_SURFACE_RE})"
)
_COMPARE_MARKERS = re.compile(r"(비교|대조|각각|공통점|차이점|상이점|다른점)")
_POLITE_ENDINGS = re.compile(
    r"(을|를)?\s*(알려\s*주세요|설명해\s*주세요|말해\s*주세요|나열해\s*주세요|보여\s*주세요|"
    r"알려\s*줘|설명해\s*줘|말해\s*줘|나열해\s*줘|보여\s*줘)(을|를)?$"
)
_COMPARISON_ONLY_PHRASES = [
    "비교하여",
    "비교하고",
    "비교 설명해 주세요",
    "비교 설명",
    "비교 나열해 주세요",
    "비교 나열",
    "비교해 주세요",
    "비교하여 설명해 주세요",
    "비교하여 나열해 주세요",
    "비교",
    "공통점과 차이점",
    "공통점과 차이점을",
    "차이점과 공통점",
    "차이점과 공통점을",
    "공통점",
    "차이점",
    "각각",
    "각 언어의",
    "각 언어에서",
    "각 언어",
    "두 언어의",
    "두 언어에서",
    "두 언어",
    "양 언어의",
    "양 언어에서",
    "양 언어",
]
_SPLIT_ALIAS_BLACKLIST = {
    "고유어",
    "공식언어",
    "관형어",
    "교착어",
    "구어",
    "굴절어",
    "국어",
    "기능어",
    "나누어",
    "남도어",
    "남도언어",
    "단어",
    "동의어",
    "들어",
    "되어",
    "러시아어",
    "목적어",
    "민난어",
    "반의어",
    "부사어",
    "보어",
    "서술어",
    "수식어",
    "술어",
    "영어",
    "외래어",
    "의성어",
    "이루어",
    "이중언어",
    "인구어",
    "일본어",
    "일어",
    "있어",
    "접두어",
    "접미어",
    "접촉언어",
    "조어",
    "종결어",
    "주어",
    "중국어",
    "차용어",
    "친족어",
    "표준어",
    "피수식어",
    "한국어",
    "한어",
}
_NON_LANGUAGE_ENDINGS = ("되어", "들어", "있어", "붙어", "이루어", "만들어", "떨어")


@dataclass
class ExtractorStageResult:
    raw: str
    parsed: Dict[str, Any]
    repaired: bool
    parse_error: str


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def contains_hangul(text: str) -> bool:
    return bool(re.search(r"[\uac00-\ud7a3]", str(text or "")))


def _compact_alias(value: str) -> str:
    return re.sub(r"[\s\-_]+", "", str(value or ""))


def _extractor_fallback(query_id: str, query_text: str, error: str) -> Dict[str, Any]:
    return {
        "query_id": query_id,
        "original_query": query_text,
        "decision": "NO_SPLIT",
        "confidence": 0.0,
        "comparison_present": False,
        "dialect_like": False,
        "topic_like": False,
        "contextual_only": False,
        "uncertain": True,
        "shared_ask": "",
        "targets": [],
        "abstain_reason": f"extractor_parse_failure: {error}",
        "notes": "",
    }


@lru_cache(maxsize=4)
def _build_extractor_chain(request_timeout_sec: int):
    llm = get_llm(profile="retrieval", request_timeout_sec=int(request_timeout_sec))
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PROMPT_EXTRACTOR_SYSTEM),
            ("human", PROMPT_EXTRACTOR_USER),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return llm, chain


def _invoke_extractor(*, query_id: str, query_text: str, request_timeout_sec: int) -> ExtractorStageResult:
    llm, chain = _build_extractor_chain(int(request_timeout_sec))
    raw = ""
    parse_error = ""
    try:
        raw = normalize_text(chain.invoke({"query_id": query_id, "query_text": query_text}) or "")
        parsed = parse_json_strict(raw)
        if isinstance(parsed, dict):
            return ExtractorStageResult(raw=raw, parsed=parsed, repaired=False, parse_error="")
        parse_error = "json_not_object"
    except Exception as exc:
        parse_error = str(exc)

    try:
        repaired_raw, _attempts, ok = repair_json_text(
            llm=llm,
            raw_text=raw,
            schema_hint=json.dumps(EXTRACTOR_SCHEMA, ensure_ascii=False, indent=2),
            retries=1,
        )
        raw = normalize_text(repaired_raw)
        if ok:
            parsed = parse_json_strict(raw)
            if isinstance(parsed, dict):
                return ExtractorStageResult(raw=raw, parsed=parsed, repaired=True, parse_error=parse_error)
            parse_error = "json_not_object_after_repair"
    except Exception as exc:
        parse_error = f"{parse_error} | repair:{exc}"

    return ExtractorStageResult(
        raw=raw,
        parsed=_extractor_fallback(query_id, query_text, parse_error),
        repaired=False,
        parse_error=parse_error,
    )


def _normalize_target_label(text: str) -> str:
    value = normalize_text(text)
    value = re.sub(r"[\"'`“”‘’]", "", value)
    value = re.sub(r"\(([^()]*)\)", r" \1 ", value)
    value = re.sub(r"(에서|의|는|은|을|를|과|와|및)$", "", value)
    return normalize_text(value)


def _explicit_language_pair(query_text: str) -> List[Dict[str, Any]]:
    q = normalize_text(query_text)
    pairs: List[Dict[str, Any]] = []
    for match in _LANGUAGE_PAIR_RE.finditer(q):
        left = _normalize_target_label(match.group("a"))
        right = _normalize_target_label(match.group("b"))
        if not left or not right:
            continue
        pairs.append(
            {
                "left": left,
                "right": right,
                "left_surface": normalize_text(match.group("a")),
                "right_surface": normalize_text(match.group("b")),
                "span": [match.start(), match.end()],
                "text": match.group(0),
            }
        )
    return pairs


def _is_dialect_like(text: str) -> bool:
    value = normalize_text(text).lower()
    return any(marker in value for marker in ["방언", "dialect", "사투리"])


def is_dialect_like_surface(text: str) -> bool:
    return _is_dialect_like(text)


def _strip_html_text(path: Path, *, limit: int = 2000) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[: max(200, int(limit))]


def _looks_like_non_language_token(token: str) -> bool:
    norm = _normalize_target_label(token)
    if not norm:
        return True
    if norm in _SPLIT_ALIAS_BLACKLIST:
        return True
    if any(norm.endswith(suffix) for suffix in _NON_LANGUAGE_ENDINGS):
        return True
    return False


@lru_cache(maxsize=1)
def _split_recovery_alias_index() -> Dict[str, Dict[str, int]]:
    base = Path("data/ltdb/doc")
    out: Dict[str, Dict[str, int]] = {}
    if not base.exists():
        return out

    for lang_dir in sorted(base.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang_id = lang_dir.name
        freq: Counter[str] = Counter()
        for html_path in lang_dir.rglob("*.html"):
            body = _strip_html_text(html_path, limit=1600)
            if not body:
                continue
            early = body[:240]
            late = body[240:]
            for token in re.findall(r"[가-힣]{1,20}(?:어|언어)", early):
                norm = _normalize_target_label(token)
                if _looks_like_non_language_token(norm):
                    continue
                freq[norm] += 5
            for token in re.findall(r"[가-힣]{1,20}(?:어|언어)", late):
                norm = _normalize_target_label(token)
                if _looks_like_non_language_token(norm):
                    continue
                freq[norm] += 1

        alias_scores: Dict[str, int] = {}
        for token, count in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])):
            if count < 1:
                continue
            alias_scores[token] = int(count)
            if len(alias_scores) >= 12:
                break
        if alias_scores:
            out[lang_id] = alias_scores
    return out


@lru_cache(maxsize=1)
def _split_recovery_self_alias_index() -> Dict[str, set[str]]:
    base = Path("data/ltdb/doc")
    out: Dict[str, set[str]] = {}
    if not base.exists():
        return out

    for lang_dir in sorted(base.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang_id = lang_dir.name
        basic_page = lang_dir / "0_general" / "01_basic" / "01_basic.html"
        body = _strip_html_text(basic_page, limit=2400)
        if not body:
            continue
        early = body[:900]
        freq: Counter[str] = Counter()
        for token in re.findall(r"[가-힣]{1,20}(?:어|언어)", early):
            norm = _normalize_target_label(token)
            if _looks_like_non_language_token(norm):
                continue
            freq[norm] += 1
        if not freq:
            continue
        aliases = {token for token, _count in freq.most_common(6)}
        if aliases:
            out[lang_id] = aliases
    return out


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            ))
        prev = curr
    return prev[-1]


def _resolve_via_split_recovery_aliases(
    *,
    label: str,
    anchor_span: str,
    metadata_store: MetadataStore | None,
) -> str | None:
    if metadata_store is None:
        return None

    candidate_tokens = _label_resolution_candidates(label, anchor_span)
    core_name_hits: list[str] = []
    for candidate in candidate_tokens:
        normalized_candidate = _normalize_target_label(candidate)
        if not normalized_candidate:
            continue
        candidate_forms = {normalized_candidate, normalize_text(_compact_alias(normalized_candidate))}
        for lang_id, row in metadata_store.lang_rows.items():
            core_forms: set[str] = set()
            for value in (
                row.get("core_name_ko", ""),
                row.get("core_name_en", ""),
                row.get("lang_id", ""),
                row.get("glottocode", ""),
            ):
                normalized_value = _normalize_target_label(value)
                if not normalized_value:
                    continue
                core_forms.add(normalized_value)
                core_forms.add(normalize_text(_compact_alias(normalized_value)))
            if candidate_forms & core_forms:
                core_name_hits.append(lang_id)
    core_name_hits = sorted(set(core_name_hits))
    if len(core_name_hits) == 1:
        return core_name_hits[0]

    alias_index = _split_recovery_alias_index()
    direct_scores: Counter[str] = Counter()
    for candidate in candidate_tokens:
        normalized_candidate = _normalize_target_label(candidate)
        if not normalized_candidate:
            continue
        for lang_id, aliases in alias_index.items():
            if normalized_candidate in aliases:
                direct_scores[lang_id] += int(aliases.get(normalized_candidate, 0))
    if direct_scores:
        ranked_direct = sorted(direct_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        if len(ranked_direct) == 1 or ranked_direct[0][1] > ranked_direct[1][1]:
            return ranked_direct[0][0]

    korean_candidates = [
        _normalize_target_label(token)
        for token in candidate_tokens
        if contains_hangul(token)
    ]
    fuzzy_scores: Counter[str] = Counter()
    for candidate in korean_candidates:
        if not candidate:
            continue
        for lang_id, aliases in alias_index.items():
            for alias, weight in aliases.items():
                if not contains_hangul(alias):
                    continue
                distance = _levenshtein_distance(candidate, alias)
                if distance <= 2:
                    fuzzy_scores[lang_id] += max(1, (3 - distance)) * int(weight)
    if not fuzzy_scores:
        return None
    ranked_fuzzy = sorted(fuzzy_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ranked_fuzzy) == 1 or ranked_fuzzy[0][1] >= (ranked_fuzzy[1][1] * 2):
        return ranked_fuzzy[0][0]
    return None


def _label_resolution_candidates(label: str, anchor_span: str) -> List[str]:
    raw_values = [normalize_text(label), normalize_text(anchor_span)]
    candidates: List[str] = []
    for raw in raw_values:
        if not raw:
            continue
        candidates.append(raw)
        for piece in re.split(r"[\s/]+", raw):
            piece = normalize_text(piece)
            if piece and piece not in candidates:
                candidates.append(piece)
        normalized = _normalize_target_label(raw)
        if normalized and normalized not in candidates:
            candidates.append(normalized)
        for piece in re.split(r"[\s/]+", normalized):
            piece = _normalize_target_label(piece)
            if piece and piece not in candidates:
                candidates.append(piece)
        bare = re.sub(r"\([^()]*\)", " ", raw)
        bare = _normalize_target_label(bare)
        if bare and bare not in candidates:
            candidates.append(bare)
        for piece in re.split(r"[\s/]+", bare):
            piece = _normalize_target_label(piece)
            if piece and piece not in candidates:
                candidates.append(piece)
        parenthetical = re.findall(r"\(([^()]*)\)", raw)
        for token in parenthetical:
            token = normalize_text(token)
            if token and token not in candidates:
                candidates.append(token)
            normalized_token = _normalize_target_label(token)
            if normalized_token and normalized_token not in candidates:
                candidates.append(normalized_token)
    return [c for c in candidates if c]


def resolve_target_label_to_lang_id(
    *,
    label: str,
    anchor_span: str,
    metadata_store: MetadataStore | None,
    recovery_mode: str = "strict",
) -> str | None:
    if metadata_store is None:
        return None
    resolved: List[str] = []
    for candidate in _label_resolution_candidates(label, anchor_span):
        lang_id = metadata_store.resolve_lang(candidate)
        if lang_id:
            resolved.append(lang_id)
    uniq = sorted(set(resolved))
    if str(recovery_mode).strip().lower() == "recovered":
        recovered = _resolve_via_split_recovery_aliases(
            label=label,
            anchor_span=anchor_span,
            metadata_store=metadata_store,
        )
        if recovered:
            return recovered
    if len(uniq) == 1:
        return uniq[0]
    return None


def _shared_ask_clean(shared_ask: str, targets: Sequence[str]) -> str:
    ask = normalize_text(shared_ask)
    previous = None
    while ask and ask != previous:
        previous = ask
        ask = ask.strip(" .?!,;:")
        ask = _COMPARE_MARKERS.sub("", ask)
        ask = ask.strip(" .?!,;:")
        ask = _POLITE_ENDINGS.sub("", ask)
        ask = ask.strip(" .?!,;:")
        for target in targets:
            if target:
                ask = ask.replace(target, " ")
        ask = normalize_text(ask)
        # The renderer supplies the outer predicate, so trailing object particles
        # from extractor spans like "... 형태를" should not survive as "...를를".
        ask = re.sub(r"(을|를)$", "", ask).strip()
        ask = ask.strip(" .?!,;:")
    ask = ask.replace(",", " ")
    ask = normalize_text(ask)
    return ask


def _shared_ask_usable(shared_ask: str, original_query: str) -> bool:
    ask = normalize_text(shared_ask)
    if not ask:
        return False
    if len(ask) < 3:
        return False
    if contains_hangul(original_query) and not contains_hangul(ask):
        return False
    if _COMPARE_MARKERS.search(ask):
        return False
    return True


def _shared_ask_from_surface(query_text: str, explicit_pair: Dict[str, Any], targets: Sequence[str]) -> str:
    original = normalize_text(query_text)
    pair_text = normalize_text(explicit_pair.get("text", ""))
    if not original or not pair_text:
        return ""
    ask = original.replace(pair_text, " ", 1)
    ask = re.sub(r"^\s*의\s*", "", ask)
    ask = re.sub(r"^\s*에서\s*", "", ask)
    for phrase in _COMPARISON_ONLY_PHRASES:
        ask = ask.replace(phrase, " ")
    ask = _shared_ask_clean(ask, targets)
    return ask


def _surface_preserving_cleanup(text: str) -> str:
    cleaned = normalize_text(text)
    for phrase in _COMPARISON_ONLY_PHRASES:
        cleaned = cleaned.replace(phrase, " ")
    cleaned = re.sub(r"\b(each|both)\s+language(s)?\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*,\s*", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"(을|를)\s+(을|를)\s+", r"\1 ", cleaned)
    cleaned = re.sub(r"(목록|형태|체계)을\s+(각|두|양)\s+언어", r"\1", cleaned)
    cleaned = re.sub(r"(을|를)\s+\.", ".", cleaned)
    cleaned = cleaned.replace(" .", ".")
    cleaned = normalize_text(cleaned)
    return cleaned


def _render_branch_query_surface_preserving(
    *,
    query_text: str,
    explicit_pair: Dict[str, Any],
    lang_label: str,
    other_label: str,
    fallback_shared_ask: str,
) -> tuple[str, str, bool]:
    original = normalize_text(query_text)
    pair_text = normalize_text(explicit_pair.get("text", ""))
    branch_query = original
    if pair_text and pair_text in original:
        branch_query = original.replace(pair_text, _normalize_target_label(lang_label), 1)
    branch_query = _surface_preserving_cleanup(branch_query)
    if other_label:
        branch_query = normalize_text(branch_query.replace(_normalize_target_label(other_label), " "))
    topical_anchor = _shared_ask_from_surface(query_text, explicit_pair, [lang_label, other_label]) or normalize_text(fallback_shared_ask)
    anchor_retained = bool(topical_anchor and topical_anchor in branch_query)
    if (
        not branch_query
        or branch_query == original
        or (other_label and _normalize_target_label(other_label) in branch_query)
        or (contains_hangul(original) and not contains_hangul(branch_query))
        or any(marker in branch_query for marker in ("각 언어", "두 언어", "양 언어"))
        or not anchor_retained
        or not re.search(r"(알려\s*주세요|설명해\s*주세요|말해\s*주세요|나열해\s*주세요|보여\s*주세요|인가요\?|어디인가요\?)\.?$", branch_query)
    ):
        fallback = _render_branch_query(lang_label=lang_label, shared_ask=fallback_shared_ask)
        return fallback, topical_anchor, bool(topical_anchor and topical_anchor in fallback)
    return branch_query, topical_anchor, anchor_retained


def _render_branch_query(*, lang_label: str, shared_ask: str) -> str:
    label = _normalize_target_label(lang_label)
    ask = normalize_text(shared_ask)
    if not label or not ask:
        return ""
    last = ask[-1]
    particle = "를"
    if re.search(r"[가-힣]$", last):
        particle = "을" if ((ord(last) - 0xAC00) % 28) else "를"
    return normalize_text(f"{label}에서 {ask}{particle} 설명해 주세요.")


def validate_materialized_branch_queries(
    original_query: str,
    branch_queries: Sequence[Dict[str, Any]],
    target_labels: Sequence[str],
) -> Dict[str, bool]:
    original = normalize_text(original_query)
    queries = [normalize_text(item.get("query", "")) for item in branch_queries]
    exactly_two = len(branch_queries) == 2
    non_empty = exactly_two and all(bool(q) for q in queries)
    distinct = exactly_two and len(set(queries)) == 2
    different_from_original = exactly_two and all(q != original for q in queries)
    no_comparison_branch = exactly_two and all(
        normalize_text(str(item.get("target_label", ""))).lower() not in {"comparison", "비교", "contrast"}
        for item in branch_queries
    )
    english_scaffold = False
    if contains_hangul(original):
        english_scaffold = any(not contains_hangul(q) for q in queries)
    same_language_as_original = not english_scaffold
    target_leakage_low = True
    if exactly_two:
        a, b = [_normalize_target_label(v) for v in target_labels]
        if len({a, b}) != 2:
            target_leakage_low = False
        else:
            if a and b and ((b in queries[0]) or (a in queries[1])):
                target_leakage_low = False
    comparative_residue_low = all(not _COMPARE_MARKERS.search(q) for q in queries)
    return {
        "exactly_two_language_branches": exactly_two,
        "branch_query_nonempty": non_empty,
        "branch_queries_distinct": distinct,
        "branch_queries_different_from_original": different_from_original,
        "no_comparison_branch": no_comparison_branch,
        "same_language_as_original": same_language_as_original,
        "english_scaffold_free": not english_scaffold,
        "target_leakage_low": target_leakage_low,
        "comparative_residue_low": comparative_residue_low,
    }


def materialized_contract_ok(flags: Dict[str, bool]) -> bool:
    required = [
        "exactly_two_language_branches",
        "branch_query_nonempty",
        "branch_queries_distinct",
        "branch_queries_different_from_original",
        "no_comparison_branch",
        "same_language_as_original",
        "english_scaffold_free",
        "target_leakage_low",
        "comparative_residue_low",
    ]
    return all(bool(flags.get(key, False)) for key in required)


def _same_language_pair(labels: Sequence[str]) -> bool:
    normed = [_normalize_target_label(v) for v in labels if _normalize_target_label(v)]
    return len(normed) == 2 and len(set(normed)) == 1


def materialize_language_split_plan(
    *,
    query_id: str,
    query_text: str,
    request_timeout_sec: int,
    metadata_store: MetadataStore | None = None,
    split_mode: str = "strict",
    renderer_mode: str = "template",
) -> Dict[str, Any]:
    cache_key = (
        str(query_id),
        normalize_text(query_text),
        int(request_timeout_sec),
        "with_store" if metadata_store is not None else "no_store",
        str(split_mode).strip().lower(),
        str(renderer_mode).strip().lower(),
    )
    cached = _PLAN_CACHE.get(cache_key)
    if cached is not None:
        return json.loads(json.dumps(cached, ensure_ascii=False))

    def _finish(result: Dict[str, Any]) -> Dict[str, Any]:
        _PLAN_CACHE[cache_key] = result
        return json.loads(json.dumps(result, ensure_ascii=False))

    extractor = _invoke_extractor(query_id=query_id, query_text=query_text, request_timeout_sec=request_timeout_sec)
    parsed = extractor.parsed
    policy_flags = {
        "comparison_present": bool(parsed.get("comparison_present", False)),
        "dialect_like": bool(parsed.get("dialect_like", False)),
        "topic_like": bool(parsed.get("topic_like", False)),
        "contextual_only": bool(parsed.get("contextual_only", False)),
        "uncertain": bool(parsed.get("uncertain", False)),
    }

    answer_targets = parsed.get("targets") or []
    if not isinstance(answer_targets, list):
        answer_targets = []
    answer_targets = answer_targets[:2]

    explicit_pairs = _explicit_language_pair(query_text)
    target_labels = [_normalize_target_label(t.get("label") or t.get("anchor_span") or "") for t in answer_targets]
    target_spans = [normalize_text(t.get("anchor_span", "")) for t in answer_targets]
    strict_mode = str(split_mode).strip().lower() != "recovered"

    if strict_mode:
        if parsed.get("decision") != "TWO_LANGUAGE_SPLIT":
            return _finish({
                "should_split": False,
                "split_count": 0,
                "axis": "none",
                "extracted_targets": [],
                "extracted_target_spans": [],
                "shared_ask": "",
                "branch_queries": [],
                "abstain_reason": str(parsed.get("abstain_reason") or "extractor_no_split"),
                "cleanliness_flags": {},
                "policy_flags": policy_flags,
                "notes": str(parsed.get("notes") or ""),
                "extractor_output": parsed,
                "extractor_raw": extractor.raw,
                "extractor_repaired": extractor.repaired,
                "prompt_bundle": PROMPT_BUNDLE,
            })

        if len(answer_targets) != 2:
            return _finish({
                "should_split": False,
                "split_count": 0,
                "axis": "none",
                "extracted_targets": [],
                "extracted_target_spans": [],
                "shared_ask": "",
                "branch_queries": [],
                "abstain_reason": "not_exactly_two_targets",
                "cleanliness_flags": {},
                "policy_flags": policy_flags,
                "notes": str(parsed.get("notes") or ""),
                "extractor_output": parsed,
                "extractor_raw": extractor.raw,
                "extractor_repaired": extractor.repaired,
                "prompt_bundle": PROMPT_BUNDLE,
            })

        for span in target_spans:
            if not span or span not in normalize_text(query_text):
                return _finish({
                    "should_split": False,
                    "split_count": 0,
                    "axis": "none",
                    "extracted_targets": target_labels,
                    "extracted_target_spans": target_spans,
                    "shared_ask": "",
                    "branch_queries": [],
                    "abstain_reason": "target_span_not_explicit_in_query",
                    "cleanliness_flags": {},
                    "policy_flags": policy_flags,
                    "notes": str(parsed.get("notes") or ""),
                    "extractor_output": parsed,
                    "extractor_raw": extractor.raw,
                    "extractor_repaired": extractor.repaired,
                    "prompt_bundle": PROMPT_BUNDLE,
                })

        if _same_language_pair(target_labels):
            return _finish({
                "should_split": False,
                "split_count": 0,
                "axis": "none",
                "extracted_targets": target_labels,
                "extracted_target_spans": target_spans,
                "shared_ask": "",
                "branch_queries": [],
                "abstain_reason": "same_language_pair_rejected",
                "cleanliness_flags": {},
                "policy_flags": policy_flags,
                "notes": str(parsed.get("notes") or ""),
                "extractor_output": parsed,
                "extractor_raw": extractor.raw,
                "extractor_repaired": extractor.repaired,
                "prompt_bundle": PROMPT_BUNDLE,
            })

        if any(_is_dialect_like(value) for value in [*target_labels, *target_spans]):
            policy_flags["dialect_like"] = True
        if any(policy_flags.get(flag, False) for flag in ["dialect_like", "topic_like", "contextual_only", "uncertain"]):
            reasons = [name for name, active in policy_flags.items() if active and name in {"dialect_like", "topic_like", "contextual_only", "uncertain"}]
            return _finish({
                "should_split": False,
                "split_count": 0,
                "axis": "none",
                "extracted_targets": target_labels,
                "extracted_target_spans": target_spans,
                "shared_ask": "",
                "branch_queries": [],
                "abstain_reason": ";".join(reasons) if reasons else "policy_gate_rejected",
                "cleanliness_flags": {},
                "policy_flags": policy_flags,
                "notes": str(parsed.get("notes") or ""),
                "extractor_output": parsed,
                "extractor_raw": extractor.raw,
                "extractor_repaired": extractor.repaired,
                "prompt_bundle": PROMPT_BUNDLE,
            })

        if not explicit_pairs:
            return _finish({
                "should_split": False,
                "split_count": 0,
                "axis": "none",
                "extracted_targets": target_labels,
                "extracted_target_spans": target_spans,
                "shared_ask": "",
                "branch_queries": [],
                "abstain_reason": "no_explicit_language_pair_in_query",
                "cleanliness_flags": {},
                "policy_flags": policy_flags,
                "notes": str(parsed.get("notes") or ""),
                "extractor_output": parsed,
                "extractor_raw": extractor.raw,
                "extractor_repaired": extractor.repaired,
                "prompt_bundle": PROMPT_BUNDLE,
            })

        explicit = explicit_pairs[0]
        explicit_targets = [_normalize_target_label(explicit["left"]), _normalize_target_label(explicit["right"])]
        resolved_target_ids = [
            resolve_target_label_to_lang_id(
                label=label,
                anchor_span=span,
                metadata_store=metadata_store,
                recovery_mode="strict",
            )
            for label, span in zip(target_labels, target_spans, strict=False)
        ]
        resolved_explicit_ids = [
            resolve_target_label_to_lang_id(
                label=label,
                anchor_span=surface,
                metadata_store=metadata_store,
                recovery_mode="strict",
            )
            for label, surface in zip(
                explicit_targets,
                [explicit.get("left_surface", explicit["left"]), explicit.get("right_surface", explicit["right"])],
                strict=False,
            )
        ]
        if any(not lang_id for lang_id in resolved_explicit_ids):
            return _finish({
                "should_split": False,
                "split_count": 0,
                "axis": "none",
                "extracted_targets": target_labels,
                "extracted_target_spans": target_spans,
                "resolved_target_lang_ids": resolved_target_ids,
                "shared_ask": "",
                "branch_queries": [],
                "abstain_reason": "target_label_unresolved",
                "cleanliness_flags": {},
                "policy_flags": policy_flags,
                "notes": str(parsed.get("notes") or ""),
                "extractor_output": parsed,
                "extractor_raw": extractor.raw,
                "extractor_repaired": extractor.repaired,
                "prompt_bundle": PROMPT_BUNDLE,
            })

        if len(set(str(v) for v in resolved_explicit_ids if v)) != 2:
            return _finish({
                "should_split": False,
                "split_count": 0,
                "axis": "none",
                "extracted_targets": target_labels,
                "extracted_target_spans": target_spans,
                "resolved_target_lang_ids": resolved_explicit_ids,
                "shared_ask": "",
                "branch_queries": [],
                "abstain_reason": "same_language_pair_rejected",
                "cleanliness_flags": {},
                "policy_flags": policy_flags,
                "notes": str(parsed.get("notes") or ""),
                "extractor_output": parsed,
                "extractor_raw": extractor.raw,
                "extractor_repaired": extractor.repaired,
                "prompt_bundle": PROMPT_BUNDLE,
            })

        if any(resolved_target_ids) and set(v for v in resolved_target_ids if v) != set(v for v in resolved_explicit_ids if v):
            return _finish({
                "should_split": False,
                "split_count": 0,
                "axis": "none",
                "extracted_targets": target_labels,
                "extracted_target_spans": target_spans,
                "resolved_target_lang_ids": resolved_target_ids,
                "shared_ask": "",
                "branch_queries": [],
                "abstain_reason": "extractor_targets_do_not_match_explicit_pair",
                "cleanliness_flags": {},
                "policy_flags": policy_flags,
                "notes": str(parsed.get("notes") or ""),
                "extractor_output": parsed,
                "extractor_raw": extractor.raw,
                "extractor_repaired": extractor.repaired,
                "prompt_bundle": PROMPT_BUNDLE,
            })

        shared_ask = _shared_ask_clean(str(parsed.get("shared_ask") or ""), explicit_targets)
        if not _shared_ask_usable(shared_ask, query_text):
            return _finish({
                "should_split": False,
                "split_count": 0,
                "axis": "none",
                "extracted_targets": explicit_targets,
                "extracted_target_spans": target_spans,
                "shared_ask": shared_ask,
                "branch_queries": [],
                "abstain_reason": "shared_ask_not_usable",
                "cleanliness_flags": {},
                "policy_flags": policy_flags,
                "notes": str(parsed.get("notes") or ""),
                "extractor_output": parsed,
                "extractor_raw": extractor.raw,
                "extractor_repaired": extractor.repaired,
                "prompt_bundle": PROMPT_BUNDLE,
            })

        ordered_targets = explicit_targets
        ordered_target_lang_ids = [str(v) for v in resolved_explicit_ids]
        branches = [
            {
                "branch_id": f"Q{i+1}",
                "target_label": target,
                "target_lang_id": ordered_target_lang_ids[i],
                "query": _render_branch_query(lang_label=target, shared_ask=shared_ask),
                "anchor_retained": True,
                "topic_anchor": shared_ask,
            }
            for i, target in enumerate(ordered_targets)
        ]
        cleanliness_flags = validate_materialized_branch_queries(query_text, branches, ordered_targets)
        should_split = materialized_contract_ok(cleanliness_flags)
        return _finish({
            "should_split": should_split,
            "split_count": 2 if should_split else 0,
            "axis": "language" if should_split else "none",
            "extracted_targets": ordered_targets,
            "extracted_target_spans": target_spans,
            "resolved_target_lang_ids": ordered_target_lang_ids,
            "shared_ask": shared_ask,
            "branch_queries": branches if should_split else [],
            "abstain_reason": "" if should_split else ";".join(
                [name for name, active in cleanliness_flags.items() if not active]
            ),
            "cleanliness_flags": cleanliness_flags,
            "policy_flags": policy_flags,
            "notes": str(parsed.get("notes") or ""),
            "extractor_output": parsed,
            "extractor_raw": extractor.raw,
            "extractor_repaired": extractor.repaired,
            "prompt_bundle": PROMPT_BUNDLE,
        })

    if not explicit_pairs:
        return _finish({
            "should_split": False,
            "split_count": 0,
            "axis": "none",
            "extracted_targets": target_labels,
            "extracted_target_spans": target_spans,
            "shared_ask": "",
            "branch_queries": [],
            "abstain_reason": "no_explicit_language_pair_in_query",
            "cleanliness_flags": {},
            "policy_flags": policy_flags,
            "notes": str(parsed.get("notes") or ""),
            "extractor_output": parsed,
            "extractor_raw": extractor.raw,
            "extractor_repaired": extractor.repaired,
            "prompt_bundle": PROMPT_BUNDLE,
        })

    explicit = explicit_pairs[0]
    explicit_surfaces = [
        normalize_text(explicit.get("left_surface", explicit["left"])),
        normalize_text(explicit.get("right_surface", explicit["right"])),
    ]
    explicit_targets = [_normalize_target_label(v) for v in explicit_surfaces]
    if _same_language_pair(explicit_targets):
        return _finish({
            "should_split": False,
            "split_count": 0,
            "axis": "none",
            "extracted_targets": explicit_targets,
            "extracted_target_spans": explicit_surfaces,
            "shared_ask": "",
            "branch_queries": [],
            "abstain_reason": "same_language_pair_rejected",
            "cleanliness_flags": {},
            "policy_flags": policy_flags,
            "notes": str(parsed.get("notes") or ""),
            "extractor_output": parsed,
            "extractor_raw": extractor.raw,
            "extractor_repaired": extractor.repaired,
            "prompt_bundle": PROMPT_BUNDLE,
        })

    if any(_is_dialect_like(value) for value in [*explicit_targets, *explicit_surfaces, *target_labels, *target_spans]):
        policy_flags["dialect_like"] = True
    if any(policy_flags.get(flag, False) for flag in ["dialect_like", "topic_like", "contextual_only"]):
        reasons = [name for name, active in policy_flags.items() if active and name in {"dialect_like", "topic_like", "contextual_only"}]
        return _finish({
            "should_split": False,
            "split_count": 0,
            "axis": "none",
            "extracted_targets": explicit_targets,
            "extracted_target_spans": explicit_surfaces,
            "shared_ask": "",
            "branch_queries": [],
            "abstain_reason": ";".join(reasons) if reasons else "policy_gate_rejected",
            "cleanliness_flags": {},
            "policy_flags": policy_flags,
            "notes": str(parsed.get("notes") or ""),
            "extractor_output": parsed,
            "extractor_raw": extractor.raw,
            "extractor_repaired": extractor.repaired,
            "prompt_bundle": PROMPT_BUNDLE,
        })

    resolved_explicit_ids: List[str | None] = []
    for idx, (label, surface) in enumerate(zip(explicit_targets, explicit_surfaces, strict=False)):
        resolved = resolve_target_label_to_lang_id(
            label=label,
            anchor_span=surface,
            metadata_store=metadata_store,
            recovery_mode="recovered",
        )
        if not resolved and idx < len(target_labels):
            resolved = resolve_target_label_to_lang_id(
                label=target_labels[idx],
                anchor_span=target_spans[idx] if idx < len(target_spans) else "",
                metadata_store=metadata_store,
                recovery_mode="recovered",
            )
        resolved_explicit_ids.append(resolved)

    if not any(resolved_explicit_ids):
        return _finish({
            "should_split": False,
            "split_count": 0,
            "axis": "none",
            "extracted_targets": explicit_targets,
            "extracted_target_spans": explicit_surfaces,
            "resolved_target_lang_ids": resolved_explicit_ids,
            "shared_ask": "",
            "branch_queries": [],
            "abstain_reason": "target_label_unresolved",
            "cleanliness_flags": {},
            "policy_flags": policy_flags,
            "notes": str(parsed.get("notes") or ""),
            "extractor_output": parsed,
            "extractor_raw": extractor.raw,
            "extractor_repaired": extractor.repaired,
            "prompt_bundle": PROMPT_BUNDLE,
        })

    if any(not lang_id for lang_id in resolved_explicit_ids):
        return _finish({
            "should_split": False,
            "split_count": 0,
            "axis": "none",
            "extracted_targets": explicit_targets,
            "extracted_target_spans": explicit_surfaces,
            "resolved_target_lang_ids": resolved_explicit_ids,
            "shared_ask": "",
            "branch_queries": [],
            "abstain_reason": "target_pair_not_fully_resolved",
            "cleanliness_flags": {},
            "policy_flags": policy_flags,
            "notes": str(parsed.get("notes") or ""),
            "extractor_output": parsed,
            "extractor_raw": extractor.raw,
            "extractor_repaired": extractor.repaired,
            "prompt_bundle": PROMPT_BUNDLE,
        })

    ordered_target_lang_ids = [str(v) for v in resolved_explicit_ids if v]
    if len(set(ordered_target_lang_ids)) != 2:
        return _finish({
            "should_split": False,
            "split_count": 0,
            "axis": "none",
            "extracted_targets": explicit_targets,
            "extracted_target_spans": explicit_surfaces,
            "resolved_target_lang_ids": ordered_target_lang_ids,
            "shared_ask": "",
            "branch_queries": [],
            "abstain_reason": "same_language_pair_rejected",
            "cleanliness_flags": {},
            "policy_flags": policy_flags,
            "notes": str(parsed.get("notes") or ""),
            "extractor_output": parsed,
            "extractor_raw": extractor.raw,
            "extractor_repaired": extractor.repaired,
            "prompt_bundle": PROMPT_BUNDLE,
        })

    shared_ask = _shared_ask_clean(str(parsed.get("shared_ask") or ""), explicit_targets)
    if not _shared_ask_usable(shared_ask, query_text):
        shared_ask = _shared_ask_from_surface(query_text, explicit, explicit_targets)
    if not _shared_ask_usable(shared_ask, query_text):
        return _finish({
            "should_split": False,
            "split_count": 0,
            "axis": "none",
            "extracted_targets": explicit_targets,
            "extracted_target_spans": explicit_surfaces,
            "resolved_target_lang_ids": ordered_target_lang_ids,
            "shared_ask": shared_ask,
            "branch_queries": [],
            "abstain_reason": "shared_ask_not_usable",
            "cleanliness_flags": {},
            "policy_flags": policy_flags,
            "notes": str(parsed.get("notes") or ""),
            "extractor_output": parsed,
            "extractor_raw": extractor.raw,
            "extractor_repaired": extractor.repaired,
            "prompt_bundle": PROMPT_BUNDLE,
        })

    branches = []
    for i, target in enumerate(explicit_targets):
        other = explicit_targets[1 - i]
        if str(renderer_mode).strip().lower() == "surface_preserving":
            branch_query, topic_anchor, anchor_retained = _render_branch_query_surface_preserving(
                query_text=query_text,
                explicit_pair=explicit,
                lang_label=explicit_targets[i],
                other_label=explicit_targets[1 - i],
                fallback_shared_ask=shared_ask,
            )
        else:
            branch_query = _render_branch_query(lang_label=target, shared_ask=shared_ask)
            topic_anchor = shared_ask
            anchor_retained = bool(topic_anchor and topic_anchor in branch_query)
        branches.append(
            {
                "branch_id": f"Q{i+1}",
                "target_label": target,
                "target_lang_id": ordered_target_lang_ids[i],
                "query": branch_query,
                "anchor_retained": bool(anchor_retained),
                "topic_anchor": topic_anchor,
            }
        )

    cleanliness_flags = validate_materialized_branch_queries(query_text, branches, explicit_targets)
    cleanliness_flags["anchor_retention_all"] = all(bool(branch.get("anchor_retained", False)) for branch in branches)
    should_split = materialized_contract_ok(cleanliness_flags)
    return _finish({
        "should_split": should_split,
        "split_count": 2 if should_split else 0,
        "axis": "language" if should_split else "none",
        "extracted_targets": explicit_targets,
        "extracted_target_spans": explicit_surfaces,
        "resolved_target_lang_ids": ordered_target_lang_ids,
        "shared_ask": shared_ask,
        "branch_queries": branches if should_split else [],
        "abstain_reason": "" if should_split else ";".join(
            [name for name, active in cleanliness_flags.items() if not active]
        ),
        "cleanliness_flags": cleanliness_flags,
        "policy_flags": policy_flags,
        "notes": str(parsed.get("notes") or ""),
        "extractor_output": parsed,
        "extractor_raw": extractor.raw,
        "extractor_repaired": extractor.repaired,
        "prompt_bundle": PROMPT_BUNDLE,
    })
