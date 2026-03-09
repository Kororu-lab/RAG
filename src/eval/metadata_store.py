from __future__ import annotations

import csv
import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _norm(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _split_pipe(value: str) -> list[str]:
    raw = _norm(value)
    if not raw:
        return []
    out: list[str] = []
    for part in raw.split("|"):
        item = part.strip()
        if item:
            out.append(item)
    return out


def _normalize_alias(value: Any) -> str:
    raw = _norm(value).lower()
    if not raw:
        return ""
    raw = html.unescape(raw)
    raw = raw.replace("/", " ").replace("\\", " ")
    raw = re.sub(r"[\"'`’“”·•]", "", raw)
    raw = re.sub(r"[()\[\]{}.,:;!?]", " ", raw)
    raw = re.sub(r"[\s\-_]+", " ", raw).strip()
    return raw


def _compact_alias(value: str) -> str:
    return re.sub(r"[\s\-_]+", "", value)


def _is_hangul(value: str) -> bool:
    return bool(re.search(r"[가-힣]", value))


def _add_alias_candidate(
    *,
    alias_candidates: dict[str, list[tuple[str, str]]],
    lang_id: str,
    alias: str,
    source: str,
) -> None:
    normalized = _normalize_alias(alias)
    if not normalized:
        return
    if len(normalized) <= 1:
        return
    alias_candidates.setdefault(lang_id, []).append((normalized, source))


def _extract_lang_page_aliases(lang_page: Path) -> list[str]:
    if not lang_page.exists():
        return []
    try:
        text = lang_page.read_text(encoding="utf-8")
    except Exception:
        return []

    out: list[str] = []
    for slug in re.findall(r"https?://en\.wikipedia\.org/wiki/([A-Za-z0-9_\-()%]+)", text):
        slug_clean = html.unescape(slug).strip()
        if not slug_clean:
            continue
        out.append(slug_clean.replace("_", " "))
        out.append(slug_clean.replace("_", " ").replace("-", " "))
        if slug_clean.lower().endswith("_language"):
            out.append(slug_clean[:-9].replace("_", " "))
        if slug_clean.lower().endswith("-language"):
            out.append(slug_clean[:-9].replace("-", " "))
    for glot_id in re.findall(r"glottolog\.org/resource/languoid/id/([a-z0-9]+)", text.lower()):
        out.append(glot_id)
    return out


def _extract_basic_page_aliases(basic_page: Path) -> list[str]:
    if not basic_page.exists():
        return []
    try:
        text = basic_page.read_text(encoding="utf-8")
    except Exception:
        return []

    out: list[str] = []
    patterns = [
        r"<li>\s*영문\s*:\s*([^<]+)</li>",
        r"<li>\s*한글\s*:\s*([^<]+)</li>",
        r"<li>\s*국문\s*:\s*([^<]+)</li>",
    ]
    for pat in patterns:
        for match in re.findall(pat, text, flags=re.IGNORECASE):
            value = html.unescape(match).strip()
            if value:
                out.append(value)

    # Fallback for files that only contain narrative text (no explicit "영문/한글 :").
    if not out:
        body = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
        body = re.sub(r"<style[\s\S]*?</style>", " ", body, flags=re.IGNORECASE)
        body = re.sub(r"<[^>]+>", " ", body)
        body = html.unescape(body)
        body = re.sub(r"\s+", " ", body).strip()
        sample = body[:5000]

        ko_tokens = re.findall(r"[가-힣]{2,20}(?:어|언어)", sample)
        blacklist = {"언어", "한어", "영어", "중국어", "한국어", "러시아어"}
        freq: dict[str, int] = {}
        for tok in ko_tokens:
            if tok in blacklist:
                continue
            freq[tok] = freq.get(tok, 0) + 1
        for tok, count in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])):
            if count >= 2:
                out.append(tok)
            if len(out) >= 4:
                break

        if len(out) < 2:
            for eng in re.findall(r"([A-Za-z][A-Za-z\- ]{2,40}\s+language)", sample, flags=re.IGNORECASE):
                out.append(eng.strip())
                if len(out) >= 4:
                    break
    return out


def _alias_variants(alias: str) -> set[str]:
    base = _normalize_alias(alias)
    if not base:
        return set()
    out: set[str] = {base}

    compact = _compact_alias(base)
    if len(compact) >= 3:
        out.add(compact)

    out.add(base.replace("-", " "))
    out.add(base.replace("_", " "))

    if _is_hangul(base):
        no_space = base.replace(" ", "")
        if len(no_space) >= 2:
            out.add(no_space)
        if base.endswith(" 언어"):
            stem = base[:-3].strip()
            if stem:
                out.add(stem)
                out.add(f"{stem}어")
        elif base.endswith("어"):
            out.add(f"{base} 언어")
        else:
            out.add(f"{base}어")
            out.add(f"{base} 언어")
    else:
        if base.endswith(" language"):
            stem = base[:-9].strip()
            if stem:
                out.add(stem)
                out.add(_compact_alias(stem))
        else:
            out.add(f"{base} language")

    return {v for v in out if v and len(v.strip()) > 1}


def _is_special_risk_alias(alias: str, family_terms: set[str], region_terms: set[str]) -> bool:
    a = _normalize_alias(alias)
    if not a:
        return True
    if a in {"wa", "yi", "miao", "zhuang"}:
        return True
    if a in family_terms or a in region_terms:
        return True
    compact = _compact_alias(a)
    if re.fullmatch(r"[a-z]{1,3}", compact or ""):
        return True
    if re.fullmatch(r"[0-9a-z]{1,2}", compact or ""):
        return True
    return False


@dataclass(frozen=True)
class MetadataStore:
    lang_rows: dict[str, dict[str, str]]
    alias_to_lang: dict[str, str]
    alias_index: dict[str, list[dict[str, Any]]]
    parent_topics_by_lang: dict[str, list[str]]
    child_topics_by_lang: dict[str, list[str]]
    family_to_langs: dict[str, list[str]]
    region_to_langs: dict[str, list[str]]
    topic_pairs_by_lang: dict[str, list[str]]

    @classmethod
    def from_csv(cls, csv_path: Path) -> "MetadataStore":
        lang_rows: dict[str, dict[str, str]] = {}
        alias_to_lang: dict[str, str] = {}
        alias_index: dict[str, list[dict[str, Any]]] = {}
        alias_candidates: dict[str, list[tuple[str, str]]] = {}
        parent_topics_by_lang: dict[str, list[str]] = {}
        child_topics_by_lang: dict[str, list[str]] = {}
        topic_pairs_by_lang: dict[str, list[str]] = {}
        family_to_langs_raw: dict[str, set[str]] = {}
        region_to_langs_raw: dict[str, set[str]] = {}
        family_terms: set[str] = set()
        region_terms: set[str] = set()

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                lang_id = _norm(row.get("lang_id")).lower()
                if not lang_id:
                    continue
                lang_rows[lang_id] = {k: _norm(v) for k, v in row.items() if k}

                core_en = _norm(row.get("core_name_en")).lower()
                core_ko = _norm(row.get("core_name_ko")).lower()
                glottocode = _norm(row.get("glottocode")).lower()
                _add_alias_candidate(alias_candidates=alias_candidates, lang_id=lang_id, alias=lang_id, source="lang_id")
                _add_alias_candidate(alias_candidates=alias_candidates, lang_id=lang_id, alias=core_en, source="core_name_en")
                _add_alias_candidate(alias_candidates=alias_candidates, lang_id=lang_id, alias=core_ko, source="core_name_ko")
                _add_alias_candidate(alias_candidates=alias_candidates, lang_id=lang_id, alias=glottocode, source="glottocode")

                glottolog_ref = _norm(row.get("glottolog_ref"))
                if glottolog_ref:
                    for slug in re.findall(r"/([A-Za-z0-9_\-()%]+)$", glottolog_ref):
                        _add_alias_candidate(
                            alias_candidates=alias_candidates,
                            lang_id=lang_id,
                            alias=slug.replace("_", " "),
                            source="glottolog_ref_slug",
                        )

                parent_topics = _split_pipe(_norm(row.get("parent_topics")))
                child_topics = _split_pipe(_norm(row.get("child_topics")))
                parent_topics_by_lang[lang_id] = parent_topics
                child_topics_by_lang[lang_id] = child_topics
                topic_pairs_by_lang[lang_id] = _split_pipe(_norm(row.get("topic_pairs_parent_child")))

                family_id = _norm(row.get("family_id")).lower() or _norm(row.get("lang_family")).lower()
                if family_id:
                    family_to_langs_raw.setdefault(family_id, set()).add(lang_id)
                    family_terms.add(_normalize_alias(family_id))

                region = _norm(row.get("region")).lower()
                if region:
                    region_to_langs_raw.setdefault(region, set()).add(lang_id)
                    region_terms.add(_normalize_alias(region))
                # Keep region_detail searchable via region index for deterministic lookup.
                for detail in [d.strip().lower() for d in _norm(row.get("region_detail")).split(",") if d.strip()]:
                    region_to_langs_raw.setdefault(detail, set()).add(lang_id)
                    region_terms.add(_normalize_alias(detail))

                for term in (_norm(row.get("lang_family")), _norm(row.get("region"))):
                    normalized = _normalize_alias(term)
                    if normalized:
                        if term == _norm(row.get("lang_family")):
                            family_terms.add(normalized)
                        else:
                            region_terms.add(normalized)

                if glottocode:
                    lang_page = Path("data/ltdb/lang") / f"{glottocode}.html"
                    for alias in _extract_lang_page_aliases(lang_page):
                        _add_alias_candidate(
                            alias_candidates=alias_candidates,
                            lang_id=lang_id,
                            alias=alias,
                            source="lang_page",
                        )

                basic_page = Path("data/ltdb/doc") / lang_id / "0_general/01_basic/01_basic.html"
                for alias in _extract_basic_page_aliases(basic_page):
                    source = "basic_name_ko" if _is_hangul(alias) else "basic_name_en"
                    _add_alias_candidate(
                        alias_candidates=alias_candidates,
                        lang_id=lang_id,
                        alias=alias,
                        source=source,
                    )

        dedupe_guard: set[tuple[str, str, str]] = set()
        for lang_id, candidates in alias_candidates.items():
            for alias, source in candidates:
                for variant in _alias_variants(alias):
                    normalized = _normalize_alias(variant)
                    if not normalized:
                        continue
                    key = (normalized, lang_id, source)
                    if key in dedupe_guard:
                        continue
                    dedupe_guard.add(key)
                    risk = _is_special_risk_alias(normalized, family_terms, region_terms)
                    alias_index.setdefault(normalized, []).append(
                        {
                            "lang_id": lang_id,
                            "source": source,
                            "risk": risk,
                        }
                    )

        for alias, entries in alias_index.items():
            langs = sorted({str(e.get("lang_id")) for e in entries if e.get("lang_id")})
            has_risk = any(bool(e.get("risk", False)) for e in entries)
            if len(langs) == 1 and not has_risk:
                alias_to_lang.setdefault(alias, langs[0])

        family_to_langs = {
            family_id: sorted(langs) for family_id, langs in family_to_langs_raw.items()
        }
        region_to_langs = {
            region: sorted(langs) for region, langs in region_to_langs_raw.items()
        }

        return cls(
            lang_rows=lang_rows,
            alias_to_lang=alias_to_lang,
            alias_index=alias_index,
            parent_topics_by_lang=parent_topics_by_lang,
            child_topics_by_lang=child_topics_by_lang,
            family_to_langs=family_to_langs,
            region_to_langs=region_to_langs,
            topic_pairs_by_lang=topic_pairs_by_lang,
        )

    def resolve_lang(self, token: str) -> str | None:
        key = _normalize_alias(token)
        if not key:
            return None
        direct = self.alias_to_lang.get(key)
        if direct:
            return direct
        compact = _compact_alias(key)
        if compact:
            compact_direct = self.alias_to_lang.get(compact)
            if compact_direct:
                return compact_direct
            compact_candidates = self.alias_index.get(compact, [])
            compact_langs = sorted({str(e.get("lang_id")) for e in compact_candidates if e.get("lang_id")})
            if len(compact_langs) == 1:
                return compact_langs[0]
        candidates = self.alias_index.get(key, [])
        langs = sorted({str(e.get("lang_id")) for e in candidates if e.get("lang_id")})
        if len(langs) == 1:
            return langs[0]
        return None

    def parent_topics(self, lang_id: str) -> list[str]:
        return list(self.parent_topics_by_lang.get(_norm(lang_id).lower(), []))

    def child_topics(self, lang_id: str) -> list[str]:
        return list(self.child_topics_by_lang.get(_norm(lang_id).lower(), []))

    def langs_for_family(self, family_id: str) -> list[str]:
        return list(self.family_to_langs.get(_norm(family_id).lower(), []))

    def langs_for_region(self, region: str) -> list[str]:
        return list(self.region_to_langs.get(_norm(region).lower(), []))


def load_metadata_store(preferred_path: str | None = None) -> MetadataStore | None:
    candidates: list[Path] = []
    if preferred_path:
        candidates.append(Path(preferred_path))
    candidates.extend(
        [
            Path("metadata.enriched.csv"),
            Path("metadata.csv"),
            Path("data/metadata.enriched.csv"),
            Path("data/metadata.csv"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return MetadataStore.from_csv(candidate)
    return None
