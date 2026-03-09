from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

GLOTTO_RE = re.compile(r"\b([a-z]{4}[0-9]{4})\b", re.IGNORECASE)
GLOTTO_URL_RE = re.compile(
    r"https?://(?:www\.)?glottolog\.org/resource/languoid/id/([a-z]{4}[0-9]{4})",
    re.IGNORECASE,
)
WIKI_URL_RE = re.compile(
    r"https?://(?P<lang>en|ko)\.wikipedia\.org/wiki/(?P<slug>[^\s\"'<>]+)",
    re.IGNORECASE,
)
KO_LABEL_PATTERNS = [
    re.compile(r"(?:국문|한글|한국어)\s*[:：]\s*([^<\n\r]+)"),
]
EN_LABEL_PATTERNS = [
    re.compile(r"(?:영어)\s*[:：]\s*([^<\n\r]+)"),
]


def _norm(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _split_source_files(raw: str) -> list[str]:
    text = _norm(raw)
    if not text:
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return ""


def _extract_glottocodes_from_text(text: str) -> set[str]:
    if not text:
        return set()
    lowered = text.lower()
    out = {m.group(1).lower() for m in GLOTTO_RE.finditer(lowered)}
    out.update(m.group(1).lower() for m in GLOTTO_URL_RE.finditer(lowered))
    return out


def _extract_label_value(patterns: list[re.Pattern[str]], html_text: str) -> str:
    if not html_text:
        return ""
    for pattern in patterns:
        match = pattern.search(html_text)
        if not match:
            continue
        value = html.unescape(match.group(1)).strip()
        value = re.sub(r"\s+", " ", value)
        value = value.strip(" \t\r\n-–—:;,.·")
        if value:
            return value
    return ""


def _name_from_wiki_slug(slug: str) -> str:
    if not slug:
        return ""
    value = html.unescape(slug).replace("_", " ")
    value = re.sub(r"\s*\(.*?\)\s*$", "", value)
    value = re.sub(r"\s+language\s*$", "", value, flags=re.IGNORECASE)
    value = value.strip()
    return value


def _extract_name_from_url_text(text: str) -> str:
    value = _norm(text)
    if not value:
        return ""
    match = WIKI_URL_RE.search(value)
    if not match:
        return ""
    return _name_from_wiki_slug(match.group("slug"))


def _is_url_like(text: str) -> bool:
    value = _norm(text).lower()
    return value.startswith("http://") or value.startswith("https://")


def _normalize_name_candidate(text: str) -> str:
    value = _norm(text)
    if not value:
        return ""
    if _is_url_like(value):
        return _extract_name_from_url_text(value)
    return value


def _load_override_csv(path: Path | None) -> dict[str, dict[str, str]]:
    if not path or not path.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lang_id = _norm(row.get("lang_id"))
            if not lang_id:
                continue
            out[lang_id] = {
                "glottocode": _norm(row.get("glottocode")),
                "iso639_3": _norm(row.get("iso639_3")),
                "lang_family": _norm(row.get("lang_family")),
                "region": _norm(row.get("region")),
                "region_detail": _norm(row.get("region_detail")),
                "core_name_en": _norm(row.get("core_name_en")),
                "core_name_ko": _norm(row.get("core_name_ko")),
            }
    return out


def _build_lang_dir_index(lang_dir: Path) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    if not lang_dir.exists():
        return index

    for html_file in sorted(lang_dir.glob("*.html")):
        text = _read_text(html_file)
        if not text:
            continue

        candidates = sorted(_extract_glottocodes_from_text(text))
        file_stem = html_file.stem.lower()
        glottocode = file_stem if GLOTTO_RE.fullmatch(file_stem) else ""
        if not glottocode and len(candidates) == 1:
            glottocode = candidates[0]
        if not glottocode:
            continue

        info = index.setdefault(glottocode, {})
        for match in WIKI_URL_RE.finditer(text):
            wiki_lang = match.group("lang").lower()
            slug = match.group("slug")
            url = match.group(0)
            if wiki_lang == "en":
                info.setdefault("wikipedia_en_url", url)
                name_en = _name_from_wiki_slug(slug)
                if name_en:
                    info.setdefault("wikipedia_en_name", name_en)
            elif wiki_lang == "ko":
                info.setdefault("wikipedia_ko_url", url)
                name_ko = _name_from_wiki_slug(slug)
                if name_ko:
                    info.setdefault("wikipedia_ko_name", name_ko)

    return index


def _collect_reference_glottocodes(reference_dir: Path) -> set[str]:
    out: set[str] = set()
    if not reference_dir.exists():
        return out
    for html_file in sorted(reference_dir.glob("*.html")):
        out.update(_extract_glottocodes_from_text(_read_text(html_file)))
    return out


def _collect_doc_glottocode_candidates(
    source_files: list[str],
    *,
    ltdb_root: Path,
    file_cache: dict[str, set[str]],
) -> list[str]:
    codes: set[str] = set()
    for rel_path in source_files:
        cached = file_cache.get(rel_path)
        if cached is None:
            text = _read_text(ltdb_root / rel_path)
            cached = _extract_glottocodes_from_text(text)
            file_cache[rel_path] = cached
        codes.update(cached)
    return sorted(codes)


def _basic_doc_path(lang_id: str, ltdb_root: Path) -> Path:
    return ltdb_root / "doc" / lang_id / "0_general" / "01_basic" / "01_basic.html"


def _set_if_blank(
    row: dict[str, str],
    key: str,
    value: str,
    *,
    overwrite: bool,
    changed_keys: Counter,
) -> bool:
    val = _norm(value)
    if not val:
        return False

    current = _norm(row.get(key, ""))
    if current and not overwrite:
        return False
    if current == val:
        return False

    row[key] = val
    changed_keys[key] += 1
    return True


def enrich_rows(
    rows: list[dict[str, str]],
    *,
    ltdb_root: Path,
    lang_dir: Path,
    reference_dir: Path,
    override_map: dict[str, dict[str, str]] | None = None,
    overwrite: bool = False,
    allow_ko_wiki_fallback: bool = True,
) -> tuple[list[dict[str, str]], list[dict[str, str]], Counter]:
    lang_index = _build_lang_dir_index(lang_dir)
    reference_glottocodes = _collect_reference_glottocodes(reference_dir)
    override_map = override_map or {}

    changed_keys: Counter = Counter()
    review_rows: list[dict[str, str]] = []
    file_cache: dict[str, set[str]] = {}

    for row in rows:
        lang_id = _norm(row.get("lang_id"))
        source_files = _split_source_files(row.get("source_files", ""))

        doc_candidates = _collect_doc_glottocode_candidates(
            source_files, ltdb_root=ltdb_root, file_cache=file_cache
        )
        basic_doc = _basic_doc_path(lang_id, ltdb_root)
        basic_text = _read_text(basic_doc)

        override = override_map.get(lang_id, {})
        current_glottocode = _norm(row.get("glottocode"))

        resolved_glottocode = ""
        method = ""
        confidence = ""

        if current_glottocode:
            resolved_glottocode = current_glottocode.lower()
        elif _norm(override.get("glottocode")):
            resolved_glottocode = _norm(override.get("glottocode")).lower()
            method = "override"
            confidence = "high"
        elif len(doc_candidates) == 1:
            resolved_glottocode = doc_candidates[0]
            method = "doc_token"
            confidence = "high"

        if resolved_glottocode:
            _set_if_blank(
                row,
                "glottocode",
                resolved_glottocode,
                overwrite=overwrite,
                changed_keys=changed_keys,
            )
            _set_if_blank(
                row,
                "core_glottocode",
                resolved_glottocode,
                overwrite=overwrite,
                changed_keys=changed_keys,
            )
            if method:
                _set_if_blank(
                    row,
                    "mapping_method",
                    method,
                    overwrite=overwrite,
                    changed_keys=changed_keys,
                )
            if confidence:
                _set_if_blank(
                    row,
                    "mapping_confidence",
                    confidence,
                    overwrite=overwrite,
                    changed_keys=changed_keys,
                )

        # Override for metadata attributes (fill-only unless overwrite)
        for key in ("iso639_3", "lang_family", "region", "region_detail"):
            if _norm(override.get(key)):
                _set_if_blank(
                    row,
                    key,
                    _norm(override.get(key)),
                    overwrite=overwrite,
                    changed_keys=changed_keys,
                )

        lang_info = lang_index.get(_norm(row.get("glottocode")).lower(), {})

        # Legacy cleanup: some rows contain full URL in core_name_en.
        existing_core_name_en = _norm(row.get("core_name_en"))
        if _is_url_like(existing_core_name_en):
            normalized_from_url = _extract_name_from_url_text(existing_core_name_en)
            replacement = normalized_from_url if normalized_from_url else ""
            if existing_core_name_en != replacement:
                row["core_name_en"] = replacement
                changed_keys["core_name_en"] += 1

        # core_name_en fill policy: explicit doc label -> override csv -> lang wikipedia slug
        core_name_en = _norm(row.get("core_name_en"))
        if not core_name_en or overwrite:
            en_from_doc = _normalize_name_candidate(_extract_label_value(EN_LABEL_PATTERNS, basic_text))
            en_from_override = _normalize_name_candidate(_norm(override.get("core_name_en")))
            en_from_wiki = _normalize_name_candidate(_norm(lang_info.get("wikipedia_en_name")))
            candidate_en = en_from_doc or en_from_override or en_from_wiki
            if candidate_en:
                _set_if_blank(
                    row,
                    "core_name_en",
                    candidate_en,
                    overwrite=overwrite,
                    changed_keys=changed_keys,
                )

        # core_name_ko fill policy: explicit doc label -> override csv -> ko wiki slug(optional)
        core_name_ko = _norm(row.get("core_name_ko"))
        if not core_name_ko or overwrite:
            ko_from_doc = _extract_label_value(KO_LABEL_PATTERNS, basic_text)
            ko_from_override = _norm(override.get("core_name_ko"))
            ko_from_wiki = _norm(lang_info.get("wikipedia_ko_name")) if allow_ko_wiki_fallback else ""
            candidate_ko = ko_from_doc or ko_from_override or ko_from_wiki
            if candidate_ko:
                _set_if_blank(
                    row,
                    "core_name_ko",
                    candidate_ko,
                    overwrite=overwrite,
                    changed_keys=changed_keys,
                )

        # Review-only diagnostics
        if not _norm(row.get("glottocode")):
            issue = "unresolved_glottocode"
            details = ""
            if len(doc_candidates) > 1:
                issue = "ambiguous_doc_glottocode"
                details = ",".join(doc_candidates)
            review_rows.append(
                {
                    "lang_id": lang_id,
                    "issue": issue,
                    "details": details,
                    "doc_glottocode_candidates": json.dumps(doc_candidates, ensure_ascii=False),
                    "reference_glottocode_hits": json.dumps(
                        sorted(c for c in doc_candidates if c in reference_glottocodes),
                        ensure_ascii=False,
                    ),
                }
            )

        if not _norm(row.get("core_name_en")):
            review_rows.append(
                {
                    "lang_id": lang_id,
                    "issue": "missing_core_name_en",
                    "details": "",
                    "doc_glottocode_candidates": json.dumps(doc_candidates, ensure_ascii=False),
                    "reference_glottocode_hits": "[]",
                }
            )

        if not _norm(row.get("core_name_ko")):
            review_rows.append(
                {
                    "lang_id": lang_id,
                    "issue": "missing_core_name_ko",
                    "details": "",
                    "doc_glottocode_candidates": json.dumps(doc_candidates, ensure_ascii=False),
                    "reference_glottocode_hits": "[]",
                }
            )

    return rows, review_rows, changed_keys


def _read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def _write_csv_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_review_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    else:
        fieldnames = [
            "lang_id",
            "issue",
            "details",
            "doc_glottocode_candidates",
            "reference_glottocode_hits",
        ]
    _write_csv_rows(path, rows, fieldnames)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich data/metadata.csv conservatively.")
    parser.add_argument("--metadata-csv", default="data/metadata.csv")
    parser.add_argument("--output-csv", default="data/metadata.enriched.csv")
    parser.add_argument("--review-csv", default="data/metadata.enrich_review.csv")
    parser.add_argument("--ltdb-root", default="data/ltdb")
    parser.add_argument("--lang-dir", default="data/ltdb/lang")
    parser.add_argument("--reference-dir", default="data/ltdb/reference")
    parser.add_argument("--override-csv", default="")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing non-empty values.",
    )
    parser.add_argument(
        "--disable-ko-wiki-fallback",
        action="store_true",
        help="Do not use ko-wikipedia slug as fallback for core_name_ko.",
    )
    args = parser.parse_args()

    metadata_csv = Path(args.metadata_csv)
    output_csv = Path(args.output_csv)
    review_csv = Path(args.review_csv)
    ltdb_root = Path(args.ltdb_root)
    lang_dir = Path(args.lang_dir)
    reference_dir = Path(args.reference_dir)
    override_csv = Path(args.override_csv) if _norm(args.override_csv) else None

    rows, fieldnames = _read_csv_rows(metadata_csv)
    override_map = _load_override_csv(override_csv)

    enriched_rows, review_rows, changed_keys = enrich_rows(
        rows,
        ltdb_root=ltdb_root,
        lang_dir=lang_dir,
        reference_dir=reference_dir,
        override_map=override_map,
        overwrite=args.overwrite,
        allow_ko_wiki_fallback=not args.disable_ko_wiki_fallback,
    )

    _write_csv_rows(output_csv, enriched_rows, fieldnames)
    _write_review_csv(review_csv, review_rows)

    print(f"[metadata-enrich] rows={len(enriched_rows)}")
    print(f"[metadata-enrich] changed_fields={dict(changed_keys)}")
    print(f"[metadata-enrich] review_rows={len(review_rows)}")
    print(f"[metadata-enrich] output_csv={output_csv}")
    print(f"[metadata-enrich] review_csv={review_csv}")


if __name__ == "__main__":
    main()
