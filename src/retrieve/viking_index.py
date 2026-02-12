"""Lightweight taxonomy index built from ./data/ltdb/doc directory structure."""

import os
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field


@dataclass
class TaxonomyIndex:
    """In-memory index of the LTDB document taxonomy.

    Directory layout: doc/<language>/<category>/<phenomenon>/<file>.html
    Example: doc/bashkir/4_ss/48_tame/bashkir_48_tame.html
    """
    languages: Set[str] = field(default_factory=set)
    # code -> short name, e.g. "1_phonology" -> "phonology"
    categories: Dict[str, str] = field(default_factory=dict)
    # code -> short name, e.g. "48_tame" -> "tame"
    phenomena: Dict[str, str] = field(default_factory=dict)
    # phenomenon code -> parent category code
    phenomenon_to_category: Dict[str, str] = field(default_factory=dict)


_cached_index: Optional[TaxonomyIndex] = None


def build_taxonomy_index(data_path: str) -> TaxonomyIndex:
    """Scan data_path/doc/ and build a deterministic taxonomy index."""
    doc_dir = os.path.join(data_path, "doc")
    index = TaxonomyIndex()

    if not os.path.isdir(doc_dir):
        print(f"[VikingIndex] doc directory not found: {doc_dir}")
        return index

    for lang in sorted(os.listdir(doc_dir)):
        lang_dir = os.path.join(doc_dir, lang)
        if not os.path.isdir(lang_dir):
            continue
        index.languages.add(lang)

        for cat in os.listdir(lang_dir):
            cat_dir = os.path.join(lang_dir, cat)
            if not os.path.isdir(cat_dir):
                continue
            cat_name = cat.split("_", 1)[1] if "_" in cat else cat
            index.categories[cat] = cat_name

            for phenom in os.listdir(cat_dir):
                phenom_dir = os.path.join(cat_dir, phenom)
                if not os.path.isdir(phenom_dir):
                    continue
                phenom_name = phenom.split("_", 1)[1] if "_" in phenom else phenom
                index.phenomena[phenom] = phenom_name
                index.phenomenon_to_category[phenom] = cat

    print(f"[VikingIndex] Built: {len(index.languages)} languages, "
          f"{len(index.categories)} categories, {len(index.phenomena)} phenomena")
    return index


def get_taxonomy_index(data_path: str) -> TaxonomyIndex:
    """Return cached taxonomy index, building on first call."""
    global _cached_index
    if _cached_index is None:
        _cached_index = build_taxonomy_index(data_path)
    return _cached_index


def normalize_name(name: str) -> str:
    """Case-insensitive normalization for matching."""
    return name.lower().strip()
