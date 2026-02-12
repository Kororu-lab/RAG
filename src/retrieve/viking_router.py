"""Viking-style query router for search-space restriction.

Routes queries to a narrowed scope (languages, categories, phenomena) using
lexical matching against the LTDB taxonomy. No LLM calls — deterministic only.

Lexicon mappings are loaded from an external YAML file (config/viking_lexicon.yaml)
via viking_lexicon_loader. No hardcoded dictionaries.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from src.retrieve.viking_index import TaxonomyIndex
from src.retrieve.viking_lexicon_loader import load_viking_lexicon
import src.utils


@dataclass
class VikingScope:
    """Computed search scope for a query."""
    languages: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    phenomena: List[str] = field(default_factory=list)
    confidence: str = "low"  # low, medium, high
    fallback_plan: str = "widen"
    path_patterns: List[str] = field(default_factory=list)  # SQL LIKE patterns
    trace: List[str] = field(default_factory=list)  # Audit trail


def _get_lexicons() -> tuple:
    """Load lexicons from external YAML via config path."""
    config = src.utils.load_config()
    lexicon_path = (config.get("retrieval", {})
                    .get("viking", {})
                    .get("lexicon_path", "config/viking_lexicon.yaml"))
    return load_viking_lexicon(lexicon_path)


def compute_scope(
    query: str,
    metadata: Dict[str, Any],
    index: TaxonomyIndex,
) -> VikingScope:
    """Compute Viking scope from query + metadata. No LLM calls."""
    phenomenon_lexicon, category_lexicon = _get_lexicons()

    scope = VikingScope()
    q_lower = query.lower()

    # 1. Languages — reuse metadata already extracted by LLM in nodes.py
    if metadata and metadata.get("lang"):
        lang = metadata["lang"]
        if isinstance(lang, list):
            scope.languages = [l for l in lang if l in index.languages]
        elif lang in index.languages:
            scope.languages = [lang]
        if scope.languages:
            scope.trace.append(f"lang={scope.languages}")

    # 2. Phenomenon detection via lexicon (longest-match-first)
    matched_phenomena: set = set()
    for term in sorted(phenomenon_lexicon, key=len, reverse=True):
        if term in q_lower:
            for code in phenomenon_lexicon[term]:
                if code in index.phenomena:
                    matched_phenomena.add(code)

    if matched_phenomena:
        scope.phenomena = sorted(matched_phenomena)
        scope.confidence = "high" if len(matched_phenomena) <= 3 else "medium"
        scope.trace.append(f"phenomena={scope.phenomena}")

    # 3. Category-level fallback when no phenomena matched
    if not scope.phenomena:
        matched_cats: set = set()
        for term, cat_code in category_lexicon.items():
            if term in q_lower and cat_code in index.categories:
                matched_cats.add(cat_code)
        if matched_cats:
            scope.categories = sorted(matched_cats)
            scope.confidence = "medium"
            scope.trace.append(f"categories={scope.categories}")

    # 4. No match → low confidence, full search space
    if not scope.phenomena and not scope.categories:
        scope.confidence = "low"
        scope.fallback_plan = "widen"
        scope.trace.append("no_match")

    # 5. Build SQL LIKE patterns
    scope.path_patterns = _build_path_patterns(scope)
    return scope


def _build_path_patterns(scope: VikingScope) -> List[str]:
    """Build SQL LIKE patterns from scope for source_file matching."""
    patterns: List[str] = []

    if scope.phenomena:
        for phenom in scope.phenomena:
            if scope.languages:
                for lang in scope.languages:
                    patterns.append(f"doc/{lang}/%/{phenom}/%")
            else:
                patterns.append(f"doc/%/%/{phenom}/%")
    elif scope.categories:
        for cat in scope.categories:
            if scope.languages:
                for lang in scope.languages:
                    patterns.append(f"doc/{lang}/{cat}/%")
            else:
                patterns.append(f"doc/%/{cat}/%")
    elif scope.languages:
        for lang in scope.languages:
            patterns.append(f"doc/{lang}/%")

    return patterns


def widen_scope(scope: VikingScope, index: TaxonomyIndex) -> VikingScope:
    """Widen scope one step for soft fallback.

    Strategy chain:
      specific phenomena → parent category → language only → unrestricted
    """
    # Step 1: phenomena → parent category
    if scope.phenomena:
        parent_cats = set()
        for phenom in scope.phenomena:
            parent = index.phenomenon_to_category.get(phenom)
            if parent:
                parent_cats.add(parent)
        if parent_cats:
            scope.categories = sorted(parent_cats)
        scope.phenomena = []
        scope.trace.append(f"widened→categories={scope.categories}")
    # Step 2: category → language only
    elif scope.categories:
        scope.categories = []
        scope.trace.append("widened→lang_only")
    # Step 3: language only → unrestricted
    elif scope.languages:
        scope.languages = []
        scope.trace.append("widened→unrestricted")

    scope.path_patterns = _build_path_patterns(scope)
    if not scope.path_patterns:
        scope.fallback_plan = "exhausted"
    return scope
