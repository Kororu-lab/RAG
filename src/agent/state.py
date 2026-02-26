from typing import Any, Dict, List, TypedDict
from langchain_core.documents import Document

class GraphState(TypedDict, total=False):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        documents: list of documents (L1/L0 mixed)
        generation: LLM generation
        web_search: whether to add web search (Optional)
        search_count: number of search attempts (Loop limit)
    """
    question: str
    documents: List[Document]
    generation: str
    web_search: bool
    search_count: int
    hallucination_status: str # 'pass' or 'fail'
    warnings: List[str]
    timeout_locations: List[str]
    retrieval_diagnostics: Dict[str, Any]
