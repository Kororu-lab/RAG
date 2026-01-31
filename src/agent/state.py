from typing import List, TypedDict
from langchain_core.documents import Document

class GraphState(TypedDict):
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
