from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser

from src.llm.factory import get_llm

# --- 1. Grade Documents Chain ---

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

def get_grade_chain():
    llm = get_llm("retrieval")
    
    # Keep a permissive relevance gate: drop clear noise, keep partially useful docs.
    system = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
It does not need to be a stringent test. The goal is to filter out erroneous retrievals.

Return only 'yes' or 'no'. Nothing else."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: {question}",
            ),
        ]
    )

    retrieval_grader = grade_prompt | llm | StrOutputParser()
    return retrieval_grader

# --- 2. Rewrite Query Chain ---

def get_rewrite_chain():
    llm = get_llm("retrieval")

    system = """You are a query re-writer for retrieval.
Rules:
1) Preserve the original intent exactly.
2) Preserve explicitly mentioned languages/entities exactly.
3) Do not broaden scope, add new subtopics, or add new languages.
4) Improve clarity and keyword match only.
Return ONLY the rewritten question. Do not explain."""
     
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ]
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()
    return question_rewriter

# --- 3. Hallucination Checker Chain ---

def get_hallucination_chain():
    llm = get_llm("retrieval")
    
    system = """You are a contradiction-focused fact-checker for RAG outputs.
Given documents and an answer, decide only whether there are clear unsupported/fabricated details or contradictions.

Decision policy:
- yes: no clear contradiction/fabrication; cited claims are consistent with document evidence.
- ambiguous: wording is generalized/abstractive, but no clear fabricated specific fact.
- no: any clearly fabricated specific detail, citation mismatch, or contradiction with documents.

Important:
1) Do NOT fail for concise paraphrase if meaning is preserved.
2) Be conservative with 'no'; use 'no' only when evidence clearly conflicts or is absent for a specific factual claim.
3) If inline citation tags are present, verify claim-tag consistency.
Return ONLY one word: 'yes', 'ambiguous', or 'no'."""
    
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Documents: \n{documents} \n\n Answer: \n{generation} \n\n Judgment (yes/ambiguous/no):"),
        ]
    )
    
    checker = hallucination_prompt | llm | StrOutputParser()
    return checker
