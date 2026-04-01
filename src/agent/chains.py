from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser

from src.llm.factory import get_llm
from src.utils import load_config

# --- 1. Grade Documents Chain ---

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes', 'maybe', or 'no'"
    )

def get_grade_chain():
    llm = get_llm("retrieval")
    rag_cfg = load_config().get("rag", {}) or {}
    strict_relevance_judge_enabled = bool(
        rag_cfg.get("strict_relevance_judge_enabled", True)
    )

    if strict_relevance_judge_enabled:
        system = """You are a careful relevance grader for retrieval.
Decide whether a retrieved document is useful evidence for answering the user question.

Use these labels:
- yes: directly relevant evidence for the asked topic, requested feature, or requested examples.
- maybe: partially relevant or one-side supporting evidence that is still useful to keep.
- no: off-topic, wrong phenomenon, or only broad background with no meaningful support.

Important rules:
- For comparison questions, a document may still be relevant even if it covers only one compared language/entity, as long as it directly supports one side of the requested comparison.
- For split or branch-local retrieval, judge the document against the effective branch question when provided.
- For inventory, paradigm, or feature-summary questions, exact section pages and nearby explanatory chunks about the same requested phenomenon can be relevant even if they are not exhaustive on their own.
- For phenomenon-internal sub-sections, keep partial but on-topic evidence.
  Examples:
  - For a consonant-system question, the overview section, the single-consonant section, and the digraph/cluster section are all relevant.
  - For a vowel-system question, the overview section, subtype inventory sections, and distribution sections can all be relevant.
  - If a document enumerates members, counts, subtypes, or examples of the asked phenomenon, prefer 'yes' or 'maybe' over 'no'.

Reject documents when they are only loosely related because they:
- mention the same language/entity but discuss a different subtopic,
- come from the same broad grammar area without addressing the asked phenomenon,
- contain generic background that does not support a core claim in the answer,
- provide examples unrelated to the requested topic.

If the user asks about information structure, topic markers, focus, alignment, word order, or examples, keep only documents that directly support those notions.

Return only one label: 'yes', 'maybe', or 'no'. Nothing else."""
    else:
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
