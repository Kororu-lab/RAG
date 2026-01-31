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
    
    # Fallback to string parsing for broader compatibility (DeepSeek/Ollama)
    system = """You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
    
    Return only 'yes' or 'no'. Nothing else."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | llm | StrOutputParser()
    return retrieval_grader

# --- 2. Rewrite Query Chain ---

def get_rewrite_chain():
    llm = get_llm("retrieval")

    system = """You are a question re-writer that converts an input question to a better version that is optimized 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning. 
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
    
    system = """You are a fact-checker. Given a set of documents and an answer, determine if the answer is grounded in the documents.
    
    Return 'yes' if the answer is clearly grounded in the documents.
    Return 'ambiguous' if the answer is reasonable but relies on inference or general knowledge not explicitly in the docs.
    Return 'no' if the answer contradicts the documents or contains specific facts not found in the documents (hallucination).
    
    Return ONLY one word: 'yes', 'ambiguous', or 'no'.
    """
    
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Documents: \n{documents} \n\n Answer: \n{generation} \n\n Judgment (yes/ambiguous/no):"),
        ]
    )
    
    checker = hallucination_prompt | llm | StrOutputParser()
    return checker
