from typing import Any, Dict
from src.agent.state import GraphState
from src.retrieve.rag_retrieve import RAGRetriever, extract_query_metadata
from src.agent.chains import get_grade_chain, get_rewrite_chain, get_hallucination_chain
from src.llm.rag_generate import generate_answer

def retrieve_node(state: GraphState):
    """
    Retrieve documents (L1 + L0) based on the question.
    Features: Dynamic Top-K, Accumulative Retrieval.
    """
    print("---RETRIEVE---")
    question = state["question"]
    search_count = state.get("search_count", 0)
    existing_docs = state.get("documents", []) # Accumulative

    # Unload LLM to free VRAM for embeddings
    try:
        from src.utils import LLMUtility
        LLMUtility.unload_model("retrieval")
    except Exception:
        pass

    # Dynamic Top-K scaling
    base_k = 5
    base_top_n = 5
    
    broad_keywords = ["모든", "목록", "종류", "전체", "list", "types", "all", "overview", "언어들", "데이터베이스", "db", "비교", "차이", "features", "특징", "구조"]
    if any(k in question.lower() for k in broad_keywords):
        print(f"  - Dynamic Scaling: Broad query detected ('{question}').")
        k = 30
        top_n = 15 # Allow more docs to pass reranker
        print(f"    -> Increasing k={k}, top_n={top_n}")
    else:
        print(f"  - Dynamic Scaling: Specific query detected. Keeping base k={base_k}, top_n={base_top_n}.")
        k = base_k
        top_n = base_top_n


    retriever = RAGRetriever()
    
    # Metadata Extraction Logic (New)
    print("  - Analyzing Query Metadata...")
    metadata = extract_query_metadata(question, retriever.config)
    
    new_docs = retriever.retrieve_documents(question, k=k, top_n=top_n, metadata=metadata)
    
    # Merge and deduplicate
    combined_docs = []
    seen_ids = set()
    for d in existing_docs + new_docs:
        ref_id = d.metadata.get('ref_id')
        if ref_id and ref_id not in seen_ids:
            combined_docs.append(d)
            seen_ids.add(ref_id)
        elif not ref_id: # fallback
             combined_docs.append(d)

    print(f"  - Total Documents (Accumulated): {len(combined_docs)}")
    
    return {"documents": combined_docs, "question": question, "search_count": search_count}

def grade_documents_node(state: GraphState):
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    print("---CHECK DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    
    grader = get_grade_chain()
    filtered_docs = []
    
    relevance_count = 0
    
    for d in documents:
        source_info = f"[Source]: {d.metadata.get('source_file', 'unknown')}"
        rich_content = f"{source_info}\n{d.page_content}"
        
        score = grader.invoke({"question": question, "document": rich_content})
        
        # Robust parsing for string output
        grade = str(score).lower().strip()
        
        if "yes" in grade:
            print(f"  - GRADED DOCUMENT: RELEVANT ({d.metadata.get('ref_id')}) [{source_info}]")
            filtered_docs.append(d)
            relevance_count += 1
        else:
            print(f"  - GRADED DOCUMENT: NOT RELEVANT ({d.metadata.get('ref_id')})")
            continue
            
    return {"documents": filtered_docs, "question": question}

def generate_node(state: GraphState):
    """
    Generate answer using the retrieved documents.
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # Format Context
    text_context_lines = []
    vision_context_lines = []
    
    for d in documents:
        if d.metadata.get('type') == 'vision':
            vision_context_lines.append(d.page_content)
        else:
            level = d.metadata.get('level', '0')
            prefix = "[Summary]" if str(level) == '1' else "[Detail]"
            source_info = f"[Source: {d.metadata.get('source_file', 'unknown')}]"
            text_context_lines.append(f"{prefix} Ref: {d.metadata.get('ref_id', 'Chunk')} {source_info}\n{d.page_content}")

    context_str = "\n\n".join(text_context_lines)
    if vision_context_lines:
        context_str += "\n\n[Visual Evidence Found]\n" + "\n---\n".join(vision_context_lines)
        
    if not context_str.strip():
        context_str = "No relevant documents found."
        
    generation = generate_answer(query=question, context=context_str)
    return {"documents": documents, "question": question, "generation": generation}

def check_hallucination_node(state: GraphState):
    """
    Checks if the generated answer is grounded in the documents.
    """
    print("---CHECK HALLUCINATION---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    checker = get_hallucination_chain()
    
    # Prepare documents text for checker
    full_text = "\n".join([d.page_content for d in documents])
    
    if len(full_text) > 30000:
        docs_text = full_text[:30000] + "... (truncated)"
    else:
        docs_text = full_text
        
    # Check for valid refusal
    refusal_keywords = ["정보가 부족", "알 수 없습니다", "문맥에 나타나 있지 않으", "제공된 문맥", "information is missing"]
    if any(k in generation for k in refusal_keywords):
        print("  - DECISION: GROUNDED REFUSAL (Pass)")
        return {"documents": documents, "generation": generation, "hallucination_status": "pass"}
    
    score = checker.invoke({"documents": docs_text, "generation": generation})
    grade = str(score).lower().strip()
    
    # Tiered Logic
    if "yes" in grade:
        print("  - DECISION: GROUNDED (Pass)")
        return {"documents": documents, "generation": generation, "hallucination_status": "pass"}
        
    elif "ambiguous" in grade:
        print("  - DECISION: AMBIGUOUS -> PASS WITH WARNING")
        warning_msg = "\n\n> [!WARNING] (Uncertainty): This answer is based on inference or partial information. Please verify specific details."
        new_generation = generation + warning_msg
        return {"documents": documents, "generation": new_generation, "hallucination_status": "pass"}
        
    else:
        print("  - DECISION: HALLUCINATION DETECTED (Fail)")
        return {"documents": documents, "generation": generation, "hallucination_status": "fail"}


def rewrite_query_node(state: GraphState):
    """
    Transform the query to produce a better question.
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    search_count = state.get("search_count", 0)
    
    rewriter = get_rewrite_chain()
    better_question = rewriter.invoke({"question": question})
    
    print(f"  - Old Query: {question}")
    print(f"  - New Query: {better_question}")
    
    return {"documents": documents, "question": better_question, "search_count": search_count + 1}
