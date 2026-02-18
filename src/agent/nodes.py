import threading
from typing import Any, Dict

import src.utils
from src.agent.state import GraphState
from src.retrieve.rag_retrieve import RAGRetriever, extract_query_metadata
from src.agent.chains import get_grade_chain, get_rewrite_chain, get_hallucination_chain
from src.llm.rag_generate import generate_answer

def _build_structured_metadata(d) -> Dict[str, str]:
    """
    Build structured metadata from source_file split.
    Fallback is used only when split fields are missing.
    """
    source_file = str(d.metadata.get("source_file", "unknown"))
    parts = [p for p in source_file.split("/") if p]

    language = d.metadata.get("lang")
    if not language:
        if len(parts) >= 2 and parts[0] == "doc":
            language = parts[1]
        else:
            language = "unknown"

    filename = parts[-1] if parts else "unknown"

    topic = parts[-2] if len(parts) >= 2 else ""
    if not topic:
        # Error-only fallback
        fallback_header = str(d.metadata.get("original_header", "")).strip()
        topic = fallback_header or "unknown"

    return {
        "Language": str(language),
        "Filename": str(filename),
        "Topic": str(topic),
        "SourceFile": source_file,
    }


def _format_structured_block(meta: Dict[str, str]) -> str:
    return (
        "[Structured Metadata]\n"
        f"Language: {meta['Language']}\n"
        f"Filename: {meta['Filename']}\n"
        f"Topic: {meta['Topic']}\n"
        f"SourceFile: {meta['SourceFile']}"
    )

def _get_generation_timeout_sec(default_sec: int = 60) -> int:
    try:
        config = src.utils.load_config()
        sec = int(config.get("rag", {}).get("generation_timeout_sec", default_sec))
        return sec if sec > 0 else default_sec
    except Exception:
        return default_sec

def _generate_with_timeout(query: str, context: str, timeout_sec: int) -> str:
    result = {"value": None, "error": None}
    timeout_fallback = (
        f"정보가 부족합니다\n\n(답변 생성 시간 제한 {timeout_sec}초를 초과했습니다.)"
    )

    def _worker():
        try:
            result["value"] = generate_answer(
                query=query,
                context=context,
                request_timeout_sec=timeout_sec,
            )
        except Exception as e:
            result["error"] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)

    if t.is_alive():
        return timeout_fallback

    if result["error"] is not None:
        if "timeout" in str(result["error"]).lower():
            return timeout_fallback
        raise result["error"]

    return result["value"]


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

    # Dynamic Top-K scaling (3-tier)
    # Base: 15 for normal queries
    # Mid-broad: 30 for topic-specific linguistics queries  
    # Super-broad: 60 for exhaustive queries (all, compare, list)
    
    super_broad_keywords = [
        "모든", "목록", "전체", "list", "all", "overview", "언어들",
        "데이터베이스", "db", "비교", "compare", "차이", "difference"
    ]
    
    mid_broad_keywords = [
        # Topic-specific linguistics terms
        "aspect", "시상", "tense", "양태", "modality", "mood",
        "연동", "serial", "svc", "구문", "construction",
        "음운", "phonology", "phoneme", "자음", "모음", "vowel", "consonant",
        "어순", "word order", "화제", "topic", "주어", "subject",
        "격", "case", "표지", "marker", "접사", "affix",
        "종류", "types", "features", "특징", "구조", "structure"
    ]
    
    q_lower = question.lower()
    
    if any(kw in q_lower for kw in super_broad_keywords):
        k, top_n = 60, 25
        print(f"  - Dynamic Scaling: Super-broad query detected.")
        print(f"    -> k={k}, top_n={top_n}")
    elif any(kw in q_lower for kw in mid_broad_keywords):
        k, top_n = 30, 15
        print(f"  - Dynamic Scaling: Mid-broad (topic-specific) query detected.")
        print(f"    -> k={k}, top_n={top_n}")
    else:
        k, top_n = 15, 10
        print(f"  - Dynamic Scaling: Normal query. k={k}, top_n={top_n}")


    with RAGRetriever() as retriever:
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

    # Skip grading if disabled via config
    config = src.utils.load_config()
    if config.get("rag", {}).get("skip_grading", False):
        print("---SKIP DOCUMENT GRADING (disabled)---")
        return {"documents": documents, "question": question}

    grader = get_grade_chain()
    filtered_docs = []
    
    relevance_count = 0
    
    for d in documents:
        structured_meta = _build_structured_metadata(d)
        source_info = f"[Source]: {structured_meta['SourceFile']}"
        rich_content = (
            f"{_format_structured_block(structured_meta)}\n"
            "[Document Content]\n"
            f"{d.page_content}"
        )
        
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
            structured_meta = _build_structured_metadata(d)
            level = d.metadata.get('level', '0')
            prefix = "[Summary]" if str(level) == '1' else "[Detail]"
            source_info = f"[Source: {structured_meta['SourceFile']}]"
            text_context_lines.append(
                f"{prefix} Ref: {d.metadata.get('ref_id', 'Chunk')} {source_info}\n"
                f"{_format_structured_block(structured_meta)}\n"
                "[Document Content]\n"
                f"{d.page_content}"
            )

    context_str = "\n\n".join(text_context_lines)
    if vision_context_lines:
        context_str += "\n\n[Visual Evidence Found]\n" + "\n---\n".join(vision_context_lines)
        
    if not context_str.strip():
        context_str = "No relevant documents found."

    timeout_sec = _get_generation_timeout_sec(default_sec=60)
    generation = _generate_with_timeout(query=question, context=context_str, timeout_sec=timeout_sec)
    return {"documents": documents, "question": question, "generation": generation}

def check_hallucination_node(state: GraphState):
    """
    Checks if the generated answer is grounded in the documents.
    """
    print("---CHECK HALLUCINATION---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # Skip hallucination check if disabled via config
    config = src.utils.load_config()
    if config.get("rag", {}).get("skip_hallucination", False):
        print("---SKIP HALLUCINATION CHECK (disabled)---")
        return {"documents": documents, "generation": generation, "hallucination_status": "pass"}

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
