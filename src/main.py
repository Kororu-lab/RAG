import os
import sys
import yaml
import subprocess
import logging

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils import load_config

# Configure Logging
logging.basicConfig(level=logging.WARNING)



def get_router_llm(config):
    """Initializes a lightweight LLM instance for routing."""
    llm_cfg = config.get("llm_retrieval", config.get("llm", {}))
    return ChatOllama(
        model=llm_cfg.get("model_name"),
        base_url=llm_cfg.get("base_url"),
        temperature=0
    )

def classify_intent(llm, query):
    """
    Classifies the user query into GRAPH or VECTOR.
    GRAPH: Lists, relations, genealogy, counts, specific metadata lookups.
    VECTOR: Grammar, examples, definitions, translations (unstructured text).
    """
    template = """Classify the following query into one of two categories: GRAPH or VECTOR.

Definitions:
- GRAPH: Questions asking for lists of entities, relationships (e.g., family, region), counts, or finding specific languages based on metadata attributes.
- VECTOR: Questions asking for grammar rules, sentence examples, translations, definitions, phonology clarifications, or general linguistic descriptions.

Query: {query}

Constraint: Return ONLY the category name (GRAPH or VECTOR). Do not add any punctuation.

Category:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    try:
        category = chain.invoke({"query": query}).strip().upper()
        # Fallback parsing
        if "GRAPH" in category: return "GRAPH"
        return "VECTOR"
    except Exception as e:
        print(f"[Router Error] {e}. Defaulting to VECTOR.")
        return "VECTOR"

def run_vector_pipeline(query, interactive=False):
    """Executes the standard Vector RAG pipeline (Process Isolation)."""
    print(f"\n[Router] Routing to VECTOR RAG Pipeline...")
    
    # Stage 1: Retrieval
    print("[Stage 1] Retrieving Context...")
    ret_code = subprocess.call(["uv", "run", "src/retrieve/rag_retrieve.py", query])
    
    if ret_code != 0:
        print("Retrieval failed.")
        return

    # Stage 2: Generation
    print("[Stage 2] Generating Answer...")
    subprocess.call(["uv", "run", "src/llm/rag_generate.py"])

def run_graph_pipeline(query):
    """Executes the Graph RAG script."""
    print(f"\n[Router] Routing to GRAPH RAG Pipeline...")
    subprocess.call(["uv", "run", "src/retrieve/graph_search.py", "--query", query])

def main():
    print("==================================================")
    print("  LTDB v1.3 Intelligent Orchestrator")
    print("==================================================")
    
    config = load_config()
    enable_graph = config.get("rag", {}).get("enable_graph", True)
    
    if not enable_graph:
        print("[Info] Graph RAG is DISABLED by config.")
    
    llm = None
    if enable_graph:
        print("Initializing Semantic Router...")
        llm = get_router_llm(config)

    while True:
        try:
            query = input("\nEnter query (or 'q' to quit): ").strip()
        except EOFError:
            break
            
        if query.lower() in ['q', 'quit', 'exit']:
            print("Exiting...")
            break
        
        if not query:
            continue

        if not enable_graph:
            run_vector_pipeline(query)
            continue

        # Semantic Routing
        intent = classify_intent(llm, query)
        print(f"[Router] Detected Intent: {intent}")
        
        if intent == "GRAPH":
            run_graph_pipeline(query)
        else:
            run_vector_pipeline(query)

if __name__ == "__main__":
    main()
