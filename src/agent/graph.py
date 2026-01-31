import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langgraph.graph import END, StateGraph
from src.agent.state import GraphState
from src.agent.nodes import retrieve_node, grade_documents_node, generate_node, rewrite_query_node, check_hallucination_node

# --- Conditional Edge Logic ---

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.
    Features: Soft Fallback (Generate anyway after max retries).
    """
    print("---DECIDE TO GENERATE---")
    state["documents"]
    question = state["question"]
    
    filtered_documents = state["documents"]
    search_count = state.get("search_count", 0)

    if not filtered_documents:
        # All documents filtered out
        if search_count >= 3:
             # Soft Fallback: Try generating with what we have (or "No documents" message)
             print("---DECISION: MAX RETRIES REACHED -> FORCE GENERATE (SOFT FALLBACK)---")
             return "generate"
        else:
             print("---DECISION: ALL DOCUMENTS IRRELEVANT -> REWRITE QUERY---")
             return "rewrite"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

def check_hallucination(state):
    """
    Determines if the generation is grounded.
    """
    print("---DECIDE: HALLUCINATION CHECK---")
    status = state.get("hallucination_status", "fail")
    
    if status == "pass":
        print("---DECISION: PASS -> END---")
        return "useful"
    else:
        print("---DECISION: HALLUCINATION -> REWRITE---")
       
        search_count = state.get("search_count", 0)
        if search_count >= 3:
             print("---DECISION: MAX RETRIES (HALLUCINATION LOOP) -> END (WITH WARNING)---")
             return "useful" # Accept it as is to break loop

        return "not useful"

# --- Graph Contruction ---

workflow = StateGraph(GraphState)

# Define Nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("generate", generate_node)
workflow.add_node("rewrite", rewrite_query_node)
workflow.add_node("check_hallucination", check_hallucination_node)

# Build Graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "rewrite": "rewrite",
        "generate": "generate",
    },
)
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", "check_hallucination")
workflow.add_conditional_edges(
    "check_hallucination",
    check_hallucination,
    {
        "useful": END,
        "not useful": "rewrite",
    },
)

# Compile
app = workflow.compile()

# --- Execution Entry Point ---
if __name__ == "__main__":
    import pprint
    
    # Test Interaction
    print("Running Corrective RAPTOR Agent...")
    while True:
        query = input("\nEnter query (or 'q' to quit): ").strip()
        if query.lower() in ["q", "quit"]:
            break
            
        inputs = {"question": query, "search_count": 0}
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"Node '{key}':")
                
                # Only print generation at the generation step to avoid duplicates
                if key == "generate" and "generation" in value:
                    print(f"\n>> 답변 (Generation):\n{value['generation']}")
                    
                    # Print References
                    print("\n[Reference Sources]")
                    seen_refs = set()
                    if "documents" in value:
                        for doc in value["documents"]:
                            ref_id = doc.metadata.get('ref_id', 'Unknown')
                            if ref_id not in seen_refs:
                                level = doc.metadata.get('level', '0')
                                prefix = "[Summary/L1]" if str(level) == '1' else "[Detail/L0]"
                                print(f"- {prefix} {ref_id}")
                                seen_refs.add(ref_id)
                    print("="*50)
                            
                elif "question" in value and key == "rewrite":
                    print(f"  - Rewritten Query: {value['question']}")
            print("\n---\n")
            
