
import os
import sys
import json
import yaml
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.llm.factory import get_llm
from src.utils import LLMUtility, load_config



def get_rag_chain():
    """
    Creates and returns the RAG generation chain.
    """
    llm = get_llm("retrieval")

    system_text = """당신은 언어학 데이터베이스(LTDB) 전문 연구원입니다.
1. **문맥 기반 답변 (Truthfulness)**
   - 제공된 [Context]에 있는 내용만을 바탕으로 답변을 구성하십시오.
   - 문맥에 없는 내용은 추측하여 생성하지 말고, 정보가 없으면 솔직히 명시하십시오.

2. **외부 지식 사용 자제 (No Erasure, No Invention)**
   - 문맥에 **없는** 언어를 억지로 채워넣지 마십시오. (Hallucination 방지)
   - 반대로, **문맥에 있는** 유용한 정보는 절대 누락하지 말고 충분히 활용하십시오.

3. **정보가 부족할 때**
   - 질문을 해결하기에 문맥이 턱없이 부족할 때만 "정보가 부족합니다"라고 답하십시오.
   - 조금이라도 관련된 내용이 있다면 최대한 그 내용을 바탕으로 답변해 주십시오.

4. **언어**
   - 한국어로 답변하십시오.
   - 전문 용어는 영어/원어를 병기하십시오."""

    human_text = """[Context]
{context}

[Question]
{question}

[Constraints]
1. **Strict Context Adherence**: Answer ONLY based on the provided [Context].
2. **NO Hallucinated Links**: Do NOT provide external links (e.g., Wikipedia, Glottolog) unless explicitly in [Context].
3. **NO Meta-Commentary**: Just answer content.
4. **Language**: Answer in **Korean**.
5. **Tone**: Academic, precise. 

Answer:"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_text),
        HumanMessagePromptTemplate.from_template(human_text)
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain

def generate_answer(query: str, context: str) -> str:
    """
    Generates an answer string programmatically.
    """
    chain = get_rag_chain()
    response = chain.invoke({"context": context, "question": query})
    return response

def generate():
    """
    CLI Entry point reading from context.json
    """
    if not os.path.exists("context.json"):
        print("Error: context.json not found. Run rag_retrieve.py first.")
        return

    with open("context.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    query = data['query']
    context = data['context']
    references = data['references']
    
    config = load_config()
    llm_cfg = config.get("llm_retrieval", config.get("llm", {}))
    llm_model = llm_cfg.get("model_name")
    
    print(f"Stage 2: Generating Answer with {llm_model}...")
    
    chain = get_rag_chain()
    
    print(f"Model: {llm_model}")
    print("-" * 50)
    
    # Stream output
    stream_active = False
    try:
        full_response = ""
        for chunk in chain.stream({"context": context, "question": query}):
            print(chunk, end="", flush=True)
            full_response += chunk
            stream_active = True
    except Exception as e:
        print(f"\\n[Error during streaming] {e}")

    if not stream_active:
        print("\\n[Warning] Stream returned empty. Retrying with invoke()...")
        try:
            response = chain.invoke({"context": context, "question": query})
            print(response)
        except Exception as e:
            print(f"\\n[Error during invoke] {e}")
            
    print("\\n" + "-" * 50)
    
    # Print References
    print("\\n[Reference Sources]")
    for ref in references:
        if ref['type'] == 'text':
            level = ref.get('level', '0')
            prefix = "[Summary/L1]" if str(level) == '1' else "[Detail/L0]"
            ref_id = ref.get('ref_id', 'Unknown')
            preview = ref.get('preview', '')
            
            print(f"- {prefix} {ref_id}")
            if preview:
                print(f"  Preview: {preview}")
                
        elif ref['type'] == 'vision':
            print(f"- [Vision] [{ref.get('element_type')}] Score: {ref.get('score'):.2f}")
            print(f"  File: {ref.get('file')}")
            print(f"  Image: {ref.get('image')}")
            
    print("=" * 50)
    
    # Cleanup
    print("Stage 2 Complete. Unloading LLM...")
    LLMUtility.unload_model("retrieval")

if __name__ == "__main__":
    generate()
