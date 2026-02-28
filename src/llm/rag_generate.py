
import os
import sys
import json
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.llm.factory import get_llm
from src.utils import LLMUtility, load_config



def _get_grounding_settings() -> dict:
    config = load_config()
    rag_cfg = config.get("rag", {}) or {}
    answer_format_style = str(rag_cfg.get("answer_format_style", "fluent_with_table")).strip().lower()
    if answer_format_style not in {"fluent_with_table", "concise_bullets"}:
        answer_format_style = "fluent_with_table"
    return {
        "grounding_mode": str(rag_cfg.get("grounding_mode", "strict_extractive")).strip().lower(),
        "require_inline_citations": bool(rag_cfg.get("require_inline_citations", True)),
        "max_claims_per_answer": int(rag_cfg.get("max_claims_per_answer", 12)),
        "missing_info_phrase": str(rag_cfg.get("missing_info_phrase", "정보가 부족합니다")).strip() or "정보가 부족합니다",
        "answer_format_style": answer_format_style,
    }


def get_rag_chain(request_timeout_sec: int | None = None):
    """
    Creates and returns the RAG generation chain.
    """
    llm = get_llm("retrieval", request_timeout_sec=request_timeout_sec)
    settings = _get_grounding_settings()
    grounding_mode = settings["grounding_mode"]
    require_inline_citations = settings["require_inline_citations"]
    max_claims_per_answer = max(1, settings["max_claims_per_answer"])
    missing_info_phrase = settings["missing_info_phrase"]
    answer_format_style = settings["answer_format_style"]

    if grounding_mode == "strict_extractive":
        system_text = """당신은 LTDB 문맥 기반 추출 응답기입니다.
핵심 원칙:
1) [Context]에 명시된 정보만 사용하십시오.
2) [Context]에 없는 사실은 절대 추가하지 마십시오.
3) 각 주장(문장/핵심 bullet) 끝에 반드시 근거 ref 태그를 붙이십시오.
4) 태그는 반드시 [lang/file.html:chunk_id] 형식으로만 쓰십시오.
   예) [archi/12_vowel.html:0]
5) [Context]의 Ref ID에서 위 canonical 부분만 사용하고, 제목/설명을 태그에 넣지 마십시오.
6) 근거가 없으면 해당 항목은 간단히 '자료 없음'으로 표시하십시오.
7) 과도한 확장 설명, 추측, 외부 상식 보충을 금지합니다.
8) 한국어로 간결하고 정확하게 답하십시오."""
    else:
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

    citation_rule = (
        "6. **Inline Evidence Tags**: 모든 주장 끝에 [lang/file.html:chunk_id] 태그를 붙이십시오."
        if require_inline_citations
        else "6. **Inline Evidence Tags**: 가능하면 [lang/file.html:chunk_id] 태그를 붙이십시오."
    )
    style_rule = (
        "9. **Format**: 짧은 도입(1-2문장) + 필요 시 비교표 1개 + 짧은 결론(1-2문장). "
        "표의 모든 데이터 행 끝에 근거 태그를 붙이고, 표 밖의 문장도 문장마다 근거 태그를 붙이십시오."
        if answer_format_style == "fluent_with_table"
        else "9. **Format**: 핵심 불릿 위주로 간결하게 작성하십시오."
    )

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
{citation_rule}
7. **Claim Count Limit**: 핵심 주장/불릿은 최대 {max_claims_per_answer}개.
8. 근거 없는 항목은 "{missing_info_phrase}" 또는 "자료 없음"으로 짧게 명시.
{style_rule}

Answer:"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_text),
        HumanMessagePromptTemplate.from_template(human_text)
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain

def generate_answer(
    query: str,
    context: str,
    request_timeout_sec: int | None = None,
) -> str:
    """
    Generates an answer string programmatically.
    """
    chain = get_rag_chain(request_timeout_sec=request_timeout_sec)
    settings = _get_grounding_settings()
    response = chain.invoke(
        {
            "context": context,
            "question": query,
            "citation_rule": (
                "필수"
                if settings["require_inline_citations"]
                else "권장"
            ),
            "max_claims_per_answer": max(1, settings["max_claims_per_answer"]),
            "missing_info_phrase": settings["missing_info_phrase"],
            "style_rule": (
                "9. **Format**: 짧은 도입(1-2문장) + 필요 시 비교표 1개 + 짧은 결론(1-2문장). 표의 모든 데이터 행 끝에 근거 태그를 붙이고, 표 밖의 문장도 문장마다 근거 태그를 붙이십시오."
                if settings["answer_format_style"] == "fluent_with_table"
                else "9. **Format**: 핵심 불릿 위주로 간결하게 작성하십시오."
            ),
        }
    )
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
    settings = _get_grounding_settings()
    invoke_payload = {
        "context": context,
        "question": query,
        "citation_rule": (
            "필수"
            if settings["require_inline_citations"]
            else "권장"
        ),
        "max_claims_per_answer": max(1, settings["max_claims_per_answer"]),
        "missing_info_phrase": settings["missing_info_phrase"],
        "style_rule": (
            "9. **Format**: 짧은 도입(1-2문장) + 필요 시 비교표 1개 + 짧은 결론(1-2문장). 표의 모든 데이터 행 끝에 근거 태그를 붙이고, 표 밖의 문장도 문장마다 근거 태그를 붙이십시오."
            if settings["answer_format_style"] == "fluent_with_table"
            else "9. **Format**: 핵심 불릿 위주로 간결하게 작성하십시오."
        ),
    }
    
    print(f"Model: {llm_model}")
    print("-" * 50)
    
    # Stream output
    stream_active = False
    try:
        full_response = ""
        for chunk in chain.stream(invoke_payload):
            print(chunk, end="", flush=True)
            full_response += chunk
            stream_active = True
    except Exception as e:
        print(f"\\n[Error during streaming] {e}")

    if not stream_active:
        print("\\n[Warning] Stream returned empty. Retrying with invoke()...")
        try:
            response = chain.invoke(invoke_payload)
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
