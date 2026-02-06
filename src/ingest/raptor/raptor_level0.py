import os
import sys
import glob
import json
import re
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Add project root to sys.path to resolve 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.utils import load_config, LLMUtility
from src.ingest.xsampa_converter import xsampa_to_ipa
from src.llm.factory import get_llm


CONFIG = load_config()
DATA_PATH = CONFIG["project"]["data_path"]
DB_PATH = CONFIG["project"]["db_path"]
llm_cfg = CONFIG.get("llm_ingestion", CONFIG.get("llm", {}))
LLM_MODEL = llm_cfg.get("model_name")
BASE_URL = llm_cfg.get("base_url")

from bs4 import BeautifulSoup, NavigableString

def parse_html_sections(file_path):
    """
    Parses an HTML file and extracts sections based on h1, h2, h3 tags.
    Uses leaf-node traversal to avoid text duplication.
    Returns a list of tuples: (header_text, content_text)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Pre-convert X-SAMPA tags to preserve tone superscripts
    for tag in soup.find_all(class_='xsampa'):
        if tag.string:
            converted = xsampa_to_ipa(tag.string)
            tag.string.replace_with(converted)

    sections = []
    current_header = "Intro"
    current_content = []

    # Iterate over all elements using descendants to catch everything
    if soup.body:
        for element in soup.body.descendants:
            # Check for Header Tags -> content switch
            if element.name in ["h1", "h2", "h3"]:
                # Save previous section if it has content
                if current_content:
                    sections.append((current_header, " ".join(current_content).strip()))
                    current_content = []
                current_header = element.get_text(strip=True)
            
            # 'element' is the h1 tag -> switch context.
            elif isinstance(element, NavigableString):
                parent_name = element.parent.name
                if parent_name not in ["h1", "h2", "h3", "script", "style"]:
                    text = element.strip()
                    if text:
                        current_content.append(text)
    
    # Append the last section
    if current_content:
        sections.append((current_header, " ".join(current_content).strip()))

    return sections

def summarize_section(llm, text, header, language):
    """
    Generates a Level 0 summary using an LLM with Masking Strategy.
    """
    # Mask numbers to protect from X-SAMPA conversion
    masked_text = re.sub(r'(\d+)', r'⦗NUM:\1⦘', text)

    prompt = ChatPromptTemplate.from_template("""
You are a specialized linguistics data summarizer.
The input text contains **X-SAMPA** phonetic encoding.
**CRITICAL**: Numbers are masked as `⦗NUM:123⦘` to protect them from incorrect conversion.

[Rules]
1. **PROTECT NUMBERS**: When you see `⦗NUM:378⦘`, output `378` (plain digits).
   - **NEVER** convert these digits to IPA/X-SAMPA (e.g., do NOT write `³`, `¹`, `ɤ`, `ɵ`).
   - Output the digits exactly as they are.
2. **CONVERT X-SAMPA**: Convert other X-SAMPA symbols to IPA contextually.
3. **NO META-COMMENTARY**: Do not mention "masked numbers" or "safety tokens". Just use the value.
4. **NO HALLUCINATION**: Do NOT generate programming code, tutorials, or unrelated text. This is LINGUISTICS data.
5. **SUMMARIZE**: Summarize in **Korean**.

[Metadata]
- Language: {language}
- Section Header: {header}

[Text Content]
{text}

[Instructions]
1.  **Summary**: Write a concise, factual summary in **Korean**.
2.  **Key Terms**: Keep linguistic terms in **English** or explain them.
3.  **Keywords**: At the end, add `[Keywords]: term1, term2, ...`

[Response Format]
[Summary]: <Korean summary>
[Keywords]: <term1, term2, ...>
""")
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "language": language,
        "header": header,
        "text": masked_text
    })

    # Unmask numbers in output
    unmasked_response = re.sub(r'⦗NUM:(\d+)⦘', r'\1', response)
    
    return unmasked_response

def _is_garbage_output(text: str) -> bool:
    """
    Detects if the output is garbage (infinite repetition or irrelevant AI text).
    """
    if len(text) < 100:
        return False
        
    # Repetition check
    words = text.split()
    if len(text) > 300 and len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.1: # < 10% unique words
            return True
            
    # N-gram repetition
    if len(text) > 200:
        tail = text[-100:]
        if text.count(tail) > 2: # Repeated 3+ times
            return True
            
    # Known hallucination keywords
    bad_keywords = [
        "Artificial Intelligence", "Future of Work", "Machine Learning", 
        "Job displacement", "일의 미래", "인공지능",
        "Node.js", "Express", "MongoDB", "checklist", "starter code", 
        "Clarifying Questions", "High‑Level Plan", "Example Implementation"
    ]
    for kw in bad_keywords:
        if kw in text: 
             return True
             
    return False

def summarize_section_robust(llm, text, header, language) -> str:
    """
    Robust wrapper for summarize_section with retries and garbage detection.
    """
    attempts = 0
    max_attempts = 3
    
    while attempts < max_attempts:
        try:
            if attempts > 0:
                # 재시도 시 model temerature 변경(필요시 추가)
                pass 

            response = summarize_section(llm, text, header, language)
            
            if _is_garbage_output(response):
                print(f"  [WARN] Garbage detected for {header} ({language}). Retrying... ({attempts+1}/{max_attempts})")
                attempts += 1
                continue
                
            return response
            
        except Exception as e:
            print(f"  [ERR] Exception during summarization: {e}")
            attempts += 1
            
    print(f"  [FAIL] Failed to generate valid summary for {header} after retries.")
    return text[:500] + "... (Validation Failed)"

def process_file(file_path, llm):
    """
    Processes a single file: Parse -> X-SAMPA Conv -> Summarize/Bypass -> Return Data
    """
    rel_path = os.path.relpath(file_path, DATA_PATH)
    parts = rel_path.split(os.sep)
    # Expected doc/LANGUAGE/topic/...
    lang = parts[1] if len(parts) > 1 else "unknown"
    
    sections = parse_html_sections(file_path)
    file_data = []

    for i, (header, raw_content) in enumerate(sections):
        content = raw_content.replace("ERROR: no translation line", "").strip()
        content_len = len(content)

        if content_len < 10:
            continue  # Noise
        
        summary = ""
        keywords = ""
        node_type = "L0_base"
        
        if 10 <= content_len < 100:
            # Bypass LLM
            summary = content.strip()
            summary = " ".join(summary.split())
            keywords = "[Auto-Bypassed: Short Text]"
        else:
            # Full LLM Summarization
            full_response = summarize_section_robust(llm, content, header, lang)
            
            # Robust Regex Parsing for Keywords
            keyword_match = re.search(r'\[Keywords\]:\s*(.*)', full_response, re.IGNORECASE | re.DOTALL)
            if keyword_match:
                keywords = keyword_match.group(1).strip()
                # Remove the keyword part from the summary
                summary = full_response[:keyword_match.start()].replace("[Summary]:", "").strip()
            else:
                summary = full_response.replace("[Summary]:", "").strip()
                
            if "Validation Failed" in summary:
                keywords = "error"

        file_data.append({
            "source_file": rel_path,
            "chunk_id": i,
            "lang": lang,
            "level": 0,
            "type": node_type,
            "original_header": header,
            "summary": summary,
            "keywords": keywords,
            "original_content": content 
        })
        
    return file_data

def main():
    print(f"Initializing Matrix RAPTOR Ingestion...")
    print(f"Data Source: {DATA_PATH}")
    print(f"LLM: {LLM_MODEL} (Base URL: {BASE_URL})")

    # Initialize LLM
    print(f"Initializing LLM (Ingestion Profile)...")
    llm = get_llm("ingestion")

    # Find all HTML files
    all_files = glob.glob(os.path.join(DATA_PATH, "doc", "**", "*.html"), recursive=True)
    all_files.sort() # Ensure deterministic order
    
    print(f"Found {len(all_files)} HTML files to process.")
    
    # Setup Output
    # Setup Output
    output_file = os.path.join("data", "raptor", "matrix_raptor_L0_data.jsonl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Initialized output file: {output_file}")
    
    with open(output_file, "w", encoding="utf-8") as f_out:
        for idx, file_path in enumerate(all_files):
            # Console progress
            print(f"[{idx+1}/{len(all_files)}] Processing {os.path.basename(file_path)}...", end="\r")
            
            file_results = process_file(file_path, llm)
            
            for item in file_results:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                
    print(f"\n\nIngestion Complete. Data saved to {output_file}")
    # Cleanup
    LLMUtility.unload_model("ingestion")

if __name__ == "__main__":
    main()
