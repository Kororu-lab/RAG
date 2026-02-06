import os
import sys
import json
import re
from typing import List, Dict, Any
# from langchain_ollama import ChatOllama # Replaced by factory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Ensure src is resolvable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.utils import load_config
from src.llm.factory import get_llm

def load_l0_data(file_path: str) -> List[Dict]:
    """Loads Level 0 data from JSONL."""
    data = []
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return []
        
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data

def group_data(l0_data: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Groups data vertically (by language) and horizontally (by topic).
    """
    groups = {
        "vertical": {},   # Key: lang
        "horizontal": {}  # Key: topic
    }
    
    for item in l0_data:
        # Vertical Grouping (Language)
        lang = item.get("lang", "unknown")
        if lang not in groups["vertical"]:
            groups["vertical"][lang] = []
        groups["vertical"][lang].append(item)
        
        # Horizontal Grouping (Topic)
        # Path: doc/{lang}/{category}/{topic}/{filename}
        source_file = item.get("source_file", "")
        parts = source_file.split("/")
        
        # Topic is usually the directory containing the file
        # e.g.) doc/abaza/1_phonology/12_vowel/12_vowel.html -> topic: 12_vowel
        if len(parts) >= 2:
            topic = parts[-2]
            
            # Simple validation to exclude root level folders if path is short
            if topic not in [lang, "doc"]:
                if topic not in groups["horizontal"]:
                    groups["horizontal"][topic] = []
                groups["horizontal"][topic].append(item)
                
    return groups

def create_context_text(items: List[Dict], max_chars: int = 30000) -> str:
    """
    Concatenates summaries with citations. Truncates if too long.
    Format: [Ref: chunk_id] summary
    """
    context_parts = []
    current_length = 0
    
    for item in items:
        # Generate a unique reference ID (e.g., source_file:chunk_id or just chunk_id if processed per file)
        # Using simple index or ID to save tokens
        ref_id = item.get("chunk_id", "0")
        src_lang = item.get("lang", "")
        summary = item.get("summary", "")
        
        entry = f"[Ref: {src_lang}-{ref_id}] {summary}\n"
        
        if current_length + len(entry) > max_chars:
            break
            
        context_parts.append(entry)
        current_length += len(entry)
        
    return "\n".join(context_parts)

def summarize_group(llm, group_type: str, group_key: str, items: List[Dict]) -> Dict:
    """
    Generates Level 1 summary for a group.
    """
    # SKIP if too few items
    if len(items) < 2:
        return None
        
    context = create_context_text(items)
    
    # Define Prompts
    if group_type == "vertical":
        # Language Overview
        system_prompt = """You are a linguistics expert.
Your task is to synthesize the provided details into a comprehensive **Language Overview** for the language: {key}.
The input consists of aggregated chunks from various linguistic topics (phonology, grammar, etc.).

[Constraints]
1. **NO Repetition**: Do not repeat the same sentence loop.
2. **NO Hallucination**: Do not include outside information (e.g. AI, Future of Work). Use ONLY the provided context.
3. **NO Simple Listing**: Do NOT simply enumerate every single entity/file. Synthesize common patterns, group features, and highlight notable exceptions.
4. **Consistency**: If the context is empty/irrelevant, state "insufficient information".

[Instructions]
1. **Overview**: Write a structured summary covering the key linguistic features found in the text.
2. **Structure**: Organize by major categories (e.g., Phonology, Morphology, Syntax) if data permits.
3. **Citation**: Cite sources using the provided tags (e.g. [Ref: lang-id]) to back up claims.
4. **Language**: Write the output in **Korean**.
"""
    else:
        # Horizontal Comparative Analysis
        system_prompt = """You are a linguistics expert.
Your task is to perform a **Cross-linguistic Comparative Analysis** on the topic: {key}.
The input contains data from multiple languages regarding this topic.

[Constraints]
1. **NO Repetition**: Do not repeat the same sentence loop.
2. **NO Hallucination**: Do not include outside information (e.g. AI, Future of Work). Use ONLY the provided context.
3. **NO Simple Listing**: Do NOT simply enumerate every language one by one. Group languages by shared features or typological patterns.

[Instructions]
1. **Compare & Contrast**: Analyze how this feature is realized differently across the languages.
2. **Patterns**: Identify common patterns or unique anomalies.
3. **Citation**: Cite sources using the provided tags (e.g. [Ref: lang-id]) to specifically attribute features to languages.
4. **Language**: Write the output in **Korean**.
"""

    format_prompt = """
[Input Text]
{context}

[Output Format]
[Summary]: <Detailed Korean summary>
[Keywords]: <comma, separated, english, keywords>
"""
    
    full_template = system_prompt + format_prompt
    prompt = ChatPromptTemplate.from_template(full_template)
    chain = prompt | llm | StrOutputParser()
    
    attempts = 0
    max_attempts = 2
    
    # Batching for large clusters
    if len(items) > 15:
        print(f"  [INFO] Large cluster detected ({len(items)} items). Switching to Batch Summarization.")
        return summarize_in_batches(llm, group_type, group_key, items, system_prompt, format_prompt)

    while attempts < max_attempts:
        try:
            print(f"  > Summarizing {group_type} group: {group_key} (Items: {len(items)})... (Attempt {attempts+1})")
            
            # Dynamic Parameter Adjustment on Retry
            if attempts > 0:
                # Use safer parameters on retry
                 chain_retry = prompt | llm.bind(temperature=0.4, repeat_penalty=1.2) | StrOutputParser()
                 response = chain_retry.invoke({
                    "key": group_key,
                    "context": context
                })
            else:
                 response = chain.invoke({
                    "key": group_key,
                    "context": context
                })
            
            # Validation
            if _is_garbage_output(response):
                # Isolate corrupt sample
                corrupt_dir = "corrupt_samples"
                if not os.path.exists(corrupt_dir):
                    os.makedirs(corrupt_dir)
                
                TIMESTAMP = re.sub(r'[^0-9]', '', str(os.times()[4])) # Simple hash-like timestamp
                corrupt_filename = f"{corrupt_dir}/{group_key}_attempt{attempts}_{TIMESTAMP}.txt"
                with open(corrupt_filename, "w", encoding="utf-8") as f:
                    f.write(f"Key: {group_key}\nAttempt: {attempts}\nReason: Garbage Detected\n\n{response}")
                
                print(f"  [WARN] Garbage detected for {group_key}. Isolated to {corrupt_filename}. Retrying...")
                attempts += 1
                continue
                
            # Parse Output
            summary_text = response
            keywords = ""
            
            if "[Keywords]:" in response:
                parts = response.split("[Keywords]:")
                summary_text = parts[0].replace("[Summary]:", "").strip()
                keywords = parts[1].strip()
            elif "[Summary]:" in response:
                 summary_text = response.replace("[Summary]:", "").strip()
                 
            # Store precise references for recursive retrieval
            child_refs = [
                {"source_file": item.get("source_file"), "chunk_id": item.get("chunk_id")}
                for item in items
            ]
            
            return {
                "level": 1,
                "type": f"L1_{'lang' if group_type == 'vertical' else 'topic'}",
                "group_key": group_key,
                "summary": summary_text,
                "keywords": keywords,
                "child_chunks": child_refs, # New Schema
                "child_files": list(set([item.get("source_file") for item in items])) # Keep for legacy/debug
            }
            
        except Exception as e:
            print(f"Error summarizing {group_key}: {e}")
            attempts += 1
            
    print(f"  [FAIL] Failed to generate valid summary for {group_key} after retries.")
    return None

def summarize_in_batches(llm, group_type, group_key, items, system_prompt, format_prompt) -> Dict:
    """
    Handles large clusters by summarizing in batches and then synthesizing.
    """
    BATCH_SIZE = 10
    batches = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    
    intermediate_summaries = []
    print(f"  > Batch Processing: Split {len(items)} items into {len(batches)} batches.")
    
    # Batch summarization
    batch_prompt_template = """You are a linguistics expert.
Synthesize the key points of these linguistic chunks into a concise intermediate summary.
Focus on extracting factual patterns and features.

Topic: {key}
Context: {context}

Summary:"""
    batch_prompt = ChatPromptTemplate.from_template(batch_prompt_template)
    batch_chain = batch_prompt | llm | StrOutputParser()
    
    for i, batch in enumerate(batches):
        print(f"    - Processing Batch {i+1}/{len(batches)}...")
        ctx = create_context_text(batch)
        try:
            res = batch_chain.invoke({"key": group_key, "context": ctx})
            intermediate_summaries.append(f"[Batch {i+1} Summary]: {res}")
        except Exception as e:
            print(f"      [Error] Batch {i+1} failed: {e}")
            
    # Final Synthesis
    print(f"  > Synthesizing {len(intermediate_summaries)} intermediate summaries...")
    final_context = "\n\n".join(intermediate_summaries)
    
    # Use the original prompts but with the intermediate summaries as context
    synthesis_instruction = "\n[Synthesis Instruction]\nSynthesize the provided intermediate summaries into one coherent, structured L1 node. Do not list them one by one. Merge common patterns."
    
    full_template = system_prompt + synthesis_instruction + format_prompt
    final_prompt = ChatPromptTemplate.from_template(full_template)
    final_chain = final_prompt | llm | StrOutputParser()
    
    try:
        response = final_chain.invoke({
            "key": group_key,
            "context": final_context
        })
        
        # Reuse parsing logic (can extract to helper if needed, but duplicating for safety here)
        summary_text = response
        keywords = ""
        if "[Keywords]:" in response:
            parts = response.split("[Keywords]:")
            summary_text = parts[0].replace("[Summary]:", "").strip()
            keywords = parts[1].strip()
        elif "[Summary]:" in response:
             summary_text = response.replace("[Summary]:", "").strip()

        child_refs = [
            {"source_file": item.get("source_file"), "chunk_id": item.get("chunk_id")}
            for item in items
        ]

        return {
            "level": 1,
            "type": f"L1_{'lang' if group_type == 'vertical' else 'topic'}",
            "group_key": group_key,
            "summary": summary_text,
            "keywords": keywords,
            "child_chunks": child_refs,
            "child_files": list(set([item.get("source_file") for item in items]))
        }
        
    except Exception as e:
        print(f"  [FAIL] Final synthesis failed for {group_key}: {e}")
        return None

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
    bad_keywords = ["Artificial Intelligence", "Future of Work", "Machine Learning", "Job displacement", "일의 미래", "인공지능"]
    for kw in bad_keywords:
        if kw in text and "Linguistics" not in text: 
             return True
             
    return False

def main():
    config = load_config()
    
    # Initialize LLM via Factory
    print(f"Initializing LLM (Ingestion Profile)...")
    llm = get_llm("ingestion")

    # Load L0 Data
    l0_file = os.path.join("data", "raptor", "matrix_raptor_L0_data.jsonl")
    print(f"Loading {l0_file}...")
    l0_data = load_l0_data(l0_file)
    print(f"Loaded {len(l0_data)} chunks.")
    
    # Group Data
    groups = group_data(l0_data)
    print(f"Vertical Groups (Languages): {len(groups['vertical'])}")
    print(f"Horizontal Groups (Topics): {len(groups['horizontal'])}")
    
    # Process Groups
    l1_results = []
    
    # Vertical Processing
    total_vertical = len(groups["vertical"])
    print(f"\nProcessing Vertical Groups (Languages)... Total: {total_vertical}")
    for i, (lang, items) in enumerate(groups["vertical"].items()):
        print(f"[{i+1}/{total_vertical}] Summarizing Vertical: {lang}")
        res = summarize_group(llm, "vertical", lang, items)
        if res:
            l1_results.append(res)
            
    # Horizontal Processing
    total_horizontal = len(groups["horizontal"])
    print(f"\nProcessing Horizontal Groups (Topics)... Total: {total_horizontal}")
    for i, (topic, items) in enumerate(groups["horizontal"].items()):
        print(f"[{i+1}/{total_horizontal}] Summarizing Horizontal: {topic}")
        res = summarize_group(llm, "horizontal", topic, items)
        if res:
            l1_results.append(res)
            
    # Save Output
    output_file = os.path.join("data", "raptor", "matrix_raptor_L1_data.jsonl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"\nSaving {len(l1_results)} Level 1 summaries to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in l1_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print("Done.")

if __name__ == "__main__":
    main()
