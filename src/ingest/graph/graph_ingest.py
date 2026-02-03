import os
import sys
import glob
import yaml
import json
import logging
import re
import difflib
from tqdm import tqdm
from bs4 import BeautifulSoup
from neo4j import GraphDatabase
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.utils import load_config

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data Quality Configuration
BLOCKLIST = {'unknown', 'none', 'null', 'n/a', 'language', 'dialect', 'region', 'family'}



def connect_neo4j(config):
    neo4j_conf = config.get("neo4j", {})
    uri = neo4j_conf.get("uri", "bolt://localhost:7687")
    username = neo4j_conf.get("username", "neo4j")
    password = neo4j_conf.get("password", "password")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
        return driver
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        return None

def setup_database(driver):
    """
    Sets up constraints to ensure data uniqueness and prevent duplicates.
    """
    queries = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Language) REQUIRE l.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Family) REQUIRE f.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Region) REQUIRE r.name IS UNIQUE"
    ]
    
    print("Setting up Neo4j constraints...")
    with driver.session() as session:
        for q in queries:
            try:
                session.run(q)
            except Exception as e:
                logger.warning(f"Constraint creation warning: {e}")

def extract_text_from_html(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            # Get text and limit to 2000 chars - Metadata usually at top
            text = soup.get_text(separator=' ', strip=True)
            return text[:2000]
    except Exception as e:
        logger.warning(f"Error reading {file_path}: {e}")
        return ""

def load_valid_languages(data_path):
    """
    Loads strictly allowed language names from the directory structure.
    Returns a set of lower-cased names and a mapping to their canonical form (Title Case).
    """
    doc_dir = os.path.join(data_path, "doc")
    if not os.path.exists(doc_dir):
        return set(), {}
        
    dirs = [d for d in os.listdir(doc_dir) if os.path.isdir(os.path.join(doc_dir, d))]
    valid_set = set(d.lower() for d in dirs)
    canonical_map = {d.lower(): d.title() for d in dirs} # e.g. 'takpa' -> 'Takpa'
    
    # Add manual mapping for special cases if needed (e.g. 'bu-nao' -> 'Bu-Nao')
    # Use original directory casing if possible, but ls returned lowercase usually.
    # We will title case them for the graph: 'takpa' -> 'Takpa'.
    
    return valid_set, canonical_map

def clean_entity_name(name, valid_set=None, canonical_map=None, type="Language"):
    """
    Normalizes entity names.
    For 'Language' type, it enforces STRICT matching against valid_set.
    """
    if not name:
        return None
    
    # Basic Clean
    name = re.sub(r'\s*\(.*?\)\s*', '', name)
    name = name.strip()
    
    if not name:
        return None

    # Check Blocklist
    if name.lower() in BLOCKLIST or len(name) < 2:
        return None

    # Strict Allowlist Enforcement for Languages
    if type == "Language" and valid_set is not None:
        name_lower = name.lower()
        
        # 1. Exact Match
        if name_lower in valid_set:
            return canonical_map[name_lower]
        
        # 2. Fuzzy Match (handle typos or minor variations)
        matches = difflib.get_close_matches(name_lower, valid_set, n=1, cutoff=0.6)
        if matches:
            matched_name = matches[0]
            # print(f"DEBUG: Mapped '{name}' -> '{canonical_map[matched_name]}'")
            return canonical_map[matched_name]
        else:
            # print(f"DEBUG: Rejected '{name}' (Not in allowlist)")
            return None # Reject if not in allowlist
            
    # For Family/Region, just Title Case
    return name.title()

def extract_metadata_with_llm(llm, text):
    # Modified Prompt: Enforce English Standardization to prevent fragmentation
    prompt_template = """You are a Strict Comparative Linguist.
Analyze the text below and extract metadata ONLY for the PRIMARY language discussed.

Rules:
1. SUBJECT FOCUS: Extract only the main language of the document.
2. INCLUSIVITY & SPECIFICITY: Treat dialects, varieties, or vernaculars as valid Language entities if they are the primary subject of the text.
   - Clarification: Extract the specific variety name rather than the umbrella language group.
3. LANGUAGE STANDARDIZATION (CRITICAL): 
   - Output the **Standard English Academic Name** (e.g., Glottolog/ISO 639-3 reference name).
   - **Translate** any non-English names (Korean, Chinese, etc.) into English.
   - Example: If text says '춰나-먼바어', output 'Takpa' or 'Cona Monpa'. If '몽골어', output 'Mongolian'.
4. NEGATIVE CONSTRAINT: Do NOT extract languages mentioned purely for comparison (e.g., ignore "Unlike Mongolian...").
5. PRECISION: If specific info is missing, return null. DO NOT GUESS.

Output Format: Strict JSON only. No markdown.

Fields:
- "language": Target language name (In English)
- "family": Language family (In English)
- "region": Geographic region (In English)

Text:
{text}
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    
    try:
        response = chain.invoke({"text": text})
        content = response.content.strip()
        # Clean up if LLM wraps in markdown
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "")
        
        data = json.loads(content)
        return data
    except Exception as e:
        logger.warning(f"LLM extraction failed: {e}")
        return None

def ingest_to_graph(driver, data, valid_set, canonical_map):
    if not data:
        return 0
    
    # Preprocessing & Cleaning
    # Strict enforcement for Language
    lang = clean_entity_name(data.get("language"), valid_set, canonical_map, type="Language")
    
    # Loose enforcement for Family/Region (no allowlist passed)
    family = clean_entity_name(data.get("family"), type="Other")
    region = clean_entity_name(data.get("region"), type="Other")
    
    # Strict Ingestion Rule: We need at least a valid Language name to proceed
    if not lang:
        return 0

    # Dynamic Cypher Generation based on available data
    # Only merge relations if the target entity is valid (not None/Unknown)
    
    with driver.session() as session:
        try:
            # 1. Merge Language Node (Central Anchor)
            session.run("MERGE (l:Language {name: $lang})", parameters={"lang": lang})
            
            # 2. Merge Family Relationship
            if family:
                session.run("""
                    MATCH (l:Language {name: $lang})
                    MERGE (f:Family {name: $family})
                    MERGE (l)-[:BELONGS_TO]->(f)
                """, parameters={"lang": lang, "family": family})
                
            # 3. Merge Region Relationship
            if region:
                session.run("""
                    MATCH (l:Language {name: $lang})
                    MERGE (r:Region {name: $region})
                    MERGE (l)-[:SPOKEN_IN]->(r)
                """, parameters={"lang": lang, "region": region})
                
            return 1
        except Exception as e:
            logger.error(f"Neo4j ingest error for {lang}: {e}")
            return 0

def main():
    config = load_config()
    data_path = config["project"]["data_path"]
    doc_dir = os.path.join(data_path, "doc")
    
    # Load Valid Languages Allowlist
    print("Loading valid language list from directory structure...")
    valid_set, canonical_map = load_valid_languages(data_path)
    print(f"Loaded {len(valid_set)} valid languages for strict enforcement.")
    
    # Initialize LLM
    llm_cfg = config.get("llm_ingestion", config.get("llm", {}))
    llm_name = llm_cfg.get("model_name")
    base_url = llm_cfg.get("base_url")
    print(f"Initializing LLM: {llm_name}...")
    llm = ChatOllama(model=llm_name, base_url=base_url, temperature=0)

    # Initialize Neo4j
    print("Connecting to Neo4j...")
    driver = connect_neo4j(config)
    if not driver:
        sys.exit(1)
        
    # Setup Data Quality Constraints
    setup_database(driver)

    # Find HTML files
    print(f"Scanning for HTML files in {doc_dir}...")
    html_files = glob.glob(os.path.join(doc_dir, "**/*.html"), recursive=True)
    print(f"Found {len(html_files)} files.")

    processed_count = 0
    
    # Process files
    for file_path in tqdm(html_files, desc="Ingesting to Graph"):
        text = extract_text_from_html(file_path)
        if not text:
            continue
            
        metadata = extract_metadata_with_llm(llm, text)
        if metadata:
            if ingest_to_graph(driver, metadata, valid_set, canonical_map):
                processed_count += 1
    
    driver.close()
    print("="*50)
    print(f"Processing Complete.")
    print(f"Valid Entities Ingested: {processed_count}/{len(html_files)}")
    print("Graph Build Finished.")

if __name__ == "__main__":
    main()
