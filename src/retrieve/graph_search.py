import os
import sys
import yaml
import logging
import argparse
import re
import difflib
from neo4j import GraphDatabase, exceptions
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global Cache for Entity Names
VALID_LANGUAGES = []

def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "..", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

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

def fetch_valid_entities(driver):
    """
    Fetches valid Language names for Auto-Correction.
    """
    global VALID_LANGUAGES
    query = "MATCH (l:Language) RETURN DISTINCT l.name AS name ORDER BY l.name"
    try:
        with driver.session() as session:
            result = session.run(query)
            VALID_LANGUAGES = [record["name"] for record in result if record["name"]]
            print(f"[Init] Loaded {len(VALID_LANGUAGES)} valid language names for Auto-Correction.")
    except Exception as e:
        logger.warning(f"Failed to fetch entity list: {e}")
        VALID_LANGUAGES = []

def correct_cypher_query(query):
    """
    Analyzing the generated Cypher query and fuzzy-matching entity names against the DB whitelist.
    This fixes LLM hallucinations.
    """
    if not VALID_LANGUAGES or not query:
        return query

    # Regex to find names inside quotes: "Name" or 'Name'
    # We look for names likely to be Languages (simple heuristic)
    matches = re.findall(r'[\"\'](.*?)[\"\']', query)
    
    corrected_query = query
    for name in matches:
        # Skip Cypher keywords or short strings
        if len(name) < 3: continue
        
        # If exact match exists, skip
        if name in VALID_LANGUAGES:
            continue
            
        # Find closest match
        closest = difflib.get_close_matches(name, VALID_LANGUAGES, n=1, cutoff=0.7)
        if closest:
            best_match = closest[0]
            if best_match != name:
                print(f"[Auto-Correct] Replacing '{name}' with '{best_match}'")
                corrected_query = corrected_query.replace(name, best_match)
    
    return corrected_query

def generate_cypher(llm, question):
    # Prompt: Focus on generating a syntacticly correct query.
    # We rely on Python post-processing for strict name matching.
    template = """You are a Neo4j Cypher expert.
Convert the user's question into a Cypher query.

Schema:
- Nodes: (l:Language), (f:Family), (r:Region)
- Relations: [:BELONGS_TO], [:SPOKEN_IN]
- Property: name (string)

**Strategy:**
1. Try to translate the user's input (Korean/Phonetic) to the Standard English Academic name.
2. Use `toLower(n.name) = toLower("English Name")` or `CONTAINS` for best results.
3. **Korean Mapping Tips**:
   - '춰나-먼바' -> 'Chona-Memba' (or 'Cona Monpa')
   - '동어' -> 'Dong'

Question:
{question}

Constraint: Return only the Cypher query string. No markdown.

Cypher Query:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({"question": question})
        cleaned_query = response.strip().replace("```cypher", "").replace("```", "").strip()
        return cleaned_query
    except Exception as e:
        logger.error(f"LLM Cypher generation failed: {e}")
        return None

def execute_query(driver, query):
    if not query:
        return None
    
    # 1. Apply Auto-Correction before execution
    final_query = correct_cypher_query(query)
    
    try:
        with driver.session() as session:
            print(f"[Debug] Executing Cypher: {final_query}")
            result = session.run(final_query)
            return [record.data() for record in result]
    except exceptions.CypherSyntaxError as e:
        logger.error(f"Cypher Syntax Error: {e}")
        return f"Error: Invalid Cypher Query Generated. {e}"
    except Exception as e:
        logger.error(f"Query Execution Error: {e}")
        return f"Error executing query: {e}"

def main():
    parser = argparse.ArgumentParser(description="Neo4j Graph Search")
    parser.add_argument("--query", type=str, help="Search query (natural language)")
    args = parser.parse_args()

    config = load_config()
    
    llm_name = config["llm"]["model_name"]
    base_url = config["llm"]["base_url"]
    
    driver = connect_neo4j(config)
    if not driver:
        print("Could not connect to database. Exiting.")
        sys.exit(1)

    # 1. Fetch Valid Entities for Auto-Correction
    fetch_valid_entities(driver)

    # CLI Mode
    if args.query:
        llm = ChatOllama(model=llm_name, base_url=base_url, temperature=0)
        print(f"Graph Search Query: {args.query}")
        cypher = generate_cypher(llm, args.query)
        if cypher:
            results = execute_query(driver, cypher)
            if isinstance(results, list):
                if not results:
                    print("No results found.")
                else:
                    print(f"\nFound {len(results)} results:")
                    for idx, row in enumerate(results, 1):
                        print(f"{idx}. {row}")
            else:
                print(results)
        else:
            print("Failed to generate query.")
        driver.close()
        return

    # Interactive Mode
    print(f"Initializing LLM ({llm_name}) for Graph Search...")
    llm = ChatOllama(model=llm_name, base_url=base_url, temperature=0)
    
    print("\n" + "="*50)
    print("  Neo4j Knowledge Graph Search (Type 'q' to quit)")
    print("="*50)

    while True:
        try:
            question = input("\nEnter your question: ").strip()
        except EOFError:
            break
            
        if question.lower() in ['q', 'quit', 'exit']:
            print("Exiting...")
            break
        
        if not question:
            continue

        print("Generating Cypher query...")
        cypher = generate_cypher(llm, question)
        
        if cypher:
            results = execute_query(driver, cypher)
            if isinstance(results, list):
                if not results:
                    print("No results found.")
                else:
                    print(f"\nFound {len(results)} results:")
                    for idx, row in enumerate(results, 1):
                        print(f"{idx}. {row}")
            else:
                print(results)
        else:
            print("Failed to generate query.")

    driver.close()

if __name__ == "__main__":
    main()