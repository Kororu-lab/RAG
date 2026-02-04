import os
from langchain_community.chat_models import ChatOllama
# Optional imports for flexible migration
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None
try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

from src.utils import load_config

def get_llm(profile: str = "retrieval"):
    """
    Factory function to get LLM instance based on profile ('ingestion' or 'retrieval').
    """
    config = load_config()
    
    # Select config section
    section_key = f"llm_{profile}"
    if section_key not in config:
        # Fallback for backward compatibility or error
        print(f"[WARN] Config section '{section_key}' not found. Defaulting to 'llm_ingestion' or 'llm'.")
        if "llm_ingestion" in config:
            cfg = config["llm_ingestion"]
        else:
            cfg = config.get("llm", {}) # Legacy fallback
    else:
        cfg = config[section_key]

    provider = cfg.get("provider", "ollama").lower()
    model_name = cfg.get("model_name")
    temperature = cfg.get("temperature", 0.1) # 0.1, 필요 부분에서 0으로 override
    base_url = cfg.get("base_url")
    
    # 1. Ollama
    if provider == "ollama":
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            num_ctx=cfg.get("num_ctx", 4096),
            keep_alive="30m" # Keep alive for performance
        )
    
    # 2. OpenAI / DeepSeek
    elif provider in ["openai", "deepseek"]:
        if not ChatOpenAI:
            raise ImportError("langchain_openai not installed. Run `pip install langchain-openai`.")
        
        api_key_conf = cfg.get("api_key_env", "OPENAI_API_KEY")
        
        # Logic: Try treating as Env Var first. If None, check if it looks like a key.
        api_key = os.getenv(api_key_conf)
        if not api_key:
             # If env lookup failed, maybe the user put the key directly?
             if api_key_conf and api_key_conf.startswith("sk-"):
                 api_key = api_key_conf
        
        if not api_key:
             raise ValueError(f"API Key not found. Checked env var '{api_key_conf}' and direct value.")

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key
        )

    # 3. Anthropic
    elif provider == "anthropic":
        if not ChatAnthropic:
            raise ImportError("langchain_anthropic not installed. Run `pip install langchain-anthropic`.")
        
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
