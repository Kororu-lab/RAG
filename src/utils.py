import requests
import json
import os
import yaml

def load_config():
    """
    Loads configuration from config.yaml by searching up from the current script.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    while True:
        config_path = os.path.join(current_dir, "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        
        parent = os.path.dirname(current_dir)
        if parent == current_dir: # Reached root
            raise FileNotFoundError("config.yaml not found in directory tree.")
        current_dir = parent


class LLMUtility:
    @staticmethod
    def unload_model(profile: str = None):
        """
        Unloads model if provider is Ollama.
        profile: 'ingestion' or 'retrieval'. If None, tries keys in order.
        """
        config = load_config()
        
        # Determine target section
        target_cfg = {}
        if profile:
            section = f"llm_{profile}"
            target_cfg = config.get(section, {})
        
        # Fallback/Auto-detect if no profile or empty config found
        if not target_cfg:
             # Try ingestion first, then retrieval, then legacy
             target_cfg = config.get("llm_ingestion", config.get("llm_retrieval", config.get("llm", {})))
             
        provider = target_cfg.get("provider", "ollama").lower()
        model_name = target_cfg.get("model_name")
        base_url = target_cfg.get("base_url", "http://localhost:11434")

        # Logic: Only Ollama needs explicit unload to free VRAM
        if provider == "ollama" and model_name:
            if not base_url: base_url = "http://localhost:11434"
            
            url = f"{base_url}/api/generate"
            payload = {
                "model": model_name,
                "keep_alive": 0
            }
            try:
                response = requests.post(url, json=payload)
                if response.status_code == 200:
                    print(f"[LLMUtility] Successfully unloaded Ollama model: {model_name}")
                else:
                    print(f"[LLMUtility] Failed to unload model {model_name}: {response.text}")
            except Exception as e:
                print(f"[LLMUtility] Error unloading model: {e}")
        else:
            # API providers (OpenAI, etc.) do not hold VRAM state locally
            pass
