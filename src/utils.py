import requests
import json
import os
import yaml
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy


_CONFIG_OVERRIDE: ContextVar[dict | None] = ContextVar("_CONFIG_OVERRIDE", default=None)


@contextmanager
def use_config_override(cfg: dict | None):
    """Temporarily override load_config() within the current context."""
    token = _CONFIG_OVERRIDE.set(deepcopy(cfg) if cfg is not None else None)
    try:
        yield
    finally:
        _CONFIG_OVERRIDE.reset(token)


def _safe_import_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


def resolve_torch_device(configured_device: str | None = "auto") -> str:
    requested = (configured_device or "auto").lower()
    torch = _safe_import_torch()

    cuda_available = False
    mps_available = False

    if torch is not None:
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False

        try:
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is not None:
                mps_available = bool(mps_backend.is_available())
        except Exception:
            mps_available = False

    if requested == "cuda":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"

    if requested == "mps":
        if mps_available:
            return "mps"
        if cuda_available:
            return "cuda"
        return "cpu"

    if requested == "cpu":
        return "cpu"

    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def resolve_vision_device_and_dtype(config: dict) -> tuple[str, "torch.dtype"]:
    torch = _safe_import_torch()
    if torch is None:
        raise RuntimeError("torch is required for vision device and dtype resolution.")

    vision_cfg = config.get("vision", {}) or {}
    embedding_cfg = config.get("embedding", {}) or {}

    requested = vision_cfg.get("device") or embedding_cfg.get("device") or "auto"
    resolved = resolve_torch_device(requested)

    if resolved == "mps" and vision_cfg.get("mps_fallback_to_cpu", True):
        return "cpu", torch.float32
    if resolved == "cuda":
        return "cuda", torch.bfloat16
    if resolved == "mps":
        return "mps", torch.float16
    return "cpu", torch.float32


def resolve_vision_store_dtype(config: dict) -> "torch.dtype":
    torch = _safe_import_torch()
    if torch is None:
        raise RuntimeError("torch is required for vision dtype resolution.")

    vision_cfg = config.get("vision", {}) or {}
    requested = str(vision_cfg.get("ingest_store_dtype", "float32")).lower()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(requested, torch.float32)


def clear_torch_cache() -> None:
    torch = _safe_import_torch()
    if torch is None:
        return

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return
    except Exception:
        pass

    try:
        mps_backend = getattr(torch.backends, "mps", None)
        mps_module = getattr(torch, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            if mps_module is not None and hasattr(mps_module, "empty_cache"):
                mps_module.empty_cache()
    except Exception:
        pass


def load_config():
    """
    Loads configuration from config.yaml by searching up from the current script.
    """
    override = _CONFIG_OVERRIDE.get()
    if override is not None:
        return deepcopy(override)

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
                response = requests.post(url, json=payload, timeout=(2, 5))
                if response.status_code == 200:
                    print(f"[LLMUtility] Successfully unloaded Ollama model: {model_name}")
                else:
                    print(f"[LLMUtility] Failed to unload model {model_name}: {response.text}")
            except requests.Timeout as e:
                print(f"[LLMUtility] Timeout unloading model {model_name}: {e}")
            except Exception as e:
                print(f"[LLMUtility] Error unloading model: {e}")
        else:
            # API providers (OpenAI, etc.) do not hold VRAM state locally
            pass
