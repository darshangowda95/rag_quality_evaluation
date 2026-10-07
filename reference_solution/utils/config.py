"""Configuration management for Lab 2.4."""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration for RAG evaluation."""

    # Paths
    BASE_DIR = Path(__file__).parent.parent.parent
    EVAL_DATASET_PATH = os.getenv("EVAL_DATASET_PATH", "dataset/evaluation_questions.json")
    GROUND_TRUTH_PATH = os.getenv("GROUND_TRUTH_PATH", "dataset/ground_truth.json")
    PROMPT_TEMPLATES_PATH = os.getenv("PROMPT_TEMPLATES_PATH", "dataset/prompt_templates.json")
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "dataset/fixtures/seed_store")
    REPORTS_DIR = os.getenv("REPORTS_DIR", "reports")

    # LLM API
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    # Mock RAG defaults
    DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "512"))
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
    DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.3"))

    # Optimization
    CHUNK_SIZES: List[int] = [int(x) for x in os.getenv("CHUNK_SIZES", "256,512,1024").split(",")]
    TOP_K_VALUES: List[int] = [int(x) for x in os.getenv("TOP_K_VALUES", "3,5,10").split(",")]

    # Live LLM call budget (mock RAG client's answer generation - no offline
    # fallback; see integration/rag_client.py)
    GENERATION = {
        "max_retries": 2,
        "retry_backoff_seconds": 2,
        "llm_timeout_seconds": 30,
        "max_tokens_per_response": 400,
    }

    @classmethod
    def has_llm_api(cls) -> bool:
        """Check if a live OpenRouter key is configured."""
        return bool(cls.OPENROUTER_API_KEY.strip())

    @classmethod
    def get_absolute_path(cls, relative_path: str) -> Path:
        """Convert relative path to absolute."""
        return cls.BASE_DIR / relative_path
