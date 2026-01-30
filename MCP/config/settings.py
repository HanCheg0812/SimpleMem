"""
Settings configuration for SimpleMem MCP Server
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from functools import lru_cache


@dataclass
class Settings:
    """Application settings"""

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # JWT Configuration
    jwt_secret_key: str = field(default_factory=lambda: os.getenv(
        "JWT_SECRET_KEY",
        "simplemem-secret-key-change-in-production"
    ))
    jwt_algorithm: str = "HS256"
    jwt_expiration_days: int = 30

    # Encryption for API Keys
    encryption_key: str = field(default_factory=lambda: os.getenv(
        "ENCRYPTION_KEY",
        "simplemem-encryption-key-32bytes!"  # Must be 32 bytes for AES-256
    ))

    # Database Paths
    data_dir: str = field(default_factory=lambda: os.getenv(
        "DATA_DIR",
        "./data"
    ))
    lancedb_path: str = field(default_factory=lambda: os.getenv(
        "LANCEDB_PATH",
        "./data/lancedb"
    ))
    user_db_path: str = field(default_factory=lambda: os.getenv(
        "USER_DB_PATH",
        "./data/users.db"
    ))

    # LLM API Configuration (OpenAI-compatible: OpenRouter, DashScope, etc.)
    openrouter_base_url: str = field(default_factory=lambda: os.getenv(
        "API_BASE_URL",
        "https://openrouter.ai/api/v1"
    ))
    llm_model: str = field(default_factory=lambda: os.getenv(
        "LLM_MODEL",
        "openai/gpt-4.1-mini"
    ))
    embedding_model: str = field(default_factory=lambda: os.getenv(
        "EMBEDDING_MODEL",
        "text-embedding-v3"
    ))
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

    # Memory Building Configuration
    window_size: int = 20
    overlap_size: int = 2

    # Retrieval Configuration
    semantic_top_k: int = 25
    keyword_top_k: int = 5
    enable_planning: bool = True
    enable_reflection: bool = True
    max_reflection_rounds: int = 2

    # LLM Configuration
    llm_temperature: float = 0.1
    llm_max_retries: int = 3
    use_streaming: bool = True

    def __post_init__(self):
        """Ensure directories exist"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.lancedb_path, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
