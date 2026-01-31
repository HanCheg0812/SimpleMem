"""
Embedding utilities - Generate vector embeddings using SentenceTransformers or API
Supports both local models and OpenAI-compatible embedding APIs
"""
from typing import List, Optional
import numpy as np
import config


class EmbeddingModel:
    """
    Embedding model supporting both local (SentenceTransformers) and API modes
    """
    def __init__(self, model_name: Optional[str] = None, use_optimization: bool = True):
        self.use_optimization = use_optimization

        # Determine provider mode
        provider = getattr(config, "EMBEDDING_PROVIDER", "local")

        if provider == "api":
            self._init_api_embedding()
        else:
            self.model_name = model_name or config.EMBEDDING_MODEL
            print(f"Loading embedding model: {self.model_name}")
            if self.model_name.lower().startswith("qwen"):
                self._init_qwen3_sentence_transformer()
            else:
                self._init_standard_sentence_transformer()

    def _init_api_embedding(self) -> None:
        """Initialize API-based embedding"""
        try:
            from openai import OpenAI

            api_key = getattr(config, "EMBEDDING_API_KEY", None) or config.OPENAI_API_KEY
            api_base = getattr(config, "EMBEDDING_API_BASE", "https://openrouter.ai/api/v1")
            self.api_model = getattr(config, "EMBEDDING_API_MODEL", "qwen/qwen3-embedding-8b")

            self.client = OpenAI(api_key=api_key, base_url=api_base)
            self.dimension = getattr(config, "EMBEDDING_DIMENSION", 4096)
            self.model_type = "api"
            self.model_name = self.api_model
            self.supports_query_prompt = False

            print(f"API Embedding initialized: {self.api_model}")
            print(f"  Base URL: {api_base}")
            print(f"  Dimension: {self.dimension}")

        except ImportError:
            print("OpenAI package not installed. Run: pip install openai")
            raise
        except Exception as e:
            print(f"Failed to initialize API embedding: {e}")
            raise

    def _init_qwen3_sentence_transformer(self) -> None:
        """Initialize Qwen3 model using SentenceTransformers"""
        try:
            from sentence_transformers import SentenceTransformer

            qwen3_models = {
                "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
                "qwen3-4b": "Qwen/Qwen3-Embedding-4B",
                "qwen3-8b": "Qwen/Qwen3-Embedding-8B"
            }

            model_path = qwen3_models.get(self.model_name.lower(), self.model_name)
            print(f"Loading Qwen3 model via SentenceTransformers: {model_path}")

            if self.use_optimization:
                try:
                    self.model = SentenceTransformer(
                        model_path,
                        model_kwargs={
                            "attn_implementation": "flash_attention_2",
                            "device_map": "auto"
                        },
                        tokenizer_kwargs={"padding_side": "left"},
                        trust_remote_code=True
                    )
                    print("Qwen3 loaded with flash_attention_2 optimization")
                except Exception as e:
                    print(f"Flash attention failed ({e}), using standard loading...")
                    self.model = SentenceTransformer(model_path, trust_remote_code=True)
            else:
                self.model = SentenceTransformer(model_path, trust_remote_code=True)

            self.dimension = self.model.get_sentence_embedding_dimension()
            self.model_type = "qwen3_sentence_transformer"
            self.supports_query_prompt = hasattr(self.model, "prompts") and "query" in getattr(self.model, "prompts", {})

            print(f"Qwen3 model loaded successfully with dimension: {self.dimension}")
            if self.supports_query_prompt:
                print("Query prompt support detected")

        except Exception as e:
            print(f"Failed to load Qwen3 model: {e}")
            print("Falling back to default SentenceTransformers model...")
            self._fallback_to_sentence_transformer()

    def _init_standard_sentence_transformer(self) -> None:
        """Initialize standard SentenceTransformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.model_type = "sentence_transformer"
            self.supports_query_prompt = False
            print(f"SentenceTransformer model loaded with dimension: {self.dimension}")
        except Exception as e:
            print(f"Failed to load SentenceTransformer model: {e}")
            raise

    def _fallback_to_sentence_transformer(self) -> None:
        """Fallback to default SentenceTransformer model"""
        fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Using fallback model: {fallback_model}")
        self.model_name = fallback_model
        self._init_standard_sentence_transformer()

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Encode list of texts to vectors

        Args:
            texts: List of texts to encode
            is_query: Whether these are query texts (for Qwen3 prompt optimization)

        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        if self.model_type == "api":
            return self._encode_api(texts)
        elif self.model_type == "qwen3_sentence_transformer" and self.supports_query_prompt and is_query:
            return self._encode_with_query_prompt(texts)
        else:
            return self._encode_standard(texts)

    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        """Encode single text"""
        return self.encode([text], is_query=is_query)[0]

    def encode_query(self, queries: List[str]) -> np.ndarray:
        """Encode queries with optimal settings"""
        return self.encode(queries, is_query=True)

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """Encode documents (no query prompt)"""
        return self.encode(documents, is_query=False)

    def _encode_api(self, texts: List[str]) -> np.ndarray:
        """Encode texts using OpenAI-compatible API"""
        try:
            # Handle dimension parameter for models that support it
            kwargs = {"model": self.api_model, "input": texts}

            # Some providers support dimensions parameter
            if self.dimension and self.dimension != 4096:
                kwargs["dimensions"] = self.dimension

            response = self.client.embeddings.create(**kwargs)
            embeddings = [item.embedding for item in response.data]

            # Normalize embeddings
            embeddings = np.array(embeddings)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-10)

            return embeddings

        except Exception as e:
            print(f"API embedding failed: {e}")
            raise

    def _encode_with_query_prompt(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Qwen3 query prompt"""
        try:
            embeddings = self.model.encode(
                texts,
                prompt_name="query",
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return embeddings
        except Exception as e:
            print(f"Query prompt encoding failed: {e}, falling back to standard encoding")
            return self._encode_standard(texts)

    def _encode_standard(self, texts: List[str]) -> np.ndarray:
        """Encode texts using standard method"""
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings
