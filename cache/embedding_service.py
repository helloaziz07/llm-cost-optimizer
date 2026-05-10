"""
Embedding Service for Semantic Cache
Supports Vertex AI (service account JSON), Gemini API key, and simulation mode.
"""
import os
import numpy as np
import hashlib
import logging
from typing import List

from config import config

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Handles text embedding using Vertex AI, Gemini API, or simulation"""

    def __init__(self):
        self.dimension = config.EMBEDDING_DIM
        self._client = None
        self._auth_method = None

        # Only initialize if NOT simulating
        if not config.SIMULATE_EMBEDDINGS:
            self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate embedding client based on auth method"""

        if config.AUTH_METHOD == "vertex_ai":
            self._init_vertex_ai()
        elif config.AUTH_METHOD == "api_key":
            self._init_gemini_api_key()
        else:
            logger.warning(f"Unknown AUTH_METHOD: {config.AUTH_METHOD}, using simulation")

    def _init_vertex_ai(self):
        """Initialize Vertex AI for embeddings"""
        try:
            from google.cloud import aiplatform

            # Set credentials path if provided
            if config.GOOGLE_APPLICATION_CREDENTIALS:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.GOOGLE_APPLICATION_CREDENTIALS

            if not config.GCP_PROJECT_ID:
                logger.warning("GCP_PROJECT_ID not set, falling back to simulation for embeddings")
                return

            aiplatform.init(
                project=config.GCP_PROJECT_ID,
                location=config.GCP_LOCATION,
            )

            self._client = "vertex_ai"
            self._auth_method = "vertex_ai"
            self.model = config.EMBEDDING_MODEL
            logger.info(f"Vertex AI embeddings initialized (model={self.model})")

        except ImportError:
            logger.warning("google-cloud-aiplatform not installed for embeddings, using simulation. "
                           "Install with: pip install google-cloud-aiplatform")
        except Exception as e:
            logger.error(f"Vertex AI embeddings init error: {e}")

    def _init_gemini_api_key(self):
        """Initialize with legacy Gemini API key for embeddings"""
        try:
            import google.generativeai as genai
            if config.GEMINI_API_KEY:
                genai.configure(api_key=config.GEMINI_API_KEY)
                self._client = genai
                self._auth_method = "api_key"
                self.model = config.EMBEDDING_MODEL
                logger.info("Gemini embeddings initialized with API key")
            else:
                logger.warning("GEMINI_API_KEY not set, using simulation for embeddings")
        except ImportError:
            logger.warning("google-generativeai not installed for embeddings, using simulation")

    def normalize_text(self, text: str) -> str:
        """Normalize input text for consistent embeddings"""
        return " ".join(text.lower().strip().split())

    def _simulate_embedding(self, text: str) -> np.ndarray:
        """
        Generate a deterministic simulated embedding based on text hash.
        Same text always produces same embedding for cache matching.
        """
        normalized = self.normalize_text(text)

        # Create a deterministic seed from the text
        text_hash = hashlib.md5(normalized.encode()).hexdigest()
        seed = int(text_hash[:8], 16)

        # Generate deterministic random embedding
        rng = np.random.RandomState(seed)
        embedding = rng.randn(self.dimension).astype(np.float32)

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    async def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""

        # Always simulate if flag is set or no client configured
        if config.SIMULATE_EMBEDDINGS or self._client is None:
            return self._simulate_embedding(query)

        if self._auth_method == "vertex_ai":
            return await self._vertex_ai_embed(query)
        else:
            return await self._gemini_api_key_embed(query)

    async def _vertex_ai_embed(self, query: str) -> np.ndarray:
        """Generate embedding using Vertex AI"""
        try:
            from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

            normalized_query = self.normalize_text(query)
            model = TextEmbeddingModel.from_pretrained(self.model)

            embeddings = model.get_embeddings(
                [TextEmbeddingInput(normalized_query, "RETRIEVAL_QUERY")]
            )

            embedding = np.array(embeddings[0].values, dtype=np.float32)

            # L2 normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            logger.error(f"Vertex AI embedding error, using simulation: {e}")
            return self._simulate_embedding(query)

    async def _gemini_api_key_embed(self, query: str) -> np.ndarray:
        """Generate embedding using Gemini API key (legacy)"""
        try:
            normalized_query = self.normalize_text(query)

            # Ensure model name has proper prefix for Gemini API
            model_name = self.model
            if not model_name.startswith("models/"):
                model_name = f"models/{model_name}"

            result = self._client.embed_content(
                model=model_name,
                content=normalized_query,
                task_type="retrieval_query"
            )
            embedding = np.array(result['embedding'], dtype=np.float32)

            # L2 normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding
        except Exception as e:
            # Fallback to simulation on any error
            logger.error(f"Gemini embedding API error, using simulation: {e}")
            return self._simulate_embedding(query)

    async def embed_queries(self, queries: List[str]) -> np.ndarray:
        """Generate embeddings for multiple queries"""
        embeddings = []
        for query in queries:
            emb = await self.embed_query(query)
            embeddings.append(emb)
        return np.array(embeddings, dtype=np.float32)

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        return float(np.dot(embedding1, embedding2))
