"""
LLM Service — Unified LLM caller with simulation and real API support.
Supports Vertex AI (service account JSON) and legacy Gemini API key auth.
"""
import os
import time
import random
import logging
from typing import Optional, Tuple, Dict, Any

from config import config
from batching.model_catalog import get_model_info, get_model_cost

logger = logging.getLogger(__name__)


class LLMService:
    """
    Handles all LLM API calls.
    - Simulation mode: returns realistic template responses (no API needed)
    - Vertex AI mode: uses service account JSON for authentication
    - API Key mode: uses Gemini API key (legacy)
    """

    def __init__(self):
        self._client = None
        self._auth_method = None

        # Only initialize if not in simulation mode
        if not config.SIMULATE_LLM:
            self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate LLM client based on auth method"""

        if config.AUTH_METHOD == "vertex_ai":
            self._init_vertex_ai()
        elif config.AUTH_METHOD == "api_key":
            self._init_gemini_api_key()
        else:
            logger.warning(f"Unknown AUTH_METHOD: {config.AUTH_METHOD}, falling back to simulation")

    def _init_vertex_ai(self):
        """Initialize Vertex AI client with service account JSON"""
        try:
            from google.cloud import aiplatform
            from vertexai.generative_models import GenerativeModel

            # Set credentials path if provided
            if config.GOOGLE_APPLICATION_CREDENTIALS:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.GOOGLE_APPLICATION_CREDENTIALS

            if not config.GCP_PROJECT_ID:
                logger.warning("GCP_PROJECT_ID not set, falling back to simulation")
                return

            # Initialize Vertex AI
            aiplatform.init(
                project=config.GCP_PROJECT_ID,
                location=config.GCP_LOCATION,
            )

            self._client = "vertex_ai"
            self._auth_method = "vertex_ai"
            logger.info(f"Vertex AI initialized (project={config.GCP_PROJECT_ID}, location={config.GCP_LOCATION})")

        except ImportError:
            logger.warning("google-cloud-aiplatform not installed, falling back to simulation. "
                           "Install with: pip install google-cloud-aiplatform")
        except Exception as e:
            logger.error(f"Vertex AI init error: {e}")

    def _init_gemini_api_key(self):
        """Initialize with legacy Gemini API key"""
        try:
            import google.generativeai as genai
            if config.GEMINI_API_KEY:
                genai.configure(api_key=config.GEMINI_API_KEY)
                self._client = genai
                self._auth_method = "api_key"
                logger.info("Gemini API initialized with API key")
            else:
                logger.warning("GEMINI_API_KEY not set, falling back to simulation")
        except ImportError:
            logger.warning("google-generativeai not installed, using simulation")

    async def generate_response(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate LLM response. Returns dict with response text and metrics.

        Returns:
            {
                "response": str,
                "input_tokens": int,
                "output_tokens": int,
                "cost": float,
                "latency_ms": float,
                "model_used": str,
                "simulated": bool
            }
        """
        model_name = model_name or config.LLM_MODEL

        # Simulation mode
        if config.SIMULATE_LLM or self._client is None:
            return self._simulate_response(prompt, model_name)

        # Real API call based on auth method
        if self._auth_method == "vertex_ai":
            return await self._vertex_ai_response(prompt, model_name, max_tokens, temperature)
        else:
            return await self._gemini_api_key_response(prompt, model_name, max_tokens, temperature)

    def _simulate_response(self, prompt: str, model_name: str) -> Dict[str, Any]:
        """Generate a simulated response with realistic metrics"""
        start = time.time()

        # Generate realistic simulated response based on intent
        prompt_lower = prompt.lower()

        if any(x in prompt_lower for x in ["code", "function", "python", "program"]):
            response = (
                "Here's a well-structured implementation:\n\n"
                "```python\ndef solution(data):\n"
                "    # Process input data\n"
                "    result = []\n"
                "    for item in data:\n"
                "        processed = transform(item)\n"
                "        result.append(processed)\n"
                "    return result\n```\n\n"
                "This implementation handles edge cases and follows best practices."
            )
        elif any(x in prompt_lower for x in ["explain", "what is", "how does"]):
            response = (
                "This concept involves several key aspects:\n\n"
                "1. **Core Principle**: The fundamental mechanism operates through "
                "a systematic process of analysis and synthesis.\n\n"
                "2. **Key Components**: Multiple interconnected elements work together "
                "to achieve the desired outcome.\n\n"
                "3. **Practical Application**: In real-world scenarios, this translates "
                "to improved efficiency and better results.\n\n"
                "The key takeaway is that understanding the underlying principles "
                "enables more effective application."
            )
        elif any(x in prompt_lower for x in ["summarize", "brief", "short"]):
            response = (
                "In summary: The core idea centers on optimization and efficiency. "
                "Key points include systematic analysis, practical methodology, "
                "and measurable improvement in outcomes."
            )
        else:
            response = (
                "Based on the analysis of your query, here is a comprehensive response:\n\n"
                "The subject involves multiple dimensions that need consideration. "
                "Through careful evaluation of the available information, we can identify "
                "key patterns and insights. The recommended approach combines analytical "
                "reasoning with practical considerations to achieve optimal results.\n\n"
                "Key recommendations:\n"
                "- Apply structured methodology\n"
                "- Consider edge cases\n"
                "- Iterate based on feedback"
            )

        latency_ms = (time.time() - start) * 1000 + random.uniform(50, 200)

        # Realistic token estimation
        input_tokens = len(prompt.split()) * 2
        output_tokens = len(response.split()) * 2

        # Calculate cost using catalog pricing
        cost = get_model_cost(model_name, input_tokens, output_tokens)
        if cost is None:
            cost = config.calculate_cost(input_tokens, output_tokens)

        return {
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "latency_ms": round(latency_ms, 2),
            "model_used": model_name,
            "simulated": True
        }

    async def _vertex_ai_response(
        self,
        prompt: str,
        model_name: str,
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Call Vertex AI Gemini API with service account authentication"""
        start_time = time.time()

        try:
            from vertexai.generative_models import GenerativeModel, GenerationConfig

            model = GenerativeModel(model_name)

            response = model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                )
            )

            latency_sec = time.time() - start_time
            latency_ms = latency_sec * 1000

            # Extract usage metadata
            usage = response.usage_metadata if hasattr(response, 'usage_metadata') else None

            input_tokens = usage.prompt_token_count if usage else len(prompt.split()) * 2
            output_tokens = usage.candidates_token_count if usage else 0

            # Get response text
            response_text = response.text if hasattr(response, 'text') else str(response)
            if not output_tokens:
                output_tokens = len(response_text.split()) * 2

            # Calculate cost
            cost = get_model_cost(model_name, input_tokens, output_tokens)
            if cost is None:
                cost = config.calculate_cost(input_tokens, output_tokens)

            return {
                "response": response_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "latency_ms": round(latency_ms, 2),
                "model_used": model_name,
                "simulated": False
            }

        except Exception as e:
            logger.error(f"Vertex AI LLM error: {e}")
            logger.info("Falling back to simulation mode")
            result = self._simulate_response(prompt, model_name)
            result["error"] = str(e)
            result["simulated"] = True
            return result

    async def _gemini_api_key_response(
        self,
        prompt: str,
        model_name: str,
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Call real Gemini API with API key (legacy method)"""
        start_time = time.time()

        try:
            # Ensure model name has proper prefix for Gemini
            gemini_model_name = model_name
            if not model_name.startswith("models/"):
                gemini_model_name = f"models/{model_name}"

            model = self._client.GenerativeModel(gemini_model_name)

            # Generate response
            response = model.generate_content(
                prompt,
                generation_config=self._client.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                )
            )

            latency_sec = time.time() - start_time
            latency_ms = latency_sec * 1000

            # Extract usage metadata
            usage = response.usage_metadata if hasattr(response, 'usage_metadata') else None

            input_tokens = usage.prompt_token_count if usage else len(prompt.split()) * 2
            output_tokens = usage.candidates_token_count if usage else 0

            # Get response text
            response_text = response.text if hasattr(response, 'text') else str(response)
            if not output_tokens:
                output_tokens = len(response_text.split()) * 2

            # Calculate cost
            cost = get_model_cost(model_name, input_tokens, output_tokens)
            if cost is None:
                cost = config.calculate_cost(input_tokens, output_tokens)

            return {
                "response": response_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "latency_ms": round(latency_ms, 2),
                "model_used": model_name,
                "simulated": False
            }

        except Exception as e:
            logger.error(f"LLM API error: {e}")
            # Fallback to simulation on error
            logger.info("Falling back to simulation mode")
            result = self._simulate_response(prompt, model_name)
            result["error"] = str(e)
            result["simulated"] = True
            return result
