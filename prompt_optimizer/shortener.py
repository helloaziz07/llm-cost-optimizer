"""
LLM-based Prompt Shortener
Uses Vertex AI or Gemini API to intelligently shorten prompts while preserving meaning.
Falls back to rule-based shortening in simulation mode.
"""
import os
import logging
from config import config

logger = logging.getLogger(__name__)

# Initialize the appropriate client at module level
_shortener_client = None
_shortener_auth = None

if not config.SIMULATE_SHORTENER:
    if config.AUTH_METHOD == "vertex_ai":
        try:
            from google.cloud import aiplatform
            from vertexai.generative_models import GenerativeModel

            if config.GOOGLE_APPLICATION_CREDENTIALS:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.GOOGLE_APPLICATION_CREDENTIALS

            if config.GCP_PROJECT_ID:
                aiplatform.init(
                    project=config.GCP_PROJECT_ID,
                    location=config.GCP_LOCATION,
                )
                _shortener_client = "vertex_ai"
                _shortener_auth = "vertex_ai"
                logger.info("Vertex AI initialized for prompt shortener")
        except ImportError:
            logger.warning("google-cloud-aiplatform not installed for shortener")
        except Exception as e:
            logger.error(f"Vertex AI shortener init error: {e}")

    elif config.AUTH_METHOD == "api_key":
        try:
            import google.generativeai as genai
            if config.GEMINI_API_KEY:
                genai.configure(api_key=config.GEMINI_API_KEY)
                _shortener_client = genai
                _shortener_auth = "api_key"
                logger.info("Gemini API key initialized for prompt shortener")
        except ImportError:
            logger.warning("google-generativeai not installed for shortener")


def shorten_prompt(text: str) -> str:
    """
    Shorten the prompt using LLM while keeping meaning intact.
    Falls back to returning original text if API is unavailable or simulation mode.
    """
    # Simulation mode — rule-based shortening
    if config.SIMULATE_SHORTENER or _shortener_client is None:
        shortened = text
        verbose_patterns = [
            ("in order to", "to"),
            ("due to the fact that", "because"),
            ("at this point in time", "now"),
            ("in the event that", "if"),
            ("for the purpose of", "for"),
        ]
        for pattern, replacement in verbose_patterns:
            shortened = shortened.replace(pattern, replacement)
        return shortened

    try:
        shortening_prompt = f"""
        Shorten this prompt without changing meaning or context.
        Remove redundant words, politeness, and filler phrases.
        Keep the structure clear and concise.
        Original prompt: {text}
        Return only the shortened version.
        """

        if _shortener_auth == "vertex_ai":
            from vertexai.generative_models import GenerativeModel
            model = GenerativeModel(config.LLM_MODEL)
            response = model.generate_content(shortening_prompt)
            return response.text.strip()
        else:
            model = _shortener_client.GenerativeModel(config.LLM_MODEL)
            response = model.generate_content(shortening_prompt)
            return response.text.strip()

    except Exception as e:
        logger.error(f"LLM shortening error: {e}")
        return text  # fallback to original
