"""
Token Counter
Counts tokens for various models using tiktoken.
"""
import tiktoken


def count_tokens(prompt: str, model_name: str = "gpt-4o") -> int:
    """
    Count tokens in a prompt using tiktoken.
    Defaults to GPT-4o tokenization as a reasonable approximation.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(prompt))
    except Exception:
        # Fallback: rough approximation (1 token ≈ 4 characters)
        return len(prompt) // 4 + 1
