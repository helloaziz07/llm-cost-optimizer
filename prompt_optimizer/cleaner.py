"""
Structural Prompt Cleaner
Removes filler words, excessive whitespace, and normalizes text.
"""
import re


def clean_prompt(prompt: str) -> str:
    """
    Perform basic structural cleaning of the prompt:
    - Remove excessive spaces, line breaks, filler words
    - Normalize punctuation
    - Keep content meaning intact
    """
    # Remove extra whitespace and newlines
    prompt = re.sub(r'\s+', ' ', prompt).strip()

    # Remove filler or redundant phrases
    fillers = [
        "please", "kindly", "can you", "could you",
        "I want to know", "tell me", "I would like to",
        "I need you to", "would you mind"
    ]
    for f in fillers:
        prompt = re.sub(r'\b' + re.escape(f) + r'\b', '', prompt, flags=re.IGNORECASE)

    # Remove duplicate spaces again
    prompt = re.sub(r'\s{2,}', ' ', prompt)

    return prompt.strip()
