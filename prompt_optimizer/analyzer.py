"""
Prompt Complexity Analyzer
Classifies prompts by intent, complexity, and expected output characteristics.
Rule-based for speed (no API calls).
"""
from typing import Dict, Any


def classify_intent(prompt: str) -> Dict[str, Any]:
    """Rule-based intent classification (fast, no API calls)"""
    p_lower = prompt.lower()

    # Intent Type
    if any(x in p_lower for x in ["why", "analyze", "explain", "reason", "derive", "justify"]):
        intent = "reasoning"
    elif any(x in p_lower for x in ["summarize", "in short", "briefly", "tldr"]):
        intent = "summarization"
    elif any(x in p_lower for x in ["code", "program", "function", "python", "java", "javascript", "algorithm", "debug"]):
        intent = "coding"
    elif any(x in p_lower for x in ["dataset", "analyze data", "plot", "visualize", "csv", "chart"]):
        intent = "data_analysis"
    elif any(x in p_lower for x in ["story", "poem", "creative", "write about"]):
        intent = "creative_writing"
    elif any(x in p_lower for x in ["who", "what", "when", "where", "fact", "define"]):
        intent = "factual_answering"
    elif any(x in p_lower for x in ["conversation", "chat", "talk", "discuss"]):
        intent = "conversation"
    elif any(x in p_lower for x in ["classify", "category", "type of", "label"]):
        intent = "classification"
    else:
        intent = "general"

    # Complexity Level based on word count and keywords
    word_count = len(p_lower.split())
    has_complex_keywords = any(x in p_lower for x in [
        "step-by-step", "comprehensive", "detailed", "in-depth",
        "multiple", "compare", "contrast", "evaluate"
    ])

    if word_count < 10 and not has_complex_keywords:
        complexity = "low"
    elif word_count < 30 or (word_count < 50 and not has_complex_keywords):
        complexity = "medium"
    else:
        complexity = "high"

    # Expected Output Length
    if any(x in p_lower for x in ["short", "summary", "brief", "quick", "one line"]):
        output_len = "short"
    elif any(x in p_lower for x in ["explain", "detailed", "step", "comprehensive", "in depth"]):
        output_len = "long"
    else:
        output_len = "medium"

    # Latency Tolerance (inverse of expected output)
    latency_map = {"short": "low", "medium": "medium", "long": "high"}
    latency = latency_map[output_len]

    # Compliance Need
    compliance = any(x in p_lower for x in [
        "personal data", "pii", "privacy", "gdpr", "confidential",
        "sensitive", "policy", "terms", "legal"
    ])

    return {
        "intent_type": intent,
        "complexity_level": complexity,
        "expected_output_length": output_len,
        "latency_tolerance": latency,
        "compliance_needed": compliance
    }


def analyze_complexity(prompt: str) -> Dict[str, Any]:
    """
    Analyze prompt complexity.
    Uses rule-based classification (fast, no API calls).
    """
    return classify_intent(prompt)
