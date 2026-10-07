"""Shared text-matching helpers for the deterministic evaluation metrics.

PREFILLED. The mock RAG client's generation has no offline fallback (see
integration/rag_client.py) - a refusal now comes back in the live model's
own words, not always the exact fixed string the hardcoded "no contexts
retrieved" path uses. `is_refusal()` recognizes both.
"""

REFUSAL_TEXT = "I cannot answer this question based on the available policy documents."

_REFUSAL_PHRASES = (
    "cannot answer",
    "cannot provide",
    "cannot find",
    "do not contain",
    "does not contain",
    "does not include",
    "do not include",
    "no information",
    "not available in",
    "unable to answer",
    "unable to provide",
    "i don't have",
    "i do not have",
)


def is_refusal(answer: str) -> bool:
    """True if `answer` is a refusal - the exact fixed string, or the live
    model declining in its own words.

    Args:
        answer: The generated answer text.

    Returns:
        bool: True if the answer is empty or reads as a refusal.
    """
    if not answer or not answer.strip():
        return True
    lowered = answer.lower()
    return any(phrase in lowered for phrase in _REFUSAL_PHRASES)
