"""Grounding validator - deterministic refusal-correctness check.

PREFILLED. Checks the other half of grounding that FaithfulnessChecker
does not: not "is the answer's content hallucinated", but "did the system
correctly recognize whether this question is answerable at all" - refusing
an out-of-scope question, and not refusing one it should answer.
"""

from app.utils.text import is_refusal


class GroundingValidator:
    """Checks that refusal behavior matches a question's answerability."""

    def validate(self, category: str, retrieved_contexts: list[str], answer: str) -> bool:
        """Check that an out-of-scope question was refused, and an
        in-scope one was not.

        Args:
            category: The EvaluationCase's category (e.g. "out_of_scope").
            retrieved_contexts: Contexts the mock RAG client retrieved.
            answer: The mock RAG client's answer text.

        Returns:
            bool: True if the refusal/answer decision matches the
                question's category.
        """
        refused = (not retrieved_contexts) or is_refusal(answer)
        if category == "out_of_scope":
            return refused
        return not refused
