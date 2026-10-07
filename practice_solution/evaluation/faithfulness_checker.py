"""Faithfulness checker - R3 implementation.

Detects whether answers are grounded in retrieved context (no hallucinations).
"""

from typing import Tuple
from app.utils.llm_judge import LLMJudge
from app.utils.config import Config
from app.utils.text import is_refusal


class FaithfulnessChecker:
    """Checks if answers are faithful to retrieved context."""

    def __init__(self, use_llm: bool = False):
        """Initialize faithfulness checker.

        Args:
            use_llm: Use LLM-as-judge if True, else deterministic
        """
        self.use_llm = use_llm and Config.has_llm_api()
        self.llm_judge = None
        if self.use_llm:
            self.llm_judge = LLMJudge(Config.OPENROUTER_API_KEY, Config.OPENROUTER_MODEL)

    def check_faithfulness(self, answer: str, retrieved_contexts: list[str]) -> Tuple[bool, str]:
        """Check if answer is faithful to contexts.

        Args:
            answer: Generated answer
            retrieved_contexts: Retrieved contexts

        Returns:
            Tuple of (is_faithful, evidence_string)
        """
        if not retrieved_contexts:
            # No context provided, so cannot be grounded
            return False, "No context provided for grounding"

        if is_refusal(answer):
            # Refusal is faithful
            return True, "Answer correctly refuses without context"

        if self.use_llm:
            context_combined = "\n\n".join(retrieved_contexts)
            is_faithful = self.llm_judge.check_faithfulness(answer, context_combined)
            evidence = "LLM judge determined answer is grounded" if is_faithful else "LLM judge found hallucinations"
            return is_faithful, evidence
        else:
            # Deterministic: check if key phrases from answer appear in context
            answer_phrases = self._extract_key_phrases(answer)
            context_combined = " ".join(retrieved_contexts).lower()

            grounded_count = 0
            for phrase in answer_phrases:
                if phrase in context_combined:
                    grounded_count += 1

            if not answer_phrases:
                return True, "Answer is very short, likely faithful"

            grounded_ratio = grounded_count / len(answer_phrases)

            if grounded_ratio >= 0.5:  # At least 50% of phrases found
                return True, f"Grounded: {grounded_count}/{len(answer_phrases)} key phrases found in context"
            else:
                return False, f"Hallucination suspected: only {grounded_count}/{len(answer_phrases)} key phrases found in context"

    def _extract_key_phrases(self, text: str) -> list[str]:
        """Extract key phrases from text for faithfulness checking."""
        # Remove common words and split into phrases
        text = text.lower()
        # Remove citations like [S1]
        import re
        text = re.sub(r'\[s\d+\]', '', text)

        # Split by sentences, then extract significant words
        sentences = [s.strip() for s in text.split('.')
if s.strip()]
        key_phrases = []
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'but', 'this', 'that', 'with', 'from', 'by'}

        for sent in sentences:
            words = [w for w in sent.split() if w not in stopwords and len(w) > 3]
            if len(words) >= 2:
                # Create bigrams and trigrams
                for i in range(len(words)):
                    if i + 1 < len(words):
                        key_phrases.append(f"{words[i]} {words[i+1]}")
                    if i + 2 < len(words):
                        key_phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")

        return key_phrases[:20]  # Limit to top 20 phrases
