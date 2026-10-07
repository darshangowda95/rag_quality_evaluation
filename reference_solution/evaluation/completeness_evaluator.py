"""Completeness evaluator - R4 implementation.

Measures whether answers address all parts of multi-part questions.
"""

from typing import List
from app.utils.llm_judge import LLMJudge
from app.utils.config import Config
from app.utils.text import is_refusal


class CompletenessEvaluator:
    """Evaluates completeness of answers."""

    def __init__(self, use_llm: bool = False):
        """Initialize completeness evaluator.

        Args:
            use_llm: Use LLM-as-judge if True, else deterministic
        """
        self.use_llm = use_llm and Config.has_llm_api()
        self.llm_judge = None
        if self.use_llm:
            self.llm_judge = LLMJudge(Config.OPENROUTER_API_KEY, Config.OPENROUTER_MODEL)

    def evaluate_completeness(self, question: str, answer: str) -> float:
        """Evaluate if answer addresses all parts of question.

        Args:
            question: The question
            answer: Generated answer

        Returns:
            Completeness score 0.0-1.0
        """
        if is_refusal(answer):
            return 0.0  # Refusal is incomplete

        if self.use_llm:
            return self.llm_judge.score_completeness(question, answer)
        else:
            # Deterministic: keyword coverage approach
            question_parts = self._identify_question_parts(question)
            if not question_parts:
                return 1.0  # Single-part question, if answered at all it's complete

            # Check how many parts are addressed with partial credit
            answer_lower = answer.lower()
            part_scores = []

            for part_keywords in question_parts:
                # Count how many keywords from this part appear in answer
                matched_keywords = sum(1 for kw in part_keywords if kw in answer_lower)
                if part_keywords:
                    part_score = matched_keywords / len(part_keywords)
                    # Boost score if at least half the keywords match
                    if matched_keywords >= len(part_keywords) / 2:
                        part_score = min(1.0, part_score + 0.2)
                    part_scores.append(part_score)
                else:
                    part_scores.append(0.0)

            base_score = sum(part_scores) / len(part_scores) if part_scores else 0.0

            # Bonus: if answer is substantial (multiple sentences), boost score
            if len(answer.split('.')) > 1 or len(answer.split()) > 15:
                base_score = min(1.0, base_score + 0.1)

            return base_score

    def _identify_question_parts(self, question: str) -> List[List[str]]:
        """Identify parts of a multi-part question.

        Returns:
            List of keyword lists, one per question part
        """
        question_lower = question.lower()

        # Split by common multi-part indicators
        parts = []

        # Check for "and" separating parts (most common)
        if " and " in question_lower:
            segments = question_lower.split(" and ")
            for seg in segments:
                keywords = self._extract_keywords(seg)
                if keywords:
                    parts.append(keywords)
            return parts  # Return early if found parts via "and"

        # Check for numbered/bulleted lists
        if any(marker in question_lower for marker in ['1)', '2)', '(1)', '(2)', 'first', 'second', 'third']):
            # Has explicit parts
            for marker in ['1)', '2)', '3)', '(1)', '(2)', '(3)']:
                if marker in question_lower:
                    idx = question_lower.find(marker)
                    next_idx = len(question_lower)
                    for next_marker in ['2)', '3)', '4)', '(2)', '(3)', '(4)']:
                        if next_marker in question_lower[idx+2:]:
                            next_idx = question_lower.find(next_marker, idx+2)
                            break
                    part_text = question_lower[idx:next_idx]
                    keywords = self._extract_keywords(part_text)
                    if keywords:
                        parts.append(keywords)
            if parts:
                return parts

        # Check for question words indicating multiple aspects
        question_words = ['what', 'when', 'where', 'who', 'why', 'how']
        question_word_indices = [(qw, question_lower.find(qw)) for qw in question_words if qw in question_lower]

        if len(question_word_indices) > 1:
            # Sort by position
            question_word_indices.sort(key=lambda x: x[1])
            # Split into segments between question words
            for i, (qw, idx) in enumerate(question_word_indices):
                next_idx = question_word_indices[i+1][1] if i+1 < len(question_word_indices) else len(question_lower)
                segment = question_lower[idx:next_idx]
                keywords = self._extract_keywords(segment)
                if keywords:
                    parts.append(keywords)
            if parts:
                return parts

        # If no parts detected, treat as single-part
        return []

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text segment."""
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'but', 'this', 'that'}
        words = text.lower().split()
        # Keep question words (what, when, who, etc.) and meaningful words
        keywords = []
        question_words = {'what', 'when', 'where', 'who', 'why', 'how'}
        for w in words:
            clean_word = w.strip('?.,!;:')
            if clean_word in question_words or (len(clean_word) > 3 and clean_word not in stopwords):
                keywords.append(clean_word)
        return keywords[:8]  # Top 8 keywords per part
