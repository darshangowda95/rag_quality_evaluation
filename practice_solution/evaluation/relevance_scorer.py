"""Relevance scorer - R2 implementation.

Measures whether retrieved documents are relevant to the question.
"""

import numpy as np
from typing import List
from app.utils.llm_judge import LLMJudge
from app.utils.config import Config


class RelevanceScorer:
    """Scores relevance of retrieved documents to question."""

    def __init__(self, use_llm: bool = False):
        """Initialize relevance scorer.

        Args:
            use_llm: Use LLM-as-judge if True, else deterministic
        """
        self.use_llm = use_llm and Config.has_llm_api()
        self.llm_judge = None
        if self.use_llm:
            self.llm_judge = LLMJudge(Config.OPENROUTER_API_KEY, Config.OPENROUTER_MODEL)

    def _embed(self, text: str) -> np.ndarray:
        """Simple deterministic embedding using word overlap."""
        words = set(text.lower().split())
        embedding = np.zeros(100)
        # Use consistent hashing for each word
        for word in words:
            # Create a stable hash for each word
            word_hash = sum(ord(c) for c in word)
            idx = word_hash % 100
            embedding[idx] += 1.0
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity."""
        return float(np.dot(v1, v2))

    def score_relevance(self, question: str, retrieved_contexts: List[str]) -> float:
        """Score relevance of retrieved contexts to question.

        Args:
            question: The question
            retrieved_contexts: List of retrieved document texts

        Returns:
            Relevance score 0.0-1.0 (average across all contexts)
        """
        if not retrieved_contexts:
            return 0.0

        if self.use_llm:
            scores = [self.llm_judge.score_relevance(question, ctx) for ctx in retrieved_contexts]
            return sum(scores) / len(scores)
        else:
            # Deterministic: cosine similarity
            question_emb = self._embed(question)
            scores = []
            for ctx in retrieved_contexts:
                ctx_emb = self._embed(ctx)
                score = self._cosine_similarity(question_emb, ctx_emb)
                scores.append(max(0.0, min(1.0, score)))  # Clamp to [0, 1]
            return sum(scores) / len(scores)
