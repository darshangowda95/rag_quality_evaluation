"""Standalone mock RAG service for evaluation.

This is a self-contained RAG implementation for Lab 2.4 evaluation testing.
NO external dependencies on Lab 2.3 or other labs.

Retrieval is deterministic (hash-based embedding, see `_embed`) so the same
question always retrieves the same contexts. Generation is a live LLM call
with no offline fallback - `OPENROUTER_API_KEY` is required to answer any
question that retrieved at least one context; a question with no retrieved
contexts is refused before the LLM is ever called (see `generate`).
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.utils.data_classes import RAGResponse, RetrievalConfig
from app.utils.config import Config
from app.utils.llm_judge import LLMJudge


class MockRAGService:
    """Standalone mock RAG service for evaluation."""

    def __init__(self):
        """Initialize mock RAG service."""
        self.contexts = self._load_contexts()
        self.llm_judge = LLMJudge(Config.OPENROUTER_API_KEY, Config.OPENROUTER_MODEL)

    def _load_contexts(self) -> Dict[str, Any]:
        """Load ground truth contexts."""
        path = Config.get_absolute_path(Config.GROUND_TRUTH_PATH)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("contexts", {})

    def _embed(self, text: str) -> np.ndarray:
        """Simple deterministic embedding (hash-based)."""
        # Simple word-based embedding for deterministic behavior.
        # sum(ord(c)...) instead of hash(word): Python's built-in hash()
        # is randomized per-process (PYTHONHASHSEED), so it silently broke
        # the "deterministic" retrieval this is meant to provide - results
        # varied on every restart. ord()-based hashing is stable.
        words = text.lower().split()
        embedding = np.zeros(100)
        for i, word in enumerate(words[:100]):
            word_hash = sum(ord(c) for c in word)
            embedding[i % 100] += (word_hash % 1000) / 1000.0
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity."""
        dot_product = np.dot(v1, v2)
        return float(dot_product)  # Already normalized

    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into character chunks of at most chunk_size.

        A context shorter than chunk_size returns as a single chunk - most
        of this corpus's policy passages (150-455 chars) only actually get
        split at chunk_size=256, not 512 or 1024.
        """
        if chunk_size <= 0 or len(text) <= chunk_size:
            return [text]
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def retrieve(self, question: str, config: RetrievalConfig) -> List[Dict[str, Any]]:
        """Retrieve relevant contexts.

        Each context is split into config.chunk_size chunks before
        embedding; a context's score is its single best-matching chunk's
        score, so chunk_size genuinely changes retrieval - splitting a
        context dilutes each chunk's standalone signal, which can help or
        hurt depending on where the matching words land.

        Args:
            question: Question to retrieve for
            config: Retrieval configuration

        Returns:
            List of retrieved contexts with scores
        """
        question_emb = self._embed(question)

        # Score all contexts
        scored = []
        for ctx_id, ctx_data in self.contexts.items():
            text = ctx_data.get("text", "")
            chunks = self._chunk_text(text, config.chunk_size)
            score = max(
                self._cosine_similarity(question_emb, self._embed(chunk))
                for chunk in chunks)

            if score >= config.similarity_threshold:
                scored.append({
                    "id": ctx_id,
                    "text": text,
                    "score": score,
                    "topics": ctx_data.get("topics", [])
                })

        # Sort by score and return top_k
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:config.top_k]

    def generate(self, question: str, contexts: List[Dict[str, Any]], prompt_template: str = "") -> str:
        """Generate answer from contexts via the live LLM.

        No contexts means refuse immediately, before the LLM is ever
        called - same as an empty-retrieval refusal in a real RAG service.
        Otherwise the live LLM is required; there is no offline fallback,
        so a missing key or a failed call raises GenerationError (see
        `utils/llm_judge.py`) instead of degrading to an extractive answer.

        Args:
            question: Question
            contexts: Retrieved contexts
            prompt_template: System prompt (optional)

        Returns:
            Generated answer

        Raises:
            GenerationError: No API key is configured, or every retry
                attempt failed.
        """
        if not contexts:
            return "I cannot answer this question based on the available policy documents."

        context_text = "\n\n".join([f"[S{i+1}] {ctx['text']}" for i, ctx in enumerate(contexts)])

        system_prompt = prompt_template if prompt_template else (
            "You are a banking policy assistant. Answer the question using ONLY information "
            "from the provided context. Cite sources as [S1], [S2] etc."
        )

        full_prompt = f"""{system_prompt}

Context:
{context_text}

Question: {question}

Answer:"""

        return self.llm_judge.judge(
            full_prompt, max_tokens=Config.GENERATION["max_tokens_per_response"])

    def query(self, question: str, config: Optional[RetrievalConfig] = None, prompt_template: str = "") -> RAGResponse:
        """Full RAG pipeline: retrieve + generate.

        Args:
            question: Question
            config: Retrieval configuration (optional)
            prompt_template: System prompt (optional)

        Returns:
            RAGResponse with answer and metadata
        """
        if config is None:
            config = RetrievalConfig(
                chunk_size=Config.DEFAULT_CHUNK_SIZE,
                top_k=Config.DEFAULT_TOP_K,
                similarity_threshold=Config.DEFAULT_SIMILARITY_THRESHOLD
            )

        # Retrieve
        contexts = self.retrieve(question, config)

        # Generate
        answer = self.generate(question, contexts, prompt_template)

        # Return response
        return RAGResponse(
            question=question,
            answer=answer,
            retrieved_contexts=[ctx["text"] for ctx in contexts],
            metadata={
                "config": {
                    "chunk_size": config.chunk_size,
                    "top_k": config.top_k,
                    "similarity_threshold": config.similarity_threshold
                },
                "retrieval_scores": [ctx["score"] for ctx in contexts],
                "num_retrieved": len(contexts),
                "mode": "refused" if not contexts else "llm"
            }
        )


class RagClient:
    """Client wrapper for mock RAG service."""

    def __init__(self):
        """Initialize RAG client."""
        self.service = MockRAGService()

    def query(self, question: str, config: Optional[RetrievalConfig] = None, prompt_template: str = "") -> RAGResponse:
        """Query RAG service.

        Args:
            question: Question
            config: Retrieval configuration
            prompt_template: System prompt

        Returns:
            RAGResponse
        """
        return self.service.query(question, config, prompt_template)
