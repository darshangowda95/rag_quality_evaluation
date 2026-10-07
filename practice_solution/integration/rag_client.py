"""Standalone mock RAG service - PROVIDED for learners.

This is a complete mock RAG service provided for Lab 2.4.
You do NOT need to modify this file.

Retrieval is deterministic (hash-based embedding, see `_embed`). Generation
is a live LLM call with no offline fallback - `OPENROUTER_API_KEY` is
required to answer any question that retrieved at least one context; a
question with no retrieved contexts is refused before the LLM is ever
called (see `generate`).
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.utils.data_classes import RAGResponse, RetrievalConfig
from app.utils.config import Config
from app.utils.llm_judge import LLMJudge


class MockRAGService:
    """Standalone mock RAG service (PROVIDED - no modifications needed)."""

    def __init__(self):
        self.contexts = self._load_contexts()
        self.llm_judge = LLMJudge(Config.OPENROUTER_API_KEY, Config.OPENROUTER_MODEL)

    def _load_contexts(self) -> Dict[str, Any]:
        path = Config.get_absolute_path(Config.GROUND_TRUTH_PATH)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("contexts", {})

    def _embed(self, text: str) -> np.ndarray:
        words = text.lower().split()
        embedding = np.zeros(100)
        for i, word in enumerate(words[:100]):
            # sum(ord(c)...) instead of hash(word): Python's built-in hash()
            # is randomized per-process (PYTHONHASHSEED), so it silently
            # broke the "deterministic" retrieval this is meant to provide -
            # results varied on every restart. ord()-based hashing is stable.
            word_hash = sum(ord(c) for c in word)
            embedding[i % 100] += (word_hash % 1000) / 1000.0
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.dot(v1, v2))

    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Split into character chunks of at most chunk_size (1 chunk if text already fits)."""
        if chunk_size <= 0 or len(text) <= chunk_size:
            return [text]
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def retrieve(self, question: str, config: RetrievalConfig) -> List[Dict[str, Any]]:
        """A context's score is its single best-matching chunk's score, so
        chunk_size genuinely changes retrieval."""
        question_emb = self._embed(question)
        scored = []
        for ctx_id, ctx_data in self.contexts.items():
            text = ctx_data.get("text", "")
            chunks = self._chunk_text(text, config.chunk_size)
            score = max(self._cosine_similarity(question_emb, self._embed(chunk)) for chunk in chunks)
            if score >= config.similarity_threshold:
                scored.append({"id": ctx_id, "text": text, "score": score, "topics": ctx_data.get("topics", [])})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:config.top_k]

    def generate(self, question: str, contexts: List[Dict[str, Any]], prompt_template: str = "") -> str:
        """No contexts -> refuse before ever calling the LLM. Otherwise the
        live LLM is required; no offline fallback (raises GenerationError
        on a missing key or exhausted retries - see utils/llm_judge.py)."""
        if not contexts:
            return "I cannot answer this question based on the available policy documents."
        context_text = "\n\n".join([f"[S{i+1}] {ctx['text']}" for i, ctx in enumerate(contexts)])
        system_prompt = prompt_template if prompt_template else "You are a banking policy assistant. Answer using ONLY information from the provided context."
        full_prompt = f"{system_prompt}\n\nContext:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
        return self.llm_judge.judge(full_prompt, max_tokens=Config.GENERATION["max_tokens_per_response"])

    def query(self, question: str, config: Optional[RetrievalConfig] = None, prompt_template: str = "") -> RAGResponse:
        if config is None:
            config = RetrievalConfig(chunk_size=Config.DEFAULT_CHUNK_SIZE, top_k=Config.DEFAULT_TOP_K, similarity_threshold=Config.DEFAULT_SIMILARITY_THRESHOLD)
        contexts = self.retrieve(question, config)
        answer = self.generate(question, contexts, prompt_template)
        return RAGResponse(
            question=question,
            answer=answer,
            retrieved_contexts=[ctx["text"] for ctx in contexts],
            metadata={"config": {"chunk_size": config.chunk_size, "top_k": config.top_k, "similarity_threshold": config.similarity_threshold}, "retrieval_scores": [ctx["score"] for ctx in contexts], "num_retrieved": len(contexts), "mode": "refused" if not contexts else "llm"}
        )


class RagClient:
    """Client wrapper for mock RAG service."""

    def __init__(self):
        self.service = MockRAGService()

    def query(self, question: str, config: Optional[RetrievalConfig] = None, prompt_template: str = "") -> RAGResponse:
        return self.service.query(question, config, prompt_template)
