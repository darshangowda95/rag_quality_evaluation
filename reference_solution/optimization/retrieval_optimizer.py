"""Retrieval optimizer - Task 2 implementation.

Optimizes chunk size and top_k parameters for best retrieval performance.
"""

from typing import List, Dict, Any
from app.utils.data_classes import EvaluationCase, RetrievalConfig, OptimizationResult
from app.integration.rag_client import RagClient
from app.evaluation.relevance_scorer import RelevanceScorer
from app.utils.config import Config


class RetrievalOptimizer:
    """Optimizes retrieval configuration through grid search."""

    def __init__(self):
        """Initialize retrieval optimizer."""
        self.rag_client = RagClient()
        self.relevance_scorer = RelevanceScorer(use_llm=False)  # Use deterministic for speed

    def optimize(self, evaluation_cases: List[EvaluationCase], quick_mode: bool = False) -> OptimizationResult:
        """Optimize retrieval configuration.

        Args:
            evaluation_cases: Test cases for evaluation
            quick_mode: If True, run only 1 iteration for fast validation (default: False)

        Returns:
            OptimizationResult with best configuration
        """
        # Use quick mode for validation (1 iteration), or full grid search for actual optimization
        if quick_mode:
            chunk_sizes = [Config.DEFAULT_CHUNK_SIZE]  # Just 1 value
            top_k_values = [Config.DEFAULT_TOP_K]      # Just 1 value
        else:
            chunk_sizes = Config.CHUNK_SIZES
            top_k_values = Config.TOP_K_VALUES

        print(f"Running grid search: {len(chunk_sizes)} chunk sizes x {len(top_k_values)} top_k values...")

        baseline_config = RetrievalConfig(
            chunk_size=Config.DEFAULT_CHUNK_SIZE,
            top_k=Config.DEFAULT_TOP_K
        )
        baseline_score = self._evaluate_config(baseline_config,evaluation_cases)

        best_score = baseline_score
        best_config = baseline_config
        all_results = []

        # Grid search
        for chunk_size in chunk_sizes:
            for top_k in top_k_values:
                config = RetrievalConfig(chunk_size=chunk_size, top_k=top_k)
                score = self._evaluate_config(config, evaluation_cases)

                result = {
                    "chunk_size": chunk_size,
                    "top_k": top_k,
                    "relevance_score": score
                }
                all_results.append(result)

                print(f"  chunk_size={chunk_size}, top_k={top_k}: relevance={score:.3f}")

                if score > best_score:
                    best_score = score
                    best_config = config

        improvement = best_score - baseline_score
        improvement_pct = (improvement / baseline_score * 100) if baseline_score else 0.0
        print(f"\nBest config: chunk_size={best_config.chunk_size}, top_k={best_config.top_k}")
        print(f"Improvement: {improvement:.3f} ({improvement_pct:.1f}%)")

        return OptimizationResult(
            best_config={
                "chunk_size": best_config.chunk_size,
                "top_k": best_config.top_k
            },
            baseline_score=baseline_score,
            optimized_score=best_score,
            improvement=improvement,
            all_results=all_results
        )

    def _evaluate_config(self, config: RetrievalConfig, cases: List[EvaluationCase]) -> float:
        """Evaluate a single configuration."""
        scores = []
        for case in cases[:10]:  # Sample for speed
            response = self.rag_client.query(case.question, config=config)
            if response.retrieved_contexts:
                score = self.relevance_scorer.score_relevance(case.question, response.retrieved_contexts)
                scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0
