"""Prompt optimizer - Task 3 implementation.

A/B tests different prompt variations for best faithfulness and completeness.
"""

import json
from typing import List, Dict, Any
from pathlib import Path
from app.utils.data_classes import EvaluationCase, OptimizationResult
from app.integration.rag_client import RagClient
from app.evaluation.faithfulness_checker import FaithfulnessChecker
from app.evaluation.completeness_evaluator import CompletenessEvaluator
from app.utils.config import Config


class PromptOptimizer:
    """Optimizes system prompt through A/B testing."""

    def __init__(self):
        """Initialize prompt optimizer."""
        self.rag_client = RagClient()
        self.faithfulness_checker = FaithfulnessChecker(use_llm=False)
        self.completeness_evaluator = CompletenessEvaluator(use_llm=False)

    def optimize(self, evaluation_cases: List[EvaluationCase], quick_mode: bool = False) -> OptimizationResult:
        """Optimize prompt through A/B testing.

        Args:
            evaluation_cases: Test cases
            quick_mode: If True, test only baseline prompt for fast validation (default: False)

        Returns:
            OptimizationResult with best prompt
        """
        prompts = self._load_prompt_templates()

        # In quick mode, only test the baseline prompt
        if quick_mode:
            prompts = {k: v for k, v in prompts.items() if k == "baseline"}

        print(f"Testing {len(prompts)} prompt variations...")

        best_score = 0.0
        best_prompt_id = ""
        best_prompt_text = ""
        all_results = []

        for prompt_id, prompt_data in prompts.items():
            prompt_text = prompt_data["system_prompt"]
            score = self._evaluate_prompt(prompt_text, evaluation_cases)

            result = {
                "prompt_id": prompt_id,
                "prompt_name": prompt_data["name"],
                "combined_score": score
            }
            all_results.append(result)

            print(f"  {prompt_id}: score={score:.3f}")

            if score > best_score:
                best_score = score
                best_prompt_id = prompt_id
                best_prompt_text = prompt_text

        baseline_score = next((r["combined_score"] for r in all_results if r["prompt_id"] == "baseline"), 0.5)
        improvement = best_score - baseline_score
        improvement_pct = (improvement / baseline_score * 100) if baseline_score else 0.0

        print(f"\nBest prompt: {best_prompt_id}")
        print(f"Improvement: {improvement:.3f} ({improvement_pct:.1f}%)")

        return OptimizationResult(
            best_config={
                "prompt_id": best_prompt_id,
                "prompt_text": best_prompt_text
            },
            baseline_score=baseline_score,
            optimized_score=best_score,
            improvement=improvement,
            all_results=all_results
        )

    def _load_prompt_templates(self) -> Dict[str, Any]:
        """Load prompt templates from JSON."""
        path = Config.get_absolute_path(Config.PROMPT_TEMPLATES_PATH)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("templates", {})

    def _evaluate_prompt(self, prompt_text: str, cases: List[EvaluationCase]) -> float:
        """Evaluate a single prompt."""
        faithfulness_scores = []
        completeness_scores = []

        for case in cases[:10]:  # Sample for speed
            response = self.rag_client.query(case.question, prompt_template=prompt_text)

            # Faithfulness
            is_faithful, _ = self.faithfulness_checker.check_faithfulness(
                response.answer, response.retrieved_contexts
            )
            faithfulness_scores.append(1.0 if is_faithful else 0.0)

            # Completeness
            completeness = self.completeness_evaluator.evaluate_completeness(
                case.question, response.answer
            )
            completeness_scores.append(completeness)

        # Combined score (average of both metrics)
        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
        avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0

        return (avg_faithfulness + avg_completeness) / 2.0
