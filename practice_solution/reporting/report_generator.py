"""Report generator - Task 4 implementation.

TASK 4: Implement evaluation report generation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from app.utils.data_classes import EvaluationCase, EvaluationReport
from app.evaluation.dataset_loader import EvaluationDatasetLoader
from app.evaluation.relevance_scorer import RelevanceScorer
from app.evaluation.faithfulness_checker import FaithfulnessChecker
from app.evaluation.completeness_evaluator import CompletenessEvaluator
from app.evaluation.grounding_validator import GroundingValidator
from app.optimization.retrieval_optimizer import RetrievalOptimizer
from app.optimization.prompt_optimizer import PromptOptimizer
from app.integration.rag_client import RagClient
from app.utils.config import Config


class ReportGenerator:
    """Generates evaluation reports."""

    def __init__(self):
        """Initialize report generator."""
        self.dataset_loader = EvaluationDatasetLoader()
        self.rag_client = RagClient()
        self.relevance_scorer = RelevanceScorer(use_llm=False)
        self.faithfulness_checker = FaithfulnessChecker(use_llm=False)
        self.completeness_evaluator = CompletenessEvaluator(use_llm=False)
        self.grounding_validator = GroundingValidator()

    def generate(self, output_path: str = None) -> EvaluationReport:
        """Generate complete evaluation report.

        Args:
            output_path: Path to save report (optional)

        Returns:
            EvaluationReport
        """
        print("=== RAG Evaluation Pipeline ===\n")

        # Load dataset
        print("[1/5] Loading evaluation dataset...")
        dataset_path = Config.get_absolute_path(Config.EVAL_DATASET_PATH)
        cases = self.dataset_loader.load(str(dataset_path))
        print(f"  Loaded {len(cases)} test cases\n")

        # Baseline evaluation
        print("[2/5] Running baseline evaluation...")
        baseline_metrics = self._run_evaluation(cases, is_baseline=True)
        print(f"  Baseline: relevance={baseline_metrics['relevance']:.3f}, "
              f"faithfulness={baseline_metrics['faithfulness']:.3f}, "
              f"completeness={baseline_metrics['completeness']:.3f}, "
              f"grounding_accuracy={baseline_metrics['grounding_accuracy']:.3f}\n")

        # Optimization
        print("[3/5] Running optimization...")
        retrieval_result = RetrievalOptimizer().optimize(cases)
        prompt_result = PromptOptimizer().optimize(cases)

        # Optimized evaluation
        print("\n[4/5] Running optimized evaluation...")
        optimized_metrics = self._run_evaluation(
            cases,
            is_baseline=False,
            best_retrieval_config=retrieval_result.best_config,
            best_prompt=prompt_result.best_config["prompt_text"]
        )
        print(f"  Optimized: relevance={optimized_metrics['relevance']:.3f}, "
              f"faithfulness={optimized_metrics['faithfulness']:.3f}, "
              f"completeness={optimized_metrics['completeness']:.3f}, "
              f"grounding_accuracy={optimized_metrics['grounding_accuracy']:.3f}\n")

        # Calculate improvements - every metric _run_evaluation returns, not
        # a fixed list, so a new metric never has to be added here by hand.
        print("[5/5] Generating report...")
        improvement_deltas = {
            metric: optimized_metrics[metric] - baseline_metrics[metric]
            for metric in baseline_metrics
        }

        report = EvaluationReport(
            timestamp=datetime.now().isoformat(),
            baseline_metrics=baseline_metrics,
            optimized_metrics=optimized_metrics,
            improvement_deltas=improvement_deltas,
            optimization_details={
                "retrieval": retrieval_result.best_config,
                "prompt": prompt_result.best_config
            }
        )

        # Save report
        if output_path:
            self._save_report(report, output_path)

        print("\n=== Report Complete ===")
        return report

    def _run_evaluation(self, cases: List[EvaluationCase], is_baseline: bool = True,
                       best_retrieval_config: Dict = None, best_prompt: str = "") -> Dict[str, float]:
        """Run evaluation (PROVIDED - helper method)."""
        from app.utils.data_classes import RetrievalConfig
        if is_baseline:
            config = RetrievalConfig()
            prompt = ""
        else:
            config = RetrievalConfig(chunk_size=best_retrieval_config["chunk_size"], top_k=best_retrieval_config["top_k"])
            prompt = best_prompt
        relevance_scores = []
        faithfulness_scores = []
        completeness_scores = []
        grounding_scores = []
        for case in cases[:15]:
            response = self.rag_client.query(case.question, config=config, prompt_template=prompt)
            if response.retrieved_contexts:
                rel_score = self.relevance_scorer.score_relevance(case.question, response.retrieved_contexts)
                relevance_scores.append(rel_score)
            is_faithful, _ = self.faithfulness_checker.check_faithfulness(response.answer, response.retrieved_contexts)
            faithfulness_scores.append(1.0 if is_faithful else 0.0)
            comp_score = self.completeness_evaluator.evaluate_completeness(case.question, response.answer)
            completeness_scores.append(comp_score)
            is_grounded = self.grounding_validator.validate(case.category, response.retrieved_contexts, response.answer)
            grounding_scores.append(1.0 if is_grounded else 0.0)
        return {
            "relevance": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0,
            "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0,
            "completeness": sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0,
            "grounding_accuracy": sum(grounding_scores) / len(grounding_scores) if grounding_scores else 0.0
        }
