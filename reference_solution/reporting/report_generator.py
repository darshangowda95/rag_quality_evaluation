"""Report generator - Task 4 implementation.

Generates structured evaluation reports comparing baseline vs optimized performance.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
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
        """Run evaluation on all cases."""
        from app.utils.data_classes import RetrievalConfig

        if is_baseline:
            config = RetrievalConfig()
            prompt = ""
        else:
            config = RetrievalConfig(
                chunk_size=best_retrieval_config["chunk_size"],
                top_k=best_retrieval_config["top_k"]
            )
            prompt = best_prompt

        relevance_scores = []
        faithfulness_scores = []
        completeness_scores = []
        grounding_scores = []

        for case in cases[:15]:  # Sample for speed
            response = self.rag_client.query(case.question, config=config, prompt_template=prompt)

            # Relevance
            if response.retrieved_contexts:
                rel_score = self.relevance_scorer.score_relevance(case.question, response.retrieved_contexts)
                relevance_scores.append(rel_score)

            # Faithfulness
            is_faithful, _ = self.faithfulness_checker.check_faithfulness(
                response.answer, response.retrieved_contexts
            )
            faithfulness_scores.append(1.0 if is_faithful else 0.0)

            # Completeness
            comp_score = self.completeness_evaluator.evaluate_completeness(
                case.question, response.answer
            )
            completeness_scores.append(comp_score)

            # Grounding: did refusal/answer match this question's category?
            is_grounded = self.grounding_validator.validate(
                case.category, response.retrieved_contexts, response.answer
            )
            grounding_scores.append(1.0 if is_grounded else 0.0)

        return {
            "relevance": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0,
            "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0,
            "completeness": sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0,
            "grounding_accuracy": sum(grounding_scores) / len(grounding_scores) if grounding_scores else 0.0
        }

    def _save_report(self, report: EvaluationReport, output_path: str):
        """Save report as JSON and Markdown."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = output_path
        with open(json_path, 'w', encoding='utf-8') as f:
            report_dict = {
                "timestamp": report.timestamp,
                "baseline_metrics": report.baseline_metrics,
                "optimized_metrics": report.optimized_metrics,
                "improvement_deltas": report.improvement_deltas,
                "optimization_details": report.optimization_details
            }
            json.dump(report_dict, f, indent=2)

        # Save Markdown
        md_path = output_path.with_suffix('.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(self._format_markdown(report))

        print(f"  Report saved: {json_path}")
        print(f"  Markdown saved: {md_path}")

    def _format_markdown(self, report: EvaluationReport) -> str:
        """Format report as Markdown."""

        def pct(delta, baseline):
            return (delta / baseline * 100) if baseline else 0.0

        deltas = report.improvement_deltas
        direction = "improved" if min(deltas.values()) >= 0 else "changed"
        summary_pcts = ", ".join(
            f"{metric} {pct(delta, report.baseline_metrics[metric]):+.1f}%"
            for metric, delta in deltas.items())

        md = f"""# RAG Evaluation Report

**Generated:** {report.timestamp}

## Executive Summary

Optimization {direction} the four quality metrics: {summary_pcts}.

## Baseline Performance

| Metric | Score |
|--------|-------|
| Relevance | {report.baseline_metrics['relevance']:.3f} |
| Faithfulness | {report.baseline_metrics['faithfulness']:.3f} |
| Completeness | {report.baseline_metrics['completeness']:.3f} |
| Grounding Accuracy | {report.baseline_metrics['grounding_accuracy']:.3f} |

## Optimized Performance

| Metric | Score | Improvement |
|--------|-------|-------------|
| Relevance | {report.optimized_metrics['relevance']:.3f} | +{report.improvement_deltas['relevance']:.3f} ({pct(report.improvement_deltas['relevance'], report.baseline_metrics['relevance']):.1f}%) |
| Faithfulness | {report.optimized_metrics['faithfulness']:.3f} | +{report.improvement_deltas['faithfulness']:.3f} ({pct(report.improvement_deltas['faithfulness'], report.baseline_metrics['faithfulness']):.1f}%) |
| Completeness | {report.optimized_metrics['completeness']:.3f} | +{report.improvement_deltas['completeness']:.3f} ({pct(report.improvement_deltas['completeness'], report.baseline_metrics['completeness']):.1f}%) |
| Grounding Accuracy | {report.optimized_metrics['grounding_accuracy']:.3f} | +{report.improvement_deltas['grounding_accuracy']:.3f} ({pct(report.improvement_deltas['grounding_accuracy'], report.baseline_metrics['grounding_accuracy']):.1f}%) |

## Optimization Details

**Best Retrieval Configuration:**
- Chunk size: {report.optimization_details['retrieval']['chunk_size']}
- Top K: {report.optimization_details['retrieval']['top_k']}

**Best Prompt:**
- Prompt ID: {report.optimization_details['prompt']['prompt_id']}

## Recommendations

1. Deploy optimized configuration to production
2. Monitor these metrics continuously
3. Re-evaluate quarterly as policies change
"""
        return md
