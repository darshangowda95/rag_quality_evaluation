"""Main evaluation runner for Lab 2.4.

Run This script to execute the full RAG evaluation pipeline.
"""

import argparse
import sys
from pathlib import Path

# Setup path and bootstrap
sys.path.insert(0, str(Path(__file__).parent))
import bootstrap
bootstrap.use_tree()

from app.reporting.report_generator import ReportGenerator
from app.utils.config import Config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RAG Evaluation Pipeline")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run baseline evaluation only"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run optimization only"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full pipeline (baseline + optimize + report)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for report (default: reports/evaluation_report.json)"
    )

    args = parser.parse_args()

    # Default to full if no specific mode chosen
    if not (args.baseline or args.optimize or args.full):
        args.full = True

    # Set default output path
    if args.output is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(Config.REPORTS_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / f"evaluation_report_{timestamp}.json")

    print("=" * 60)
    print("RAG Evaluation Pipeline - Lab 2.4")
    print("=" * 60)
    print(f"Solution tree: {bootstrap.active_tree()}")
    print(f"Output: {args.output}")
    print("=" * 60)
    print()

    try:
        generator = ReportGenerator()

        if args.full:
            # Run complete pipeline
            report = generator.generate(output_path=args.output)
            print_summary(report)

        elif args.baseline:
            # Baseline only
            from app.evaluation.dataset_loader import EvaluationDatasetLoader
            loader = EvaluationDatasetLoader()
            dataset_path = Config.get_absolute_path(Config.EVAL_DATASET_PATH)
            cases = loader.load(str(dataset_path))

            print("[Baseline Evaluation Only]")
            baseline_metrics = generator._run_evaluation(cases, is_baseline=True)
            print(f"\nBaseline Results:")
            for metric, score in baseline_metrics.items():
                print(f"  {metric}: {score:.3f}")

        elif args.optimize:
            # Optimization only
            from app.evaluation.dataset_loader import EvaluationDatasetLoader
            from app.optimization.retrieval_optimizer import RetrievalOptimizer
            from app.optimization.prompt_optimizer import PromptOptimizer

            loader = EvaluationDatasetLoader()
            dataset_path = Config.get_absolute_path(Config.EVAL_DATASET_PATH)
            cases = loader.load(str(dataset_path))

            print("[Optimization Only]")
            print("\nOptimizing retrieval...")
            retrieval_result = RetrievalOptimizer().optimize(cases)
            print(f"Best config: {retrieval_result.best_config}")

            print("\nOptimizing prompts...")
            prompt_result = PromptOptimizer().optimize(cases)
            print(f"Best prompt: {prompt_result.best_config['prompt_id']}")

        print("\n" + "=" * 60)
        print("Evaluation complete!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def print_summary(report):
    """Print report summary."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print("\nBaseline Performance:")
    for metric, score in report.baseline_metrics.items():
        print(f"  {metric:15s}: {score:.3f}")

    print("\nOptimized Performance:")
    for metric, score in report.optimized_metrics.items():
        delta = report.improvement_deltas[metric]
        pct = (delta / report.baseline_metrics[metric] * 100) if report.baseline_metrics[metric] > 0 else 0
        print(f"  {metric:15s}: {score:.3f} (+{delta:.3f}, +{pct:.1f}%)")

    print("\nOptimization Details:")
    print(f"  Retrieval: chunk_size={report.optimization_details['retrieval']['chunk_size']}, "
          f"top_k={report.optimization_details['retrieval']['top_k']}")
    print(f"  Prompt: {report.optimization_details['prompt']['prompt_id']}")


if __name__ == "__main__":
    main()
