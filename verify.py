"""Lightweight verification script for Lab 2.4.

Quick checks to verify lab setup and implementation.
"""

import sys
import io
from pathlib import Path

# Fix Windows Unicode encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Setup path and bootstrap
sys.path.insert(0, str(Path(__file__).parent))
import bootstrap

print("=" * 60)
print("Lab 2.4 Verification")
print("=" * 60)

# Check 1: Bootstrap
print("\n[1/6] Checking bootstrap...")
try:
    tree = bootstrap.use_tree()
    print(f"   ✓ Active tree: {tree}")
except Exception as e:
    print(f"   ❌ Bootstrap failed: {e}")
    sys.exit(1)

# Check 2: Dataset
print("\n[2/6] Checking evaluation dataset...")
try:
    from app.evaluation.dataset_loader import EvaluationDatasetLoader
    from app.utils.config import Config

    loader = EvaluationDatasetLoader()
    dataset_path = Config.get_absolute_path(Config.EVAL_DATASET_PATH)
    cases = loader.load(str(dataset_path))

    print(f"   ✓ Loaded {len(cases)} test cases")

    if len(cases) < 25:
        print(f"   ⚠ Warning: Expected at least 25 cases, found {len(cases)}")
except Exception as e:
    print(f"   ❌ Dataset load failed: {e}")
    sys.exit(1)

# Check 3: Metrics
print("\n[3/6] Checking metrics modules...")
try:
    from app.evaluation.relevance_scorer import RelevanceScorer
    from app.evaluation.faithfulness_checker import FaithfulnessChecker
    from app.evaluation.completeness_evaluator import CompletenessEvaluator
    from app.evaluation.grounding_validator import GroundingValidator

    scorer = RelevanceScorer(use_llm=False)
    checker = FaithfulnessChecker(use_llm=False)
    evaluator = CompletenessEvaluator(use_llm=False)
    validator = GroundingValidator()

    print("   ✓ Relevance scorer")
    print("   ✓ Faithfulness checker")
    print("   ✓ Completeness evaluator")
    print("   ✓ Grounding validator")
except Exception as e:
    print(f"   ❌ Metrics import failed: {e}")
    print("   ℹ The relevance/faithfulness/completeness/grounding metrics "
          "are prefilled, not a guided task - this points to a real import "
          "error, not an unfinished TODO.")

# Check 4: Optimizers
print("\n[4/6] Checking optimizer modules...")
try:
    from app.optimization.retrieval_optimizer import RetrievalOptimizer
    from app.optimization.prompt_optimizer import PromptOptimizer

    print("   ✓ Retrieval optimizer")
    print("   ✓ Prompt optimizer")
except Exception as e:
    print(f"   ❌ Optimizer import failed: {e}")
    if tree == "practice_solution":
        print("   ℹ This is expected if you haven't implemented Task 2 "
              "(retrieval optimizer) or Task 3 (prompt optimizer) yet")

# Check 5: Reporter
print("\n[5/6] Checking report generator...")
try:
    from app.reporting.report_generator import ReportGenerator

    print("   ✓ Report generator")
except Exception as e:
    print(f"   ❌ Report generator import failed: {e}")
    if tree == "practice_solution":
        print("   ℹ This is expected if you haven't implemented Task 4 "
              "(report generator) yet")

# Check 6: Integration
print("\n[6/6] Checking RAG client...")
try:
    from app.integration.rag_client import RagClient

    client = RagClient()
    print("   ✓ RAG client initialized")

    # Try a test query
    from app.utils.data_classes import RetrievalConfig
    config = RetrievalConfig(chunk_size=512, top_k=5)
    response = client.query("What is KYC?", config=config)

    print(f"   ✓ Test query successful (retrieved {len(response.retrieved_contexts)} contexts)")
except Exception as e:
    print(f"   ❌ RAG client failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 60)
print("Verification Summary")
print("=" * 60)
print(f"Solution tree: {tree}")
print(f"Dataset: {len(cases)} test cases")
print("\nNext steps:")
if tree == "practice_solution":
    print("  1. Implement Tasks 1-4 (follow the TODOs in practice_solution/)")
    print("  2. Run: pytest -v")
    print("  3. Run: python run_evaluation.py --full")
else:
    print("  1. Run tests: pytest -v")
    print("  2. Run evaluation: python run_evaluation.py --full")
    print("  3. Switch to practice: export RAG_EVAL=practice_solution")

print("\n" + "=" * 60)
print("✓ Verification complete")
print("=" * 60)
