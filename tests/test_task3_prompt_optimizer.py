"""Tests for Task 3: Prompt optimizer.

optimize() queries the mock RAG client once per (prompt, case) pair, and
the mock RAG client's generation is a live LLM call with no offline
fallback - so optimize() is run once per test module (not once per test)
via the module-scoped `result` fixture below, to avoid tripling the live
call count for three independent assertions on the same result.
"""

import pytest
from app.optimization.prompt_optimizer import PromptOptimizer
from app.evaluation.dataset_loader import EvaluationDatasetLoader


@pytest.fixture(scope="module")
def sample_cases():
    """Load sample evaluation cases."""
    loader = EvaluationDatasetLoader()
    dataset_path = "dataset/evaluation_questions.json"
    cases = loader.load(dataset_path)
    return cases[:5]


@pytest.fixture(scope="module")
def result(sample_cases):
    """Run the optimizer once per module; every test below checks this same result."""
    return PromptOptimizer().optimize(sample_cases)


def test_optimizer_tests_multiple_prompts(result):
    """Test that optimizer tests multiple prompt variations."""
    # Should have tested at least 3 prompts (baseline + variations)
    assert len(result.all_results) >= 3, f"Optimizer should test multiple prompts, tested {len(result.all_results)}"


def test_optimizer_selects_best_prompt(result):
    """Test that optimizer selects the best prompt."""
    assert "prompt_id" in result.best_config, "Best config should have prompt_id"
    assert "prompt_text" in result.best_config, "Best config should have prompt_text"
    assert len(result.best_config["prompt_text"]) > 0, "Prompt text should not be empty"


def test_metrics_improve_over_baseline(result):
    """Test that optimization shows improvement."""
    # Optimized should be >= baseline
    assert result.optimized_score >= result.baseline_score, \
        f"Optimized score ({result.optimized_score}) should not be worse than baseline ({result.baseline_score})"
