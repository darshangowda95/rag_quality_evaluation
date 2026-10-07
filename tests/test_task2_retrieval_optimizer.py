"""Tests for Task 2: Retrieval optimizer.

optimize() queries the mock RAG client once per (config, case) pair, and
the mock RAG client's generation is a live LLM call with no offline
fallback - so optimize() is run once per test module (not once per test)
via the module-scoped `result` fixture below, to avoid tripling the live
call count for three independent assertions on the same result.
"""

import pytest
from app.optimization.retrieval_optimizer import RetrievalOptimizer
from app.evaluation.dataset_loader import EvaluationDatasetLoader


@pytest.fixture(scope="module")
def sample_cases():
    """Load sample evaluation cases."""
    loader = EvaluationDatasetLoader()
    dataset_path = "dataset/evaluation_questions.json"
    cases = loader.load(dataset_path)
    return cases[:5]  # Use first 5 for testing


@pytest.fixture(scope="module")
def result(sample_cases):
    """Run the optimizer once per module; every test below checks this same result."""
    return RetrievalOptimizer().optimize(sample_cases)


def test_optimizer_tries_multiple_configs(result):
    """Test that optimizer tests multiple configurations."""
    # Should have tested at least 2 configurations
    assert len(result.all_results) >= 2, "Optimizer should test multiple configs"


def test_optimizer_improves_baseline(result):
    """Test that optimizer finds improvement over baseline."""
    # Optimized should be >= baseline (equal is acceptable if baseline is already optimal)
    assert result.optimized_score >= result.baseline_score, \
        f"Optimized ({result.optimized_score}) should not be worse than baseline ({result.baseline_score})"


def test_best_config_returned(result):
    """Test that best configuration is returned."""
    assert "chunk_size" in result.best_config, "Best config should have chunk_size"
    assert "top_k" in result.best_config, "Best config should have top_k"
    assert result.best_config["chunk_size"] > 0, "Chunk size should be positive"
    assert result.best_config["top_k"] > 0, "Top K should be positive"
