"""Pytest fixtures for Lab 2.4 tests."""

import pytest
import sys
from pathlib import Path

# Add lab root to path
lab_root = Path(__file__).parent.parent
sys.path.insert(0, str(lab_root))

# Import bootstrap and use appropriate tree
import bootstrap
bootstrap.use_tree()  # Uses RAG_EVAL environment variable

from app.evaluation.dataset_loader import EvaluationDatasetLoader
from app.utils.data_classes import EvaluationCase


@pytest.fixture
def sample_evaluation_case():
    """Provide a sample evaluation case for testing."""
    return EvaluationCase(
        id="test_001",
        question="How long must KYC records be retained?",
        expected_answer="KYC records must be retained for 5 years after account closure.",
        ground_truth_context_ids=["kyc_retention_001"],
        category="factual_recall",
        difficulty="beginner"
    )


@pytest.fixture
def sample_contexts():
    """Provide sample retrieved contexts."""
    return [
        "KYC customer identification records must be retained for a minimum of 5 years following account closure.",
        "Retention requirements comply with BSA/AML regulations and may be extended for investigations."
    ]


@pytest.fixture
def dataset_loader():
    """Provide dataset loader instance."""
    return EvaluationDatasetLoader()


@pytest.fixture
def evaluation_dataset_path():
    """Provide path to evaluation dataset."""
    return str(lab_root / "dataset" / "evaluation_questions.json")
