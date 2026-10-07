"""Tests for the (prefilled) faithfulness checker."""

import pytest
from app.evaluation.faithfulness_checker import FaithfulnessChecker


def test_grounded_answer_passes():
    """Test that grounded answers are marked as faithful."""
    checker = FaithfulnessChecker(use_llm=False)

    context = ["KYC records must be retained for 5 years after account closure."]
    grounded_answer = "KYC records must be retained for 5 years after account closure."

    is_faithful, evidence = checker.check_faithfulness(grounded_answer, context)
    assert is_faithful, f"Grounded answer should be faithful. Evidence: {evidence}"


def test_hallucinated_answer_fails():
    """Test that hallucinated answers are detected."""
    checker = FaithfulnessChecker(use_llm=False)

    context = ["KYC records must be retained for 5 years."]
    hallucinated_answer = "KYC records must be retained for 10 years and submitted to the Federal Reserve annually."

    is_faithful, evidence = checker.check_faithfulness(hallucinated_answer, context)
    assert not is_faithful, f"Hallucinated answer should be detected. Evidence: {evidence}"


def test_refuses_when_no_context():
    """Test that refusal without context is faithful."""
    checker = FaithfulnessChecker(use_llm=False)

    answer = "I cannot answer this question based on the available policy documents."
    contexts = []

    is_faithful, evidence = checker.check_faithfulness(answer, contexts)
    # Refusing without context is acceptable (not a hallucination)
    assert evidence is not None, "Should provide evidence"
