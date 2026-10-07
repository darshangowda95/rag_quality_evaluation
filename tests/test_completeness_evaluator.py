"""Tests for the (prefilled) completeness evaluator."""

import pytest
from app.evaluation.completeness_evaluator import CompletenessEvaluator


def test_complete_answer_scores_high():
    """Test that complete answers score high."""
    evaluator = CompletenessEvaluator(use_llm=False)

    question = "What is the wire limit and what documentation is required?"
    complete_answer = "The wire limit is $5,000 and documentation includes ID verification and source of funds."

    score = evaluator.evaluate_completeness(question, complete_answer)
    assert score >= 0.7, f"Complete answer should score high, got {score}"


def test_partial_answer_scores_low():
    """Test that partial answers score lower."""
    evaluator = CompletenessEvaluator(use_llm=False)

    question = "What is the wire limit and what documentation is required?"
    partial_answer = "The wire limit is $5,000."  # Missing documentation part

    score = evaluator.evaluate_completeness(question, partial_answer)
    assert score < 0.7, f"Partial answer should score lower, got {score}"


def test_multi_part_question():
    """Test evaluation of multi-part questions."""
    evaluator = CompletenessEvaluator(use_llm=False)

    question = "What is the retention period and who must approve exceptions?"
    answer_both = "The retention period is 5 years and exceptions require compliance officer approval."
    answer_one = "The retention period is 5 years."

    score_both = evaluator.evaluate_completeness(question, answer_both)
    score_one = evaluator.evaluate_completeness(question, answer_one)

    assert score_both > score_one, "Answer addressing both parts should score higher"
