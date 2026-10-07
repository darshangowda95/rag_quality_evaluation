"""Tests for the (prefilled) grounding validator."""

import pytest
from app.evaluation.grounding_validator import GroundingValidator


def test_out_of_scope_refusal_is_correctly_grounded():
    """Test that refusing an out-of-scope question counts as grounded."""
    validator = GroundingValidator()

    answer = "I cannot answer this question based on the available policy documents."
    is_grounded = validator.validate("out_of_scope", [], answer)
    assert is_grounded, "Refusing an out-of-scope question should be correctly grounded"


def test_out_of_scope_live_wording_is_recognized():
    """Test that a live model's own refusal wording is still recognized."""
    validator = GroundingValidator()

    answer = "The provided context does not include any information about this topic."
    is_grounded = validator.validate("out_of_scope", ["some unrelated context"], answer)
    assert is_grounded, "A live-model refusal in its own words should still count as grounded"


def test_out_of_scope_answered_anyway_is_not_grounded():
    """Test that answering an out-of-scope question (not refusing) is ungrounded."""
    validator = GroundingValidator()

    answer = "You can invest in stocks by opening a brokerage account."
    is_grounded = validator.validate("out_of_scope", ["some context"], answer)
    assert not is_grounded, "Answering an out-of-scope question should not be grounded"


def test_in_scope_answer_is_correctly_grounded():
    """Test that answering an in-scope question (not refusing) is grounded."""
    validator = GroundingValidator()

    answer = "KYC records must be retained for 5 years after account closure."
    is_grounded = validator.validate("factual_recall", ["KYC retention context"], answer)
    assert is_grounded, "Answering an in-scope question should be correctly grounded"


def test_in_scope_wrongly_refused_is_not_grounded():
    """Test that refusing an in-scope question (that should be answered) is ungrounded."""
    validator = GroundingValidator()

    answer = "I cannot answer this question based on the available policy documents."
    is_grounded = validator.validate("factual_recall", ["KYC retention context"], answer)
    assert not is_grounded, "Wrongly refusing an in-scope question should not be grounded"
