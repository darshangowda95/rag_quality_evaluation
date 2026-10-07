"""Tests for the (prefilled) relevance scorer."""

import pytest
from app.evaluation.relevance_scorer import RelevanceScorer


def test_relevance_score_in_range():
    """Test that relevance scores are in [0.0, 1.0]."""
    scorer = RelevanceScorer(use_llm=False)

    question = "What is the wire transfer limit?"
    contexts = ["Daily wire transfer limit is $5,000 for domestic transfers."]

    score = scorer.score_relevance(question, contexts)
    assert 0.0 <= score <= 1.0, f"Score {score} not in valid range [0.0, 1.0]"


def test_relevant_scores_higher():
    """Test that relevant documents score higher than irrelevant ones."""
    scorer = RelevanceScorer(use_llm=False)

    question = "What is the wire transfer limit?"

    relevant_context = ["The daily wire transfer limit for retail customers is $5,000."]
    irrelevant_context = ["The bank's headquarters is located in New York City."]

    relevant_score = scorer.score_relevance(question, relevant_context)
    irrelevant_score = scorer.score_relevance(question, irrelevant_context)

    assert relevant_score > irrelevant_score, \
        f"Relevant score ({relevant_score}) should be higher than irrelevant ({irrelevant_score})"


def test_irrelevant_scores_lower():
    """Test that completely irrelevant documents score low."""
    scorer = RelevanceScorer(use_llm=False)

    question = "What is the KYC retention period?"
    contexts = ["The weather today is sunny with a chance of rain."]

    score = scorer.score_relevance(question, contexts)
    assert score < 0.5, f"Irrelevant context scored too high: {score}"
