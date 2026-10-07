"""Data structures for RAG evaluation.

PROVIDED - These data classes are used throughout the evaluation pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class EvaluationCase:
    """A single test case for RAG evaluation."""
    id: str
    question: str
    expected_answer: str
    ground_truth_context_ids: List[str]
    category: str = "factual_recall"
    difficulty: str = "beginner"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalConfig:
    """Configuration for RAG retrieval."""
    chunk_size: int = 512
    top_k: int = 5
    similarity_threshold: float = 0.3


@dataclass
class RAGResponse:
    """Response from RAG system."""
    question: str
    answer: str
    retrieved_contexts: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a single response."""
    relevance_score: float  # 0.0-1.0
    is_faithful: bool
    faithfulness_evidence: str
    completeness_score: float  # 0.0-1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Result of optimization process."""
    best_config: Dict[str, Any]
    baseline_score: float
    optimized_score: float
    improvement: float
    all_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    timestamp: str
    baseline_metrics: Dict[str, float]
    optimized_metrics: Dict[str, float]
    improvement_deltas: Dict[str, float]
    optimization_details: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
