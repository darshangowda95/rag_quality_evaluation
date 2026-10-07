"""Evaluation dataset loader - Task 1 implementation.

TASK 1: Implement the dataset loader.

Complete the TODOs below to load evaluation test cases from JSON.
"""

import json
from pathlib import Path
from typing import List

from app.utils.data_classes import EvaluationCase


class EvaluationDatasetLoader:
    """Loads evaluation datasets from JSON files."""

    def load(self, dataset_path: str) -> List[EvaluationCase]:
        """Load evaluation dataset from JSON file.

        Args:
            dataset_path: Path to evaluation_questions.json

        Returns:
            List of EvaluationCase objects

        Raises:
            ValueError: If required fields are missing
            FileNotFoundError: If dataset file not found

        Example:
            >>> loader = EvaluationDatasetLoader()
            >>> cases = loader.load("dataset/evaluation_questions.json")
            >>> len(cases)
            27
            >>> cases[0].question
            'How long must KYC customer identification records be retained?'
        """
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        content = path.read_text(encoding="utf-8")
        data = json.loads(content)

        # Extract test cases
        if "test_cases" not in data:
            raise ValueError("Dataset missing 'test_cases' field")

        test_cases = data["test_cases"]

        # Create EvaluationCase objects
        cases = []
        for item in test_cases:
            # Validate required fields
            required = ["id", "question", "expected_answer", "ground_truth_context_ids"]
            missing = [f for f in required if f not in item]
            if missing:
                raise ValueError(f"Test case {item.get('id', '?')} missing fields: {missing}")

            case = EvaluationCase(
                id=item["id"],
                question=item["question"],
                expected_answer=item["expected_answer"],
                ground_truth_context_ids=item["ground_truth_context_ids"],
                category=item.get("category", "factual_recall"),
                difficulty=item.get("difficulty", "beginner"),
                metadata=item
            )
            cases.append(case)

        return cases

