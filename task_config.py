"""Task configuration for Lab 2.4 - RAG Evaluation & Optimization.

Defines the 4 guided tasks. Read by streamlit_app.py to render the Task
Tracker, and by the README's Guided Tasks section.

`requirements` (R1, R5, R6, R7) is this lab's original internal brief
identifier per task, kept for traceability back to the module brief -
always use `number` (or "Task N") for anything user-facing.

The relevance/faithfulness/completeness/grounding scorers (originally
R2/R3/R4) are prefilled, not a guided task - see
evaluation/relevance_scorer.py, faithfulness_checker.py,
completeness_evaluator.py, and grounding_validator.py. They still have
their own tests and are exercised by every other task, but there is no
TODO to implement in them.
"""

TASKS = [
    {
        "id": "task_1",
        "number": 1,
        "title": "Load Evaluation Dataset",
        "requirements": ["R1"],
        "description": (
            "Implement `EvaluationDatasetLoader.load()` in `evaluation/dataset_loader.py`. "
            "Load the 27 banking policy test questions from JSON, validate required fields, "
            "and return structured EvaluationCase objects. This dataset will be used to "
            "systematically measure RAG quality across diverse question types."
        ),
        "lesson_files": ["evaluation/dataset_loader.py"],
        "test": "tests/test_task1_dataset_loader.py",
        "estimated_time": "30 minutes"
    },
    {
        "id": "task_2",
        "number": 2,
        "title": "Optimize Retrieval Configuration",
        "requirements": ["R5"],
        "description": (
            "Implement `RetrievalOptimizer.optimize()` in `optimization/retrieval_optimizer.py`. "
            "Run grid search over chunk_size [256, 512, 1024] and top_k [3, 5, 10] parameters - "
            "chunk_size genuinely changes retrieval, since the mock RAG service splits each "
            "policy passage into chunks of that size before scoring. For each configuration, "
            "query the mock RAG service (integration/rag_client.py) and measure relevance "
            "scores. Return the configuration that maximizes relevance. This automates what "
            "would otherwise be manual trial-and-error tuning."
        ),
        "lesson_files": ["optimization/retrieval_optimizer.py"],
        "test": "tests/test_task2_retrieval_optimizer.py",
        "estimated_time": "30 minutes"
    },
    {
        "id": "task_3",
        "number": 3,
        "title": "Optimize Prompts",
        "requirements": ["R6"],
        "description": (
            "Implement `PromptOptimizer.optimize()` in `optimization/prompt_optimizer.py`. "
            "Test 5 prompt variations (baseline, strict_grounding, citation_focused, "
            "structured_response, risk_aware) from dataset/prompt_templates.json. For each "
            "prompt, run evaluation measuring faithfulness and completeness. Return the prompt "
            "that maximizes combined score. Demonstrates A/B testing for prompt engineering."
        ),
        "lesson_files": ["optimization/prompt_optimizer.py"],
        "test": "tests/test_task3_prompt_optimizer.py",
        "estimated_time": "30 minutes"
    },
    {
        "id": "task_4",
        "number": 4,
        "title": "Generate Evaluation Report",
        "requirements": ["R7"],
        "description": (
            "Implement `ReportGenerator.generate()` in `reporting/report_generator.py`. "
            "Run the complete pipeline: (1) baseline evaluation with default config, "
            "(2) optimization to find best retrieval+prompt settings, (3) optimized evaluation "
            "with best config, (4) comparison showing before/after metrics and improvement "
            "deltas, across all four metrics (relevance, faithfulness, completeness, "
            "grounding_accuracy - the prefilled `_run_evaluation()` helper already collects "
            "all four, so this just passes them through). Generate both JSON (for programmatic "
            "use) and Markdown (for humans) reports. This provides quantitative evidence of "
            "quality improvement for stakeholders."
        ),
        "lesson_files": ["reporting/report_generator.py"],
        "test": "tests/test_task4_report_generator.py",
        "estimated_time": "30 minutes"
    },
]

TASK_GUIDES = {
    "task_1": (
        "Run `pytest tests/test_task1_dataset_loader.py`. Your implementation should load "
        "dataset/evaluation_questions.json and return 27 EvaluationCase objects with validated "
        "required fields (id, question, expected_answer, ground_truth_context_ids)."
    ),
    "task_2": (
        "Use the RAG client wrapper in integration/rag_client.py (a standalone mock service, "
        "no external project needed) to query it with different chunk_size and top_k values. "
        "The grid search pattern is straightforward: nested loops over parameters, measuring "
        "relevance for each combination, tracking the best."
    ),
    "task_3": (
        "Similar to Task 2 but testing prompt variations instead of retrieval configs. Load "
        "prompt templates from dataset/prompt_templates.json, query RAG service with each prompt, "
        "measure faithfulness+completeness, select the prompt with highest combined score. "
        "A/B testing made systematic."
    ),
    "task_4": (
        "Orchestrate the full evaluation pipeline. Run baseline first (default settings), then "
        "run optimizers from Tasks 2-3, then run optimized evaluation, then calculate deltas. "
        "Format as JSON (json.dumps with indent=2) and Markdown table. The report is your "
        "deliverable proving optimization worked."
    ),
}

# Requirement to task mapping for coverage verification. R2/R3/R4 are
# prefilled (see module docstring) and intentionally absent here - they
# have tests but no guided task.
REQUIREMENT_TASK_MAP = {
    "R1": ["task_1"],
    "R5": ["task_2"],
    "R6": ["task_3"],
    "R7": ["task_4"],
}
