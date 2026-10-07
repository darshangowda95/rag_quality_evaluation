"""Implementation-tree selector for Lab 2.4 RAG Evaluation.

PROVIDED — you do not need to edit this file.

Both `practice_solution/` and `reference_solution/` have identical layouts and
import each other's modules through the neutral namespace `app.*` (for example
`from app.evaluation.dataset_loader import EvaluationDatasetLoader`). This shim
points that namespace at whichever tree you want to run.

Order of precedence:
    1. an explicit argument to `use_tree(...)`
    2. the RAG_EVAL environment variable
    3. "practice_solution" (your workspace)

`run_evaluation.py`, `verify.py` and test files all call `use_tree()` before
importing anything under `app.*`.
"""

from __future__ import annotations

import importlib
import os
import sys

VALID_TREES = ("reference_solution", "practice_solution")


def use_tree(tree_name: str | None = None) -> str:
    """Alias `app` to the requested solution tree and return the tree name.

    Args:
        tree_name (str | None): "reference_solution" or "practice_solution".
            Falls back to RAG_EVAL, then "practice_solution".

    Returns:
        str: The tree name that `app` now points at.
    """
    tree_name = tree_name or os.getenv("RAG_EVAL", "practice_solution")
    if tree_name not in VALID_TREES:
        raise ValueError(
            f"Unknown solution tree {tree_name!r}. Choose one of {VALID_TREES}.")

    # Drop any previously aliased app.* modules so switching trees is clean.
    for name in [n for n in sys.modules if n == "app" or n.startswith("app.")]:
        del sys.modules[name]

    package = importlib.import_module(tree_name)
    sys.modules["app"] = package
    return tree_name


def active_tree() -> str:
    """Return the tree name `app` currently points at, or "" if unset."""
    package = sys.modules.get("app")
    return getattr(package, "__name__", "") if package is not None else ""
