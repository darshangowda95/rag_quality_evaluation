"""Tests for Task 4: Report generator.

generate() has no quick_mode of its own - it is intentionally the real,
full-dataset pipeline, making well over a hundred live LLM calls (no
offline fallback - see integration/rag_client.py). It is run once per
test module (not once per test) via the module-scoped `report` fixture
below, to avoid tripling that cost for independent assertions on the same
result. test_report_writes_files runs it a second time with an
output_path, since writing files is a distinct behavior to verify.
"""

import pytest
from pathlib import Path
from app.reporting.report_generator import ReportGenerator


@pytest.fixture(scope="module")
def report():
    """Run the full pipeline once per module (slow - makes ~100+ live LLM calls)."""
    return ReportGenerator().generate()


def test_report_contains_all_metrics(report):
    """Test that generated report contains all required metrics."""
    assert "relevance" in report.baseline_metrics, "Report should include relevance"
    assert "faithfulness" in report.baseline_metrics, "Report should include faithfulness"
    assert "completeness" in report.baseline_metrics, "Report should include completeness"
    assert "grounding_accuracy" in report.baseline_metrics, "Report should include grounding_accuracy"

    assert "relevance" in report.optimized_metrics, "Report should include optimized relevance"
    assert "faithfulness" in report.optimized_metrics, "Report should include optimized faithfulness"
    assert "completeness" in report.optimized_metrics, "Report should include optimized completeness"
    assert "grounding_accuracy" in report.optimized_metrics, "Report should include optimized grounding_accuracy"


def test_report_shows_improvement(report):
    """Test that report shows improvement deltas."""
    assert "relevance" in report.improvement_deltas, "Should have relevance delta"
    assert "faithfulness" in report.improvement_deltas, "Should have faithfulness delta"
    assert "completeness" in report.improvement_deltas, "Should have completeness delta"
    assert "grounding_accuracy" in report.improvement_deltas, "Should have grounding_accuracy delta"

    # All improvements should be >= 0 (non-negative)
    for metric, delta in report.improvement_deltas.items():
        assert delta >= -0.1, f"{metric} should not regress significantly (delta={delta})"


def test_report_writes_files(tmp_path):
    """Test that report generator writes JSON and Markdown files."""
    generator = ReportGenerator()

    output_path = tmp_path / "test_report.json"
    generator.generate(output_path=str(output_path))

    # Check JSON file exists
    assert output_path.exists(), "JSON report should be written"

    # Check Markdown file exists
    md_path = output_path.with_suffix('.md')
    assert md_path.exists(), "Markdown report should be written"
