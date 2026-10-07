"""Optional UI: task tracker, self-check, and the full report pipeline.

    streamlit run streamlit_app.py

OPTIONAL: every task is verifiable headlessly with `pytest` (see the
README). This app adds a "Test Your Code" tab where every task is checked
in isolation, on tiny synthetic inputs built into the check itself, and a
"Reports" tab that runs the real full pipeline end to end.

Tasks 2-4 call the mock RAG client's `generate()`, which makes a live
OpenRouter call with no offline fallback - `OPENROUTER_API_KEY` must be
set in `.env` for those checks to pass.

Nothing in the lab depends on this file. It is prefilled; you do not edit
it as part of any task.
"""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

import bootstrap  # noqa: E402
from task_config import TASKS  # noqa: E402

st.set_page_config(page_title="RAG Quality Evaluation Lab", page_icon="📊",
                   layout="wide")


def get_module(impl: str, dotted: str):
    """(Re)import one `app.<dotted>` module against the chosen tree."""
    bootstrap.use_tree(impl)
    return importlib.import_module(f"app.{dotted}")


# Verdict states, in the order a task moves through them:
NOT_IMPLEMENTED, INCORRECT, ERROR, PASS = "not_implemented", "incorrect", "error", "pass"

_BADGE = {
    PASS: ("success", "✅ Correct"),
    INCORRECT: ("warning", "⚠️ Runs, but not correct yet"),
    NOT_IMPLEMENTED: ("info", "⏳ Not implemented yet"),
    ERROR: ("error", "❌ Crashed"),
}


def render_verdict(status: str, detail: str, value=None) -> None:
    kind, label = _BADGE[status]
    getattr(st, kind)(f"{label} - {detail}")
    if value is not None:
        if isinstance(value, (dict, list)):
            st.json(value)
        else:
            st.code(str(value))


# -----------------------------------------------------------------------
# One self-contained checker per task. Each builds its own tiny input, so
# testing task N never requires any other task to be done first. Tasks
# 2-4 call the mock RAG client's generate(), a live LLM call with no
# offline fallback - those raise GenerationError (shown as "Crashed")
# without a real OPENROUTER_API_KEY in .env.
# -----------------------------------------------------------------------

def check_task_1(impl: str, questions_json: str):
    """Task 1: EvaluationDatasetLoader.load() on a tiny synthetic JSON file."""
    try:
        loader_mod = get_module(impl, "evaluation.dataset_loader")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "try_it.json"
            path.write_text(questions_json, encoding="utf-8")
            loader = loader_mod.EvaluationDatasetLoader()
            cases = loader.load(str(path))
    except NotImplementedError as e:
        return NOT_IMPLEMENTED, str(e), None
    except Exception as e:  # noqa: BLE001
        return ERROR, f"{type(e).__name__}: {e}", None

    if not cases:
        return NOT_IMPLEMENTED, "load() returned no cases", cases
    value = [{"id": c.id, "question": c.question,
              "ground_truth_context_ids": c.ground_truth_context_ids} for c in cases]
    checks = {
        "returned 2 cases": len(cases) == 2,
        "each case has id/question/expected_answer/ground_truth_context_ids": all(
            hasattr(c, "id") and hasattr(c, "question")
            and hasattr(c, "expected_answer") and hasattr(c, "ground_truth_context_ids")
            for c in cases),
    }
    if all(checks.values()):
        return PASS, f"loaded {len(cases)} cases with all required fields", value
    failed = [name for name, ok in checks.items() if not ok]
    return INCORRECT, "failed: " + ", ".join(failed), value


def _make_case(question: str):
    MockCase = type("EvaluationCase", (), {})
    case = MockCase()
    case.id = "try_it_001"
    case.question = question
    return case


def check_task_2(impl: str, question: str):
    """Task 2: RetrievalOptimizer.optimize(), quick_mode=True, one synthetic case."""
    try:
        optimizer_mod = get_module(impl, "optimization.retrieval_optimizer")
        optimizer = optimizer_mod.RetrievalOptimizer()
        result = optimizer.optimize([_make_case(question)], quick_mode=True)
    except NotImplementedError as e:
        return NOT_IMPLEMENTED, str(e), None
    except Exception as e:  # noqa: BLE001
        return ERROR, f"{type(e).__name__}: {e}", None

    if result is None:
        return NOT_IMPLEMENTED, "optimize() returned None", None
    value = {"best_config": result.best_config, "baseline_score": result.baseline_score,
              "optimized_score": result.optimized_score, "num_configs_tested": len(result.all_results)}
    checks = {
        "tested at least 1 configuration": len(result.all_results) >= 1,
        "best_config has a positive chunk_size": result.best_config.get("chunk_size", 0) > 0,
        "best_config has a positive top_k": result.best_config.get("top_k", 0) > 0,
        "optimized_score is not worse than baseline": result.optimized_score >= result.baseline_score,
    }
    if all(checks.values()):
        return PASS, "optimize() returns a best_config no worse than baseline", value
    failed = [name for name, ok in checks.items() if not ok]
    return INCORRECT, "failed: " + ", ".join(failed), value


def check_task_3(impl: str, question: str):
    """Task 3: PromptOptimizer.optimize(), quick_mode=True, one synthetic case."""
    try:
        optimizer_mod = get_module(impl, "optimization.prompt_optimizer")
        optimizer = optimizer_mod.PromptOptimizer()
        result = optimizer.optimize([_make_case(question)], quick_mode=True)
    except NotImplementedError as e:
        return NOT_IMPLEMENTED, str(e), None
    except Exception as e:  # noqa: BLE001
        return ERROR, f"{type(e).__name__}: {e}", None

    if result is None:
        return NOT_IMPLEMENTED, "optimize() returned None", None
    value = {"best_config": result.best_config, "baseline_score": result.baseline_score,
              "optimized_score": result.optimized_score, "num_prompts_tested": len(result.all_results)}
    checks = {
        "tested at least 1 prompt": len(result.all_results) >= 1,
        "best_config has a non-empty prompt_text": bool(result.best_config.get("prompt_text")),
        "optimized_score is not worse than baseline": result.optimized_score >= result.baseline_score,
    }
    if all(checks.values()):
        return PASS, "optimize() returns a best prompt no worse than baseline", value
    failed = [name for name, ok in checks.items() if not ok]
    return INCORRECT, "failed: " + ", ".join(failed), value


def check_task_4(impl: str, questions_json: str):
    """Task 4: ReportGenerator.generate(), on a tiny 2-question dataset and a
    bounded grid (1 chunk_size x 1 top_k), so this stays fast even though
    generate() has no quick_mode of its own by design.

    Only one bootstrap.use_tree() call here (via get_module, once) - a
    second call would drop and re-import app.utils.config again, handing
    back a different Config class than the one report_generator.py's own
    `from app.utils.config import Config` resolves to, so patching the
    first one's attributes would silently do nothing.
    """
    try:
        config_mod = get_module(impl, "utils.config")
        Config = config_mod.Config
        generator_mod = importlib.import_module("app.reporting.report_generator")

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "try_it_questions.json"
            dataset_path.write_text(questions_json, encoding="utf-8")

            original = {
                "EVAL_DATASET_PATH": Config.EVAL_DATASET_PATH,
                "CHUNK_SIZES": Config.CHUNK_SIZES,
                "TOP_K_VALUES": Config.TOP_K_VALUES,
            }
            Config.EVAL_DATASET_PATH = str(dataset_path)
            Config.CHUNK_SIZES = [Config.DEFAULT_CHUNK_SIZE]
            Config.TOP_K_VALUES = [Config.DEFAULT_TOP_K]
            try:
                generator = generator_mod.ReportGenerator()
                report = generator.generate()
            finally:
                for key, val in original.items():
                    setattr(Config, key, val)
    except NotImplementedError as e:
        return NOT_IMPLEMENTED, str(e), None
    except Exception as e:  # noqa: BLE001
        return ERROR, f"{type(e).__name__}: {e}", None

    if report is None:
        return NOT_IMPLEMENTED, "generate() returned None", None
    value = {"baseline_metrics": report.baseline_metrics, "optimized_metrics": report.optimized_metrics,
              "improvement_deltas": report.improvement_deltas}
    checks = {
        "baseline_metrics has all 4 metrics": set(report.baseline_metrics) >= {"relevance", "faithfulness", "completeness", "grounding_accuracy"},
        "optimized_metrics has all 4 metrics": set(report.optimized_metrics) >= {"relevance", "faithfulness", "completeness", "grounding_accuracy"},
        "improvement_deltas has all 4 metrics": set(report.improvement_deltas) >= {"relevance", "faithfulness", "completeness", "grounding_accuracy"},
    }
    if all(checks.values()):
        return PASS, "generate() returns baseline + optimized metrics and deltas", value
    failed = [name for name, ok in checks.items() if not ok]
    return INCORRECT, "failed: " + ", ".join(failed), value


# -----------------------------------------------------------------------
# Compact test block per task: minimal inputs (pre-filled, editable), one
# button, one verdict. The verdict is cached in session_state (keyed by
# task + impl) so it stays on screen after you click a different task's
# button, instead of vanishing on the next rerun.
# -----------------------------------------------------------------------

def run_and_render(key: str, impl: str, label: str, compute) -> None:
    state_key = f"verdict_{key}_{impl}"
    if st.button(label, key=f"btn_{key}_{impl}"):
        st.session_state[state_key] = compute()
    if state_key in st.session_state:
        render_verdict(*st.session_state[state_key])


_DEFAULT_QUESTIONS_JSON = json.dumps({
    "test_cases": [
        {"id": "try_001", "question": "How long must KYC records be retained?",
         "expected_answer": "Five years.", "ground_truth_context_ids": ["kyc_retention_001"]},
        {"id": "try_002", "question": "What is the daily wire transfer limit?",
         "expected_answer": "250,000 USD.", "ground_truth_context_ids": ["wire_limit_001"]},
    ]
}, indent=2)


def block_task_1(impl: str, number: int) -> None:
    st.caption("Testing: `EvaluationDatasetLoader.load()` on a 2-question JSON "
               "file you can edit - no real dataset file or other task needed.")
    questions_json = st.text_area("evaluation_questions.json content", height=220,
                                   value=_DEFAULT_QUESTIONS_JSON, key="t1_json")
    run_and_render("task_1", impl, f"Test Task {number}",
                   lambda: check_task_1(impl, questions_json))


def block_task_2(impl: str, number: int) -> None:
    st.caption("Testing: `RetrievalOptimizer.optimize(quick_mode=True)` on one "
               "synthetic question - calls the mock RAG client's live LLM "
               "generator, so this needs OPENROUTER_API_KEY in .env.")
    question = st.text_input("question", value="What is KYC?", key="t2_q")
    run_and_render("task_2", impl, f"Test Task {number}",
                   lambda: check_task_2(impl, question))


def block_task_3(impl: str, number: int) -> None:
    st.caption("Testing: `PromptOptimizer.optimize(quick_mode=True)` on one "
               "synthetic question - calls the mock RAG client's live LLM "
               "generator, so this needs OPENROUTER_API_KEY in .env.")
    question = st.text_input("question", value="What is KYC?", key="t3_q")
    run_and_render("task_3", impl, f"Test Task {number}",
                   lambda: check_task_3(impl, question))


def block_task_4(impl: str, number: int) -> None:
    st.caption("Testing: `ReportGenerator.generate()` on a 2-question dataset "
               "you can edit, with the retrieval grid bounded to 1 config so "
               "this stays fast. Makes several live LLM calls (needs "
               "OPENROUTER_API_KEY in .env) - may take under a minute.")
    questions_json = st.text_area("evaluation_questions.json content", height=220,
                                   value=_DEFAULT_QUESTIONS_JSON, key="t4_json")
    run_and_render("task_4", impl, f"Test Task {number}",
                   lambda: check_task_4(impl, questions_json))


BLOCKS = {
    "task_1": block_task_1, "task_2": block_task_2,
    "task_3": block_task_3, "task_4": block_task_4,
}


def _task_passed(task_id: str, impl: str) -> bool:
    return st.session_state.get(f"verdict_{task_id}_{impl}", (None,))[0] == PASS


st.title("📊 Lab 2.4: RAG Quality Evaluation & Optimization")
impl = st.sidebar.radio(
    "Implementation", ["practice_solution", "reference_solution"],
    help="practice_solution runs your code; reference_solution runs the answer.")
st.sidebar.caption("Tasks 2-4 call the mock RAG client's live LLM generator - "
                   "OPENROUTER_API_KEY must be set in .env; there is no offline fallback.")

tab_overview, tab_test, tab_reports = st.tabs(
    ["Overview", "Test Your Code", "Reports"])

with tab_overview:
    st.caption("This is the one place that lists what to implement, where - "
               "\"Test Your Code\" only tests it.")
    for task in TASKS:
        with st.expander(f"Task {task['number']} - {task['title']}"):
            st.write(task["description"])
            st.caption("File(s): " + ", ".join(f"`{f}`" for f in task["lesson_files"]))
            st.caption("Test: `" + task["test"] + "`")

with tab_test:
    st.caption("Each check below is self-contained - it builds its own tiny "
               "input and calls your function directly, so you can test any "
               "task in any order, regardless of what else is done.")
    progress_slot = st.empty()

    for task in TASKS:
        with st.expander(f"Task {task['number']} - {task['title']}"):
            BLOCKS[task["id"]](impl, task["number"])

    # Rendered into a slot reserved above the tasks, but filled last, so it
    # reflects this run's button click instead of lagging one click behind.
    passed = sum(1 for t in TASKS if _task_passed(t["id"], impl))
    progress_slot.progress(passed / len(TASKS),
                           text=f"{passed} / {len(TASKS)} tasks passing against {impl}")

with tab_reports:
    st.caption("Runs the real full pipeline against the real 27-question "
               "dataset: baseline eval, retrieval optimization, prompt "
               "optimization, optimized eval, then a before/after report. "
               "Makes ~100+ live LLM calls - may take a few minutes.")

    reports_dir = Path("reports")

    if st.button("🚀 Generate New Report", use_container_width=True):
        with st.spinner("Running full pipeline (this makes many live LLM "
                         "calls and may take a few minutes)..."):
            try:
                bootstrap.use_tree(impl)
                from app.reporting.report_generator import ReportGenerator
                from datetime import datetime

                reports_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = reports_dir / f"evaluation_report_{timestamp}.json"

                generator = ReportGenerator()
                report = generator.generate(output_path=str(output_path))

                if report is None:
                    st.warning("A task is not implemented yet: "
                               "ReportGenerator.generate() returned None.")
                else:
                    st.success(f"✅ Report generated: {output_path.name}")
                    st.rerun()
            except NotImplementedError as exc:
                st.warning(f"A task is not implemented yet: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ Error: {exc}")

    st.markdown("---")

    if reports_dir.exists():
        report_files = sorted(reports_dir.glob("evaluation_report_*.json"), reverse=True)
        if report_files:
            selected_report = st.selectbox(
                "Select Report", report_files, format_func=lambda p: p.name)

            with open(selected_report) as f:
                report_data = json.load(f)

            st.subheader("📊 Report Summary")

            st.markdown("**Baseline Performance:**")
            baseline = report_data["baseline_metrics"]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Relevance", f"{baseline['relevance']:.3f}")
            col2.metric("Faithfulness", f"{baseline['faithfulness']:.3f}")
            col3.metric("Completeness", f"{baseline['completeness']:.3f}")
            col4.metric("Grounding Accuracy", f"{baseline['grounding_accuracy']:.3f}")

            st.markdown("**Optimized Performance:**")
            optimized = report_data["optimized_metrics"]
            deltas = report_data["improvement_deltas"]

            def fmt_delta(metric):
                delta = deltas[metric]
                pct = (delta / baseline[metric] * 100) if baseline[metric] else 0.0
                return f"{delta:+.3f} ({pct:+.1f}%)"

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Relevance", f"{optimized['relevance']:.3f}", fmt_delta("relevance"))
            col2.metric("Faithfulness", f"{optimized['faithfulness']:.3f}", fmt_delta("faithfulness"))
            col3.metric("Completeness", f"{optimized['completeness']:.3f}", fmt_delta("completeness"))
            col4.metric("Grounding Accuracy", f"{optimized['grounding_accuracy']:.3f}", fmt_delta("grounding_accuracy"))

            st.markdown("---")
            st.markdown("**Optimization Details:**")
            st.json(report_data["optimization_details"])

            md_file = selected_report.with_suffix(".md")
            if md_file.exists():
                with st.expander("📝 View Markdown Report"):
                    st.markdown(md_file.read_text())
        else:
            st.info("No reports yet. Click 'Generate New Report' above.")
    else:
        st.info("No reports yet. Click 'Generate New Report' above.")
