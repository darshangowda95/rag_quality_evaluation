# Evaluating and Optimizing RAG Quality

## Project Overview

RAG demos stop at "it returns an answer." Production RAG has to prove the
answer is relevant, faithful to the source documents, complete, and
correctly grounded — refusing when it should, answering when it should —
before it reaches a relationship manager or a customer. This lab builds
that proof: a Python evaluation pipeline that scores RAG responses against
a 27-question banking dataset on four metrics, uses those scores to
optimize retrieval and prompt configuration, and generates a before/after
report.

This is Guided Project 4 of Module 2 ("Enterprise AI Foundations &
Retrieval-Augmented Generation") in the Nuvepro Forward Deployed Engineer
program. You implement one function per stage in `practice_solution/`; the
complete `reference_solution/` is included for comparison.

## Project Objectives

- Load a structured evaluation dataset and score responses against four
  prefilled quality metrics: relevance, faithfulness, completeness,
  grounding accuracy
- Optimize retrieval configuration (chunk size, top-k) via grid search —
  chunk size genuinely changes what gets retrieved, not just a logged
  parameter
- A/B test prompt templates against the same metrics
- Validate that out-of-scope questions get refused and in-scope questions
  don't — grounding, not just hallucination-free wording
- Generate a before/after evaluation report, in JSON and Markdown
- See a real RAG system's answers scored end to end, with no offline
  substitute standing in for the live model

## Business Scenario

The architecture review board won't approve production rollout on a demo
that "looks right" — they need measurable proof: does the assistant retrieve
relevant documents, answer faithfully (no hallucinated policy details),
cover every part of a multi-part question, and correctly decline the
questions it has no business answering (investment advice, interest rate
speculation) instead of guessing. Your job is to build that measurement,
then use it to find a better retrieval and prompt configuration than the
one you started with.

**Your role:** build an evaluation framework that scores 27 banking policy
questions (KYC, AML, wire transfers, lending — spanning factual recall,
multi-step, multi-document, edge-case, and out-of-scope queries) on four
metrics, grid-search retrieval settings and A/B-test prompts against those
same metrics, and produce a report showing the improvement.

## Technology Stack

- Python 3.14+
- numpy (cosine similarity for the relevance metric, and for the mock RAG
  client's deterministic retrieval)
- httpx (the live OpenRouter call the mock RAG client's generator makes)
- pytest (the headless verification suite)
- Streamlit (optional UI, not required)

A real OpenRouter API key is required (see
[Generation: Live LLM Required](#generation-live-llm-required)). The four
evaluation metrics themselves stay deterministic (keyword/phrase matching
for faithfulness, completeness and grounding, cosine similarity for
relevance) — it is the thing being evaluated, not the evaluator, that
calls a live model.

## The RAG Service Under Test

This lab evaluates a **standalone mock RAG service**
(`integration/rag_client.py`), not the real service built in Lab 2.3. It
ships its own fixture data (`dataset/ground_truth.json`) and its own
deterministic hash-based retrieval, so the whole pipeline runs with no
external dependency and no other lab's project needs to be present. This is
a deliberate simplification for the evaluation lesson, not a bug: the
metrics, the optimization loops, and the report format are exactly what
you'd point at a real RAG service, and the mock exists so this lab doesn't
require Lab 2.3's project to be present or correctly implemented to run.

Retrieval is deterministic — each policy passage is split into `chunk_size`
chunks and scored by its best-matching chunk, so `chunk_size` genuinely
changes what gets retrieved, not just a number carried through in
metadata. Generation is not simplified away — every question that
retrieves at least one context gets its answer from a real, live
OpenRouter call (see below).

## Generation: Live LLM Required

The mock RAG client's `generate()` always calls a real OpenRouter chat
model — there is no offline fallback. `OPENROUTER_API_KEY` must be set in
`.env` to answer any question that retrieved at least one context; without
it, `RagClient.query()` raises `GenerationError`. A question that retrieves
nothing is refused before generation ever runs, so that path alone needs no
key.

The live call goes under a retry and timeout budget (see `GENERATION` in
`utils/config.py`). Approximate cost: well under $0.10 for a full
27-question evaluation with the default small model
(`openai/gpt-4o-mini`) — an order-of-magnitude estimate; check OpenRouter's
current pricing for whatever model you configure. Tasks 2-4 each make
several live calls per run (one per question per configuration tested), and
the full pipeline (`Running the Project` below) makes on the order of 100.

## Folder Structure

```text
.
├── README.md
├── requirements.txt
├── task_config.py              Guided Tasks (feeds the Task Tracker UI)
├── bootstrap.py                 switches app.* between the two trees
├── verify.py                    headless setup/import check, one line each
├── run_evaluation.py            CLI: baseline / optimize / full pipeline
├── streamlit_app.py             optional task tracker + reports UI
├── .env                          your OpenRouter API key (already set up)
├── dataset/
│   ├── evaluation_questions.json   27 banking policy test cases
│   ├── ground_truth.json           the mock RAG service's retrievable contexts
│   └── prompt_templates.json       5 prompt variants for A/B testing
├── tests/                       pytest suite
├── practice_solution/           YOUR workspace (TODOs in four lesson files)
└── reference_solution/          instructor-complete, identical layout
```

## Folder & File Description

### practice_solution/ (Your Workspace)

| Module | Status | Why |
|--------|--------|-----|
| `integration/rag_client.py` | Study (prefilled) | The standalone mock RAG service — chunk-aware deterministic retrieval, live-LLM generation |
| `evaluation/relevance_scorer.py`, `faithfulness_checker.py`, `completeness_evaluator.py`, `grounding_validator.py` | Study (prefilled) | The four deterministic metrics every task scores against |
| `utils/config.py`, `utils/data_classes.py` | Study (prefilled) | Config loading and shared dataclasses |
| `utils/llm_judge.py` | Study (prefilled) | The live OpenRouter client the mock RAG service's generator uses |
| `utils/text.py` | Study (prefilled) | `is_refusal()` — recognizes a refusal in the live model's own words, not just the fixed string |
| the four lessons in [Guided Tasks](#guided-tasks) | **Lesson (TODO)** | Tasks 1-4 |

**Don't look at `reference_solution/` until you've attempted a task
yourself** — it's one working implementation, not the only correct one; if
your code passes the tests, it's correct even if it differs.

### tests/

One test file per task, plus one per prefilled metric, run headlessly with
`pytest`. Tasks 2-4's tests make live LLM calls — see
[Generation: Live LLM Required](#generation-live-llm-required). See
[Guided Tasks](#guided-tasks) for the per-task commands, or
[Running the Project](#running-the-project) to run the whole suite at once.

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

A `.env` with a real OpenRouter API key is already set up and required for
Tasks 2-4 (see
[Generation: Live LLM Required](#generation-live-llm-required)). If you need
to swap in your own key, edit `OPENROUTER_API_KEY` in `.env` directly.

### Launch the Task Tracker

```bash
streamlit run streamlit_app.py
```

It has three tabs — **Overview** (what to implement, where), **Test Your
Code**, and **Reports** — against whichever implementation
("practice_solution" or "reference_solution") is selected in the sidebar.
Check the **Overview** tab first to see every task before you start.

## Guided Tasks

**Test Your Code** gives every task its own self-contained check: editable
example inputs, a button that calls your function directly, and a verdict —
✅ Correct, ⚠️ Runs but not correct yet, ⏳ Not implemented yet, or ❌
Crashed — plus the actual value returned. Tasks 2-4's checks call the mock
RAG client's live generator on a tiny synthetic input, so each needs
`OPENROUTER_API_KEY` in `.env` but never needs another task done first. A
progress bar shows how many tasks currently verify as correct.

Each task is implement, then test — in order.

### Task 1 - Load Evaluation Dataset

Implement `EvaluationDatasetLoader.load()` in
`evaluation/dataset_loader.py`. Load the banking policy test questions
from JSON, validate required fields, and return structured
`EvaluationCase` objects.

**Test it:** `pytest tests/test_task1_dataset_loader.py -v` — loads all 27
cases from the real dataset, each with `id`, `question`, `expected_answer`,
`ground_truth_context_ids`

**Try it in the UI:** edit the 2-question JSON and see `load()` return it
as structured cases — no live LLM call, no other task needed.

### Task 2 - Optimize Retrieval Configuration

Implement `RetrievalOptimizer.optimize()` in
`optimization/retrieval_optimizer.py`. Run grid search over `chunk_size`
(256, 512, 1024) and `top_k` (3, 5, 10), querying the mock RAG service for
each combination and measuring relevance. `chunk_size` genuinely changes
retrieval — the mock service splits each policy passage into chunks of
that size and scores by the best-matching chunk, so a smaller chunk_size
can surface a passage a larger one would dilute into a weaker match.
Return the configuration that maximizes relevance.

**Test it:** `pytest tests/test_task2_retrieval_optimizer.py -v` — tests
multiple configs, returns a `best_config` with a positive `chunk_size` and
`top_k`, and an `optimized_score` no worse than the baseline

**Try it in the UI:** edit the question and run it in `quick_mode` (one
configuration) — needs `OPENROUTER_API_KEY`, since the mock RAG client's
generator is a live call.

### Task 3 - Optimize Prompts

Implement `PromptOptimizer.optimize()` in
`optimization/prompt_optimizer.py`. Test the 5 prompt variations in
`dataset/prompt_templates.json` (baseline, strict_grounding,
citation_focused, structured_response, risk_aware), measuring faithfulness
and completeness for each. Return the prompt that maximizes the combined
score.

**Test it:** `pytest tests/test_task3_prompt_optimizer.py -v` — tests at
least 3 prompts, returns a `best_config` with a non-empty `prompt_text`,
and an `optimized_score` no worse than the baseline

**Try it in the UI:** edit the question and run it in `quick_mode` (only
the baseline prompt) — needs `OPENROUTER_API_KEY`, same reason as Task 2.

### Task 4 - Generate Evaluation Report

Implement `ReportGenerator.generate()` in `reporting/report_generator.py`.
Run the complete pipeline — baseline evaluation, optimization (Tasks 2 and
3), optimized evaluation, then a comparison of before/after metrics across
all four metrics (relevance, faithfulness, completeness,
`grounding_accuracy` — the prefilled `_run_evaluation()` helper already
collects all four, so this is a pass-through, not new scoring logic) —
and write both a JSON report (for programmatic use) and a Markdown report
(for humans).

**Test it:** `pytest tests/test_task4_report_generator.py -v` — the
report's `baseline_metrics`/`optimized_metrics` carry all four metrics,
`improvement_deltas` doesn't regress, and both the `.json` and `.md` files
are written. `generate()` has no `quick_mode` of its own — it is
intentionally the real pipeline — so this test alone makes dozens of live
LLM calls and is the slowest in the suite.

**Try it in the UI:** edit the 2-question JSON — the check bounds the
retrieval grid to one configuration so it stays fast, but still makes
several real live calls, so it needs `OPENROUTER_API_KEY` and can take
under a minute. The **Reports** tab's "Generate New Report" button runs the
unbounded version against the real 27-question dataset instead.

### Worked Examples

The same inputs the UI's checks use, with real output captured from an
actual live run against `reference_solution` — live LLM wording (and so
`faithfulness`/`completeness`) will vary slightly run to run; `relevance`
will not, since it scores the deterministic retrieval, not the generated
text:

| Task | Input | Observed output |
|------|-------|-----------------|
| 1 | `load()` on a 2-case JSON file | 2 `EvaluationCase` objects, each with `id`, `question`, `expected_answer`, `ground_truth_context_ids` |
| 2 | `RetrievalOptimizer.optimize([case], quick_mode=True)`, question `"What is KYC?"` | `best_config={"chunk_size": 512, "top_k": 5}`; `baseline_score=optimized_score=0.0198` (quick_mode tests 1 config, so no actual choice is made) |
| 2, variant | `retrieve()` directly, question `"How long must KYC records be retained?"`, `chunk_size=512` (no splitting - passages here are under 512 chars) | top hit `wire_limits_002`, score `0.5458` — the wrong document |
| 2, variant | Same question, `chunk_size=256` (splits most passages in two) | top hit becomes `aml_patterns_001` (`0.6937`), with the actually-correct `kyc_retention_001` close behind (`0.6894`) — a smaller chunk size changed what ranked first |
| 3 | `PromptOptimizer.optimize([case], quick_mode=True)`, question `"What is KYC?"` | `best_config={"prompt_id": "baseline", "prompt_text": "..."}`; `baseline_score=optimized_score=0.5` (quick_mode tests only the baseline prompt) |
| 4 | `ReportGenerator.generate()` on a 2-question dataset, grid bounded to 1 config | `baseline_metrics={"relevance": 0.198, "faithfulness": 0.0, "completeness": 1.0, "grounding_accuracy": 1.0}`; `optimized_metrics` identical (same single config/prompt tested); `improvement_deltas` all `0.0` |
| 4, variant | `GroundingValidator().validate("out_of_scope", retrieved, answer)` for `"How do I invest in stocks through the bank?"` | the mock service still retrieves 5 contexts (the crude hash embedding doesn't discriminate out-of-scope questions by retrieval alone), but the live model's answer reads *"The provided context does not contain information on how to invest..."* — `is_refusal()` recognizes that wording, so `grounded=True` even though retrieval alone did not refuse |

Tasks 2 and 3's `quick_mode` row shows `baseline == optimized` because it
deliberately tests only one configuration/prompt — that proves the
orchestration wiring, not that optimization helps. Run the real `pytest`
suite or the **Reports** tab against the full 27-question dataset and the
real grid to see an actual best-vs-baseline comparison.

## Running the Project

Once every task passes its own test above, confirm all four at once, then
run the whole pipeline:

```bash
# All tasks at once, headlessly (makes ~100+ live LLM calls, several minutes):
pytest

# Headless PASS/FAIL summary:
python verify.py

# Full pipeline (baseline -> optimize -> report):
python run_evaluation.py --full
```

A full run prints the baseline and optimized metrics, the improvement
deltas, and writes `reports/evaluation_report_<timestamp>.json` and `.md`.

## Troubleshooting

**A test fails with an `AssertionError` on an unimplemented task**
Expected — the stubs return an empty/incomplete result (e.g. `[]`) rather
than raising, so an unfinished task fails its test with a normal
assertion, not a special exception. The TODO comments in the file name
what's missing. Switch to `reference_solution` to see a working version.

**`GenerationError: OPENROUTER_API_KEY is required...`**
There is no offline fallback for the mock RAG client's answer generation.
Confirm `OPENROUTER_API_KEY` is set in `.env`. A question that retrieves no
contexts never reaches the LLM at all, so that path alone needs no key.

**`GenerationError: live LLM call failed after 3 attempt(s)...`**
The key is set but the call itself failed every retry. Check the printed
`[llm_judge] live attempt N/3 failed: ...` lines above the error for the
real cause (a `401` means the key itself is invalid, a `402` means the
account is out of credit, a timeout means a network/firewall issue) — the
exception message only tells you to look there, not what went wrong.

**`grounding_accuracy` looks wrong for an out-of-scope question**
Retrieval itself does not refuse out-of-scope questions — the mock
service's hash-based embedding isn't discriminating enough to tell an
out-of-scope question apart from an in-scope one, so it still retrieves 5
contexts either way. What matters is whether the live model's *answer*
reads as a refusal; `utils/text.py`'s `is_refusal()` recognizes common
refusal phrasing ("does not contain", "unable to", etc.), not just the
exact fixed string. If a live answer refuses in wording that helper
doesn't recognize, that's a real gap to extend, not a bug to work around.

**Results don't match some published baseline number**
There isn't a fixed target number to match — what actually matters is that
your tests pass and that optimization measurably improves the score over
the baseline configuration, not that you hit a specific percentage.
Retrieval and relevance scoring are deterministic given the same retrieved
contexts, but the live model's answer wording is not, and the
faithfulness/completeness checks score that wording — so `pytest -k
test_report_shows_improvement` can occasionally fail on a small sample
(15 questions) even on a correct implementation, if one metric happens to
score noticeably lower on this run's wording than last run's. A failure
here on an otherwise-passing suite is normal live-model variance, not a
sign your code is wrong; re-run it if you want to confirm.

## Next Steps

The measurement discipline here — define a metric, optimize against it,
prove the improvement — applies to any AI system, not just this RAG
pipeline. In Module 3, you apply the same evaluation approach to a live
agent, this time with real production traffic instead of a fixed
27-question set.
