"""
End-to-end test of --benchmark suite with LM Studio replaced by a fake model.
Run with: python -m unittest test_suite_pipeline -v
"""
import contextlib
import io
import os
import sys
import tempfile
import types
import unittest

from offline_guard import block_network

block_network()
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))          # localbenchmark.py lives one level up

try:  # the real openai package is used when installed; the fake model replaces the client
    import openai  # noqa: F401
except ImportError:
    _stub = types.ModuleType("openai")
    _stub.OpenAI = object
    sys.modules["openai"] = _stub

import pandas as pd  # noqa: E402
import localbenchmark as lb  # noqa: E402
import suite_scoring as SS  # noqa: E402

SUITE_PATH = HERE / "suites" / "northwind_demo.jsonl"
DOCS_PATH = HERE / "company_docs"
SUITE = SS.load_suite(SUITE_PATH)
ITEM_BY_QUESTION = {lb.item_question(it): it for it in SUITE.items}

WRONG_ON = {"nw-02"}                 # answerable, answered wrongly on purpose
INVENTED_ON = {"nw-33"}              # unanswerable, answered with an invented fact


class FakeModel:
    """Answers from gold when the cited documents are in the prompt, else declines."""
    def __init__(self):
        self.loads = 0
        self.calls = 0

    def load(self, base_url, model, context, extra):
        self.loads += 1
        return {"load_config": {"context_length": context}, "instance_id": "fake-1",
                "_benchmark_load_time": 0.1}

    def solve(self, client, model, prompt, filler, max_tokens, system=None):
        self.calls += 1
        assert system == lb.SUITE_SYSTEM_PROMPT
        question = prompt.split("## Question\n", 1)[1]
        item = ITEM_BY_QUESTION[question]
        if item["id"] in INVENTED_ON:
            answer = "Contractors receive 8 weeks of paid leave."
        elif item["id"] in WRONG_ON:
            answer = "12 weeks."
        elif item["type"] == "unanswerable":
            answer = "The documents do not contain this information."
        elif all(f"### Document: {c}" in prompt for c in item.get("context", [])):
            answer = SS.gold_answer(item)
        else:
            answer = "The documents do not contain this information."
        return {"raw": answer, "code": "", "ttft": 0.5, "generation_time": 2.0,
                "decode_time": 1.0, "decode_tps": 40.0, "prompt_tokens": 3000,
                "completion_tokens": 20, "finish_reason": "stop"}


class SuitePipeline(unittest.TestCase):
    def setUp(self):
        self.fake = FakeModel()
        self.tmp = tempfile.mkdtemp()
        self.cwd = os.getcwd()
        os.chdir(self.tmp)
        self.saved = {n: getattr(lb, n) for n in
                      ["check_lm_studio", "determine_model_contexts", "unload_all_models",
                       "load_model", "get_gpu_memory", "OpenAI", "solve_problem", "run_plots"]}
        lb.check_lm_studio = lambda url: None
        lb.determine_model_contexts = lambda base, model, req: [50000]
        lb.unload_all_models = lambda url: None
        lb.load_model = self.fake.load
        lb.get_gpu_memory = lambda: (1000.0, 16000.0)
        lb.OpenAI = lambda **k: None
        lb.solve_problem = self.fake.solve
        lb.run_plots = lambda *a, **k: None

    def tearDown(self):
        for n, f in self.saved.items():
            setattr(lb, n, f)
        os.chdir(self.cwd)

    def run_main(self, *extra):
        argv = ["localbenchmark.py", "--benchmark", "suite", "--suite", str(SUITE_PATH),
                "--docs", str(DOCS_PATH), "--models", "demo/model", "--contexts", "50000"]
        sys.argv = argv + list(extra)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            lb.main()
        return out.getvalue()

    def results(self):
        return pd.read_csv(Path(self.tmp) / "suite" / "suite_results_northwind_demo.csv")

    def test_full_run_scores_all_40_items(self):
        self.run_main()
        r = self.results()
        self.assertEqual(len(r), 40)
        self.assertEqual(int(r["passed"].astype(str).str.lower().eq("true").sum()), 38)

    def test_error_labels(self):
        self.run_main()
        r = self.results().set_index("problem_id")
        self.assertEqual(r.loc["nw-02", "error_type"], "WRONG_ANSWER")
        self.assertEqual(r.loc["nw-33", "error_type"], "INVENTED_ANSWER")
        self.assertTrue(pd.isna(r.loc["nw-01", "error_type"]))

    def test_schema_has_suite_columns(self):
        self.run_main()
        cols = set(self.results().columns)
        for c in ["score", "domain", "item_type", "docs_included", "score_detail", "passed"]:
            self.assertIn(c, cols)

    def test_partial_credit_is_recorded(self):
        self.run_main()
        r = self.results().set_index("problem_id")
        self.assertEqual(r.loc["nw-24", "score"], 1.0)            # both founders named
        self.assertEqual(r.loc["nw-02", "score"], 0.0)

    def test_resume_skips_finished_items_and_model_load(self):
        self.run_main()
        loads, calls = self.fake.loads, self.fake.calls
        self.run_main()
        self.assertEqual(self.fake.loads, loads)    # config complete: model not loaded
        self.assertEqual(self.fake.calls, calls)    # no items re-asked
        self.assertEqual(len(self.results()), 40)

    def test_limited_documents_mark_missing_doc(self):
        self.run_main("--docs-limit", "3")
        r = self.results()
        self.assertFalse(r["docs_included"].astype(str).str.lower().eq("true").all())
        missing = r[r["error_type"] == "MISSING_DOC"]
        self.assertGreater(len(missing), 0)
        self.assertTrue((missing["docs_included"].astype(str).str.lower() == "false").all())

    def test_window_too_small_skips_before_loading(self):
        lb.determine_model_contexts = lambda base, model, req: [1000]
        out = self.run_main()
        self.assertIn("window too small", out)
        self.assertEqual(self.fake.loads, 0)
        self.assertFalse((Path(self.tmp) / "suite" / "suite_results_northwind_demo.csv").exists())

    def test_summary_file_and_overall_numbers(self):
        out = self.run_main()
        summary = Path(self.tmp) / "suite" / "suite_results_northwind_demo_by_domain.csv"
        self.assertTrue(summary.exists())
        self.assertIn("SUITE SUMMARY", out)
        self.assertIn("invented_answers", out)

    def test_missing_suite_argument_stops_cleanly(self):
        sys.argv = ["localbenchmark.py", "--benchmark", "suite", "--models", "demo/model"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            lb.main()
        self.assertIn("--suite", out.getvalue())


if __name__ == "__main__":
    unittest.main()
