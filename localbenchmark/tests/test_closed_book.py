"""Closed-book suites (no documents) run with the command from the user's example.
Run with: python -m unittest test_closed_book -v"""
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
sys.path.insert(0, str(HERE.parent))
try:
    import openai  # noqa: F401
except ImportError:
    _stub = types.ModuleType("openai")
    _stub.OpenAI = object
    sys.modules["openai"] = _stub
import pandas as pd  # noqa: E402
import localbenchmark as lb  # noqa: E402
import suite_scoring as SS  # noqa: E402

GK = HERE / "suites" / "general_knowledge.jsonl"
GK_ITEMS = SS.load_suite(GK).items
BY_QUESTION = {lb.item_question(it): it for it in GK_ITEMS}


class ClosedBook(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        self.tmp = tempfile.mkdtemp()
        os.chdir(self.tmp)
        self.prompts = []
        self.systems = []
        self.saved = {n: getattr(lb, n) for n in ["check_lm_studio", "determine_model_contexts",
                      "unload_all_models", "load_model", "get_gpu_memory", "OpenAI",
                      "solve_problem", "run_plots"]}
        lb.check_lm_studio = lambda url: None
        lb.determine_model_contexts = lambda base, model, req: [50000]
        lb.unload_all_models = lambda url: None
        lb.load_model = lambda base, model, ctx, extra: {
            "load_config": {"context_length": ctx}, "instance_id": "fake", "_benchmark_load_time": 0.1}
        lb.get_gpu_memory = lambda: (1000.0, 16000.0)
        lb.OpenAI = lambda **k: None
        lb.run_plots = lambda *a, **k: None
        lb.solve_problem = self.fake_solve

    def tearDown(self):
        for n, f in self.saved.items():
            setattr(lb, n, f)
        os.chdir(self.cwd)

    def fake_solve(self, client, model, prompt, filler, max_tokens, system=None):
        self.prompts.append(prompt)
        self.systems.append(system)
        question = prompt.split("## Question\n", 1)[-1]   # closed book: prompt is the question alone
        item = BY_QUESTION.get(question) or BY_QUESTION[prompt]
        return {"raw": SS.gold_answer(item) or "A short answer.", "code": "", "ttft": 0.5, "generation_time": 1.0,
                "decode_time": 1.0, "decode_tps": 40.0, "prompt_tokens": 80,
                "completion_tokens": 10, "finish_reason": "stop"}

    def run_command(self):
        # the command from the example, with no document folder present
        argv = ["localbenchmark", "--benchmark", "suite", "--suite", str(GK),
                "--models", "openai/gpt-oss-20b", "--contexts", "50000",
                "--docs", str(Path(self.tmp) / "no-such-folder")]
        sys.argv = argv
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            lb.main()
        return out.getvalue()

    def test_runs_without_documents_and_passes_gold_answers(self):
        out = self.run_command()
        self.assertIn("Closed-book run", out)
        self.assertIn("1 open-ended item(s): answered and saved, not scored", out)
        r = pd.read_csv(Path(self.tmp) / "suite" / "suite_results_general_knowledge.csv")
        self.assertEqual(len(r), len(GK_ITEMS))
        scored = r[r["error_type"].astype(str) != "NEEDS_JUDGE"]
        self.assertTrue(scored["passed"].astype(str).str.lower().eq("true").all())
        self.assertEqual(int((r["error_type"] == "NEEDS_JUDGE").sum()), 1)

    def test_prompt_has_no_document_text(self):
        self.run_command()
        self.assertTrue(self.prompts)
        for p in self.prompts:
            self.assertNotIn("Company documents", p)
        for s in self.systems:
            self.assertEqual(s, lb.SUITE_CLOSED_BOOK_PROMPT)

    def test_document_suite_is_not_closed_book(self):
        self.assertFalse(any("context" not in it for it in SS.load_suite(
            HERE / "suites" / "northwind_demo.jsonl").items))

    def test_scoring_rubric_still_returns_none(self):
        rubric = [it for it in GK_ITEMS if it["type"] == "rubric"][0]
        self.assertIsNone(SS.score_item(rubric, "anything").score)


if __name__ == "__main__":
    unittest.main()
