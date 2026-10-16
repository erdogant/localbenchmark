"""Judge model for open-ended answers. Run with: python -m unittest test_judge -v"""
import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
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
REFERENCE = ("The revolutions spread because Soviet leaders would not send troops to stop them, "
             "the Eastern bloc economies were stagnating, and mass protests in many cities made "
             "the old governments impossible to keep.")


def suite_with_reference(folder):
    """A copy of the general-knowledge suite where the open item has a reference answer.
    The tests do not depend on the copy in the suites folder, which may be older."""
    rows = [json.loads(line) for line in GK.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if row["id"] == "ev-07":
            row["reference"] = REFERENCE
    path = Path(folder) / "general_knowledge.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return path


ITEMS = {it["id"]: it for it in SS.load_suite(GK).items}
BY_QUESTION = {lb.item_question(it): it for it in ITEMS.values() if it["type"] != "rubric"}
OPEN_Q = lb.item_question(ITEMS["ev-07"])
OPEN_ANSWER = "Soviet reluctance to intervene, economic stagnation and mass protest."


class Judge(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        self.tmp = tempfile.mkdtemp()
        os.chdir(self.tmp)
        self.suite_path = suite_with_reference(self.tmp)
        self.loads = []
        self.judge_garbage_once = False
        self.saved = {n: getattr(lb, n) for n in ["check_lm_studio", "determine_model_contexts",
                      "unload_all_models", "load_model", "get_gpu_memory", "OpenAI",
                      "solve_problem", "run_plots"]}
        lb.check_lm_studio = lambda url: None
        lb.determine_model_contexts = lambda base, model, req: [50000]
        lb.unload_all_models = lambda url: None
        lb.load_model = self.load
        lb.get_gpu_memory = lambda: (1000.0, 16000.0)
        lb.OpenAI = lambda **k: None
        lb.run_plots = lambda *a, **k: None
        lb.solve_problem = self.solve

    def tearDown(self):
        for n, f in self.saved.items():
            setattr(lb, n, f)
        os.chdir(self.cwd)

    def load(self, base, model, ctx, extra):
        self.loads.append(model)
        return {"load_config": {"context_length": ctx}, "instance_id": f"{model}-inst",
                "_benchmark_load_time": 0.1}

    def solve(self, client, model, prompt, filler, max_tokens, system=None):
        reply = {"ttft": 0.1, "generation_time": 1.0, "decode_time": 1.0, "decode_tps": 40.0,
                 "prompt_tokens": 80, "completion_tokens": 10, "finish_reason": "stop"}
        if system == SS.JUDGE_SYSTEM_PROMPT:
            answer = prompt.split("<<<\n", 1)[1].split("\n>>>", 1)[0]
            if self.judge_garbage_once and OPEN_ANSWER in answer:   # only the real grading call
                self.judge_garbage_once = False
                return {**reply, "raw": "I cannot decide."}
            score = 2 if "protest" in answer else 0
            return {**reply, "raw": f'{{"score": {score}, "reason": "checked the criteria"}}'}
        if system == lb.SUITE_CLOSED_BOOK_PROMPT:
            question = prompt
            if question == OPEN_Q:
                return {**reply, "raw": OPEN_ANSWER}
            item = BY_QUESTION[question]
            return {**reply, "raw": SS.gold_answer(item)}
        raise AssertionError("unexpected system prompt")

    def command(self, *extra):
        sys.argv = ["localbenchmark", "--benchmark", "suite", "--suite", str(self.suite_path),
                    "--models", "candidate/model", "--contexts", "50000",
                    "--docs", str(Path(self.tmp) / "none")] + list(extra)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            lb.main()
        return out.getvalue()

    def results(self):
        return pd.read_csv(Path(self.tmp) / "suite" / "suite_results_general_knowledge.csv")

    def judged(self):
        return pd.read_csv(Path(self.tmp) / "suite" / "suite_results_general_knowledge_judge.csv")

    def test_default_run_does_not_load_a_judge(self):
        out = self.command()
        self.assertNotIn("judge", " ".join(self.loads))
        self.assertIn("waiting for a judge", out)
        r = self.results().set_index("problem_id")
        self.assertEqual(r.loc["ev-07", "error_type"], "NEEDS_JUDGE")

    def test_judge_scores_the_open_answer_after_the_candidate(self):
        out = self.command("--judge", "judge/model")
        self.assertEqual(self.loads, ["candidate/model", "judge/model"])
        self.assertIn("judge check: reference answer for ev-07 scored 1.0", out)
        j = self.judged()
        self.assertEqual(len(j), 1)
        self.assertEqual(j.iloc[0]["problem_id"], "ev-07")
        self.assertEqual(float(j.iloc[0]["judge_score"]), 1.0)

    def test_judged_score_reaches_the_summary(self):
        self.command("--judge", "judge/model")
        by_domain = pd.read_csv(Path(self.tmp) / "suite" / "suite_results_general_knowledge_by_domain.csv")
        world = by_domain[by_domain["domain"] == "world_events"].iloc[0]
        self.assertEqual(int(world["items"]), 7)          # all 7 world-event items, including ev-07
        self.assertEqual(int(world["passed"]), 7)

    def test_unjudged_open_answer_is_not_in_pass_rates(self):
        self.command()
        by_domain = pd.read_csv(Path(self.tmp) / "suite" / "suite_results_general_knowledge_by_domain.csv")
        world = by_domain[by_domain["domain"] == "world_events"].iloc[0]
        self.assertEqual(int(world["items"]), 6)

    def test_summary_counts_judged_answers(self):
        out = self.command("--judge", "judge/model")
        self.assertNotIn("waiting for a judge", out)
        self.assertIn("SUITE SUMMARY", out)

    def test_resume_does_not_reload_or_rejudge(self):
        self.command("--judge", "judge/model")
        self.loads.clear()
        out = self.command("--judge", "judge/model")
        self.assertIn("Already complete", out)
        self.assertEqual(self.loads, [])
        self.assertEqual(len(self.judged()), 1)

    def test_judge_added_later_to_a_finished_run(self):
        self.command()                       # candidate finished, nothing judged
        self.loads.clear()
        self.command("--judge", "judge/model")
        self.assertEqual(self.loads, ["judge/model"])   # candidate not reloaded
        self.assertEqual(len(self.judged()), 1)

    def test_unparseable_judge_reply_is_retried(self):
        self.judge_garbage_once = True
        out = self.command("--judge", "judge/model")
        self.assertIn("judge check: reference answer for ev-07 scored 1.0", out)
        j = self.judged()
        self.assertEqual(j.iloc[0]["error_type"], "PARSE_ERROR")
        self.command("--judge", "judge/model")
        j = self.judged()
        judged_ok = j[j["error_type"].isna() | (j["error_type"].astype(str) == "")]
        self.assertEqual(len(judged_ok), 1)

    def test_judge_prompt_contains_criteria_and_answer_only(self):
        prompt = SS.build_judge_prompt(ITEMS["ev-07"], OPEN_ANSWER)
        self.assertIn(ITEMS["ev-07"]["rubric"], prompt)
        self.assertIn(OPEN_ANSWER, prompt)
        self.assertNotIn("candidate/model", prompt)


if __name__ == "__main__":
    unittest.main()
