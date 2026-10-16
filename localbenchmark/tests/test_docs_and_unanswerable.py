"""Tests for docs.py (document folder) and the unanswerable scorer.
Run with: python -m unittest test_docs_and_unanswerable -v"""
import tempfile
import unittest

from offline_guard import block_network

block_network()
from pathlib import Path

import docs as D
import suite_scoring as S

UNANS = {"id": "acme-x", "domain": "company_policy", "type": "unanswerable", "prompt": "x"}


class Folder(unittest.TestCase):
    def make_folder(self, names_and_words):
        d = tempfile.mkdtemp()
        for name, words in names_and_words:
            Path(d, name).write_text(" ".join(["word"] * words), encoding="utf-8")
        return d

    def test_loads_in_name_order_and_skips_other_files(self):
        d = self.make_folder([("b.md", 10), ("a.txt", 20), ("c.pdf", 5)])
        Path(d, ".hidden.md").write_text("secret", encoding="utf-8")
        docs = D.load_docs(d)
        self.assertEqual([x.id for x in docs], ["a.txt", "b.md"])

    def test_limit_takes_first_n(self):
        d = self.make_folder([(f"doc{i:03d}.md", 5) for i in range(10)])
        self.assertEqual(len(D.load_docs(d, limit=3)), 3)
        self.assertEqual(D.load_docs(d, limit=3)[0].id, "doc000.md")

    def test_ids_keep_requested_order(self):
        d = self.make_folder([("a.md", 5), ("b.md", 5), ("c.md", 5)])
        self.assertEqual([x.id for x in D.load_docs(d, ids=["c.md", "a.md"])], ["c.md", "a.md"])

    def test_unknown_id_is_an_error(self):
        d = self.make_folder([("a.md", 5)])
        with self.assertRaises(ValueError):
            D.load_docs(d, ids=["missing.md"])

    def test_empty_or_missing_folder_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            D.load_docs(tempfile.mkdtemp() + "/nope")
        with self.assertRaises(FileNotFoundError):
            D.load_docs(tempfile.mkdtemp())

    def test_token_estimate_and_fit_report(self):
        d = self.make_folder([(f"d{i}.md", 800) for i in range(100)])
        docs = D.load_docs(d)
        self.assertEqual(docs[0].est_tokens, round(800 * 1.35))
        report = D.fit_report(docs)
        self.assertIn("100 documents", report)
        self.assertIn("120k window: fits", report)     # ~108k tokens fits 120k
        self.assertIn("50k window: does NOT fit", report)

    def test_context_block_labels_each_document(self):
        d = self.make_folder([("policy.md", 3)])
        ctx = D.build_context(D.load_docs(d))
        self.assertTrue(ctx.startswith("### Document: policy.md"))


class Unanswerable(unittest.TestCase):
    def score(self, answer):
        return S.score_item(UNANS, answer)

    def test_declines_pass(self):
        for a in ["The documents do not contain this information.",
                  "I cannot find the parental leave policy in the documents.",
                  "There is no information about that.",
                  "I don't know.",
                  "This is not specified in the provided text."]:
            self.assertEqual(self.score(a).score, 1.0, a)

    def test_invented_answers_fail(self):
        for a in ["Based on the documents, employees get 16 weeks.",
                  "Probably around 20 days.",
                  "The policy allows 12 weeks of leave."]:
            self.assertEqual(self.score(a).score, 0.0, a)

    def test_gold_answer_passes(self):
        self.assertTrue(S.score_item(UNANS, S.gold_answer(UNANS)).passed)

    def test_context_must_be_a_list_of_names(self):
        bad = {"id": "a", "domain": "d", "type": "numeric", "prompt": "p", "gold": 1,
               "context": "policy.md"}
        with self.assertRaises(ValueError):
            S.load_suite(self._write([bad]))

    def test_valid_context_is_accepted(self):
        good = {"id": "a", "domain": "d", "type": "numeric", "prompt": "p", "gold": 1,
                "context": ["policy.md"]}
        self.assertEqual(len(S.load_suite(self._write([good])).items), 1)

    def _write(self, rows):
        import json
        p = Path(tempfile.mkdtemp()) / "s.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return p


if __name__ == "__main__":
    unittest.main()
