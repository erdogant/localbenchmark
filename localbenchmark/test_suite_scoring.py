"""Tests for suite_scoring. Run with: python -m unittest test_suite_scoring -v"""
import json
import tempfile
import unittest
from pathlib import Path

import suite_scoring as S

HERE = Path(__file__).parent
SUITE = S.load_suite(HERE / "suites" / "general_knowledge.jsonl")
BY_ID = {it["id"]: it for it in SUITE.items}

KW = {"type": "keywords", "prompt": "x", "min_score": 1.0,
      "groups": [{"any": ["Chernobyl"]}, {"any": ["Ukraine", "Ukrainian SSR"]}]}
SHORT = {"type": "short", "prompt": "x", "accept": ["Canberra"], "distractors": ["Sydney"]}
NUM = {"type": "numeric", "prompt": "x", "gold": 103000, "rel_tol": 0.05}
MC = {"type": "mc", "prompt": "x", "answer": "B",
      "options": {"A": "Peru", "B": "Mongolia", "C": "Japan", "D": "Norway"}}


class Keywords(unittest.TestCase):
    def test_exact(self):
        r = S.score_item(KW, "Chernobyl, in Ukraine")
        self.assertEqual(r.score, 1.0); self.assertTrue(r.passed)

    def test_alias(self):
        item = {"type": "keywords", "prompt": "x", "groups": [{"any": ["Berlin Wall", "the Wall"]}]}
        self.assertEqual(S.score_item(item, "the Wall came down").score, 1.0)

    def test_typo_gets_partial_credit(self):
        r = S.score_item(KW, "Chernobil, in Ukraine")       # 0.889 similarity
        self.assertTrue(0.9 < r.score < 1.0)
        self.assertFalse(r.passed)                          # min_score is 1.0

    def test_plural_and_stem(self):
        item = {"type": "keywords", "prompt": "x", "groups": [{"any": ["tariff"]}]}
        self.assertEqual(S.score_item(item, "New tariffs were raised").score, 1.0)

    def test_partial_gradient(self):
        item = {"type": "keywords", "prompt": "x", "min_score": 0.5,
                "groups": [{"any": ["a1"]}, {"any": ["b2"]}, {"any": ["c3"]}, {"any": ["d4"]}]}
        r = S.score_item(item, "a1 and b2")
        self.assertEqual(r.score, 0.5); self.assertTrue(r.passed)
        self.assertFalse(S.score_item({**item, "min_score": 0.8}, "a1 and b2").passed)

    def test_negated_concept_gets_no_credit(self):
        self.assertEqual(S.score_item(KW, "Chernobyl is not in Ukraine").score, 0.5)

    def test_weights(self):
        item = {"type": "keywords", "prompt": "x", "groups": [
            {"any": ["alpha"], "weight": 3}, {"any": ["beta"], "weight": 1}]}
        self.assertEqual(S.score_item(item, "alpha").score, 0.75)


class Short(unittest.TestCase):
    def test_correct(self):
        self.assertEqual(S.score_item(SHORT, "Canberra.").score, 1.0)

    def test_negated_distractor_not_penalised(self):
        self.assertEqual(S.score_item(SHORT, "Canberra, not Sydney").score, 1.0)

    def test_no_comma_clause_split(self):
        self.assertEqual(S.score_item(SHORT, "No, Canberra is the capital").score, 1.0)

    def test_hedge_halves_score(self):
        r = S.score_item(SHORT, "Sydney or Canberra")
        self.assertEqual(r.score, 0.5); self.assertFalse(r.passed)

    def test_fuzzy_guard_short_word(self):
        item = {"type": "short", "prompt": "x", "accept": ["Paris"]}
        self.assertEqual(S.score_item(item, "Parish council").score, 0.0)

    def test_fuzzy_threshold_rejects_weak_match(self):
        item = {"type": "short", "prompt": "x", "accept": ["Berlin"]}
        self.assertEqual(S.score_item(item, "Berlim").score, 0.0)

    def test_think_block_is_ignored(self):
        self.assertEqual(S.score_item(SHORT, "<think>Sydney is big</think>Canberra").score, 1.0)


class Numeric(unittest.TestCase):
    def test_within_tolerance(self):
        self.assertEqual(S.score_item(NUM, "about 103,000 square kilometres").score, 1.0)
        self.assertEqual(S.score_item(NUM, "about 103 thousand km2").score, 1.0)

    def test_space_separated_thousands(self):
        r = S.score_item(NUM, "110 000 km2")
        self.assertTrue(0.0 < r.score < 1.0); self.assertFalse(r.passed)

    def test_exact_year(self):
        item = {"type": "numeric", "prompt": "x", "gold": 1989}
        self.assertEqual(S.score_item(item, "1990").score, 0.0)
        self.assertEqual(S.score_item(item, "It was 1989.").score, 1.0)

    def test_percent_and_million(self):
        self.assertEqual(S.score_item({"type": "numeric", "prompt": "x", "gold": 12.4,
                                       "abs_tol": 0.1}, "12.4%").score, 1.0)
        self.assertEqual(S.score_item({"type": "numeric", "prompt": "x", "gold": 2.5e6},
                                      "2.5 million").score, 1.0)

    def test_closest_number_wins(self):
        item = {"type": "numeric", "prompt": "x", "gold": 1989}
        self.assertEqual(S.score_item(item, "Between 1985 and 1989").score, 1.0)

    def test_no_number(self):
        self.assertEqual(S.score_item(NUM, "I do not know").score, 0.0)


class MultipleChoice(unittest.TestCase):
    def test_letter_forms(self):
        for answer in ["B", "(B) Mongolia", "The answer is B.", "B) Mongolia", "answer: b"]:
            self.assertEqual(S.score_item(MC, answer).score, 1.0, answer)

    def test_wrong_letter(self):
        self.assertEqual(S.score_item(MC, "C").score, 0.0)

    def test_option_text_only(self):
        self.assertEqual(S.score_item(MC, "It is Mongolia.").score, 1.0)

    def test_hedge_between_letters(self):
        self.assertEqual(S.score_item(MC, "B or C").score, 0.0)

    def test_article_is_not_a_letter(self):
        self.assertEqual(S.score_item(MC, "A landlocked country: Mongolia").score, 1.0)


class Rubric(unittest.TestCase):
    def test_needs_judge(self):
        r = S.score_item(BY_ID["ev-07"], "Some answer")
        self.assertIsNone(r.score); self.assertIsNone(r.passed)


class Loader(unittest.TestCase):
    def test_reports_bad_lines(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.jsonl"
            p.write_text(json.dumps({"id": "a", "domain": "d", "type": "numeric",
                                     "prompt": "p", "gold": 1}) + "\n"
                         + json.dumps({"id": "b", "domain": "d", "type": "mc",
                                       "prompt": "p", "options": {"A": "x"}, "answer": "Z"}) + "\n"
                         + "{not json\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                S.load_suite(p)
            msg = str(ctx.exception)
            self.assertIn("line 2", msg); self.assertIn("line 3", msg)

    def test_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "dup.jsonl"
            row = json.dumps({"id": "a", "domain": "d", "type": "numeric", "prompt": "p", "gold": 1})
            p.write_text(row + "\n" + row + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                S.load_suite(p)

    def test_example_suite_is_valid_and_gold_passes(self):
        self.assertEqual(len(SUITE.items), 26)
        self.assertEqual(S.check_gold(SUITE), [])

    def test_manifest_is_read(self):
        self.assertEqual(SUITE.name, "general_knowledge")
        self.assertEqual(SUITE.version, "0.1")


# Realistic answers a model might give. Each row: (item id, answer, should pass).
PARAPHRASES = [
    ("ctry-01", "It's Canberra, not Sydney.", True),
    ("ctry-01", "Sydney.", False),
    ("ctry-02", "The landlocked one is Mongolia.", True),
    ("ctry-02", "B", True),
    ("ctry-02", "Peru", False),
    ("ctry-03", "About 100,000 square kilometres.", True),
    ("ctry-03", "About 150,000 square kilometres.", False),
    ("ctry-04", "Pacific to the west, Atlantic to the east and the Arctic to the north.", True),
    ("ctry-04", "The Pacific and the Atlantic.", False),
    ("ctry-05", "Russia is the biggest, bigger than Canada.", True),
    ("pol-01", "Trygve Lie, the Norwegian diplomat.", True),
    ("pol-02", "1989", True),
    ("pol-02", "1991", False),
    ("pol-03", "France, Germany, Italy, Belgium, the Netherlands and Luxembourg.", True),
    ("pol-03", "France, Germany, Italy", False),
    ("pol-05", "The EU and the euro.", True),
    ("ppl-01", "Darwin wrote it.", True),
    ("ppl-02", "Armstrong did it in 1969.", True),
    ("ppl-03", "B", True),
    ("ppl-04", "Mbeki succeeded him.", True),
    ("co-01", "It was founded in 1975.", True),
    ("co-03", "The founders were Steve Jobs and Steve Wozniak.", True),
    ("co-03", "Steve Jobs", False),
    ("co-04", "Microsoft bought it in 2016.", True),
    ("co-04", "Google", False),
    ("ev-01", "1986", True),
    ("ev-02", "It is in Ukraine.", True),
    ("ev-03", "The fall of the Berlin Wall.", True),
    ("ev-04", "The United Nations, founded in 1945.", True),
    ("ev-04", "The League of Nations.", False),
    ("ev-05", "Alliances and militarism.", True),
    ("ev-05", "Imperialism.", False),
    ("ev-06", "It dissolved in December 1991.", True),
]


class Paraphrases(unittest.TestCase):
    def test_realistic_answers(self):
        wrong = []
        for item_id, answer, expected in PARAPHRASES:
            r = S.score_item(BY_ID[item_id], answer)
            if bool(r.passed) != expected:
                wrong.append((item_id, answer, r.score, expected))
        self.assertEqual(wrong, [], f"unexpected results: {wrong}")


if __name__ == "__main__":
    unittest.main()
