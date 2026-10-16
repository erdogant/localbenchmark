"""
Scoring for non-coding test suites (countries, politics, people, companies, world events).

No LLM judge. Answers are scored with normalized, fuzzy and negation-aware
matching. Every score comes with a `detail` dict, so each result can be audited.

Item types
  mc        multiple choice: a letter, "(B) text", "answer is B", or the option text
  numeric   a number within a relative or absolute tolerance, with partial credit
  short     a short fact: any accepted answer; distractors halve the score if also named
  keywords  weighted concept groups, each with aliases; partial credit per group
  rubric    open answer: reserved for a judge model, not scored here
  unanswerable  company QA: the documents do not contain the answer; the model must say so
"""

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

FUZZY_THRESHOLD = 0.88   # minimum similarity for a fuzzy phrase match
FUZZY_MIN_CHARS = 6      # shorter aliases must match exactly (Paris is not Parish)
NEGATION_WINDOW = 3      # words before a match that flip it to "not"
NEGATORS = {"not", "no", "never", "without", "neither", "nor"}
THINK_RE = re.compile(r"<think>.*?</think>|<think>.*\Z", re.S | re.I)
MIN_SCORE_DEFAULT = {"mc": 1.0, "numeric": 1.0, "short": 1.0, "keywords": 0.8,
                     "unanswerable": 1.0}
DECLINE_RE = re.compile(
    r"\b(do|does|did) not (contain|mention|say|state|specify|include|provide|cover|address)\b"
    r"|\bnot (mentioned|stated|specified|covered|addressed|available|in the (documents|text|context|provided))\b"
    r"|\bcannot (find|determine|answer|be determined|be found)\b|\bcan ?not (find|determine|answer)\b"
    r"|\bno (information|mention|record|details?)\b|\bunable to (find|determine|answer)\b"
    r"|\bthere is no (information|mention|record|answer)\b|\bnot (known|specified) in\b"
    r"|\bi (don[\u2019']?t|do not) know\b|\bi(['\u2019]m| am) not sure\b",
    re.I)

_LETTER_PATTERNS = [
    re.compile(r"^\s*\(?([A-Ea-e])\)?(?:[.):]|\s*$)"),              # "B", "B)", "(B)", "B."
    re.compile(r"(?i:answer|option|choice)\s*(?:is|:)?\s*\(?([A-Ea-e])\)?(?!\w)"),
]
NUM_RE = re.compile(
    r"(?<![\w.])(-?\d{1,3}(?:[ ,\u00a0]\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)"
    r"\s*(thousand|million|billion|trillion|bn|k|%)?(?!\w)", re.I)
SCALES = {"thousand": 1e3, "k": 1e3, "million": 1e6, "billion": 1e9,
          "bn": 1e9, "trillion": 1e12, "%": 1.0}


@dataclass
class ScoreResult:
    score: Optional[float]     # 0..1, or None when the item needs a judge
    passed: Optional[bool]
    method: str
    detail: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Text normalization and matching
# ----------------------------------------------------------------------------

def strip_think(text):
    return THINK_RE.sub("", text or "").strip()


def _fold(text):
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def _stem(tok):
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokens(text):
    text = re.sub(r"['\u2019]s\b", "", _fold(text))  # drop possessive 's
    return [_stem(t) for t in re.findall(r"[^\W_]+", text)]


def _clauses(answer):
    """Split on punctuation, so negation and matches do not cross clause boundaries."""
    parts = [tokens(p) for p in re.split(r"[.;!?,:\n]+", answer)]
    return [p for p in parts if p]


def _occurrences(alias_toks, clause):
    """Yield (strength, start) for each place the alias occurs in one clause."""
    n = len(alias_toks)
    if n == 0 or not clause:
        return
    alias_str = " ".join(alias_toks)
    fuzzy_ok = len(alias_str) >= FUZZY_MIN_CHARS
    widths = range(max(1, n - 1), n + 2) if fuzzy_ok else [n]
    for w in widths:
        for i in range(len(clause) - w + 1):
            window = clause[i:i + w]
            if w == n and window == alias_toks:
                yield 1.0, i
            elif fuzzy_ok:
                ratio = SequenceMatcher(None, alias_str, " ".join(window)).ratio()
                if ratio >= FUZZY_THRESHOLD:
                    yield ratio, i


def _negated(clause, start):
    return any(t in NEGATORS for t in clause[max(0, start - NEGATION_WINDOW):start])


def _best_match(aliases, clauses):
    """aliases: list of (text, tokens). Returns (strength 0..1, matched text or None)."""
    best, matched = 0.0, None
    for text, alias_toks in aliases:
        for clause in clauses:
            for strength, start in _occurrences(alias_toks, clause):
                if strength > best and not _negated(clause, start):
                    best, matched = strength, text
    return best, matched


# ----------------------------------------------------------------------------
# Scorers
# ----------------------------------------------------------------------------

def score_keywords(item, answer):
    clauses = _clauses(answer)
    groups = item["groups"]
    total = sum(float(g.get("weight", 1.0)) for g in groups)
    got, detail = 0.0, []
    for g in groups:
        w = float(g.get("weight", 1.0))
        strength, matched = _best_match([(a, tokens(a)) for a in g["any"]], clauses)
        got += w * strength
        detail.append({"concept": g.get("concept", g["any"][0]),
                       "strength": round(strength, 3), "matched": matched})
    return ScoreResult(got / total, None, "keywords", {"groups": detail})


def score_short(item, answer):
    clauses = _clauses(answer)
    score, matched = _best_match([(a, tokens(a)) for a in item["accept"]], clauses)
    detail = {"matched": matched}
    dist, dmatched = _best_match([(d, tokens(d)) for d in item.get("distractors", [])], clauses)
    if dist >= FUZZY_THRESHOLD:
        score *= 0.5  # the answer also names a distractor, which reads as hedging
        detail["hedged_with"] = dmatched
    return ScoreResult(score, None, "short", detail)


def parse_numbers(text):
    values = []
    for m in NUM_RE.finditer(text):
        raw = m.group(1).replace(",", "").replace(" ", "").replace("\u00a0", "")
        unit = (m.group(2) or "").lower()
        values.append(float(raw) * SCALES.get(unit, 1.0))
    return values


def score_numeric(item, answer):
    gold = float(item["gold"])
    tol = max(float(item.get("abs_tol", 0.0)), float(item.get("rel_tol", 0.0)) * abs(gold))
    candidates = parse_numbers(answer)
    if not candidates:
        return ScoreResult(0.0, None, "numeric", {"reason": "no number found"})
    best = min(candidates, key=lambda v: abs(v - gold))
    err = abs(best - gold)
    if err <= tol + 1e-12:
        score = 1.0
    elif tol > 0:  # near miss: linear credit, zero at five times the tolerance
        score = max(0.0, 1.0 - (err - tol) / (4 * tol))
    else:
        score = 0.0
    return ScoreResult(score, None, "numeric",
                       {"chosen": best, "gold": gold, "tolerance": tol,
                        "candidates": candidates[:10]})


def score_mc(item, answer):
    options = item["options"]
    letters = {m.group(1).upper() for pat in _LETTER_PATTERNS for m in pat.finditer(answer)}
    letters = {l for l in letters if l in options}
    if len(letters) > 1:
        return ScoreResult(0.0, None, "mc", {"reason": "hedged", "letters": sorted(letters)})
    if letters:
        chosen, how = letters.pop(), "letter"
    else:
        clauses = _clauses(answer)
        named = {}
        for letter, text in options.items():
            strength, _ = _best_match([(text, tokens(text))], clauses)
            if strength >= FUZZY_THRESHOLD:
                named[letter] = strength
        if len(named) != 1:
            return ScoreResult(0.0, None, "mc",
                               {"reason": "no unique option named", "options_named": sorted(named)})
        chosen, how = next(iter(named)), "option text"
    score = 1.0 if chosen == item["answer"] else 0.0
    return ScoreResult(score, None, "mc", {"chosen": chosen, "by": how})


def score_unanswerable(item, answer):
    """Full marks only if the model says the documents do not contain the answer."""
    m = DECLINE_RE.search(answer)
    return ScoreResult(1.0 if m else 0.0, None, "unanswerable",
                       {"declined": bool(m), "phrase": m.group(0) if m else None})


SCORERS = {"mc": score_mc, "numeric": score_numeric,
           "short": score_short, "keywords": score_keywords,
           "unanswerable": score_unanswerable}


def score_item(item, raw_answer):
    """Score one answer. Returns a ScoreResult; `passed` uses the item's min_score."""
    kind = item["type"]
    if kind == "rubric":
        return ScoreResult(None, None, "rubric", {"reason": "needs a judge model (not in phase 1)"})
    answer = strip_think(raw_answer)
    if not answer:
        return ScoreResult(0.0, False, kind, {"reason": "empty answer"})
    result = SCORERS[kind](item, answer)
    min_score = float(item.get("min_score", MIN_SCORE_DEFAULT[kind]))
    result.passed = result.score is not None and result.score >= min_score - 1e-9
    return result


# ----------------------------------------------------------------------------
# Suite files
# ----------------------------------------------------------------------------

REQUIRED = {"mc": ("prompt", "options", "answer"), "numeric": ("prompt", "gold"),
            "short": ("prompt", "accept"), "keywords": ("prompt", "groups"),
            "rubric": ("prompt", "rubric"), "unanswerable": ("prompt",)}


@dataclass
class Suite:
    name: str
    version: str
    path: str
    sha256: str
    items: list


def _check_item(item):
    problems = []
    for key in ("id", "type", "domain"):
        if key not in item:
            problems.append(f"missing '{key}'")
    kind = item.get("type")
    if kind not in REQUIRED:
        return problems + [f"unknown type {kind!r}"]
    for key in REQUIRED[kind]:
        if key not in item:
            problems.append(f"'{kind}' item needs '{key}'")
    if problems:
        return problems
    if kind == "mc":
        if item["answer"] not in item["options"]:
            problems.append("answer is not one of the option letters")
    if kind == "numeric":
        try:
            float(item["gold"])
        except (TypeError, ValueError):
            problems.append("gold must be a number")
    if kind == "short" and not item["accept"]:
        problems.append("accept must not be empty")
    if kind == "keywords":
        for g in item["groups"]:
            if not g.get("any"):
                problems.append("every keyword group needs a non-empty 'any' list")
    if "min_score" in item and not 0.0 <= float(item["min_score"]) <= 1.0:
        problems.append("min_score must be between 0 and 1")
    if "context" in item:
        ctx = item["context"]
        if not isinstance(ctx, list) or not all(isinstance(c, str) for c in ctx):
            problems.append("context must be a list of document file names")
    return problems


def load_suite(path):
    """Load a .jsonl suite. Raises ValueError listing every invalid line."""
    path = Path(path)
    raw = path.read_bytes()
    items, errors, seen = [], [], set()
    for n, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {n}: invalid JSON ({e.msg})")
            continue
        problems = _check_item(item)
        if item.get("id") in seen:
            problems.append("duplicate id")
        seen.add(item.get("id"))
        if problems:
            errors.append(f"line {n} ({item.get('id')}): " + "; ".join(problems))
        else:
            items.append(item)
    if errors:
        raise ValueError("invalid suite:\n  " + "\n  ".join(errors))
    manifest = path.with_name(path.stem + ".manifest.json")
    meta = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    return Suite(name=meta.get("name", path.stem), version=str(meta.get("version", "0")),
                 path=str(path), sha256=hashlib.sha256(raw).hexdigest(), items=items)


def gold_answer(item):
    """A correct answer written the way a model would write it."""
    kind = item["type"]
    if kind == "mc":
        return f"{item['answer']}) {item['options'][item['answer']]}"
    if kind == "numeric":
        return f"The answer is {item['gold']}."
    if kind == "short":
        return f"The answer is {item['accept'][0]}."
    if kind == "keywords":
        return "; ".join(g["any"][0] for g in item["groups"])
    if kind == "unanswerable":
        return "The documents do not contain this information."
    return None


def check_gold(suite):
    """Every item must pass when given its own gold answer. Returns the failures."""
    failures = []
    for item in suite.items:
        gold = gold_answer(item)
        if gold is None:
            continue
        result = score_item(item, gold)
        if not result.passed:
            failures.append((item["id"], result.score, result.detail))
    return failures


JUDGE_SYSTEM_PROMPT = (
    "You grade answers strictly against the given criteria. You do not reward length or style. "
    "You reply with JSON only.")


def build_judge_prompt(item, answer):
    """Prompt for a judge model. The judge sees the question, the criteria and the answer only."""
    return (
        f"Question:\n{item['prompt']}\n\n"
        f"Criteria:\n{item['rubric']}\n\n"
        f"Answer to grade:\n<<<\n{answer}\n>>>\n\n"
        "Score the answer:\n"
        "2 = meets the criteria\n"
        "1 = partly meets the criteria\n"
        "0 = does not meet the criteria\n"
        'Reply with JSON only, for example {"score": 1, "reason": "one short sentence"}.')


def parse_judge_reply(text):
    """(score in {0.0, 0.5, 1.0}, reason). Raises ValueError if the reply has no usable score."""
    text = strip_think(text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            score = int(obj["score"])
            if score in (0, 1, 2):
                return score / 2, str(obj.get("reason", "")).strip()
        except (ValueError, KeyError, TypeError):
            pass
    fallback = re.search(r'score"?\s*[:=]\s*([012])\b', text)
    if fallback:
        return int(fallback.group(1)) / 2, ""
    raise ValueError("no score in the judge reply")


def check_context(suite, folder):
    """(item id, file name) for every 'context' entry that is not in the document folder."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"document folder does not exist: {folder.resolve()}")
    present = {p.name for p in folder.iterdir() if p.is_file()}
    return sorted({(it["id"], c) for it in suite.items
                   for c in it.get("context", []) if c not in present})


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Validate a suite and check its gold answers.")
    ap.add_argument("suite", help="path to a .jsonl suite file")
    ap.add_argument("--docs", default=None, help="document folder: check every 'context' file exists")
    args = ap.parse_args()
    suite = load_suite(args.suite)
    print(f"{suite.name} v{suite.version}: {len(suite.items)} items, sha256 {suite.sha256[:12]}")
    print("by domain:", dict(Counter(i["domain"] for i in suite.items)))
    print("by type:  ", dict(Counter(i["type"] for i in suite.items)))
    failures = check_gold(suite)
    if failures:
        print("GOLD CHECK FAILED (fix these items; the answer key does not score full marks):")
        for item_id, score, detail in failures:
            print(f"  {item_id}: score {score}  {detail}")
        raise SystemExit(1)
    print("gold check: every scored item passes its own gold answer")
    if args.docs:
        if not Path(args.docs).is_dir():
            print(f"MISSING DOCUMENTS: the folder {Path(args.docs).resolve()} does not exist.")
            print("  Create it and copy the document files into it.")
            raise SystemExit(1)
        missing = check_context(suite, args.docs)
        if missing:
            found = [p.name for p in Path(args.docs).iterdir() if p.is_file()] if Path(args.docs).is_dir() else []
            print(f"MISSING DOCUMENTS: {len(missing)} references not found in {Path(args.docs).resolve()}")
            if not found:
                print("  The folder is empty or does not exist. Copy the document files into it.")
            print("  Files in the folder:", ", ".join(sorted(found)) or "(none)")
            for item_id, name in missing:
                print(f"  {item_id}: {name}")
            raise SystemExit(1)
        print(f"context check: every cited document exists in {args.docs}")


if __name__ == "__main__":
    main()
