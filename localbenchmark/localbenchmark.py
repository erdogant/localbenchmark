# LM Studio's documented load API does not (as far as I know) expose K/V cache
# quantization. Check the current docs; if/when a parameter exists, put it in
# load_params, e.g. {"label": "q8", "load_params": {"<param>": "q8_0"}}.
# Until then, only "default" is honest. Extra entries with empty load_params
# would just repeat identical runs.


import argparse
import csv
import json
import math
import os
import re
import struct
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path

import pandas as pd
import requests
from openai import OpenAI

try:
    from . import llm_plots
except Exception as e:
    import llm_plots

# =============================================================================
# CONFIGURATION
# =============================================================================
CACHE_CONFIGS = [{"label": "default", "load_params": {}}]
# Reasoning models burn tokens on thinking; 2048 truncates them.
DEFAULT_MAX_TOKENS = 4096
TEMPERATURE = 0.0
CODE_TIMEOUT = 10
LOAD_TIMEOUT = 900
REQUEST_TIMEOUT = 900
CHARS_PER_TOKEN = 3.0  # conservative estimate for code filler

MIN_TPS = 10.0
SET_MAX_ITERS = 3 # If k times below MIN_TPS. Then break.

# Speed probe: measures decode/prefill speed at controlled numbers of tokens
# in the KV cache, so "window size" and "tokens actually used" can be separated.
PROBE_TOKENS = [0, 2_000, 8_000, 16_000, 32_000, 64_000, 100_000, 150_000]
PROBE_MAX_TOKENS = 300
PROBE_TASK = ("Write the integers from 1 to 150 in order, separated by single spaces. Output only the numbers.")

PROBE_FIELDS = [
    "timestamp", "model", "server_model", "context_requested",
    "context_actual", "cache_label", "target_tokens", "rep",
    "prompt_tokens", "completion_tokens", "ttft", "decode_time",
    "prefill_tps", "decode_tps", "gpu_used_mb", "gpu_total_mb",
    "gpu_after_load_mb", "finish_reason", "error",
]

FIELDS = [
    "timestamp", "experiment_id", "model", "server_model", "instance_id",
    "context_requested", "context_actual", "fill_ratio", "cache_label",
    "problem_id", "passed", "error_type", "finish_reason",
    "ttft", "generation_time", "decode_time", "execution_time",
    "prompt_tokens", "completion_tokens", "decode_tokens_per_sec",
    "load_time", "gpu_memory_mb", "gpu_total_mb", "error", "code",
]


# =============================================================================
# UTILITIES
# =============================================================================

def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_name(value):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value))


def get_gpu_memory():
    """Sum memory over ALL GPUs. Returns (used_mb, total_mb) or (None, None)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()
        used = total = 0.0
        for line in out:
            u, t = [float(x.strip()) for x in line.split(",")]
            used += u
            total += t
        return used, total
    except Exception:
        return None, None


# =============================================================================
# LM STUDIO
# =============================================================================

def api(base_url, path):
    return base_url.rstrip("/") + path


def check_lm_studio(base_url):
    try:
        requests.get(api(base_url, "/api/v1/models"), timeout=5).raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Could not connect to LM Studio at {base_url}. "
            f"Is the server running?\n{e}"
        )


def get_lm_models(base_url):
    r = requests.get(api(base_url, "/api/v1/models"), timeout=10)
    r.raise_for_status()
    return r.json()


def unload_all_models(base_url):
    instances = []
    for m in get_lm_models(base_url).get("models", []):
        for inst in m.get("loaded_instances", []):
            instances.append(inst.get("id"))

    if not instances:
        return

    print(f"Unloading {len(instances)} instance(s)...")
    for instance_id in instances:
        r = requests.post(
            api(base_url, "/api/v1/models/unload"),
            json={"instance_id": instance_id},
            timeout=LOAD_TIMEOUT,
        )
        r.raise_for_status()
    time.sleep(2)  # let VRAM free up


def load_model(base_url, model, context, extra_params):
    payload = {
        "model": model,
        "context_length": context,
        "flash_attention": True,
        "offload_kv_cache_to_gpu": True,
        "echo_load_config": True,
        **extra_params,
    }

    print(f"\nLoading {model} @ ctx {context:,} ...")
    start = time.perf_counter()
    r = requests.post(api(base_url, "/api/v1/models/load"),
                      json=payload, timeout=LOAD_TIMEOUT)
    elapsed = time.perf_counter() - start

    if r.status_code != 200:
        raise RuntimeError(f"Load failed ({r.status_code}):\n{r.text}")

    data = r.json()
    data["_benchmark_load_time"] = elapsed
    print(f"Load time: {elapsed:.1f}s")
    print(json.dumps(data.get("load_config", {}), indent=2))
    return data


def get_actual_context(load_response):
    ctx = load_response.get("load_config", {}).get("context_length")
    if ctx is None:
        raise RuntimeError(
            "No 'context_length' in load_config:\n"
            + json.dumps(load_response, indent=2)
        )
    return int(ctx)


def determine_model_contexts(base_url, model, requested):
    """Clamp requested contexts to the model's maximum (max tested once)."""
    info = next(
        (m for m in get_lm_models(base_url).get("models", [])
         if m.get("key") == model),
        None,
    )
    if info is None:
        raise RuntimeError(f"Model '{model}' not found in LM Studio.")

    max_ctx = info.get("max_context_length")
    if max_ctx is None:
        raise RuntimeError(f"No max_context_length for '{model}'.")
    max_ctx = int(max_ctx)

    valid = {c for c in requested if c <= max_ctx}
    if any(c > max_ctx for c in requested):
        valid.add(max_ctx)

    valid = sorted(valid)
    print(f"{model}: max ctx {max_ctx:,} -> testing {valid}")
    return valid


# =============================================================================
# CONTEXT FILLER (this is what actually exercises the context window)
# =============================================================================

_CORPUS = None


def get_corpus(min_chars):
    """Unrelated Python source (the stdlib) used as realistic filler."""
    global _CORPUS
    if _CORPUS is not None and len(_CORPUS) >= min_chars:
        return _CORPUS

    parts, total = [], 0
    for p in sorted(Path(sysconfig.get_paths()["stdlib"]).glob("*.py")):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        chunk = f"# ===== {p.name} =====\n{text}\n"
        parts.append(chunk)
        total += len(chunk)
        if total >= min_chars:
            break

    _CORPUS = "".join(parts)
    return _CORPUS


def make_filler(chars, offset=0):
    if chars <= 0:
        return ""
    corpus = get_corpus(chars * 2)
    if len(corpus) < chars:  # corpus too small: repeat it
        corpus = corpus * (chars // max(1, len(corpus)) + 1)
    start = offset % max(1, len(corpus) - chars)
    return corpus[start:start + chars]


# =============================================================================
# BENCHMARK DATA / CODE HANDLING
# =============================================================================

def load_humaneval():
    from datasets import load_dataset
    return load_dataset("openai/openai_humaneval", split="test").to_pandas()

def extract_code(response):
    # Remove reasoning blocks (closed, or unclosed because of truncation).
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE)
    response = re.sub(r"<think>.*\Z", "", response, flags=re.DOTALL | re.IGNORECASE)

    blocks = re.findall(r"```(?:python|py)?\s*\n?(.*?)```", response,
                        flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        with_def = [b for b in blocks if "def " in b]
        pool = with_def or blocks
        return max(pool, key=len).strip()
    return response.strip()


def build_program(code, row):
    """
    Build the file to execute.

    The prompt is placed first (with a stub body) so that imports and helper
    functions from the prompt exist, then the model's code redefines the
    entry point. If the model returned only a function body, it is attached
    to the prompt's signature instead.
    """
    prompt = row["prompt"]
    entry = row["entry_point"]

    if re.search(rf"def\s+{re.escape(entry)}\s*\(", code):
        body = prompt.rstrip() + "\n    pass\n\n" + code
    else:
        if code and not code[0].isspace():
            code = "\n".join("    " + line for line in code.splitlines())
        body = prompt.rstrip() + "\n" + code

    # HumanEval's `test` only DEFINES check(); it must be called.
    return body + "\n\n" + row["test"] + f"\n\ncheck({entry})\n"


def execute_program(program, timeout=CODE_TIMEOUT):
    """
    Runs generated code in a throw-away temp dir.
    NOTE: this is not a real sandbox. Use Docker for untrusted models.
    """
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "solution.py"
        path.write_text(program, encoding="utf-8")
        try:
            r = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True, text=True, timeout=timeout,
                cwd=tmp, stdin=subprocess.DEVNULL,
            )
            return {
                "passed": r.returncode == 0,
                "execution_time": time.perf_counter() - start,
                "stderr": r.stderr[-2000:],
            }
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "execution_time": time.perf_counter() - start,
                "stderr": "TIMEOUT",
            }


# =============================================================================
# COMPANY / KNOWLEDGE SUITES: documents in the prompt, deterministic scoring
# =============================================================================
#
# Full-context mode: every selected document goes into the prompt for every
# question. Each item is scored with suite_scoring.score_item. Results go to
# their own CSV (suite_results_<suite>.csv) with extra columns for the score,
# domain, item type, and whether the item's cited documents were included.

# Suite mode needs suite_scoring.py and docs.py in the same package folder as this
# file. If they are missing, the other benchmarks still work; suite mode stops with a message.
try:
    from . import suite_scoring as SS
    from . import docs as DOCS
except ImportError:
    try:
        import suite_scoring as SS   # running this file directly from its folder
        import docs as DOCS
    except ImportError:
        SS = DOCS = None

SUITE_SYSTEM_PROMPT = (
    "You answer questions about one company using only the documents provided. "
    "If the documents do not contain the answer, say that the documents do not "
    "contain this information. Do not guess. Answer briefly.")
SUITE_CLOSED_BOOK_PROMPT = (
    "Answer the question. Be brief. If you are not sure, give your best answer.")
SUITE_QUESTION_TOKENS = 300   # estimate for the question text and template
SUITE_EXTRA_FIELDS = ["score", "domain", "item_type", "docs_included", "score_detail"]
SUITE_FIELDS = FIELDS + SUITE_EXTRA_FIELDS


def item_question(item):
    text = str(item["prompt"])
    if item.get("type") == "mc":
        options = "\n".join(f"{k}) {v}" for k, v in sorted(item["options"].items()))
        text += "\n\nOptions:\n" + options + "\n\nAnswer with the option letter and its text."
    return text


def build_suite_prompt(docs_context, item):
    return (f"## Company documents\n\n{docs_context}\n\n"
            f"## Question\n{item_question(item)}")


def row_to_item(row):
    """A DataFrame row back to a suite item, without the NaN that pandas adds for missing keys."""
    return {k: v for k, v in row.to_dict().items()
            if not (isinstance(v, float) and math.isnan(v))}


def prepare_suite(args):
    """
    Load the suite and, if it cites documents, the document folder.
    A suite without any 'context' is a closed-book test: no documents, the model answers
    from its own knowledge. Returns (suite, df, docs, docs_context).
    """
    suite = SS.load_suite(args.suite)
    closed_book = bool(getattr(args, "closed_book", False)) or not any(
        "context" in it for it in suite.items)
    args.closed_book = closed_book

    rubric = [it for it in suite.items if it.get("type") == "rubric"]
    if rubric and not getattr(args, "judge", None):
        # Make it explicit that rubric items are being skipped when no judge is provided
        print(f"Skipping {len(rubric)} rubric item(s): these are open-ended and need a judge model.")
        print(f"{len(rubric)} open-ended item(s): answered and saved, not scored. "
              "Use --judge MODEL to score them.")
    elif rubric:
        print(f"{len(rubric)} open-ended item(s) will be scored by the judge {args.judge}.")
    items = list(suite.items)

    if closed_book:
        docs, docs_context = [], ""
        print("Closed-book run: the suite cites no documents; the model answers from its own knowledge.")
    else:
        docs = DOCS.load_docs(args.docs, limit=args.docs_limit, ids=args.docs_ids)
        missing_files = SS.check_context(suite, args.docs)
        if missing_files:
            raise ValueError(f"suite cites documents that are not in {args.docs}: "
                             + ", ".join(f"{i}:{c}" for i, c in missing_files[:5]))
        docs_context = DOCS.build_context(docs)

    selected = {d.id for d in docs}
    rows = []
    for it in items:
        included = all(c in selected for c in it.get("context", []))
        rows.append({**it, "task_id": it["id"], "docs_included": included})
    df = pd.DataFrame(rows)
    return suite, df, docs, docs_context


def run_suite_experiment(df, meta, args, results_file, docs_context, force_rerun=False):
    experiment_id = experiment_id_for(
        meta["model"], meta["context_requested"], meta["cache_label"], args.fill)
    print(f"\n{'=' * 80}\nEXPERIMENT {experiment_id}\n{'=' * 80}")

    client = OpenAI(base_url=api(args.base_url, "/v1"), api_key="lm-studio",
                    timeout=REQUEST_TIMEOUT, max_retries=0)
    completed = set() if force_rerun else load_completed(results_file)
    slow_streak = 0

    for n, (_, row) in enumerate(df.iterrows()):
        pid = row["task_id"]
        if (experiment_id, pid) in completed:
            print(f"[{n + 1}/{len(df)}] {pid} ... SKIP")
            continue
        print(f"[{n + 1}/{len(df)}] {pid} ... ", end="", flush=True)

        item = row_to_item(row)
        base = {
            "timestamp": timestamp(),
            "experiment_id": experiment_id,
            "model": meta["model"],
            "server_model": meta["server_model"],
            "instance_id": meta["instance_id"],
            "context_requested": meta["context_requested"],
            "context_actual": meta["context_actual"],
            "fill_ratio": 0.0,
            "cache_label": meta["cache_label"],
            "problem_id": pid,
            "load_time": meta["load_time"],
            "domain": item.get("domain"),
            "item_type": item.get("type"),
            "docs_included": bool(item.get("docs_included")),
        }

        try:
            closed = getattr(args, "closed_book", False)
            prompt = item_question(item) if closed else build_suite_prompt(docs_context, item)
            system = SUITE_CLOSED_BOOK_PROMPT if closed else SUITE_SYSTEM_PROMPT
            llm = solve_problem(client, meta["server_model"], prompt, "",
                                args.max_tokens, system=system)
        except Exception as e:
            print("LLM ERROR")
            gpu_used, gpu_total = get_gpu_memory()
            append_row(results_file, SUITE_FIELDS, {
                **base, "passed": False, "error_type": "LLM_ERROR", "score": None,
                "gpu_memory_mb": gpu_used, "gpu_total_mb": gpu_total,
                "error": str(e)[:2000], "code": ""})
            continue

        result = SS.score_item(item, llm["raw"])
        gpu_used, gpu_total = get_gpu_memory()
        passed = bool(result.passed)

        if result.score is None:          # open-ended: the judge scores it later
            error_type, passed = "NEEDS_JUDGE", None
        elif passed:
            error_type = ""
        elif not base["docs_included"]:
            error_type = "MISSING_DOC"
        elif llm["finish_reason"] == "length":
            error_type = "TRUNCATED"
        elif not llm["raw"].strip():
            error_type = "EMPTY_RESPONSE"
        elif item.get("type") == "unanswerable":
            error_type = "INVENTED_ANSWER"
        else:
            error_type = "WRONG_ANSWER"

        tps = llm["decode_tps"]
        score_text = "-" if result.score is None else f"{result.score:.2f}"
        status = "ANSWERED" if result.score is None else ("PASS" if passed else "FAIL")
        print(f"{status}  score {score_text}"
              f"  ({llm['prompt_tokens']} prompt tok, {tps or 0:.1f} tok/s)")

        append_row(results_file, SUITE_FIELDS, {
            **base,
            "passed": passed,
            "error_type": error_type,
            "score": result.score,
            "score_detail": json.dumps(result.detail, ensure_ascii=False, default=str),
            "finish_reason": llm["finish_reason"],
            "ttft": llm["ttft"],
            "generation_time": llm["generation_time"],
            "decode_time": llm["decode_time"],
            "execution_time": None,
            "prompt_tokens": llm["prompt_tokens"],
            "completion_tokens": llm["completion_tokens"],
            "decode_tokens_per_sec": tps,
            "gpu_memory_mb": gpu_used,
            "gpu_total_mb": gpu_total,
            "error": "",
            "code": llm["raw"][:4000],
        })

        if tps is not None and tps < MIN_TPS:
            slow_streak += 1
            if slow_streak >= SET_MAX_ITERS:
                print(f"\nAborting experiment: decode TPS < {MIN_TPS} tok/s for "
                      f"{SET_MAX_ITERS} consecutive problems (last: {tps:.1f} tok/s).")
                break
        else:
            slow_streak = 0

    if args.judge:
        run_judge_stage(df, args, results_file, experiment_id)
    suite_summary(results_file, args.judge)


JUDGE_MAX_TOKENS = 1024
JUDGE_FIELDS = ["timestamp", "judge_model", "experiment_id", "problem_id", "judge_score",
                "passed", "reason", "error_type", "error", "raw"]


def judge_file_for(results_file):
    return results_file.with_name(results_file.stem + "_judge.csv")


def _judged(results_file, judge_model):
    """Judge rows for this judge model that have a score (failed judgements are retried)."""
    jf = judge_file_for(results_file)
    if not jf.exists():
        return pd.DataFrame(columns=JUDGE_FIELDS)
    j = pd.read_csv(jf, engine="python", on_bad_lines="skip")
    if j.empty:
        return j
    j = j[(j["judge_model"] == judge_model)
          & (j["error_type"].isna() | (j["error_type"].astype(str) == ""))]
    return j.drop_duplicates(["experiment_id", "problem_id"], keep="last")


def pending_judgements(results_file, experiment_id, judge_model):
    """Open-ended answers of this experiment that the judge has not scored yet."""
    if not results_file.exists():
        return []
    r = pd.read_csv(results_file, engine="python", on_bad_lines="skip")
    if r.empty or "error_type" not in r.columns:
        return []
    r = r[(r["experiment_id"] == experiment_id) & (r["error_type"].astype(str) == "NEEDS_JUDGE")]
    r = r.drop_duplicates("problem_id", keep="last")
    done = set(zip(_judged(results_file, judge_model)["experiment_id"],
                   _judged(results_file, judge_model)["problem_id"]))
    return [rec for rec in r.to_dict("records") if (experiment_id, rec["problem_id"]) not in done]


def apply_judgements(df, results_file, judge_model):
    """Replace NEEDS_JUDGE rows by the judge's score where a judgement exists."""
    j = _judged(results_file, judge_model)
    if j.empty:
        return df
    j = j.set_index(["experiment_id", "problem_id"])
    df = df.copy()
    for idx, row in df.iterrows():
        if str(row["error_type"]) != "NEEDS_JUDGE":
            continue
        key = (row["experiment_id"], row["problem_id"])
        if key in j.index:
            jr = j.loc[key]
            passed = str(jr["passed"]).lower() == "true"
            df.at[idx, "score"] = float(jr["judge_score"])
            df.at[idx, "passed"] = passed
            df.at[idx, "error_type"] = "" if passed else "WRONG_ANSWER"
    return df


def judge_calibration(client, server, items):
    """Score each reference answer with the judge. The judge must give full marks."""
    for it in items.values():
        if it.get("type") != "rubric" or not it.get("reference"):
            continue
        llm = solve_problem(client, server, SS.build_judge_prompt(it, it["reference"]), "",
                            JUDGE_MAX_TOKENS, system=SS.JUDGE_SYSTEM_PROMPT)
        try:
            score, _ = SS.parse_judge_reply(llm["raw"])
        except ValueError:
            score = None
        if score is None or score < 1.0:
            print(f"WARNING: the judge scored the reference answer of {it['id']} at {score}; "
                  "the judge results may not be reliable.")
        else:
            print(f"judge check: reference answer for {it['id']} scored 1.0")


def run_judge_stage(df, args, results_file, experiment_id):
    """Unload the candidate, load the judge, grade the saved open-ended answers, unload."""
    pending = pending_judgements(results_file, experiment_id, args.judge)
    if not pending:
        return
    items = {row["task_id"]: row_to_item(row) for _, row in df.iterrows()}
    print(f"\nJudging {len(pending)} open-ended answer(s) with {args.judge} ...")
    unload_all_models(args.base_url)
    try:
        load = load_model(args.base_url, args.judge, args.judge_context, {})
        server = load.get("instance_id") or args.judge
        client = OpenAI(base_url=api(args.base_url, "/v1"), api_key="lm-studio",
                        timeout=REQUEST_TIMEOUT, max_retries=0)
        judge_calibration(client, server, items)
        jf = judge_file_for(results_file)
        for rec in pending:
            item = items.get(rec["problem_id"])
            if item is None:
                continue
            answer = rec["code"] if isinstance(rec.get("code"), str) else ""
            base = {"timestamp": timestamp(), "judge_model": args.judge,
                    "experiment_id": experiment_id, "problem_id": rec["problem_id"]}
            print(f"  {rec['problem_id']} ... ", end="", flush=True)
            try:
                llm = solve_problem(client, server, SS.build_judge_prompt(item, answer), "",
                                    JUDGE_MAX_TOKENS, system=SS.JUDGE_SYSTEM_PROMPT)
            except Exception as e:
                print("JUDGE ERROR")
                append_row(jf, JUDGE_FIELDS, {**base, "error_type": "JUDGE_ERROR",
                                              "error": str(e)[:500]})
                continue
            try:
                score, reason = SS.parse_judge_reply(llm["raw"])
            except ValueError as e:
                print("NO SCORE IN REPLY")
                append_row(jf, JUDGE_FIELDS, {**base, "error_type": "PARSE_ERROR",
                                              "error": str(e), "raw": llm["raw"][:2000]})
                continue
            passed = score >= float(item.get("min_score", 1.0)) - 1e-9
            print(f"score {score:.1f} {'PASS' if passed else 'FAIL'}  {reason[:80]}")
            append_row(jf, JUDGE_FIELDS, {**base, "judge_score": score, "passed": passed,
                                          "reason": reason, "error_type": "", "error": "",
                                          "raw": llm["raw"][:2000]})
    finally:
        unload_all_models(args.base_url)


def suite_summary(results_file, judge_model=None):
    """Per-domain scores, and separate figures for answerable and unanswerable items."""
    if not results_file.exists():
        return
    df = pd.read_csv(results_file, engine="python", on_bad_lines="skip")
    if df.empty or "score" not in df.columns:
        return
    df = df[df["error_type"].astype(str) != "LLM_ERROR"]
    df = df.drop_duplicates(["experiment_id", "problem_id"], keep="last").copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    if judge_model:
        df = apply_judgements(df, results_file, judge_model)
    awaiting = int((df["error_type"].astype(str) == "NEEDS_JUDGE").sum())
    df = df[df["error_type"].astype(str) != "NEEDS_JUDGE"].copy()
    df["passed"] = df["passed"].astype(str).str.lower().eq("true")
    df["unanswerable"] = df["item_type"].eq("unanswerable")

    by_domain = (df.groupby(["model", "context_actual", "domain"], dropna=False)
                 .agg(items=("problem_id", "count"), passed=("passed", "sum"),
                      mean_score=("score", "mean"))
                 .reset_index())
    by_domain["pass_rate"] = 100 * by_domain["passed"] / by_domain["items"]

    overall = []
    for keys, g in df.groupby(["model", "context_actual"], dropna=False):
        ans, un = g[~g["unanswerable"]], g[g["unanswerable"]]
        overall.append({
            "model": keys[0], "context_actual": keys[1],
            "answerable_pass_rate": 100 * ans["passed"].mean() if len(ans) else float("nan"),
            "mean_score": g["score"].mean(),
            "unanswerable_declined_rate": 100 * un["passed"].mean() if len(un) else float("nan"),
            "invented_answers": int((un["error_type"] == "INVENTED_ANSWER").sum()),
            "missing_doc_items": int((g["error_type"] == "MISSING_DOC").sum()),
        })
    overall = pd.DataFrame(overall)

    out = results_file.with_name(results_file.stem + "_by_domain.csv")
    by_domain.to_csv(out, index=False)
    print("\n" + "=" * 80 + "\nSUITE SUMMARY\n" + "=" * 80)
    print(overall.round(1).to_string(index=False))
    print()
    print(by_domain.round(1).to_string(index=False))
    if awaiting:
        print(f"\nOpen-ended answers waiting for a judge: {awaiting} (use --judge MODEL to score them)")
    print(f"\nPer-domain table saved to: {out}")


# =============================================================================
# MBPP
# =============================================================================
#
# Mostly Basic Python Problems (Austin et al., 2021). The prompt shows the task
# and the visible asserts (the standard MBPP setup), so the model knows the
# function name. The score uses the same asserts; challenge_test_list is not used.

MBPP_DATASET = "google-research-datasets/mbpp"
MBPP_CONFIG = "full"  # 500 problems in the test split


def load_mbpp():
    from datasets import load_dataset
    df = load_dataset(MBPP_DATASET, MBPP_CONFIG, split="test").to_pandas()
    df["task_id"] = df["task_id"].map(lambda x: f"MBPP/{x}")
    df["prompt"] = df.apply(
        lambda r: (
            "You are an expert Python programmer, and here is your task: "
            f"{r['text']}\n"
            "Your code should pass these tests:\n\n"
            + "\n".join(list(r["test_list"]))
        ),
        axis=1,
    )
    df["setup"] = df["test_setup_code"].fillna("")
    df["tests"] = df["test_list"].map(lambda t: "\n".join(list(t)))
    return df


def build_program_mbpp(code, row):
    """Setup code, the model's code, then the asserts, run as one script."""
    parts = [row["setup"], code, row["tests"]]
    return "\n\n".join(p for p in parts if p) + "\n"


def make_program(benchmark, code, row):
    if benchmark == "mbpp":
        return build_program_mbpp(code, row)
    return build_program(code, row)


def load_benchmark(name):
    if name == "humaneval":
        return load_humaneval()
    if name == "mbpp":
        return load_mbpp()
    if name == "swebench":
        return load_swebench()
    raise ValueError(f"unknown benchmark: {name}")


# =============================================================================
# SWE-BENCH VERIFIED
# =============================================================================
#
# Two stages, because the official evaluation needs Docker:
#   1. GENERATION (no Docker): check out each repo at its base commit, give the
#      model the issue plus the files it points to, and save its unified diff.
#      Each diff is checked with `git apply --check`.
#   2. EVALUATION (needs `pip install swebench` and a running Docker daemon):
#      the diffs go to the official harness, which applies them in a container
#      and runs the tests. Only the harness decides "resolved".
# Generations are saved as they finish, so an interrupted run resumes.

SWE_DATASET = "princeton-nlp/SWE-bench_Verified"
SWE_MAX_TOKENS = 8192
SWE_CONTEXT_FRACTION = 0.5   # share of the window for repo files; the rest is for the answer
SWE_MAX_FILES = 8
SWE_EVAL_WORKERS = 4
SWE_REPO_DIR = Path("swebench_repos")
SWE_PRED_DIR = Path("swebench_predictions")
SWE_SYSTEM_PROMPT = (
    "You are an expert software engineer fixing a bug in an existing repository.\n"
    "Respond with ONLY a unified diff in git format (--- a/path, +++ b/path, @@ hunks).\n"
    "Copy context lines exactly from the files shown. Change only what the fix needs.\n"
    "Do not explain your answer."
)
SWE_GEN_FIELDS = [
    "timestamp", "experiment_id", "model", "context_actual", "instance_id", "repo",
    "ttft", "generation_time", "decode_time", "decode_tokens_per_sec",
    "prompt_tokens", "completion_tokens", "finish_reason",
    "has_patch", "patch_applies", "gpu_memory_mb", "gpu_total_mb",
    "error_type", "error", "patch",
]
SWE_GEN_ERRORS = {"LLM_ERROR", "REPO_ERROR"}  # retried on the next run
PY_PATH_RE = re.compile(r"[\w./-]+\.py\b")
IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{5,}\b")


def load_swebench():
    from datasets import load_dataset
    df = load_dataset(SWE_DATASET, split="test").to_pandas()
    df["task_id"] = df["instance_id"]
    return df


def swe_gen_file(results_file):
    return results_file.with_name(results_file.stem + "_swebench_generations.csv")


def run_git(args, cwd, timeout=600):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                          text=True, timeout=timeout)


def ensure_repo(repo):
    path = SWE_REPO_DIR / repo.replace("/", "__")
    if not (path / ".git").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  cloning {repo} (one time, can be large)...")
        r = subprocess.run(["git", "clone", f"https://github.com/{repo}.git", str(path)],
                           capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            raise RuntimeError(f"clone of {repo} failed: {r.stderr[-300:]}")
    return path


def checkout_base(path, commit):
    r = run_git(["checkout", "-f", "--detach", commit], path)
    if r.returncode != 0:  # the commit may be missing from an older clone
        run_git(["fetch", "--all", "--quiet"], path, timeout=3600)
        r = run_git(["checkout", "-f", "--detach", commit], path)
        if r.returncode != 0:
            raise RuntimeError(f"checkout {commit[:10]} failed: {r.stderr[-300:]}")
    run_git(["clean", "-fdxq"], path)


def select_context_files(path, issue):
    """
    Files the issue points at: explicit .py paths first, then files that
    contain identifiers quoted in the issue. Test files count for less.
    """
    score = {}
    for p in dict.fromkeys(PY_PATH_RE.findall(issue)):
        p = p.lstrip("./")
        if (path / p).is_file():
            score[p] = score.get(p, 0) + 5
    for ident in list(dict.fromkeys(IDENT_RE.findall(issue)))[:6]:
        r = run_git(["grep", "-l", "-w", "-I", "-e", ident, "--", "*.py"], path, timeout=120)
        for f in r.stdout.splitlines()[:40]:
            score[f] = score.get(f, 0) + (0.5 if "test" in f.lower() else 1.0)
    ranked = sorted(score, key=lambda f: (-score[f], f))
    return ranked[:SWE_MAX_FILES]


def build_swe_prompt(path, row, context_chars):
    issue = row["problem_statement"]
    blocks, used = [], 0
    for f in select_context_files(path, issue):
        try:
            text = (path / f).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        block = f"### {f}\n```python\n{text}\n```\n"
        if used + len(block) > context_chars:
            continue  # skip files that do not fit, try the next one
        blocks.append(block)
        used += len(block)
    files = "\n".join(blocks) if blocks else "(no matching files found)\n"
    return (
        f"Repository: {row['repo']} at commit {row['base_commit'][:10]}\n\n"
        f"## Issue\n{issue}\n\n"
        f"## Relevant files (current contents)\n{files}\n"
        "## Task\nWrite a unified diff that fixes this issue."
    )


def extract_diff(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*\Z", "", text, flags=re.DOTALL | re.IGNORECASE)
    fenced = re.findall(r"```(?:diff|patch)?\s*\n(.*?)```", text, flags=re.DOTALL)
    candidates = [f for f in fenced if "--- " in f and "+++ " in f]
    if candidates:
        body = max(candidates, key=len)
    else:
        m = re.search(r"^(diff --git |--- a/)", text, flags=re.MULTILINE)
        if not m:
            return ""
        body = text[m.start():]
    return body.strip("\n") + "\n"


def patch_applies(path, diff):
    if not diff.strip():
        return False
    with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False, encoding="utf-8") as f:
        f.write(diff)
        name = f.name
    try:
        return run_git(["apply", "--check", "--recount", name], path).returncode == 0
    finally:
        Path(name).unlink(missing_ok=True)


def load_swe_done(gen_file, experiment_id):
    if not gen_file.exists():
        return set()
    try:
        df = pd.read_csv(gen_file, usecols=["experiment_id", "instance_id", "error_type"],
                         engine="python", on_bad_lines="skip")
    except Exception:
        return set()
    df = df[~df["error_type"].astype(str).isin(SWE_GEN_ERRORS)]
    return set(zip(df["experiment_id"], df["instance_id"]))


def run_swebench_generation(client, meta, df, gen_file, experiment_id):
    done = load_swe_done(gen_file, experiment_id)
    context_chars = int(CHARS_PER_TOKEN * meta["context_actual"] * SWE_CONTEXT_FRACTION)

    for n, (_, row) in enumerate(df.iterrows()):
        iid = row["instance_id"]
        if (experiment_id, iid) in done:
            print(f"[{n + 1}/{len(df)}] {iid} ... SKIP")
            continue
        print(f"[{n + 1}/{len(df)}] {iid} ... ", end="", flush=True)

        base = {"timestamp": timestamp(), "experiment_id": experiment_id,
                "model": meta["model"], "context_actual": meta["context_actual"],
                "instance_id": iid, "repo": row["repo"]}
        try:
            repo_path = ensure_repo(row["repo"])
            checkout_base(repo_path, row["base_commit"])
            prompt = build_swe_prompt(repo_path, row,
                                      max(2000, context_chars - len(row["problem_statement"])))
        except Exception as e:
            print("REPO ERROR")
            append_row(gen_file, SWE_GEN_FIELDS,
                       {**base, "error_type": "REPO_ERROR", "error": str(e)[:1000]})
            continue

        try:
            llm = solve_problem(client, meta["server_model"], prompt, "",
                                SWE_MAX_TOKENS, system=SWE_SYSTEM_PROMPT)
        except Exception as e:
            print("LLM ERROR")
            append_row(gen_file, SWE_GEN_FIELDS,
                       {**base, "error_type": "LLM_ERROR", "error": str(e)[:1000]})
            continue

        patch = extract_diff(llm["raw"])
        applies = patch_applies(repo_path, patch)
        gpu_used, gpu_total = get_gpu_memory()
        print(f"{'patch' if patch else 'no patch'} "
              f"({'applies' if applies else 'does not apply'}, "
              f"{llm['decode_tps'] or 0:.1f} tok/s)")

        append_row(gen_file, SWE_GEN_FIELDS, {
            **base,
            "ttft": llm["ttft"], "generation_time": llm["generation_time"],
            "decode_time": llm["decode_time"], "decode_tokens_per_sec": llm["decode_tps"],
            "prompt_tokens": llm["prompt_tokens"], "completion_tokens": llm["completion_tokens"],
            "finish_reason": llm["finish_reason"], "has_patch": bool(patch),
            "patch_applies": applies, "gpu_memory_mb": gpu_used, "gpu_total_mb": gpu_total,
            "error_type": "" if patch else "NO_PATCH", "error": "", "patch": patch,
        })


def run_swe_harness(meta, experiment_id, real):
    """Run the official harness. Returns the set of resolved instance ids, or None."""
    try:
        import swebench  # noqa: F401
    except ImportError:
        print("  swebench is not installed (pip install swebench), and Docker must be running.\n"
              "  Patches are saved; run the same command again to evaluate them.")
        return None

    run_id = safe_name(experiment_id)[:120]
    model_name = safe_name(meta["model"])[:120]
    SWE_PRED_DIR.mkdir(exist_ok=True)
    pred = SWE_PRED_DIR / f"{run_id}.jsonl"
    with open(pred, "w", encoding="utf-8") as f:
        for rec in real.to_dict("records"):
            f.write(json.dumps({"instance_id": rec["instance_id"],
                                "model_name_or_path": model_name,
                                "model_patch": rec["patch"]}) + "\n")

    cmd = [sys.executable, "-m", "swebench.harness.run_evaluation",
           "--dataset_name", SWE_DATASET, "--predictions_path", str(pred),
           "--max_workers", str(SWE_EVAL_WORKERS), "--run_id", run_id]
    print(f"  running the official SWE-bench harness on {len(real)} patches...")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"  harness exited with code {r.returncode}; patches are kept, re-run to retry.")
        return None

    for p in Path(".").glob(f"*{run_id}*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and "resolved_ids" in data:
            return set(data["resolved_ids"])
    print(f"  no harness report containing 'resolved_ids' for run {run_id} in the current folder.")
    return None


def run_swebench_eval(meta, experiment_id, gen_file, results_file):
    if not gen_file.exists():
        return
    gens = pd.read_csv(gen_file, engine="python", on_bad_lines="skip")
    gens = gens[gens["experiment_id"] == experiment_id].drop_duplicates("instance_id", keep="last")
    gens = gens[~gens["error_type"].astype(str).isin(SWE_GEN_ERRORS)]
    if gens.empty:
        return

    already = {iid for (eid, iid) in load_completed(results_file) if eid == experiment_id}
    todo = gens[~gens["instance_id"].isin(already)]
    if todo.empty:
        print("Evaluation: all generated patches are already scored.")
        return

    has = todo["has_patch"].astype(str).str.lower().eq("true")
    resolved = set()
    if has.any():
        resolved = run_swe_harness(meta, experiment_id, todo[has])
        if resolved is None:
            return  # keep the generations; evaluate on the next run

    for rec in todo.to_dict("records"):
        iid = rec["instance_id"]
        has_patch = str(rec["has_patch"]).lower() == "true"
        ok = has_patch and iid in resolved
        append_result({
            "timestamp": timestamp(), "experiment_id": experiment_id,
            "model": meta["model"], "server_model": meta["server_model"],
            "instance_id": meta["instance_id"],
            "context_requested": meta["context_requested"],
            "context_actual": meta["context_actual"], "fill_ratio": 0.0,
            "cache_label": meta["cache_label"], "problem_id": iid, "passed": ok,
            "error_type": "" if ok else ("TEST_FAILURE" if has_patch else "NO_PATCH"),
            "finish_reason": rec.get("finish_reason"), "ttft": rec.get("ttft"),
            "generation_time": rec.get("generation_time"),
            "decode_time": rec.get("decode_time"), "execution_time": None,
            "prompt_tokens": rec.get("prompt_tokens"),
            "completion_tokens": rec.get("completion_tokens"),
            "decode_tokens_per_sec": rec.get("decode_tokens_per_sec"),
            "load_time": meta["load_time"],
            "gpu_memory_mb": rec.get("gpu_memory_mb"), "gpu_total_mb": rec.get("gpu_total_mb"),
            "error": "", "code": rec.get("patch") if isinstance(rec.get("patch"), str) else "",
        }, results_file)
    print(f"Evaluation: {len(todo)} instances scored, {sum(1 for i in todo['instance_id'] if i in resolved)} resolved.")


def run_swebench_experiment(df, meta, args, results_file):
    """Generate patches (resumable), then score the ones that are not scored yet."""
    experiment_id = experiment_id_for(meta["model"], meta["context_requested"],
                                      meta["cache_label"], args.fill)
    print(f"\n{'=' * 80}\nEXPERIMENT {experiment_id}\n{'=' * 80}")
    client = OpenAI(base_url=api(args.base_url, "/v1"), api_key="lm-studio",
                    timeout=REQUEST_TIMEOUT, max_retries=0)
    gen_file = swe_gen_file(results_file)
    run_swebench_generation(client, meta, df, gen_file, experiment_id)
    run_swebench_eval(meta, experiment_id, gen_file, results_file)


# =============================================================================
# LLM (streaming, so TTFT and decode speed are measured separately)
# =============================================================================

SYSTEM_PROMPT = (
    "You are an expert Python programmer.\n\n"
    "Solve the programming problem.\n"
    "Return only the complete Python implementation, including the function "
    "signature.\n"
    "Do not explain your answer."
)


def solve_problem(client, model, prompt, filler, max_tokens, system=None):
    if filler:
        user = (
            "Below is unrelated reference code. Ignore it.\n"
            "<reference>\n" + filler + "\n</reference>\n\n"
            "Now solve this task:\n\n" + prompt
        )
    else:
        user = prompt

    start = time.perf_counter()
    first = None
    parts, usage, finish = [], None, None

    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system or SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
        seed=42,
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        piece = getattr(delta, "content", None)
        reasoning = getattr(delta, "reasoning_content", None)
        if (piece or reasoning) and first is None:
            first = time.perf_counter()
        if piece:
            parts.append(piece)
        if choice.finish_reason:
            finish = choice.finish_reason

    end = time.perf_counter()
    text = "".join(parts)

    completion_tokens = getattr(usage, "completion_tokens", None)
    decode_time = (end - first) if first else None
    tps = None
    if completion_tokens and decode_time and decode_time > 0:
        tps = completion_tokens / decode_time

    return {
        "code": extract_code(text),
        "raw": text,
        "ttft": (first - start) if first else None,
        "generation_time": end - start,
        "decode_time": decode_time,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": completion_tokens,
        "decode_tps": tps,
        "finish_reason": finish,
    }


# =============================================================================
# RESULTS FILE
# =============================================================================

def check_results_schema(results_file):
    if not results_file.exists():
        return
    with open(results_file, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), None)
    if header != FIELDS:
        raise RuntimeError(
            f"{results_file} has a different column layout than this script.\n"
            f"Rename/move the old file or pass a new --results path."
        )


def experiment_id_for(model, context_requested, cache_label, fill):
    """
    Deterministic ID that can be computed BEFORE loading the model, so a
    finished configuration can be skipped without a slow model load.
    """
    return (
        f"{safe_name(model)}_ctx{context_requested}"
        f"_kv{safe_name(cache_label)}_fill{int(round(fill * 100))}"
    )


def repair_results_file(results_file):
    """
    If a previous run was killed while writing a row, the CSV can end with a
    torn record (no trailing newline) or fail to parse. Back the file up and
    rewrite it without the damaged record so appending/resuming is safe.
    """
    if not results_file.exists() or results_file.stat().st_size == 0:
        return

    raw = results_file.read_bytes()
    torn = not raw.endswith(b"\n")

    parse_error = None
    try:
        df = pd.read_csv(results_file)
    except Exception as e:
        parse_error = e
        df = pd.read_csv(results_file, engine="python", on_bad_lines="skip")

    if not torn and parse_error is None:
        return  # healthy

    print("Results file was damaged by an interrupted write. Repairing...")
    backup = results_file.with_suffix(".csv.bak")
    backup.write_bytes(raw)

    if torn and parse_error is None:
        df = df.iloc[:-1]  # the last record is the incomplete one

    df = df.dropna(subset=["experiment_id", "problem_id"])
    df = df[df["passed"].astype(str).str.lower().isin(["true", "false"])]
    df.to_csv(results_file, index=False)
    print(f"Kept {len(df)} valid rows (backup: {backup})")


def load_completed(results_file):
    """
    Set of (experiment_id, problem_id) that are DONE.
    Rows that failed with LLM_ERROR (server crash, timeout, ...) are NOT
    considered done, so they are retried on the next run.
    """
    if not results_file.exists():
        return set()
    try:
        df = pd.read_csv(
            results_file,
            usecols=["experiment_id", "problem_id", "error_type"],
        )
        df = df[df["error_type"].astype(str) != "LLM_ERROR"]
        return set(zip(df["experiment_id"], df["problem_id"]))
    except Exception:
        return set()


def load_summary_configs(results_file):
    """
    Set of (model, cache_label, context_requested, fill_ratio) present in the
    summary CSV.  A configuration missing from the summary is treated as not
    finished: the entire experiment for that config will be re-run even if
    some individual problem rows exist in the detailed results file.
    """
    summary_file = results_file.with_name(results_file.stem + "_summary.csv")
    if not summary_file.exists():
        return set()
    try:
        df = pd.read_csv(
            summary_file,
            usecols=["model", "cache_label", "context_requested", "fill_ratio"],
        )
        return set(
            zip(
                df["model"].astype(str),
                df["cache_label"].astype(str),
                df["context_requested"].astype(int),
                df["fill_ratio"].astype(float),
            )
        )
    except Exception:
        return set()


def append_row(path, fields, row):
    """Append one CSV row durably (flush + fsync)."""
    new_file = not path.exists() or path.stat().st_size == 0
    if not new_file:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            torn = f.read(1) != b"\n"
        if torn:  # previous run died mid-row: terminate that broken record
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n")
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k) for k in fields})
        f.flush()
        os.fsync(f.fileno())  # survive an abrupt kill


def append_result(result, results_file):
    append_row(results_file, FIELDS, result)


def write_diagnostics(diag_file, mode, meta, load, gpu_used, gpu_total):
    """One JSON line per model load: LM Studio's actual config + VRAM after load."""
    rec = {
        "timestamp": timestamp(),
        "mode": mode,
        "model": meta["model"],
        "cache_label": meta["cache_label"],
        "context_requested": meta["context_requested"],
        "context_actual": meta["context_actual"],
        "load_time": meta["load_time"],
        "gpu_after_load_mb": gpu_used,
        "gpu_total_mb": gpu_total,
        "load_config": load.get("load_config", {}),
    }
    with open(diag_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# =============================================================================
# SPEED PROBE
# =============================================================================

def probe_targets_for(context, targets):
    """Token levels that fit safely into this context window."""
    cap = int(context * 0.9) - PROBE_MAX_TOKENS - 600
    return sorted({t for t in targets if t <= cap})


def load_probe_done(probe_file):
    if not probe_file.exists():
        return set()
    try:
        df = pd.read_csv(probe_file, engine="python", on_bad_lines="skip")
        keys = ["model", "context_requested", "cache_label",
                "target_tokens", "rep"]
        df = df.dropna(subset=keys)
        df = df[df["error"].isna() | (df["error"].astype(str) == "")]
        return set(zip(df["model"], df["context_requested"].astype(int),
                       df["cache_label"], df["target_tokens"].astype(int),
                       df["rep"].astype(int)))
    except Exception:
        return set()


def probe_progress(probe_file, model, context, cache_label, args):
    done_set = load_probe_done(probe_file)
    targets = probe_targets_for(context, args.probe_tokens)
    total = len(targets) * args.probe_reps
    done = sum(
        (model, context, cache_label, t, r) in done_set
        for t in targets for r in range(args.probe_reps)
    )
    return done, total


def run_probe(meta, args, probe_file):
    """
    For each token level: fill the KV cache with N tokens of unrelated code,
    then generate a short fixed-format answer and time it.
    Filler differs per repetition, so LM Studio's prefix cache can't hide
    the prefill cost.
    """
    client = OpenAI(base_url=api(args.base_url, "/v1"), api_key="lm-studio",
                    timeout=REQUEST_TIMEOUT, max_retries=0)
    done_set = load_probe_done(probe_file)
    targets = probe_targets_for(meta["context_actual"], args.probe_tokens)

    print(f"\nProbe levels for ctx {meta['context_actual']:,}: {targets}")

    pending = [(ti, t, r) for ti, t in enumerate(targets)
               for r in range(args.probe_reps)
               if (meta["model"], meta["context_requested"],
                   meta["cache_label"], t, r) not in done_set]

    if pending:  # warm-up (not recorded): first request after load is slower
        try:
            solve_problem(client, meta["server_model"], PROBE_TASK, "",
                          PROBE_MAX_TOKENS)
        except Exception as e:
            print(f"warm-up failed: {e}")

    for ti, target, rep in pending:
        print(f"  target {target:>8,} tok  rep {rep} ... ", end="", flush=True)
        row = {
            "timestamp": timestamp(),
            "model": meta["model"],
            "server_model": meta["server_model"],
            "context_requested": meta["context_requested"],
            "context_actual": meta["context_actual"],
            "cache_label": meta["cache_label"],
            "target_tokens": target,
            "rep": rep,
            "gpu_after_load_mb": meta.get("gpu_after_load"),
        }
        filler = make_filler(int(target * CHARS_PER_TOKEN),
                             offset=rep * 104729 + ti * 7919 + 13)

        # Collect measurements. If decode is slower than 10 tok/s, repeat
        # up to 10 times to check consistency, then average. Such slow
        # rates are not used for scoring anyway.
        measurements = []
        max_iters = 1
        for attempt in range(10):
            try:
                r = solve_problem(client, meta["server_model"], PROBE_TASK,
                                  filler, PROBE_MAX_TOKENS)
            except Exception as e:
                if not measurements:
                    # First attempt failed hard: record error and move on
                    print("ERROR")
                    gpu_used, gpu_total = get_gpu_memory()
                    append_row(probe_file, PROBE_FIELDS,
                               {**row, "gpu_used_mb": gpu_used,
                                "gpu_total_mb": gpu_total, "error": str(e)[:500]})
                    break
                # Later attempt failed: stop averaging with what we have
                break

            ct, dt = r["completion_tokens"], r["decode_time"]
            decode_tps = (ct - 1) / dt if ct and ct > 1 and dt and dt > 0 else None
            prefill_tps = (r["prompt_tokens"] / r["ttft"]
                           if r["prompt_tokens"] and r["ttft"] else None)
            measurements.append({
                "r": r,
                "decode_tps": decode_tps,
                "prefill_tps": prefill_tps,
                "ct": ct,
                "dt": dt,
            })

            # After first measurement, decide whether to repeat
            if attempt == 0:
                if decode_tps is not None and decode_tps < MIN_TPS:
                    max_iters = SET_MAX_ITERS
                    print(f"(slow {decode_tps:.1f} tok/s, averaging up to 10) ", end="", flush=True)
                else:
                    break  # fast enough, no need to repeat
            if len(measurements) >= max_iters:
                break

        if not measurements:
            continue  # already recorded error

        # Average only the valid (non-None) speed measurements
        n = len(measurements)
        valid_decode = [m["decode_tps"] for m in measurements if m["decode_tps"] is not None]
        valid_prefill = [m["prefill_tps"] for m in measurements if m["prefill_tps"] is not None]
        decode_tps = (sum(valid_decode) / len(valid_decode)) if valid_decode else None
        prefill_tps = (sum(valid_prefill) / len(valid_prefill)) if valid_prefill else None

        # Use the last successful run for non-averaged fields (tokens, times)
        last = measurements[-1]
        r = last["r"]
        ct, dt = last["ct"], last["dt"]

        gpu_used, gpu_total = get_gpu_memory()

        extra = f" (avg of {n})" if n > 1 else ""
        print(f"{r['prompt_tokens']} prompt tok | prefill "
              f"{prefill_tps or 0:.0f} tok/s | decode {decode_tps or 0:.1f} tok/s"
              f"{extra} | VRAM {gpu_used}/{gpu_total} MB")

        append_row(probe_file, PROBE_FIELDS, {
            **row,
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": ct,
            "ttft": r["ttft"],
            "decode_time": dt,
            "prefill_tps": prefill_tps,
            "decode_tps": decode_tps,
            "gpu_used_mb": gpu_used,
            "gpu_total_mb": gpu_total,
            "finish_reason": r["finish_reason"],
            "error": "",
        })


# =============================================================================
# ONE EXPERIMENT (model is already loaded)
# =============================================================================

def run_experiment(df, meta, args, results_file, force_rerun=False):
    experiment_id = experiment_id_for(
        meta["model"], meta["context_requested"],
        meta["cache_label"], args.fill,
    )

    print(f"\n{'=' * 80}\nEXPERIMENT {experiment_id}\n{'=' * 80}")
    if force_rerun:
        print("Force re-run: config not present in summary CSV; "
              "ignoring any existing problem rows.")

    client = OpenAI(
        base_url=api(args.base_url, "/v1"),
        api_key="lm-studio",
        timeout=REQUEST_TIMEOUT,
        max_retries=0,
    )

    # Filler size (same for every problem unless --vary-filler).
    filler_tokens = 0
    if args.fill > 0:
        filler_tokens = min(
            int(meta["context_actual"] * args.fill),
            meta["context_actual"] - args.max_tokens - 1500,
        )
    filler_chars = max(0, int(filler_tokens * CHARS_PER_TOKEN))
    static_filler = make_filler(filler_chars)

    # When the config is missing from the summary CSV, re-run everything.
    completed = set() if force_rerun else load_completed(results_file)

    slow_streak = 0  # consecutive problems with decode_tps < MIN_TPS

    for n, (_, row) in enumerate(df.iterrows()):
        pid = row["task_id"]

        if (experiment_id, pid) in completed:
            print(f"[{n + 1}/{len(df)}] {pid} ... SKIP")
            continue

        print(f"[{n + 1}/{len(df)}] {pid} ... ", end="", flush=True)

        filler = (make_filler(filler_chars, offset=n * 7919)
                  if args.vary_filler else static_filler)

        base = {
            "timestamp": timestamp(),
            "experiment_id": experiment_id,
            "model": meta["model"],
            "server_model": meta["server_model"],
            "instance_id": meta["instance_id"],
            "context_requested": meta["context_requested"],
            "context_actual": meta["context_actual"],
            "fill_ratio": args.fill,
            "cache_label": meta["cache_label"],
            "problem_id": pid,
            "load_time": meta["load_time"],
        }

        try:
            llm = solve_problem(client, meta["server_model"],
                                row["prompt"], filler, args.max_tokens)
        except Exception as e:
            print("LLM ERROR")
            gpu_used, gpu_total = get_gpu_memory()
            append_result({**base, "passed": False, "error_type": "LLM_ERROR",
                           "gpu_memory_mb": gpu_used, "gpu_total_mb": gpu_total,
                           "error": str(e)[:2000], "code": ""}, results_file)
            continue

        execution = execute_program(make_program(args.benchmark, llm["code"], row))
        gpu_used, gpu_total = get_gpu_memory()  # sampled after inference

        passed = execution["passed"]
        if passed:
            error_type = ""
        elif execution["stderr"] == "TIMEOUT":
            error_type = "CODE_TIMEOUT"
        elif llm["finish_reason"] == "length":
            error_type = "TRUNCATED"
        elif not llm["code"]:
            error_type = "EMPTY_RESPONSE"
        else:
            error_type = "TEST_FAILURE"

        tps = llm["decode_tps"]
        print(f"{'PASS' if passed else 'FAIL'}"
              f"  ({llm['prompt_tokens']} prompt tok, "
              f"{tps or 0:.1f} tok/s)")

        append_result({
            **base,
            "passed": passed,
            "error_type": error_type,
            "finish_reason": llm["finish_reason"],
            "ttft": llm["ttft"],
            "generation_time": llm["generation_time"],
            "decode_time": llm["decode_time"],
            "execution_time": execution["execution_time"],
            "prompt_tokens": llm["prompt_tokens"],
            "completion_tokens": llm["completion_tokens"],
            "decode_tokens_per_sec": tps,
            "gpu_memory_mb": gpu_used,
            "gpu_total_mb": gpu_total,
            "error": execution["stderr"],
            "code": llm["code"],
        }, results_file)

        # Abort experiment after SET_MAX_ITERS consecutive slow problems.
        if tps is not None and tps < MIN_TPS:
            slow_streak += 1
            if slow_streak >= SET_MAX_ITERS:
                print(f"\nAborting experiment: decode TPS < {MIN_TPS} "
                      f"tok/s for {SET_MAX_ITERS} consecutive problems "
                      f"(last: {tps:.1f} tok/s).")
                break
        else:
            slow_streak = 0


# =============================================================================
# SUMMARY
# =============================================================================

def wilson(passes, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = passes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h) * 100, (c + h) * 100


def create_summary(results_file):
    if not results_file.exists():
        return
    df = pd.read_csv(results_file)
    if df.empty:
        return

    # A retried problem appears twice (old LLM_ERROR row + new row): keep latest.
    df = df.drop_duplicates(["experiment_id", "problem_id"], keep="last").copy()

    df["passed"] = df["passed"].astype(str).str.lower().eq("true")
    df["llm_error"] = df["error_type"].eq("LLM_ERROR")
    df["truncated"] = df["error_type"].eq("TRUNCATED")

    keys = ["model", "cache_label", "context_requested",
            "context_actual", "fill_ratio"]

    summary = (
        df.groupby(keys, dropna=False)
        .agg(
            problems=("problem_id", "count"),
            passed=("passed", "sum"),
            llm_errors=("llm_error", "sum"),
            truncated=("truncated", "sum"),
            avg_prompt_tokens=("prompt_tokens", "mean"),
            avg_ttft=("ttft", "mean"),
            avg_decode_tps=("decode_tokens_per_sec", "mean"),
            avg_exec_time=("execution_time", "mean"),
            load_time=("load_time", "mean"),
            gpu_peak_mb=("gpu_memory_mb", "max"),
        )
        .reset_index()
    )

    summary["pass_rate"] = summary["passed"] / summary["problems"] * 100
    ci = [wilson(p, n) for p, n in zip(summary["passed"], summary["problems"])]
    summary["ci95_low"] = [c[0] for c in ci]
    summary["ci95_high"] = [c[1] for c in ci]

    output = results_file.with_name(results_file.stem + "_summary.csv")
    summary.to_csv(output, index=False)

    print(f"\n{'=' * 80}\nSUMMARY\n{'=' * 80}")
    print(summary.sort_values("pass_rate", ascending=False).to_string(index=False))
    print(f"\nSaved to: {output}")


# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# DRY RUN: memory and speed estimates, written like a benchmark run
# =============================================================================
#
# `localbenchmark --dry-run` loads no model and sends no request. It writes to
# dryrun/: dryrun_estimates.csv, dryrun_leaderboard.csv and plots/*.png.
#
# Memory per model, quantization and context:
#   weights  = size of the GGUF file on disk (exact), else bits-per-weight estimate
#   KV cache = bytes per token from the GGUF header (layers, KV heads, head size),
#              else --kv-bytes (a default that is not measured for that model)
#   overhead = DRYRUN_OVERHEAD_GB
# Decode speed: each token reads the active weights and the whole KV cache. The part
# of the model that does not fit in VRAM is read at system RAM speed, so the ceiling
# blends GPU and RAM bandwidth by the offloaded share. It is an upper bound.

DRYRUN_OVERHEAD_GB = 0.5            # buffers and runtime; measured as ~0.5 GB for Ornith 9B
DEFAULT_VRAM_GB = 16.0              # used when nvidia-smi is not available
MODEL_PARAMS_MOE = 3                # active size for a mixture-of-experts model without one in its name
GPU_BANDWIDTH = 600                 # GB/s; replace with your card's number (--gpu-bandwidth)
RAM_BANDWIDTH = 40                  # GB/s; typical dual-channel system RAM (--ram-bandwidth), range 8 to 60
KV_BYTES_PER_TOKEN = 55_000         # default for models without a GGUF file (--kv-bytes)
KV_ELEMENT_BYTES = 2                # f16 KV cache
QUANT_BITS = {"Q3_K_M": 3.8, "Q4_K_M": 4.8, "Q5_K_M": 5.6, "Q6_K": 6.5, "Q8_0": 8.4}
QUANT_TOKEN = {"Q3_K_M": "q3km", "Q4_K_M": "q4km", "Q5_K_M": "q5km", "Q6_K": "q6k", "Q8_0": "q80"}
DRYRUN_DEFAULT_QUANTS = ["Q4_K_M", "Q6_K", "Q8_0"]   # add Q3_K_M / Q5_K_M with --quants
BENCHMARK_ITEMS = {"humaneval": 164, "mbpp": 500, "swebench": 500}
# Known models, matched on the last part of the model key (exact, lower case).
# total_b / active_b are in billions of parameters. active_b is the parameters read per
# token (equal to total_b for dense models). max_context is the model's context limit.
# Add a model here when its name does not show its size correctly. Check values against
# the model card.
MODEL_SIZE_LOOKUP = {
    "gpt-oss-20b":            {"total_b": 21.0, "active_b": 3.6},    # mixture of experts
    "phi-4":                  {"total_b": 14.0, "active_b": 14.0, "max_context": 16_000},
    "phi4":                   {"total_b": 14.0, "active_b": 14.0, "max_context": 16_000},
    "muse-glimmer":           {"total_b": 30.0, "active_b": 30.0},
    "devstral-small-2-2512":  {"total_b": 24.0, "active_b": 24.0},
}
DRYRUN_CSV = "dryrun_estimates.csv"
DRYRUN_LEADERBOARD = "dryrun_leaderboard.csv"
DRYRUN_PLOT_QUANT = "Q4_K_M"        # quantization used in the single-quantization figures
RED = "#d62728"                     # above GPU memory
GREEN = "#74c476"                   # within GPU memory
GREY = "#d9d9d9"                    # size unknown
DRYRUN_FIELDS = ["model", "label", "total_b", "active_b", "size_source", "file", "quant", "bits",
                 "context", "weights_gb", "weights_source", "kv_bytes_per_token", "kv_source",
                 "kv_gb", "total_gb", "vram_gb", "headroom_gb", "exceeds_gpu", "offload_share",
                 "read_gb_per_token", "bandwidth_used", "decode_ceiling_tps"]
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)b(?![a-z])", re.IGNORECASE)   # 27b, 3.5b, 30b
_ACTIVE_RE = re.compile(r"-a(\d+(?:\.\d+)?)b", re.IGNORECASE)        # -a3b: active parameters
_GGUF_FIXED = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
               10: "<Q", 11: "<q", 12: "<d"}
_GGUF_CACHE = {}


def _num(x, digits=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return f"{x:.{digits}f}"


def lookup_model(name):
    """Entry from MODEL_SIZE_LOOKUP for this model, or None. Matches the last part of the key."""
    return MODEL_SIZE_LOOKUP.get(name.split("/")[-1].lower())


def model_params(name):
    """(total_b, active_b, source). Lookup first, then the name. Unknown sizes are nan."""
    known = lookup_model(name)
    if known:
        return float(known["total_b"]), float(known["active_b"]), "lookup"
    low = name.lower()
    active_match = _ACTIVE_RE.search(low)
    sizes = [float(x) for x in _SIZE_RE.findall(_ACTIVE_RE.sub("", low))]
    total = max(sizes) if sizes else float("nan")
    if active_match:
        active = float(active_match.group(1))
    elif "moe" in low:
        active = float(MODEL_PARAMS_MOE)
    else:
        active = total  # dense: every parameter is read for every token (nan if unknown)
    source = "unknown" if (math.isnan(total) and math.isnan(active)) else "from name"
    return total, active, source


# ---- GGUF header: file size and KV-cache layout, without reading any weights ----

def _gguf_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


def _gguf_value(f, vtype):
    if vtype in _GGUF_FIXED:
        fmt = _GGUF_FIXED[vtype]
        (v,) = struct.unpack(fmt, f.read(struct.calcsize(fmt)))
        return v
    if vtype == 8:
        return _gguf_str(f)
    if vtype == 9:
        (etype,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        if etype == 8:
            if count > 64:  # long arrays (tokenizer vocabularies) are skipped, not read
                for _ in range(count):
                    (n,) = struct.unpack("<Q", f.read(8))
                    f.seek(n, 1)
                return None
            return [_gguf_str(f) for _ in range(count)]
        if etype in _GGUF_FIXED:
            fmt = _GGUF_FIXED[etype]
            size = struct.calcsize(fmt)
            if count > 64:
                f.seek(count * size, 1)
                return None
            return [struct.unpack(fmt, f.read(size))[0] for _ in range(count)]
        raise ValueError(f"unknown GGUF array type {etype}")
    raise ValueError(f"unknown GGUF value type {vtype}")


def read_gguf_metadata(path):
    """Key-value header of a GGUF file. Reads metadata only, never the tensors."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        (version,) = struct.unpack("<I", f.read(4))
        if version < 2:
            raise ValueError(f"GGUF version {version} is not supported")
        _, n_kv = struct.unpack("<QQ", f.read(16))
        meta = {}
        for _ in range(n_kv):
            key = _gguf_str(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            meta[key] = _gguf_value(f, vtype)
    return meta


def kv_profile(meta, elem_bytes=KV_ELEMENT_BYTES):
    """
    KV-cache bytes per token of context, from the model's architecture.
    Only attention layers keep a KV cache. Hybrid models (recurrent layers mixed with
    attention) have a full-attention interval, e.g. 4 = every fourth layer is attention.
    """
    arch = meta.get("general.architecture")
    if not arch:
        return None
    layers = meta.get(f"{arch}.block_count")
    heads = meta.get(f"{arch}.attention.head_count")
    kv_heads = meta.get(f"{arch}.attention.head_count_kv", heads)
    emb = meta.get(f"{arch}.embedding_length")
    key_len = meta.get(f"{arch}.attention.key_length") or (emb // heads if emb and heads else None)
    val_len = meta.get(f"{arch}.attention.value_length") or key_len
    if None in (layers, kv_heads, key_len, val_len):
        return None
    interval = next((v for k, v in meta.items()
                     if k.startswith(f"{arch}.") and k.endswith("full_attention_interval")
                     and isinstance(v, int) and v > 1), None)
    if isinstance(kv_heads, list):          # per-layer KV heads; 0 means no KV cache in that layer
        attn_layers = sum(1 for h in kv_heads if h)
        kv_heads_total = sum(kv_heads)
    else:
        attn_layers = layers // interval if interval else layers
        kv_heads_total = attn_layers * kv_heads
    return {"arch": arch, "layers": layers, "attention_layers": attn_layers,
            "kv_heads": kv_heads, "key_len": key_len, "val_len": val_len,
            "full_attention_interval": interval,
            "kv_bytes_per_token": elem_bytes * kv_heads_total * (key_len + val_len),
            "sliding_window": meta.get(f"{arch}.attention.sliding_window")}


def gguf_kv(path):
    """KV profile of a GGUF file, cached. None if the file cannot be read."""
    key = str(path)
    if key not in _GGUF_CACHE:
        try:
            _GGUF_CACHE[key] = kv_profile(read_gguf_metadata(path))
        except Exception:
            _GGUF_CACHE[key] = None
    return _GGUF_CACHE[key]


def scan_gguf_index(roots):
    index = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.gguf")):
            try:
                size = p.stat().st_size
            except OSError:
                continue
            rel = p.relative_to(root).as_posix().lower()
            index.append({"path": p, "name": p.name, "rel": rel,
                          "alnum": re.sub(r"[^a-z0-9]", "", p.name.lower()),
                          "relalnum": re.sub(r"[^a-z0-9]", "", rel), "size": size})
    return index


def find_gguf(index, model, quant):
    """The GGUF file of this model and quantization, or None."""
    base = model.split("/")[-1].lower()
    hits = [f for f in index if base in f["rel"] and QUANT_TOKEN[quant] in f["relalnum"]]
    return hits[0] if hits else None


def model_kv_profile(index, model):
    """KV profile from any GGUF file of this model (the architecture does not depend on quant)."""
    base = model.split("/")[-1].lower()
    for f in index:
        if base in f["rel"]:
            profile = gguf_kv(f["path"])
            if profile:
                return profile
    return None


def vram_estimate(total_b, bits, context, kv_bytes):
    weights_gb = total_b * bits / 8
    kv_gb = context * kv_bytes / 1e9
    return weights_gb, kv_gb, weights_gb + kv_gb + DRYRUN_OVERHEAD_GB


def resolve_vram(given_gb):
    if given_gb is not None:
        return float(given_gb), "given"
    try:
        _, total_mb = get_gpu_memory()
    except Exception:
        total_mb = None
    if total_mb:
        return total_mb * 1048576 / 1e9, "nvidia-smi"   # MiB to decimal GB, same unit as the file sizes
    return DEFAULT_VRAM_GB, "default"


def print_table(headers, rows):
    widths = [max([len(str(h))] + [len(str(r[i])) for r in rows])
              for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip())


def dryrun_rows(models, contexts, vram_gb, gpu_bw, ram_bw, quants, index,
                kv_default=KV_BYTES_PER_TOKEN, overhead_gb=DRYRUN_OVERHEAD_GB):
    rows = []
    for model in models:
        total_b, active_b, size_source = model_params(model)
        label = model.split("/")[-1]
        profile = model_kv_profile(index, model)
        if profile:
            kv_bpt, kv_source = profile["kv_bytes_per_token"], "GGUF file"
        else:
            kv_bpt, kv_source = kv_default, "default (not measured)"
        known = lookup_model(model)
        limit = known.get("max_context") if known else None
        model_contexts = [c for c in contexts if limit is None or c <= limit]
        for quant in quants:
            bits = QUANT_BITS[quant]
            found = find_gguf(index, model, quant)
            if found:
                weights_gb = found["size"] / 1e9
                weights_source, file_name = "GGUF file", found["name"]
            else:
                weights_gb = total_b * bits / 8
                weights_source, file_name = "estimate", ""
            active_weights_gb = weights_gb * active_b / total_b if total_b else float("nan")
            for c in model_contexts:
                kv_gb = kv_bpt * c / 1e9
                tot = weights_gb + kv_gb + overhead_gb
                exceeds = None if math.isnan(tot) else tot > vram_gb
                offload = float("nan") if math.isnan(tot) else min(1.0, max(0.0, (tot - vram_gb) / tot))
                read_gb = active_weights_gb + kv_gb
                if math.isnan(read_gb):
                    ceiling, used = float("nan"), "unknown"
                else:
                    seconds = read_gb * ((1 - offload) / gpu_bw + offload / ram_bw)
                    ceiling = 1 / seconds
                    used = "GPU" if offload == 0 else ("RAM" if offload >= 1 else "GPU+RAM")
                rows.append({"model": model, "label": label, "total_b": total_b,
                             "active_b": active_b, "size_source": size_source, "file": file_name,
                             "quant": quant, "bits": bits, "context": c,
                             "weights_gb": weights_gb, "weights_source": weights_source,
                             "kv_bytes_per_token": kv_bpt, "kv_source": kv_source, "kv_gb": kv_gb,
                             "total_gb": tot, "vram_gb": vram_gb, "headroom_gb": vram_gb - tot,
                             "exceeds_gpu": exceeds, "offload_share": offload,
                             "read_gb_per_token": read_gb, "bandwidth_used": used,
                             "decode_ceiling_tps": ceiling})
    return pd.DataFrame(rows, columns=DRYRUN_FIELDS)


def dryrun_leaderboard(df):
    """Rank-sum over decode-speed ceiling and memory headroom, for every model x quantization x
    context. Lower rank is better. Unknown sizes stay nan and are not ranked."""
    board = df.copy()
    board["score"] = (board["decode_ceiling_tps"].rank(ascending=False)
                      + board["headroom_gb"].rank(ascending=False))
    board["rank"] = board["score"].rank(method="min")
    cols = ["rank", "label", "quant", "context", "weights_source", "kv_source", "total_gb",
            "headroom_gb", "exceeds_gpu", "bandwidth_used", "decode_ceiling_tps", "score"]
    return board.sort_values(["rank", "decode_ceiling_tps"], ascending=[True, False],
                             na_position="last")[cols]


def plot_dryrun(df, board, vram_gb, plots_dir):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    plots_dir.mkdir(parents=True, exist_ok=True)
    contexts = sorted(df["context"].unique())
    cmax = int(max(contexts))
    quants = list(dict.fromkeys(df["quant"]))
    labels = list(dict.fromkeys(df["label"]))
    colours = plt.get_cmap("tab10")
    paths = []

    # 1. total GB per model x quantization (rows) against context (columns); red = above GPU memory
    df = df.assign(row=df["label"] + " " + df["quant"])
    rows = list(dict.fromkeys(df["row"]))
    pivot = df.pivot(index="row", columns="context", values="total_gb").reindex(index=rows, columns=contexts)
    rgb = np.ones(pivot.shape + (3,))
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            rgb[i, j] = matplotlib.colors.to_rgb(GREY if np.isnan(v) else (RED if v > vram_gb else GREEN))
    fig, ax = plt.subplots(figsize=(1.2 * len(contexts) + 4, 0.45 * len(rows) + 2.5))
    ax.imshow(rgb, aspect="auto")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, _num(pivot.values[i, j]), ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(contexts)), [f"{c // 1000}k" for c in contexts])
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel("Context window (tokens)")
    ax.set_title(f"Estimated total GB by context (GPU {vram_gb:.1f} GB)")
    ax.legend(handles=[Patch(color=RED, label="above GPU memory"),
                       Patch(color=GREEN, label="within GPU memory"),
                       Patch(color=GREY, label="size unknown")],
              loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    p = plots_dir / "D1_total_gb_heatmap.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # 2. total GB against context for one quantization; red markers = above GPU memory
    plot_quant = DRYRUN_PLOT_QUANT if DRYRUN_PLOT_QUANT in quants else quants[0]
    sub = df[df["quant"] == plot_quant]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    skipped = []
    for i, label in enumerate(labels):
        g = sub[sub["label"] == label].sort_values("context")
        if g["total_gb"].isna().all():
            skipped.append(label)
            continue
        colour = colours(i % 10)
        ax.plot(g["context"], g["total_gb"], color=colour, lw=1.5, label=label)
        for x, y in zip(g["context"], g["total_gb"]):
            ax.scatter(x, y, color=RED if y > vram_gb else colour, s=40, zorder=3)
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    ax.axhline(vram_gb, color="black", ls="--", lw=1)
    ax.text(cmax, vram_gb, f"GPU {vram_gb:.1f} GB ", ha="right", va="bottom", fontsize=8)
    ax.set_xticks(contexts, [f"{c // 1000}k" for c in contexts])
    ax.set_xlabel("Context window (tokens)")
    ax.set_ylabel("Estimated total GB")
    title = f"Estimated memory against context ({plot_quant})"
    if skipped:
        title += f"\nnot plotted, size unknown: {', '.join(skipped)}"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    handles, names = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=RED, label="above GPU memory"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    p = plots_dir / "D2_memory_vs_context.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # 3. speed ceiling against memory, every configuration; red = above GPU memory
    fig, ax = plt.subplots(figsize=(9, 6.5))
    plotted = df.dropna(subset=["total_gb", "decode_ceiling_tps"])
    for i, label in enumerate(labels):
        for _, r in plotted[plotted["label"] == label].iterrows():
            colour = RED if r["total_gb"] > vram_gb else colours(i % 10)
            ax.scatter(r["total_gb"], r["decode_ceiling_tps"], color=colour, s=45)
            ax.annotate(f"{label} {r['quant']} {int(r['context']) // 1000}k",
                        (r["total_gb"], r["decode_ceiling_tps"]),
                        textcoords="offset points", xytext=(4, 2), fontsize=6, color=colour)
    ax.axvline(vram_gb, color="black", ls="--", lw=1)
    ax.set_xlabel("Estimated total GB")
    ax.set_ylabel("Decode-speed ceiling (tokens/s, upper bound)")
    skipped = sorted(set(df["label"]) - set(plotted["label"]))
    title = "Speed against memory: up and to the left is better"
    if skipped:
        title += f"\nnot plotted, size unknown: {', '.join(skipped)}"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=RED, label="above GPU memory")],
              loc="upper right", fontsize=8, frameon=False)
    p = plots_dir / "D3_speed_vs_memory.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # 4. ranking of every configuration; red = above GPU memory
    ranked = board[board["score"].notna()].iloc[::-1]
    unknown = sorted(set(board["label"]) - set(ranked["label"]))
    fig, ax = plt.subplots(figsize=(10, 0.32 * max(len(ranked), 1) + 2))
    names = [f"{r.label} {r.quant} {int(r.context) // 1000}k" for r in ranked.itertuples()]
    bars = [RED if r.exceeds_gpu else "#4C78A8" for r in ranked.itertuples()]
    ax.barh(range(len(ranked)), ranked["score"], color=bars)
    ax.set_yticks(range(len(ranked)), names, fontsize=8)
    for y, r in enumerate(ranked.itertuples()):
        ax.text(r.score + 0.05, y, f"rank {int(r.rank)}  |  {r.total_gb:.1f} GB  |  "
                f"{r.decode_ceiling_tps:.0f} tok/s ceiling", va="center", fontsize=7)
    ax.set_xlabel("Rank-sum score (lower is better)")
    title = "Ranking by speed ceiling and memory headroom, per context"
    if unknown:
        title += f"\nnot ranked, size unknown: {', '.join(unknown)}"
    ax.set_title(title)
    if len(ranked):
        ax.set_xlim(0, ranked["score"].max() * 1.6)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(handles=[Patch(color=RED, label="above GPU memory"), Patch(color="#4C78A8", label="within GPU memory")],
              loc="lower right", fontsize=8, frameon=False)
    p = plots_dir / "D4_ranking.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # 5. decode-speed ceiling against context for one quantization; red = above GPU memory
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, label in enumerate(labels):
        g = sub[sub["label"] == label].sort_values("context").dropna(subset=["decode_ceiling_tps"])
        if g.empty:
            continue
        colour = colours(i % 10)
        ax.plot(g["context"], g["decode_ceiling_tps"], color=colour, lw=1.5, label=label)
        for _, r in g.iterrows():
            ax.scatter(r["context"], r["decode_ceiling_tps"], color=RED if r["total_gb"] > vram_gb else colour, s=40, zorder=3)
    ax.set_xticks(contexts, [f"{c // 1000}k" for c in contexts])
    ax.set_xlabel("Context window (tokens)")
    ax.set_ylabel("Decode-speed ceiling (tokens/s, upper bound)")
    ax.set_title(f"Speed ceiling against context ({plot_quant}); red = model above GPU memory")
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    p = plots_dir / "D5_speed_vs_context.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)
    return paths


def kv_explanation(profile):
    if not profile:
        return ""
    kv = profile["kv_heads"]
    kv_text = "sum of per-layer KV heads" if isinstance(kv, list) else f"{kv} KV heads"
    hybrid = (f", only {profile['attention_layers']} of {profile['layers']} layers hold KV "
              f"(full attention every {profile['full_attention_interval']})"
              if profile["full_attention_interval"] else f", {profile['layers']} layers")
    return (f"2 bytes x {profile['attention_layers']} attention layers x {kv_text} x "
            f"({profile['key_len']} key + {profile['val_len']} value){hybrid}")


def dryrun_console(df, models, contexts, quants, index=()):
    for model in models:
        total, active, size_source = model_params(model)
        known = lookup_model(model)
        if known and known.get("max_context"):
            print(f"{model}: context limit {known['max_context']:,} tokens; larger windows are not computed")
        sub = df[df["model"] == model]
        if sub.empty:
            print("  no context to compute (all requested contexts are above the model limit)\n")
            continue
        first = sub.iloc[0]
        profile = model_kv_profile(index, model)
        print(f"{model}\n  {_num(total)} B total, {_num(active)} B active per token ({size_source})")
        print(f"  KV cache: {int(first['kv_bytes_per_token']):,} bytes per token ({first['kv_source']})")
        if profile:
            print(f"    = {kv_explanation(profile)}")
        for quant in quants:
            hit = find_gguf(index, model, quant)
            print(f"  {quant}: " + (f"GGUF file {hit['path']}" if hit else "no GGUF file found; weights estimated"))
        rows = []
        for quant in quants:
            q = sub[sub["quant"] == quant].sort_values("context")
            rows.append([quant, _num(q["weights_gb"].iloc[0], 2), q["weights_source"].iloc[0]]
                        + [_num(v, 2) for v in q["total_gb"]])
        print_table(["quant", "weights GB", "weights from"] + [f"total GB @{c // 1000}k" for c in contexts], rows)
        print()
        rows = []
        for quant in quants:
            q = sub[sub["quant"] == quant].sort_values("context")
            rows.append([quant] + [_num(v, 0) for v in q["decode_ceiling_tps"]])
        print_table(["quant"] + [f"tok/s @{c // 1000}k" for c in contexts], rows)
        print()


def dryrun_plan(args, models, contexts):
    items = None
    if args.benchmark == "suite":
        if SS is None:
            print("Suite mode: suite_scoring.py and docs.py are not in the package folder.")
        elif not args.suite:
            print("Suite mode: --suite is missing.")
        else:
            items = len(SS.load_suite(args.suite).items)
            try:
                docs = DOCS.load_docs(args.docs, limit=args.docs_limit, ids=args.docs_ids)
                print(f"Documents: {len(docs)} files, about "
                      f"{sum(d.est_tokens for d in docs):,} tokens (estimate)")
            except Exception as e:
                print(f"Documents: not checked ({e})")
    else:
        items = BENCHMARK_ITEMS[args.benchmark]
    if items is not None and args.limit is not None:
        items = min(items, args.limit)

    configs = len(models) * len(contexts)
    print(f"\nRun plan: {configs} configurations ({len(models)} models x {len(contexts)} contexts)")
    if items is not None:
        print(f"Items per configuration: {items}   total evaluations: {configs * items:,}")


def dryrun_report(args, models, contexts):
    out_file = Path(args.results)
    out_dir = out_file.parent
    quants = list(args.quants)
    vram_gb, vram_src = resolve_vram(args.vram_gb)
    roots = list(LOCAL_MODEL_DIRS) + [Path(p) for p in (args.model_dir or [])]
    index = scan_gguf_index(roots)

    print("\n" + "=" * 80 + "\nDRY RUN: estimates only; nothing is loaded and no requests are sent\n"
          + "=" * 80)
    print(f"GPU memory: {vram_gb:.1f} GB ({vram_src}) | GPU bandwidth: {args.gpu_bandwidth:.0f} GB/s "
          f"| system RAM bandwidth: {args.ram_bandwidth:.0f} GB/s")
    print(f"GGUF files found: {len(index)} in " + (", ".join(str(r) for r in roots if Path(r).is_dir()) or "no model folder"))
    print(f"Quantizations: {', '.join(quants)}")
    print(f"Benchmark: {args.benchmark} | contexts: " + ", ".join(f"{c // 1000}k" for c in contexts))
    print()

    df = dryrun_rows(models, contexts, vram_gb, args.gpu_bandwidth, args.ram_bandwidth, quants,
                     index, kv_default=args.kv_bytes, overhead_gb=args.overhead_gb)
    df.to_csv(out_file, index=False)
    board = dryrun_leaderboard(df)
    board.to_csv(out_dir / DRYRUN_LEADERBOARD, index=False)

    dryrun_console(df, models, contexts, quants, index)
    dryrun_plan(args, models, contexts)

    print("\nLeaderboard (every model, quantization and context; speed ceiling and memory headroom):")
    print_table(["rank", "model", "quant", "context", "total GB", "headroom GB", "bandwidth", "decode ceiling tok/s"],
                [[_num(r.rank, 0), r.label, r.quant, f"{int(r.context) // 1000}k", _num(r.total_gb),
                  _num(r.headroom_gb), r.bandwidth_used, _num(r.decode_ceiling_tps, 0)]
                 for r in board.itertuples()])
    print("\nThe ranking uses estimates only. Quality needs a benchmark run.")
    print("Decode ceilings are upper bounds. Where the model does not fit in VRAM, the part in")
    print("system RAM is read at the RAM bandwidth, so the ceiling is blended by the offloaded share.")

    paths = plot_dryrun(df, board, vram_gb, out_dir / "plots")
    print(f"\nWritten to {out_dir}/: {out_file.name}, {DRYRUN_LEADERBOARD}")
    for p in paths:
        print(f"         {p}")


RESULT_NAMES = {"humaneval": "coding_benchmark_results.csv",
                "mbpp": "coding_benchmark_results_mbpp.csv",
                "swebench": "coding_benchmark_results_swebench.csv"}
LOCAL_MODEL_DIRS = [Path.home() / ".lmstudio" / "models",
                    Path.home() / ".cache" / "lm-studio" / "models"]


def resolve_output(explicit, default_name, bench_dir):
    """
    Output file for a job. A file found in the current folder is resumed first (as
    before); otherwise the benchmark folder is used. If neither has the file, it is
    created in the benchmark folder.
    """
    if explicit:
        path = Path(explicit)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    here = Path(default_name)
    if here.exists():
        print(f"Resuming from {here} (found in the current folder)")
        return here
    there = bench_dir / default_name
    if there.exists():
        print(f"Resuming from {there}")
    return there


def discover_models(base_url):
    """All local models: from the LM Studio server if it runs, otherwise from the model folders."""
    try:
        r = requests.get(api(base_url, "/api/v1/models"), timeout=3)
        r.raise_for_status()
        keys = [m["key"] for m in r.json().get("models", [])
                if m.get("key") and m.get("type", "llm") == "llm"]
        if keys:
            return sorted(set(keys)), "the LM Studio server"
    except Exception:
        pass
    keys = set()
    for root in LOCAL_MODEL_DIRS:
        if not root.is_dir():
            continue
        for gguf in root.rglob("*.gguf"):
            parts = gguf.relative_to(root).parts
            key = f"{parts[0]}/{parts[1]}" if len(parts) >= 3 else (parts[0] if len(parts) == 2 else gguf.stem)
            if "embed" not in key.lower():
                keys.add(key)
    source = "the model folders on disk (" + ", ".join(str(r) for r in LOCAL_MODEL_DIRS if r.is_dir()) + ")"
    return sorted(keys), source


def main():
    DEFAULT_LM_STUDIO_URL = "http://localhost:1234"
    CONTEXTS = [50_000, 80_000, 100_000, 120_000, 150_000, 180_000]
    MODELS = []

    parser = argparse.ArgumentParser(description="Benchmark local coding LLMs through LM Studio.")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--contexts", nargs="+", type=int, default=CONTEXTS)
    parser.add_argument("--base-url", default=DEFAULT_LM_STUDIO_URL)
    parser.add_argument("--benchmark", choices=["humaneval", "mbpp", "swebench", "suite"],
                        default="humaneval", help="Which benchmark to run.")
    parser.add_argument("--suite", default=None,
                        help="Suite file (.jsonl) for --benchmark suite, e.g. suites/northwind_demo.jsonl")
    parser.add_argument("--docs", default="company_docs",
                        help="Folder with the company documents (default: company_docs).")
    parser.add_argument("--docs-limit", type=int, default=None,
                        help="Use only the first N documents (name order).")
    parser.add_argument("--judge", default=None, metavar="MODEL",
                        help="Suite mode: score open-ended answers with this LM Studio model. "
                             "Default: open-ended answers are saved but not scored.")
    parser.add_argument("--judge-context", type=int, default=16000,
                        help="Context length for the judge model (default 16000).")
    parser.add_argument("--closed-book", action="store_true",
                        help="Suite mode: ask the questions without documents (automatic when the "
                             "suite cites no documents).")
    parser.add_argument("--docs-ids", nargs="+", default=None,
                        help="Use exactly these document file names.")
    parser.add_argument("--results", default=None,
                        help="Results CSV (default depends on --benchmark).")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N problems.")
    parser.add_argument("--plots-only", action="store_true", help="Do not run anything; just (re)generate plots from the existing results CSV.")
    parser.add_argument("--dry-run", "--dryrun", dest="dry_run", action="store_true",
                        help="Print memory and speed estimates and the run plan; load nothing.")
    parser.add_argument("--gpu-bandwidth", type=float, default=GPU_BANDWIDTH,
                        help="GPU memory bandwidth in GB/s, used by --dry-run.")
    parser.add_argument("--vram-gb", type=float, default=None,
                        help="GPU memory in GB, used by --dry-run (default: read from nvidia-smi).")
    parser.add_argument("--ram-bandwidth", type=float, default=RAM_BANDWIDTH,
                        help="System RAM bandwidth in GB/s for the part of a model that does not fit "
                             "in VRAM (--dry-run). Typical range 8 to 60.")
    parser.add_argument("--overhead-gb", type=float, default=DRYRUN_OVERHEAD_GB,
                        help="Fixed GB added to every --dry-run estimate (compute buffers, runtime). "
                             "Calibrate against the LM Studio 'Estimated Memory Usage' value.")
    parser.add_argument("--kv-bytes", type=int, default=KV_BYTES_PER_TOKEN,
                        help="KV-cache bytes per token for models without a GGUF file (--dry-run).")
    parser.add_argument("--model-dir", action="append", default=None,
                        help="Extra folder with GGUF model files for --dry-run (repeatable).")
    parser.add_argument("--quants", nargs="+", default=DRYRUN_DEFAULT_QUANTS, choices=list(QUANT_BITS),
                        help="Quantizations for --dry-run (default: Q4_K_M Q6_K Q8_0; "
                             "add Q3_K_M and Q5_K_M explicitly).")
    args = parser.parse_args()

    # Fixed defaults (no longer CLI flags)
    args.no_plots = False
    args.max_tokens = SWE_MAX_TOKENS if args.benchmark == "swebench" else DEFAULT_MAX_TOKENS
    args.fill = 0.0
    args.vary_filler = False
    args.probe = False
    args.probe_tokens = list(PROBE_TOKENS)
    args.probe_reps = 2

    if args.benchmark == "suite" and SS is None:
        print("STOP: suite mode needs suite_scoring.py and docs.py in the localbenchmark "
              "package folder (next to localbenchmark.py and llm_plots.py).")
        return
    if args.benchmark == "suite" and not args.dry_run and not args.suite:
        print("STOP: --benchmark suite needs --suite <file.jsonl>")
        return

    args.base_url = args.base_url.rstrip("/")
    contexts = sorted(set(args.contexts))

    # Every job writes into its own folder: dryrun/, humaneval/, mbpp/, swebench/, suite/
    bench_dir = Path("dryrun") if args.dry_run else Path(args.benchmark)
    bench_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir = str(bench_dir / "plots")
    if args.dry_run:
        results_file = Path(args.results) if args.results else bench_dir / DRYRUN_CSV
        probe_file = bench_dir / "speed_probe.csv"
    else:
        name = (f"suite_results_{SS.load_suite(args.suite).name}.csv" if args.benchmark == "suite"
                else RESULT_NAMES[args.benchmark])
        results_file = resolve_output(args.results, name, bench_dir)
        probe_file = resolve_output(None, "speed_probe.csv", bench_dir)
    args.results = str(results_file)
    args.probe_file = str(probe_file)
    diag_file = results_file.with_name(results_file.stem + "_diagnostics.jsonl")

    print("\n" + "=" * 80 + "\nLOCAL LLM CODING BENCHMARK\n" + "=" * 80)
    print(f"Output folder: {bench_dir}/")
    print("Contexts:", [f"{c:,}" for c in contexts])

    if args.plots_only:
        run_plots(results_file, args.plots_dir, probe_file, diag_file)
        return

    if not args.models:
        keys, source = discover_models(args.base_url)
        if not keys:
            print("STOP: no --models given and no local models were found.\n"
                  "Load a model in LM Studio, or pass --models \"openai/gpt-oss-20b\".")
            return
        args.models = keys
        print(f"No --models given: using {len(keys)} model(s) from {source}.")
    print("Models:", *args.models, sep="\n  - ")

    if args.dry_run:
        dryrun_report(args, args.models, contexts)
        return

    check_lm_studio(args.base_url)
    repair_results_file(results_file)
    if args.benchmark != "suite":
        check_results_schema(results_file)

    docs_context = None
    if args.probe:
        df = None
        print(f"\nMODE: speed probe (no HumanEval). Levels: {args.probe_tokens}, "
              f"reps: {args.probe_reps}")
    else:
        if args.benchmark == "suite":
            print(f"\nLoading suite {args.suite} and documents from {args.docs}...")
            suite, df, docs, docs_context = prepare_suite(args)
            print(f"Documents in prompt: {len(docs)} "
                  f"({DOCS.estimate_tokens(docs_context):,} tokens, estimate)")
        else:
            print(f"\nLoading {args.benchmark} dataset...")
            df = load_benchmark(args.benchmark)
        if args.limit is not None:
            df = df.head(args.limit)
        print(f"Tasks in evaluation dataset: {len(df)}")


    model_contexts = {}
    for model in args.models:
        try:
            model_contexts[model] = determine_model_contexts(
                args.base_url, model, contexts)
        except Exception as e:
            print(f"SKIPPING {model}: {e}")

    plan = [
        (m, c, kv)
        for m, ctxs in model_contexts.items()
        for c in ctxs
        for kv in CACHE_CONFIGS
    ]
    if args.benchmark == "suite":
        need = DOCS.estimate_tokens(docs_context) + SUITE_QUESTION_TOKENS
        fits = [p for p in plan if need + DOCS.ANSWER_HEADROOM <= p[1]]
        for m, c, _ in plan:
            if (m, c, _) not in fits:
                print(f"SKIP {m} @ {c:,}: documents need about {need:,} tokens, window too small")
        plan = fits

    problem_ids = [] if args.probe else list(df["task_id"])
    print(f"\nConfigurations: {len(plan)}"
          + ("" if args.probe else f"   Evaluations: {len(plan) * len(df):,}"))

    summary_configs = load_summary_configs(results_file)

    try:
        for i, (model, context, kv) in enumerate(plan, 1):
            force_rerun = False
            if args.probe:
                done, total = probe_progress(probe_file, model, context,
                                             kv["label"], args)
            else:
                exp_id = experiment_id_for(model, context, kv["label"], args.fill)
                config_key = (model, kv["label"], context, float(args.fill))
                in_summary = config_key in summary_configs

                total = len(problem_ids)
                if not in_summary:
                    # Config absent from summary CSV → force full re-run
                    # of every problem, ignoring any partial results rows.
                    force_rerun = True
                    done = 0
                else:
                    # Present in summary (including early-aborted runs) →
                    # treat as finished; do not load the model.
                    done = total

            print(f"\n{'#' * 80}\n# CONFIG {i}/{len(plan)}: {model} | "
                  f"ctx {context:,} | kv={kv['label']}\n{'#' * 80}")

            if done == total:
                print(f"Already complete ({done}/{total}) "
                      f"-> skipping, model will not be loaded.")
                if args.benchmark == "suite" and args.judge:
                    run_judge_stage(df, args, results_file,
                                    experiment_id_for(model, context, kv["label"], args.fill))
                    suite_summary(results_file, args.judge)
                continue
            if force_rerun:
                print(f"Not listed in summary CSV "
                      f"-> re-running entire experiment "
                      f"({total} problems).")
            elif done:
                print(f"Resuming: {done}/{total} already done.")

            try:
                unload_all_models(args.base_url)
                load = load_model(args.base_url, model, context,
                                  kv["load_params"])

                instance_id = load.get("instance_id")
                meta = {
                    "model": model,
                    "server_model": instance_id or model,
                    "instance_id": instance_id,
                    "context_requested": context,
                    "context_actual": get_actual_context(load),
                    "cache_label": kv["label"],
                    "load_time": load["_benchmark_load_time"],
                }
                gpu_load, gpu_total = get_gpu_memory()
                meta["gpu_after_load"] = gpu_load
                write_diagnostics(diag_file, "probe" if args.probe else "benchmark",
                                  meta, load, gpu_load, gpu_total)

                if args.probe:
                    run_probe(meta, args, probe_file)
                else:
                    if args.benchmark == "swebench":
                        run_swebench_experiment(df, meta, args, results_file)
                    elif args.benchmark == "suite":
                        run_suite_experiment(df, meta, args, results_file, docs_context,
                                             force_rerun=force_rerun)
                    else:
                        run_experiment(df, meta, args, results_file,
                                       force_rerun=force_rerun)
                    create_summary(results_file)

            except Exception as e:
                print(f"\nCONFIG ERROR: {e}\nContinuing...")

            finally:
                try:
                    unload_all_models(args.base_url)
                except Exception as e:
                    print(f"Warning: could not unload model: {e}")

    except KeyboardInterrupt:
        print("\n\nInterrupted. Every finished problem is already saved.\n"
              "Run the same command again to resume where it stopped.")

    create_summary(results_file)
    if not args.no_plots:
        run_plots(results_file, args.plots_dir, probe_file, diag_file)

    print("\n" + "=" * 80 + "\nDONE\n" + "=" * 80)
    print(f"Results: {results_file}")


def run_plots(results_file, plots_dir, probe_file=None, diag_file=None):
    try:
        llm_plots.make_all(results_file, plots_dir)
    except ImportError as e:
        print(f"Plotting skipped ({e}). Install matplotlib: pip install matplotlib")
        return
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == "__main__":
    # Examples:
    #   python llm_coding_benchmark.py --limit 10
    #   python llm_coding_benchmark.py --fill 0.5 --contexts 50000 100000
    #   python llm_coding_benchmark.py --models "google/gemma-4-26b-a4b-qat" \
    #       --fill 0.7 --vary-filler --limit 20
    main()
