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

def main():
    DEFAULT_LM_STUDIO_URL = "http://localhost:1234"
    CONTEXTS = [50_000, 80_000, 100_000, 120_000, 150_000, 180_000]
    MODELS = []

    parser = argparse.ArgumentParser(description="Benchmark local coding LLMs through LM Studio.")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--contexts", nargs="+", type=int, default=CONTEXTS)
    parser.add_argument("--base-url", default=DEFAULT_LM_STUDIO_URL)
    parser.add_argument("--benchmark", choices=["humaneval", "mbpp", "swebench"],
                        default="humaneval", help="Which benchmark to run.")
    parser.add_argument("--results", default=None,
                        help="Results CSV (default depends on --benchmark).")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N problems.")
    parser.add_argument("--plots-only", action="store_true", help="Do not run anything; just (re)generate plots from the existing results CSV.")
    args = parser.parse_args()

    # Fixed defaults (no longer CLI flags)
    args.no_plots = False
    args.max_tokens = SWE_MAX_TOKENS if args.benchmark == "swebench" else DEFAULT_MAX_TOKENS
    args.fill = 0.0
    args.vary_filler = False
    args.plots_dir = "plots"
    args.probe = False
    args.probe_tokens = list(PROBE_TOKENS)
    args.probe_reps = 2
    args.probe_file = "speed_probe.csv"

    if args.results is None:
        args.results = {"humaneval": "coding_benchmark_results.csv",
                        "mbpp": "coding_benchmark_results_mbpp.csv",
                        "swebench": "coding_benchmark_results_swebench.csv"}[args.benchmark]

    args.base_url = args.base_url.rstrip("/")
    results_file = Path(args.results)
    probe_file = Path(args.probe_file)
    diag_file = results_file.with_name(results_file.stem + "_diagnostics.jsonl")
    contexts = sorted(set(args.contexts))

    print("\n" + "=" * 80 + "\nLOCAL LLM CODING BENCHMARK\n" + "=" * 80)
    print("Models:", *args.models, sep="\n  - ")
    print("Contexts:", [f"{c:,}" for c in contexts])
    # print("Cache configs:", [c["label"] for c in CACHE_CONFIGS])
    # print(f"Fill ratio: {args.fill}  (vary filler: {args.vary_filler})")
    # if args.fill == 0:
    #     print("NOTE: fill=0 -> prompts are tiny; context size will not affect accuracy.")

    if args.plots_only:
        run_plots(results_file, args.plots_dir, probe_file, diag_file)
        return

    # Per-model context discovery (a bad model no longer aborts the run).
    if args.models is None or len(args.models) == 0:
        print("\n" + "=" * 80 )
        print('STOP: No models are specified.')
        print('localbenchmark --models "qwen3.8-27b" "openai/gpt-oss-20b" --contexts "50000" "80000"')
        print("=" * 80 )
        return

    check_lm_studio(args.base_url)
    repair_results_file(results_file)
    check_results_schema(results_file)

    if args.probe:
        df = None
        print(f"\nMODE: speed probe (no HumanEval). Levels: {args.probe_tokens}, "
              f"reps: {args.probe_reps}")
    else:
        print(f"\nLoading {args.benchmark} dataset...")
        df = load_benchmark(args.benchmark)
        if args.limit is not None:
            df = df.head(args.limit)
        print(f"Coding tasks in Evaluation dataset: {len(df)}")


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
