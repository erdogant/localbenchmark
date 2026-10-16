[![Python](https://img.shields.io/pypi/pyversions/localbenchmark)](https://img.shields.io/pypi/pyversions/localbenchmark)
[![Pypi](https://img.shields.io/pypi/v/localbenchmark)](https://pypi.org/project/localbenchmark/)
[![Docs](https://img.shields.io/badge/Sphinx-Docs-Green)](https://erdogant.github.io/localbenchmark/)
[![LOC](https://sloc.xyz/github/erdogant/localbenchmark/?category=code)](https://github.com/erdogant/localbenchmark/)
[![Downloads](https://static.pepy.tech/personalized-badge/localbenchmark?period=month&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads/month)](https://pepy.tech/project/localbenchmark)
[![Downloads](https://static.pepy.tech/personalized-badge/localbenchmark?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/localbenchmark)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/localbenchmark/blob/master/LICENSE)
[![Forks](https://img.shields.io/github/forks/erdogant/localbenchmark.svg)](https://github.com/erdogant/localbenchmark/network)
[![Issues](https://img.shields.io/github/issues/erdogant/localbenchmark.svg)](https://github.com/erdogant/localbenchmark/issues)
[![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Medium](https://img.shields.io/badge/Medium-Blog-black)](https://erdogant.github.io/localbenchmark/pages/html/Documentation.html#medium-blog)
[![Donate](https://img.shields.io/badge/Support%20this%20project-grey.svg?logo=github%20sponsors)](https://erdogant.github.io/localbenchmark/pages/html/Documentation.html#)

<div>
<a href="https://erdogant.github.io/localbenchmark/">
  <img src="https://raw.githubusercontent.com/erdogant/localbenchmark/master/docs/source/figs/logo.png"
       width="100"
       align="left" />
</a>

**localbenchmark** is a Python package for **local LLM evaluation**: coding benchmarks, long-context stress tests, speed probes, and ranking plots — driven through [LM Studio](https://lmstudio.ai/) (OpenAI-compatible API).
It answers a practical question: *can this model still write correct code when the context window is large, and is it fast enough that you would actually use it?* ⭐️Star it if you like it⭐️
</div>

<br>

---

### Key Features

| Feature | Description |
|--------|-------------|
| **Coding benchmark** | HumanEval-style tasks via LM Studio; temperature 0 for comparable runs |
| **Long context** | Test fixed windows (e.g. 50k–180k); optional context *fill* so the KV cache is exercised |
| **Speed probe** | Prefill / decode tok/s at controlled tokens-in-cache levels |
| **Resume-safe CSVs** | Durable results; skip finished configs; retry LLM errors |
| **Slow-model abort** | Stop a config after a streak of low decode speed; ranking accounts for incomplete runs |
| **Rank-sum plots** | Pass rate, decode speed, TTFT, GPU peak, context — weighted ranking (model × context and per model) |

---


### Installation

##### Install from PyPI
```bash
pip install localbenchmark
```

##### Install from GitHub
```bash
pip install git+https://github.com/erdogant/localbenchmark
```

##### Install from source
```bash
pip install git+https://github.com/erdogant/localbenchmark
```

---

### Quick start — Dry run

Make a dry run. This is ideal for a fast check of model GPU requirements and their theoretical (expected) coding speed.

```bash
# Run for all models with context window 120K and quantization of Q4_K_M

localbenchmark --dry-run --contexts 120000 --quants Q4_K_M
localbenchmark --dry-run --contexts 120000 --quants Q4_K_M --models unsloth/qwen3-coder-30b-a3b-instruct openai/gpt-oss-20b

```

Output:

```bash
rank  model                         quant   context  total GB  headroom GB  offload %  bandwidth  decode ceiling tok/s
----------------------------------------------------------------------------------------------------------------------
1     gpt-oss-20b                   Q4_K_M  120k     13.1      4.0          0%         GPU        74
2     qwen3-coder-30b-a3b-instruct  Q4_K_M  120k     18.5      -1.4         8%         GPU+RAM    21

```

---

### Quick start — coding benchmark

Start LM Studio, enable the local server, load a coding model, then:

```bash
# Full default suite (models × context windows)
localbenchmark

# Full default with base URL
localbenchmark --base-url http://localhost:1234

# Subset of models/contexts
localbenchmark --models unsloth/qwen3-coder-30b-a3b-instruct openai/gpt-oss-20b --contexts 50000 80000 100000 --base-url http://localhost:1234

```

---

Start LM Studio, enable the local server, load a coding model, then:

```bash
# Different types of benchmarks

localbenchmark --benchmark humaneval --models "openai/gpt-oss-20b"
localbenchmark --benchmark mbpp      --models "openai/gpt-oss-20b"
localbenchmark --benchmark swebench  --models "openai/gpt-oss-20b" --contexts 80000 150000

```

<a href="https://erdogant.github.io/localbenchmark/">
  <img src="https://raw.githubusercontent.com/erdogant/localbenchmark/master/docs/source/figs/09_rank_sum.png" width="75%" alt="Localbenchmark model ranking comparison">
</a>

Results are written to CSV. Configurations already present in the **summary** CSV are skipped on the next run. Configurations that abort for low tok/s stay in the summary so they are not reloaded endlessly.

**Useful flags**

| Flag | Meaning |
|------|---------|
| `--models` | One or more LM Studio model keys |
| `--contexts` | Context sizes in tokens (default includes 50k, 80k, 100k, 120k, 150k, 180k) |
| `--limit` | Run only the first N HumanEval problems |
| `--results` | Results CSV path (default depends on `--benchmark`) |
| `--benchmark` | `humaneval`, `mbpp`, `swebench` or `suite` |
| `--suite` | Suite file (`.jsonl`) for `--benchmark suite` |
| `--docs` | Document folder for suite mode (default `company_docs`) |
| `--docs-limit` | Use only the first N documents, in name order |
| `--docs-ids` | Use exactly these document file names |

---




### Custom (Company) QA

Suite mode asks questions about a set of company documents. Every selected document goes into the prompt, and we score each answer. The suite includes **unanswerable** questions: the correct answer is to say the documents do not contain the information. An invented answer scores zero, which is the most important check for company use.

#### Commands

Run the suite with LM Studio running:

```bash
Example of lines:
{"id":"ctry-01","domain":"countries","type":"short","prompt":"What is the capital of Australia?","accept":["Canberra"],"distractors":["Sydney","Melbourne"]}
{"id":"ctry-03","domain":"countries","type":"numeric","prompt":"Roughly how large is Iceland, in square kilometres?","gold":103000,"rel_tol":0.05}
{"id":"co-04","domain":"companies","type":"short","prompt":"Which company acquired LinkedIn in 2016?","accept":["Microsoft"],"distractors":["Google","Oracle"]}
{"id":"co-01","domain":"companies","type":"numeric","prompt":"In which year was Microsoft founded?","gold":1975}

localbenchmark --benchmark suite --suite general_knowledge.jsonl --models "openai/gpt-oss-20b" --contexts 120000
```

#### Run with longer/ more detailled answers

- `company_docs/`: the document folder. Any `.txt` or `.md` file can be used.
- `suites/<name>.jsonl`: one question per line. Each question names its source documents in `context`.

The demo suite (`northwind_demo`) is fictional: 12 documents for a company called Northwind Analytics and 40 questions, 30 answerable and 10 unanswerable.

```bash
Example of jsonl:

{"id": "nw-01", "domain": "hr", "type": "short", "prompt": "How many days of annual leave do full-time employees get per calendar year?", "accept": ["25"], "context": ["hr_leave_policy.md"]}

```

#### Commands

Run the suite with LM Studio running:

```bash
localbenchmark --benchmark suite --suite northwind_demo.jsonl --docs path/to/company_docs \
    --models "openai/gpt-oss-20b" --contexts 120000
```

Use fewer documents:

```bash

# first 40 files, in name order
localbenchmark --benchmark suite --suite northwind_demo.jsonl --docs path/to/company_docs \
    --models "openai/gpt-oss-20b" --docs-limit 40

localbenchmark --benchmark suite --suite northwind_demo.jsonl --docs path/to/company_docs \
    --models "openai/gpt-oss-20b" --docs-ids hr_leave_policy.md expense_policy.md
```

#### Output

- `suite_results_<suite>.csv`: one row per question, with the score (0 to 1), pass/fail, domain, item type, whether the cited documents were included, and an error label (`WRONG_ANSWER`, `INVENTED_ANSWER`, `MISSING_DOC`, `TRUNCATED`, `EMPTY_RESPONSE`, `LLM_ERROR`).
- `suite_results_<suite>_by_domain.csv`: pass rate and mean score per domain. The console also shows answerable and unanswerable results separately, and the number of invented answers.


### Plots and ranking

```bash
localbenchmarkplot --rank-agg sum      # default
localbenchmarkplot --rank-agg mean
localbenchmarkplot --rank-agg median

# Only large windows (context_actual >= 100000)
localbenchmarkplot --min-context 100000
```

**Ranking metrics** (lower combined score is better):

1. Pass rate (early aborts count unfinished work against the model)
2. Decode speed (percentage of best — large speed gaps stay large)
3. Time to first token
4. Peak GPU memory
5. Context size (percentage of best)

Pass rate and decode speed are **weighted**. Figures include pass rate vs context, tok/s, TTFT, GPU memory, quality–speed trade-off, and rank-sum panels (model × context and per model).

---

### What is measured

| Signal | Why it matters |
|--------|----------------|
| **Pass rate** | Generated code must pass the unit tests |
| **Decode speed (tok/s)** | Inference has to be usable, not only correct |
| **TTFT** | Latency before the first useful token |
| **Peak GPU memory** | Whether the setup fits your hardware |
| **Context** | Behaviour as the window grows |

Default context grid:

```text
50k → 80k → 100k → 120k → 150k → 180k
```

Any context sizes present in the results file are included in the rank plots (not limited to a hard-coded list).

---

### Resources and Links

- **Documentation:** [Website](https://erdogant.github.io/localbenchmark)
- **Issues:** [GitHub Issues](https://github.com/erdogant/localbenchmark/issues)
- **Source:** [github.com/erdogant/localbenchmark](https://github.com/erdogant/localbenchmark)

---

### Maintainer

* Erdogan Taskesen, GitHub: [erdogant](https://github.com/erdogant)
* Contributions are welcome.
* Yes! This library is entirely **free** but it runs on coffee! Feel free to support with a <a href="https://erdogant.github.io/donate/?currency=USD&amount=5">Coffee</a>.

[![Buy me a coffee](https://img.buymeacoffee.com/button-api/?text=Buy+me+a+coffee&emoji=&slug=erdogant&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff)](https://www.buymeacoffee.com/erdogant)
