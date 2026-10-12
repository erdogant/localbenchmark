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
  <img src="https://raw.githubusercontent.com/erdogant/localbenchmark/master/docs/figs/logo.png"
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

### Quick start — coding benchmark

Start LM Studio, enable the local server, load a coding model, then:

```bash
# Full default suite (models × context windows)
localbenchmark

# Full default with base url
localbenchmark --base-url http://localhost:1234

# Subset of models / contexts
localbenchmark --models qwen3.8-27b openai/gpt-oss-20b --contexts 50000 80000 100000 --base-url http://localhost:1234

```

Results are written to CSV. Configurations already present in the **summary** CSV are skipped on the next run. Configurations that abort for low tok/s stay in the summary so they are not reloaded endlessly.

**Useful flags**

| Flag | Meaning |
|------|---------|
| `--models` | One or more LM Studio model keys |
| `--contexts` | Context sizes in tokens (default includes 50k, 80k, 100k, 120k, 150k, 180k) |
| `--limit` | Run only the first N HumanEval problems |
| `--results` | Results CSV path (default `coding_benchmark_results.csv`) |

---

### Plots and ranking

```bash
localbenchmarkplot --rank-agg sum      # default
localbenchmarkplot --rank-agg mean
localbenchmarkplot --rank-agg median

# Only large windows (context_actual >= 100000)
localbenchmarkplot --min-context 100000
```

| Flag | Meaning |
|------|---------|
| `--rank-agg` | Combine per-metric ranks: `sum` (default), `mean`, or `median` |
| `--min-context` | Keep rows with `context_actual` **strictly greater** than this value |

**Ranking metrics** (lower combined score is better):

1. Pass rate (early aborts count unfinished work against the model)
2. Decode speed (percentage of best — large speed gaps stay large)
3. Time to first token
4. Peak GPU memory
5. Context size (percentage of best)

Pass rate and decode speed are **weighted ×2**. Figures include pass rate vs context, tok/s, TTFT, GPU memory, quality–speed trade-off, and rank-sum panels (model × context and per model).

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
