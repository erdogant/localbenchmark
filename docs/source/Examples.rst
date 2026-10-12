Quickstart — coding benchmark
################################

Start LM Studio, enable the local server, load a coding model, then run:

.. code-block:: console

    # Default model list × default context grid
    python -m localbenchmark.llm_coding_benchmark

    # One model, selected windows
    python -m localbenchmark.llm_coding_benchmark \
      --models openai/gpt-oss-20b \
      --contexts 50000 80000 100000 \
      --base-url http://localhost:1234

    # Fill half the context with unrelated code before each problem
    python -m localbenchmark.llm_coding_benchmark --fill 0.5

    # Only the first 20 HumanEval problems (smoke test)
    python -m localbenchmark.llm_coding_benchmark --limit 20


Speed probe
################

Measure prefill / decode speed at controlled tokens-in-cache levels (no HumanEval):

.. code-block:: console

    python -m localbenchmark.llm_coding_benchmark --probe


Plots and ranking
#################

After results exist in CSV:

.. code-block:: console

    python -m localbenchmark.plot_results \
      --results coding_benchmark_results.csv \
      --out plots \
      --rank-agg sum

    # Focus on large windows only (context_actual > 100000)
    python -m localbenchmark.plot_results --min-context 100000 --out plots


Useful CLI flags
#################

**Benchmark** (``llm_coding_benchmark``):

* ``--models`` — LM Studio model keys
* ``--contexts`` — token window sizes (default includes 50k, 80k, 100k, 120k, 150k, 180k)
* ``--fill`` — fraction of the window to fill with filler (0–0.9)
* ``--limit`` — first N HumanEval problems
* ``--results`` — results CSV path
* ``--probe`` — speed probe instead of HumanEval
* ``--no-plots`` / ``--plots-only``

**Plots** (``plot_results``):

* ``--rank-agg`` — ``sum`` (default), ``mean``, or ``median``
* ``--min-context`` — keep rows with ``context_actual`` strictly greater than this value


What is ranked
#################

Each **model × context** (and a per-model roll-up) is scored on:

1. Pass rate (early aborts count unfinished work against the model)
2. Decode speed (percentage of best)
3. Time to first token
4. Peak GPU memory
5. Context size (percentage of best)

Pass rate and decode speed are weighted **×2**. Lower combined score is better.


.. include:: add_bottom.add
