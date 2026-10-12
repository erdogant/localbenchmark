
Results, resume, and summary
''''''''''''''''''''''''''''''

Learning a full model×context grid can take many hours. **localbenchmark** writes every
finished problem to CSV immediately so a crash does not lose progress.

Results CSV
----------------

Default path: ``coding_benchmark_results.csv``.

Each row is one problem for one configuration (model, context, fill ratio, cache label).
Fields include pass/fail, error type, TTFT, decode speed, token counts, GPU memory, and the generated code.

Summary CSV
----------------

After each configuration (and at the end of a run), a summary file is written:

``coding_benchmark_results_summary.csv``

Configurations present in the summary are treated as finished on the next launch
and are **not** reloaded. Configurations missing from the summary are run again
(including early aborts that were recorded).

Resume behaviour
----------------

* Re-run the **same command** to continue where you left off
* Rows with ``error_type == LLM_ERROR`` are retried
* Other outcomes (pass, test failure, truncation, …) stay on disk
* Slow configs that hit the minimum tok/s abort are still listed in the summary
  so they are skipped on restart

Speed probe file
----------------

With ``--probe``, measurements go to ``speed_probe.csv`` (prefill/decode tok/s at
chosen tokens-in-cache levels).

Plots input
----------------

``plot_results`` reads the results CSV (and optional probe CSV) and writes figures
under ``--out`` (default ``plots/``), including rank-sum tables as CSV.


.. include:: add_bottom.add
