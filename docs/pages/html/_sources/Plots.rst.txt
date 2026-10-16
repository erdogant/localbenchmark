Plots & ranking
###############

Each benchmark run writes its figures to a ``plots`` folder inside its output folder, for example ``humaneval/plots``. To plot results again, or to plot several results files together, use ``localbenchmarkplot``.

Running the plots
*****************

After a benchmark
=================

Plots are created at the end of each run. To regenerate them without running anything:

.. code-block:: console

    localbenchmark --plots-only --benchmark humaneval

Plotting results again
======================

``localbenchmarkplot`` finds every per-problem results CSV in the current folder and its benchmark folders:

.. code-block:: console

    # Every results file found
    localbenchmarkplot

    # Only one results file
    localbenchmarkplot --results humaneval/coding_benchmark_results.csv

    # Results in another folder, with a different rank aggregation
    localbenchmarkplot --folder path/to/results --rank-agg mean

    # Large windows only (context_actual >= 100000)
    localbenchmarkplot --min-context 100000

Ranking
*******

Metrics
=======

Each model, context and quantization is scored on:

1. Pass rate (early aborts count unfinished work against the model)
2. Decode speed (percentage of best)
3. Time to first token
4. Peak GPU memory
5. Context size (percentage of best)

Pass rate and decode speed are weighted more heavily. Lower combined score is better.

Rank aggregation
================

``--rank-agg`` chooses how the ranks are combined: ``sum`` (default), ``mean`` or ``median``.

Figures
*******

The figures include pass rate against context, decode speed, time to first token, GPU memory, the quality against speed trade-off, and rank-sum panels for each model and context and for each model.

The dry run writes its own figures to ``dryrun/plots``. See :doc:`Dry_Run`.

Options
*******

============================  =========================================================
Option                        Meaning
============================  =========================================================
``--results``                 One results CSV to plot.
``--folder``                  Folder to search for results (default: current folder).
``--out``                     Output folder. Default: a ``plots`` folder next to each results file.
``--rank-agg``                ``sum`` (default), ``mean`` or ``median``.
``--min-context``             Keep rows with ``context_actual`` strictly greater than this value.
============================  =========================================================

.. include:: add_bottom.add
