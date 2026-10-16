
Results, folders, and resume
''''''''''''''''''''''''''''''

Long runs can take many hours. localbenchmark writes every finished item to CSV immediately, so a crash or Ctrl-C does not lose progress.

Output folders
--------------

Each job writes to a folder named after its benchmark, in the current directory:

* ``humaneval/``: HumanEval results
* ``mbpp/``: MBPP results
* ``swebench/``: SWE-bench results
* ``suite/``: suite results (named after the suite)
* ``dryrun/``: dry-run estimates

Each folder also has a ``plots`` subfolder with the figures for that run.

Results file
------------

The results file holds one row per item for one configuration (model, context, cache label). The file names are:

* ``humaneval/coding_benchmark_results.csv``
* ``mbpp/coding_benchmark_results_mbpp.csv``
* ``swebench/coding_benchmark_results_swebench.csv``
* ``suite/suite_results_<name>.csv``

Each row has the pass or fail result, the error label, timings, token counts, GPU memory, and the generated code or answer.

Summary file
------------

After each configuration, a summary file is written next to the results, for example ``coding_benchmark_results_summary.csv``. A configuration in the summary counts as finished. On the next run it is not loaded again. A configuration missing from the summary is run again.

Resume behaviour
----------------

* Run the **same command** again to continue where you stopped
* The program first looks for the results file in the current folder, as older versions did. If it is not there, it looks in the benchmark folder. If neither has it, a new file is created in the benchmark folder.
* Rows with ``error_type == LLM_ERROR`` are retried
* Other outcomes (pass, test failure, truncation, and so on) stay on disk
* Configurations that stopped on the minimum speed are still listed in the summary, so they are skipped on restart
* A finished configuration is not reloaded, but its open-ended answers are still judged if a judge is given

Judge and diagnostics files
---------------------------

* With ``--judge``, the judge's scores are in ``<results>_judge.csv``, one row per judged answer. A judge can be run again later with another model, and each judge's rows are kept separately.
* ``<results>_diagnostics.jsonl`` records the settings LM Studio reported for each model load.

Speed probe file
----------------

With ``--probe``, measurements go to ``speed_probe.csv`` in the benchmark folder (prefill and decode tokens/s at chosen tokens-in-cache levels).

Plots input
-----------

``localbenchmarkplot`` reads every per-problem results CSV in the current folder and its subfolders. It writes the figures and rank tables to a ``plots`` folder next to each file.

.. include:: add_bottom.add
