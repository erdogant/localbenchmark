Coding
######

Coding benchmarks run a model on programming problems through LM Studio. The generated code is executed against each problem's tests, and the results record pass rate, speed and memory.

Running a benchmark
*******************

Start LM Studio, enable its local server, and load a model. Then choose a benchmark with ``--benchmark``.

HumanEval
=========

HumanEval is the default benchmark: 164 Python functions, each checked by hidden tests.

.. code-block:: console

    localbenchmark --benchmark humaneval --models "openai/gpt-oss-20b"

MBPP
====

MBPP (Mostly Basic Python Problems) has 500 short Python tasks. The prompt shows each task together with its visible tests.

.. code-block:: console

    localbenchmark --benchmark mbpp --models "openai/gpt-oss-20b"

SWE-bench Verified
==================

SWE-bench Verified asks the model to fix real issues in real repositories, by writing a patch. Scoring runs the official harness, which needs the ``swebench`` package and Docker.

.. code-block:: console

    localbenchmark --benchmark swebench --models "openai/gpt-oss-20b" --contexts 80000 150000

Models and contexts
===================

Without ``--models``, every local model is used. Without ``--contexts``, the contexts are 50k, 80k, 100k, 120k, 150k and 180k tokens.

.. code-block:: console

    # Chosen contexts
    localbenchmark --benchmark humaneval --models "openai/gpt-oss-20b" --contexts 50000 80000

    # Every local model
    localbenchmark --benchmark humaneval

Smoke test
==========

Run only the first problems, to check the setup:

.. code-block:: console

    localbenchmark --benchmark humaneval --models "openai/gpt-oss-20b" --contexts 50000 80000 --limit 20

Results and resume
******************

Each benchmark writes to its own folder: ``humaneval/``, ``mbpp/`` or ``swebench/``. Each row of the results CSV is one problem for one configuration. Run the same command again after an interruption to continue where you stopped. See :doc:`Save and Load`.

Options
*******

============================  =========================================================
Option                        Meaning
============================  =========================================================
``--benchmark``               ``humaneval`` (default), ``mbpp`` or ``swebench``.
``--models``                  LM Studio model keys. Default: all local models.
``--contexts``                Context window sizes in tokens. Default: 50k to 180k.
``--limit``                   Run only the first N problems.
``--results``                 Results CSV path. Default: a file in the benchmark's folder.
``--base-url``                LM Studio server. Default ``http://localhost:1234``.
``--plots-only``              Only regenerate the plots from existing results.
============================  =========================================================

.. include:: add_bottom.add
