Coding
######

Coding benchmarks run a model on programming problems through LM Studio or any OpenAI-compatible server. The generated code is executed against each problem's tests, and the results record pass rate, speed and memory.

Running a benchmark
*******************

Start LM Studio and enable its local server, or start another OpenAI-compatible server (see `Other servers`_). Then choose a benchmark with ``--benchmark``.

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

Without ``--models``, every available model is used. Without ``--contexts``, the contexts are 50k, 80k, 100k, 120k, 150k and 180k tokens.

.. code-block:: console

    # Chosen contexts
    localbenchmark --benchmark humaneval --models "openai/gpt-oss-20b" --contexts 50000 80000

    # Every local model
    localbenchmark --benchmark humaneval

Listing models
===============

Show which models are available and exit:

.. code-block:: console

    localbenchmark --list
    localbenchmark --list --base-url http://localhost:8080

The models come from the server. If it is not running, the LM Studio model folders on disk are used. Embedding models are left out.

Other servers
=============

Any server with the OpenAI-compatible ``/v1/models`` and ``/v1/chat/completions`` endpoints works, for example llama.cpp, vLLM or Ollama. The API is detected automatically. Force it with ``--backend openai``, and pass ``--api-key`` if the server needs a key.

.. code-block:: console

    localbenchmark --base-url http://localhost:8080 --models my-model
    localbenchmark --base-url http://localhost:11434 --backend openai --models qwen3-coder:30b
    localbenchmark --base-url https://my-server/v1 --api-key $MY_KEY

Such a server owns its model, so localbenchmark cannot load, unload or resize it. Each model is tested once, at the context window the server reports (``max_model_len`` for vLLM, ``/props`` for llama.cpp). If the server reports none, the run is labelled with the largest ``--contexts`` value; start the server with that context size so the label is true.

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
``--models``                  Model keys. Default: all available models (see ``--list``).
``--list``                    List the available models and exit.
``--contexts``                Context window sizes in tokens. Default: 50k to 180k. Only LM Studio can change the window.
``--limit``                   Run only the first N problems.
``--results``                 Results CSV path. Default: a file in the benchmark's folder.
``--base-url``                Server URL. Default ``http://localhost:1234``.
``--backend``                 ``auto`` (default), ``lmstudio`` or ``openai``.
``--api-key``                 API key for servers that need one.
``--plots-only``              Only regenerate the plots from existing results.
============================  =========================================================

.. include:: add_bottom.add
