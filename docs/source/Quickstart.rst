Quickstart
##########

This page gets you running in a few minutes. Each category has its own page with the full details.

Setup
*****

Install the package (see :doc:`Installation`).

For coding benchmarks and knowledge suites, start LM Studio, enable its local server (default ``http://localhost:1234``), and load a model. The dry run needs no server.

Dry run
*******

Estimate memory and decode speed for each model, quantization and context. Nothing is loaded.

.. code-block:: console

    localbenchmark --dry-run --contexts 120000

See :doc:`Dry_Run`.

Coding
******

Run a coding benchmark on one model:

.. code-block:: console

    localbenchmark --benchmark humaneval --models "openai/gpt-oss-20b"

See :doc:`Coding`.

Knowledge suite
***************

Ask a set of knowledge questions, closed-book:

.. code-block:: console

    localbenchmark --benchmark suite --suite general_knowledge.jsonl --models "openai/gpt-oss-20b" --contexts 120000

See :doc:`Knowledge_Suites`.

Plots & ranking
***************

Plot every results file in the current folder and its benchmark folders:

.. code-block:: console

    localbenchmarkplot

See :doc:`Plots`.

.. include:: add_bottom.add
