Knowledge suite
###############

A knowledge suite asks a model a set of questions from a ``.jsonl`` file and scores each answer. The suite is run with ``--benchmark suite``.

Overview
********

There are two kinds of suite:

* **Closed-book suites**, such as general knowledge: the model answers from its own knowledge.
* **Document suites**, such as company QA: the selected documents are put in the prompt, and the model must answer from them.

.. code-block:: console

    # General knowledge: no documents, asked closed-book automatically
    localbenchmark --benchmark suite --suite general_knowledge.jsonl --models "openai/gpt-oss-20b" --contexts 120000

    # Company QA with a document folder
    localbenchmark --benchmark suite --suite northwind_demo.jsonl --docs path/to/company_docs --models "openai/gpt-oss-20b" --contexts 120000

The suite file is read from the path you give. A suite that cites no documents runs closed-book automatically. ``--closed-book`` forces this.

The suite file
**************

Format
======

One question per line. Each question has an ``id``, a ``domain``, a ``type`` and a ``prompt``. The other fields depend on the type:

.. code-block:: console

    {"id":"ctry-01","domain":"countries","type":"short","prompt":"What is the capital of Australia?","accept":["Canberra"],"distractors":["Sydney","Melbourne"]}
    {"id":"ctry-03","domain":"countries","type":"numeric","prompt":"Roughly how large is Iceland, in square kilometres?","gold":103000,"rel_tol":0.05}
    {"id":"nw-01","domain":"hr","type":"short","prompt":"How many days of annual leave do full-time employees get per calendar year?","accept":["25"],"context":["hr_leave_policy.md"]}

Manifest and result names
=========================

A suite can have an optional ``<name>.manifest.json`` next to it, with a name and version. Results are named after the suite: ``suite_results_<name>.csv``.

Question types and scoring
==========================

Scoring is deterministic. The only exception is the ``rubric`` type, which needs a judge model (see below).

================  ===========================================================================================================
Type              How an answer is scored
================  ===========================================================================================================
``mc``            Multiple choice: the chosen letter, ``(B) text``, ``answer is B``, or the option text.
``numeric``       A number within ``abs_tol`` or ``rel_tol`` of the gold value. Near misses get partial credit.
``short``         Any of the ``accept`` answers. Naming a ``distractors`` answer as well halves the score.
``keywords``      Weighted concept groups, each with aliases. Partial credit for each group found. Typos score by similarity.
``unanswerable``  The model must say the documents do not contain the answer. An invented answer scores 0.
``rubric``        Open-ended: scored by the judge model.
================  ===========================================================================================================

Negation is respected: "not Sydney" does not count as naming Sydney. Thinking blocks are ignored.

Each result row stores the score (0 to 1), whether the item passed, and the score details, so every decision can be checked. The pass threshold is ``min_score``: 1.0 by default, or 0.8 for ``keywords`` items.

Judge model for open-ended answers
**********************************

Open-ended (``rubric``) items need a judge. Without one, the candidate's answer is saved and marked ``NEEDS_JUDGE``, and it is left out of the pass rates. To score them, name a judge model:

.. code-block:: console

    localbenchmark --benchmark suite --suite general_knowledge.jsonl --models "openai/gpt-oss-20b" --contexts 120000 --judge "<judge-model-key>"

The run works in two steps. The candidate answers every question first. Then it is unloaded, the judge is loaded, and the judge grades the saved answers. Both models are never in memory together.

* The judge replies with 2 (meets the criteria), 1 (partly) or 0 (does not). These are stored as 1.0, 0.5 and 0.
* A ``reference`` answer on a rubric item is graded first. If the judge does not give it full marks, the run warns that the judge may not be reliable.
* Use a different model as the judge than the candidate. A model grading its own answers tends to score them high.
* ``--judge-context`` sets the judge's context length (default 16,000).
* On servers other than LM Studio nothing is loaded or unloaded: the judge must be served by the same server as the candidate.

Judgements are saved in their own file, so you can judge a finished run again with a different judge. A failed judge reply is recorded as ``PARSE_ERROR`` and tried again on the next run.

Company QA with documents
*************************

Documents folder
================

Put the documents in one folder (default ``company_docs`` in the current directory). Any ``.txt`` or ``.md`` file can be used. Documents are used in name order. The suite cites its documents with ``context``, and the run checks that every cited document exists.

.. code-block:: console

    # Use the first 40 documents only
    localbenchmark --benchmark suite --suite northwind_demo.jsonl --docs path/to/company_docs --models "openai/gpt-oss-20b" --docs-limit 40

    # Use exactly these documents
    localbenchmark --benchmark suite --suite northwind_demo.jsonl --docs path/to/company_docs --models "openai/gpt-oss-20b" --docs-ids hr_leave_policy.md expense_policy.md

Checking a suite
================

Check a suite and its document folder before a run:

.. code-block:: console

    python suite_scoring.py northwind_demo.jsonl --docs path/to/company_docs

This checks that every question is valid, that each gold answer scores full marks, and that every cited document exists.

Demo suite
==========

``northwind_demo`` is a fictional company with 12 documents and 40 questions (30 answerable, 10 unanswerable). The documents are in the ``suite/company_docs`` folder of the repository. Copy them into your own documents folder, or replace them with your own.

Results
*******

Error labels
============

When an answer fails, the results file says why:

* ``WRONG_ANSWER``: an answerable question was answered incorrectly
* ``INVENTED_ANSWER``: an unanswerable question was answered with a made-up fact (the most important check for company use)
* ``MISSING_DOC``: the document that holds the answer was not in the prompt
* ``TRUNCATED``, ``EMPTY_RESPONSE``, ``LLM_ERROR``: as in the coding benchmarks
* ``NEEDS_JUDGE``: an open-ended answer that is waiting for a judge

Output files
============

All suite files are written to a ``suite`` folder:

* ``suite/suite_results_<name>.csv``: one row per question
* ``suite/suite_results_<name>_by_domain.csv``: pass rate and mean score per domain. The console also shows answerable and unanswerable results separately.
* ``suite/suite_results_<name>_judge.csv``: judge scores, reasons and parse errors (when ``--judge`` is used)

Options
*******

============================  =========================================================
Option                        Meaning
============================  =========================================================
``--suite``                   Suite file (``.jsonl``).
``--models``                  Model keys. Default: all available models (see ``--list``).
``--list``                    List the available models and exit.
``--contexts``                Context window sizes in tokens. Default: 50k to 180k. Only LM Studio can change the window.
``--limit``                   Run only the first N questions.
``--docs``                    Document folder. Default ``company_docs``.
``--docs-limit``              Use only the first N documents, in name order.
``--docs-ids``                Use exactly these document file names.
``--closed-book``             Ask without documents, even if the suite cites some.
``--judge``                   Judge model for open-ended answers.
``--judge-context``           Context length for the judge (default 16,000).
``--base-url``                Server URL. Default ``http://localhost:1234``.
``--backend``                 ``auto`` (default), ``lmstudio`` or ``openai``.
``--api-key``                 API key for servers that need one.
============================  =========================================================

.. include:: add_bottom.add
