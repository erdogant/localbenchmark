Dry run
#######

``localbenchmark --dry-run`` estimates whether a model fits on your GPU and how fast it could decode, before you load anything. It loads no model and sends no request, so it is fast and works without LM Studio running.

Running the dry run
*******************

Basic commands
==============

.. code-block:: console

    # All local models, one context, the default quantizations (Q4_K_M, Q6_K, Q8_0)
    localbenchmark --dry-run --contexts 120000

    # Chosen models and quantizations
    localbenchmark --dry-run --contexts 120000 --quants Q4_K_M --models unsloth/qwen3-coder-30b-a3b-instruct openai/gpt-oss-20b

    # Include the smaller quantizations (off by default)
    localbenchmark --dry-run --quants Q3_K_M Q4_K_M Q5_K_M

``--dryrun`` is accepted as an alias of ``--dry-run``.

Output files
============

A dry run writes to a ``dryrun`` folder in the current directory:

* ``dryrun/dryrun_estimates.csv``: one row per model, quantization and context
* ``dryrun/dryrun_leaderboard.csv``: rank-sum ranking of every configuration
* ``dryrun/plots/``: five figures (memory heatmap, memory against context, speed against memory, ranking, speed against context)

The console table
=================

The console shows the same leaderboard:

.. code-block:: console

    rank  model           quant   context  total GB  headroom GB  bandwidth  decode ceiling tok/s
    1     gpt-oss-20b     Q4_K_M  120k     ...       ...          GPU        ...
    2     ornith-1.0-9b   Q4_K_M  120k     ...       ...          GPU        ...

The numbers depend on your GPU and models, so they are not shown here. Configurations above your GPU memory are red in the plots, and their GB value is still printed.

Estimates
*********

Memory
======

Each model, quantization and context is estimated as::

    total = weights + KV cache + overhead

* **Weights**: the size of the GGUF file, if one is found for that model and quantization in a model folder. Otherwise the size is estimated from the quantization's bits per weight.
* **KV cache**: bytes per token of context, read from the GGUF file header (layers, KV heads and head size). Only attention layers keep a KV cache, so hybrid models are handled. Models without a GGUF file use ``--kv-bytes`` (default 55,000 bytes per token), which is not measured for that model.
* **Overhead**: ``--overhead-gb`` (default 0.5 GB) for buffers and the runtime.

The GGUF files are searched in the LM Studio model folders (``~/.lmstudio/models`` and ``~/.cache/lm-studio/models``) and in any folder given with ``--model-dir``. The quantization is matched in the file name or the folder name, for example ``Q4_K_M``.

If you know the GPU memory exactly, pass it with ``--vram-gb``. Otherwise it is read from ``nvidia-smi``, with 16 GB as a fallback.

Model sizes
===========

The size of a model comes from its name (for example ``30b-a3b`` means 30B parameters, 3B active per token). Models whose size cannot be found show ``nan``, and they are not ranked.

Known models are listed in ``MODEL_SIZE_LOOKUP`` in ``localbenchmark.py``, which also gives a model's context limit. For example, ``gpt-oss-20b`` is listed with 3.6B active parameters, and ``phi-4`` with a 16,000-token limit (larger windows are not computed).

Speed
=====

The decode-speed ceiling is an upper bound. Each generated token reads the active weights and the whole KV cache, so the ceiling falls as the window grows. The part of a model that does not fit in VRAM is read at system RAM speed. The ceiling blends GPU and RAM bandwidth by the share that does not fit:

* ``--gpu-bandwidth`` (default 600 GB/s): set your card's memory bandwidth
* ``--ram-bandwidth`` (default 40 GB/s, typical range 8 to 60): system RAM speed

The ceiling ignores compute and scheduling overhead, so real speed is lower.

Ranking
*******

The dry-run leaderboard ranks every model, quantization and context by the sum of two ranks: the decode-speed ceiling and the memory headroom (GPU memory minus total GB). Lower is better. The ranking uses estimates only. Quality needs a benchmark run.

Options
*******

============================  =========================================================
Option                        Meaning
============================  =========================================================
``--dry-run``                 Run the dry run (alias ``--dryrun``).
``--models``                  Models to estimate. Default: all local models.
``--contexts``                Context windows in tokens. Default: 50k to 180k.
``--quants``                  Quantizations. Default: Q4_K_M Q6_K Q8_0.
``--vram-gb``                 GPU memory in GB.
``--gpu-bandwidth``           GPU memory bandwidth in GB/s (default 600).
``--ram-bandwidth``           System RAM bandwidth in GB/s (default 40).
``--kv-bytes``                KV bytes per token for models without a GGUF file.
``--overhead-gb``             Fixed overhead in GB (default 0.5).
``--model-dir``               Extra folder with GGUF files (repeatable).
============================  =========================================================

.. include:: add_bottom.add
