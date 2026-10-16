
Background
#############

Local coding models are often sold on two claims: strong code quality and long
context. Short-prompt leaderboards do not test both at once. **localbenchmark** is built
to measure them together under realistic local constraints.

Aim
*****

Provide a single, resume-safe evaluation pipeline that:

* Estimates **memory and decode speed** for a model before it is loaded (dry run)
* Runs **coding tasks** (HumanEval, MBPP, SWE-bench) through **LM Studio**
* Asks **knowledge questions** closed-book, and **company questions** from your own documents
* Checks whether a model **declines** questions its documents cannot answer
* Scores open-ended answers with a **judge model**, optionally
* Stresses **context windows** (e.g. 50k–180k tokens)
* Records **pass rate**, **decode speed**, **TTFT**, and **peak GPU memory**
* Produces **rank-sum plots** so quality and speed are not confused

Results
********

Each run is stored in durable CSV files, in a folder named after its benchmark.
A summary file tracks finished configurations so restarts skip completed work.
Ranking plots combine pass rate, speed, latency, memory, and context capability —
with higher weight on pass rate and decode speed.

.. include:: add_bottom.add
