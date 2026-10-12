
Coding quality
'''''''''''''''''''''

I value software quality. Higher quality software has fewer defects, better security, and better performance, which leads to happier users who can work more effectively.
Code reviews are an effective method for improving software quality. McConnell (2004) suggests that unit testing finds approximately 25% of defects, function testing 35%, integration testing 45%, and code review 55-60%.
While this means that none of these methods are good enough on their own and that they should be combined, clearly code review is an essential tool here.

This library is therefore developed with several techniques, such as coding styling, low complexity, docstrings, reviews, and unit tests.
Such conventions are helpful to improve the quality, make the code cleaner and more understandable, but also to trace future bugs, and spot syntax errors.


library
-------

The file structure of the package looks like:


.. code-block:: bash

    path/to/localbenchmark/
    ├── .editorconfig
    ├── .gitignore
    ├── CHANGELOG.rst
    ├── docs
    │   ├── source/
    │   │   ├── conf.py
    │   │   ├── index.rst
    │   │   └── ...
    │   └── figs/
    ├── LICENSE
    ├── localbenchmark
    │   ├── __init__.py
    │   ├── __version__.py
    │   ├── llm_coding_benchmark.py
    │   ├── plot_results.py
    │   ├── analyze_slowdown.py
    │   ├── LLMlight.py
    │   └── utils.py
    ├── README.md
    ├── requirements.txt
    ├── pyproject.toml
    └── tests
        ├── __init__.py
        └── test_localbenchmark.py


Style
-----

This library aims to stay compliant with the PEP-8 standards.
PEP stands for Python Enhancement Proposal and sets a baseline for the readability of Python code.
Public functions use docstrings based on numpy standards.


Complexity
----------

Developing software with lower technical debt may take extra development time, but has many advantages:

* Higher quality code
* Easier to maintain
* Less prone to bugs and errors
* Higher security


Unit tests
----------

The use of unit tests is essential to guarantee a consistent output of developed functions.
Tests live under ``tests/`` and can be run with pytest:

.. code-block:: bash

    pytest tests/test_localbenchmark.py


.. include:: add_bottom.add
