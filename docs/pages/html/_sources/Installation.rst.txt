Installation
#####################

Python **3.10+** is required. Recommended is to install ``localbenchmark`` in an isolated
environment using conda or venv.

Create Conda Environment
**************************

.. code-block:: console

    conda create -n env_localbenchmark python=3.12
    conda activate env_localbenchmark
    pip install localbenchmark


Create venv Environment
**************************

.. code-block:: console

    # Create a new project directory
    mkdir my_localbenchmark_project
    cd my_localbenchmark_project

    # Create virtual environment
    python -m venv venv

    # Activate it
    # On macOS/Linux:
    source venv/bin/activate

    # On Windows:
    venv\Scripts\activate

    # Install localbenchmark
    pip install localbenchmark


Pypi
**********************

.. code-block:: console

    # Install from PyPI:
    pip install localbenchmark

    # Force update to latest version
    pip install -U localbenchmark


Github Source
************************************

.. code-block:: console

    # Install directly from GitHub
    pip install git+https://github.com/erdogant/localbenchmark


From source (development)
************************************

.. code-block:: console

    git clone https://github.com/erdogant/localbenchmark
    cd localbenchmark
    pip install -r requirements.txt
    pip install -e .


LM Studio
**********************

The benchmarks and knowledge suites expect a running
`LM Studio <https://lmstudio.ai/>`_ server (default ``http://localhost:1234``)
with models available on the machine. A dry run does not need the server: it reads
the model files on disk. Without ``--models``, the models are taken from the
server, or from the LM Studio model folders when the server is not running.

1. Install LM Studio
2. Download one or more coding models
3. Enable the local server (Developer mode)
4. Note the model key (as shown in LM Studio) for ``--models``


Dependencies
**********************

Core packages: ``numpy``, ``pandas``, ``matplotlib``, ``requests``, ``openai``,
``datasets`` (downloads HumanEval, MBPP and SWE-bench problems), and ``adjustText``
(plot labels).

Optional: ``swebench`` for scoring SWE-bench patches. The scoring runs the official
harness in Docker, so Docker must be running.


Uninstalling
################

Remove environment
**********************

.. code-block:: console

   # List all the active environments. env_localbenchmark should be listed.
   conda env list

   # Remove the environment
   conda env remove --name env_localbenchmark

   # List all the active environments. env_localbenchmark should be absent.
   conda env list


Remove installation
**********************

.. code-block:: console

    pip uninstall localbenchmark


.. include:: add_bottom.add
