import logging
import localbenchmark.localbenchmark as localbenchmark
import localbenchmark.benchmark_plots as benchmark_plots

__author__ = 'Erdogan Taskesen'
__email__ = 'erdogant@gmail.com'
__version__ = '0.1.0'

# Setup package-level logger
_logger = logging.getLogger('localbenchmark')
_log_handler = logging.StreamHandler()
_formatter = logging.Formatter(fmt='[{asctime}] [{name:<12.12}] [{levelname:<8}] {message}', style='{', datefmt='%d-%m-%Y %H:%M:%S')
_log_handler.setFormatter(_formatter)
_log_handler.setLevel(logging.DEBUG)
if not _logger.hasHandlers():  # avoid duplicate handlers if re-imported
    _logger.addHandler(_log_handler)
_logger.setLevel(logging.DEBUG)
_logger.propagate = True  # allow submodules to inherit this handler



# module level doc-string
__doc__ = """
localbenchmark
=====================================================================

localbenchmark is for...

Example
-------
>>> from localbenchmark import localbenchmark
>>> model = localbenchmark()
>>> localbenchmark.plot()

References
----------
https://github.com/erdogant/localbenchmark

"""
