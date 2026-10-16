"""
Blocks network access while the tests run, so a CI run never downloads a model,
a dataset or a repository. A test that reaches the network fails with a clear message.

Set LOCALBENCHMARK_ALLOW_NETWORK=1 to allow network access for a manual integration run.
"""
import os
import socket


def block_network():
    if os.environ.get("LOCALBENCHMARK_ALLOW_NETWORK") == "1":
        return
    # HuggingFace libraries: use only what is already cached locally
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    def blocked(*args, **kwargs):
        raise RuntimeError(
            "network access is blocked in tests: a test tried to contact the network "
            "(model server, dataset download or repository). Mock that call in the test, "
            "or set LOCALBENCHMARK_ALLOW_NETWORK=1 for a manual run.")

    socket.socket.connect = blocked
    socket.create_connection = blocked
    socket.getaddrinfo = blocked
