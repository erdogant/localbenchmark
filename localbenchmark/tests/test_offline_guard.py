"""Checks that the offline guard blocks network access. Run with: python -m unittest test_offline_guard -v"""
import os
import socket
import unittest

from offline_guard import block_network


class Guard(unittest.TestCase):
    def test_network_is_blocked(self):
        if os.environ.get("LOCALBENCHMARK_ALLOW_NETWORK") == "1":
            self.skipTest("network allowed for this run")
        block_network()
        with self.assertRaises(RuntimeError):
            socket.create_connection(("localhost", 1234), timeout=1)
        with self.assertRaises(RuntimeError):
            socket.getaddrinfo("huggingface.co", 443)


if __name__ == "__main__":
    unittest.main()
