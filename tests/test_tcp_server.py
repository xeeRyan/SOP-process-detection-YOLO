from __future__ import annotations

import json
import socket
import unittest

from SOP_PYD import handle_client


class TcpServerTests(unittest.TestCase):
    def test_one_request_returns_response_and_closes_connection(self) -> None:
        server_side, client_side = socket.socketpair()
        client_side.settimeout(2)
        client_side.sendall(json.dumps({"command": "health"}).encode("utf-8"))

        command = handle_client(server_side, ("local", 0))
        response = json.loads(client_side.recv(262144).decode("utf-8"))
        closed = client_side.recv(1)
        client_side.close()

        self.assertEqual(command, "")
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["data"]["status"], "ok")
        self.assertEqual(closed, b"")

    def test_close_command_requests_server_shutdown(self) -> None:
        server_side, client_side = socket.socketpair()
        client_side.sendall(b"close")

        command = handle_client(server_side, ("local", 0))
        client_side.close()

        self.assertEqual(command, "close")


if __name__ == "__main__":
    unittest.main()
