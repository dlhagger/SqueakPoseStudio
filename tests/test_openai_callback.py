import io
import unittest

from squeakpose.services.openai_auth import OAuthCallbackReceiver, oauth_callback_handler

STATE = "expected-state-value"


class _FakeConnection:
    """Just enough of a socket for ``BaseHTTPRequestHandler`` without networking."""

    def __init__(self, request: bytes) -> None:
        self._request = request
        self.sent = bytearray()

    def makefile(self, mode, *_args, **_kwargs):
        return io.BytesIO(self._request)

    def sendall(self, data) -> None:
        self.sent.extend(data)


def _get(handler_class, path: str) -> tuple[int, bytes]:
    connection = _FakeConnection(f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".encode())
    handler_class(connection, ("127.0.0.1", 50000), server=None)
    response = bytes(connection.sent)
    status = int(response.split(b" ", 2)[1])
    return status, response


class OAuthCallbackReceiverTests(unittest.TestCase):
    def test_mismatched_or_missing_state_is_refused_and_waiting_continues(self):
        receiver = OAuthCallbackReceiver()
        receiver.expect(STATE)
        for path in (
            "/auth/callback?code=abc&state=wrong",
            "/auth/callback?code=abc",
            "/auth/callback?error=access_denied",
            "/auth/callback?code=abc&state=",
        ):
            with self.subTest(path=path):
                self.assertEqual(receiver.offer(path), 400)
                self.assertFalse(receiver.received)
        self.assertEqual(receiver.offer("/favicon.ico"), 404)
        self.assertFalse(receiver.received)

        self.assertEqual(receiver.offer(f"/auth/callback?code=abc&state={STATE}"), 200)
        self.assertTrue(receiver.received)
        self.assertEqual(receiver.callback, {"code": "abc", "state": STATE})
        # Later requests cannot overwrite the accepted callback.
        self.assertEqual(receiver.offer(f"/auth/callback?code=evil&state={STATE}"), 400)
        self.assertEqual(receiver.callback["code"], "abc")

    def test_matching_error_callback_is_accepted_for_reporting(self):
        receiver = OAuthCallbackReceiver(STATE)
        self.assertEqual(receiver.offer(f"/auth/callback?error=access_denied&state={STATE}"), 200)
        self.assertEqual(receiver.callback["error"], "access_denied")

    def test_nothing_is_accepted_before_the_attempt_state_is_known(self):
        receiver = OAuthCallbackReceiver()
        self.assertEqual(receiver.offer("/auth/callback?code=abc&state="), 400)
        self.assertEqual(receiver.offer("/auth/callback?code=abc&state=anything"), 400)
        self.assertFalse(receiver.received)

    def test_handler_answers_400_for_wrong_state_and_success_page_for_match(self):
        receiver = OAuthCallbackReceiver(STATE)
        handler = oauth_callback_handler(receiver)

        status, body = _get(handler, "/auth/callback?code=abc&state=wrong")
        self.assertEqual(status, 400)
        self.assertNotIn(b"Return to SqueakPose Studio", body)
        self.assertFalse(receiver.received)

        status, _body = _get(handler, "/other")
        self.assertEqual(status, 404)

        status, body = _get(handler, f"/auth/callback?code=abc&state={STATE}")
        self.assertEqual(status, 200)
        self.assertIn(b"Return to SqueakPose Studio", body)
        self.assertEqual(receiver.callback, {"code": "abc", "state": STATE})


if __name__ == "__main__":
    unittest.main()
