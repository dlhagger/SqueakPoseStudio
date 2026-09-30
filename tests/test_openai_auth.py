import json
import os
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlparse

from squeakpose.services.openai_auth import (
    ChatGPTCredentialStore,
    ChatGPTProfile,
    create_authorization_attempt,
)


class OpenAIAuthTests(unittest.TestCase):
    def test_new_registration_uses_pkce_host_id_and_plan_scopes(self):
        with TemporaryDirectory() as tmp:
            store = ChatGPTCredentialStore(tmp)
            attempt = create_authorization_attempt(
                redirect_uri="http://127.0.0.1:43210/auth/callback",
                store=store,
            )
            query = parse_qs(urlparse(attempt.authorization_url).query)

            self.assertEqual(query["client_id"], ["dynamic_agent_client"])
            self.assertEqual(query["agent_name_hint"], ["SqueakPose Studio"])
            self.assertEqual(query["ext_agent_host_id"], [attempt.host_id])
            self.assertTrue(attempt.host_id.startswith("urn:uuid:"))
            self.assertEqual(query["code_challenge_method"], ["S256"])
            self.assertIn("chatgpt.tokens.use.direct", query["scope"][0].split())
            self.assertEqual(query["redirect_uri"], [attempt.redirect_uri])

            host_path = Path(tmp) / "host.json"
            self.assertEqual(
                stat.S_IMODE(host_path.stat().st_mode),
                0o600,
            )
            self.assertEqual(store.host_id(), attempt.host_id)

    def test_returning_registration_reuses_client_and_identity_hints(self):
        with TemporaryDirectory() as tmp:
            store = ChatGPTCredentialStore(tmp)
            profile = ChatGPTProfile(
                email="researcher@example.org",
                issuer="https://auth.openai.com",
                subject="subject-1",
                client_id="oaiapp_example",
                host_id=store.host_id(),
                id_token="id-token-value",
                access_token="access-token-value",
                refresh_token="refresh-token-value",
                expires_at=1000.0,
                scopes=["chatgpt.tokens.use.direct"],
            )
            store.save_profile(profile)

            attempt = create_authorization_attempt(
                redirect_uri="http://127.0.0.1:5555/auth/callback",
                store=store,
            )
            query = parse_qs(urlparse(attempt.authorization_url).query)

            self.assertEqual(query["client_id"], ["oaiapp_example"])
            self.assertNotIn("agent_name_hint", query)
            self.assertEqual(query["id_token_hint"], ["id-token-value"])
            self.assertEqual(query["login_hint"], ["researcher@example.org"])
            self.assertEqual(attempt.returning_subject, "subject-1")

            saved = json.loads((Path(tmp) / "profile.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["client_id"], "oaiapp_example")
            if os.name != "nt":
                self.assertEqual(
                    stat.S_IMODE((Path(tmp) / "profile.json").stat().st_mode),
                    0o600,
                )


if __name__ == "__main__":
    unittest.main()
