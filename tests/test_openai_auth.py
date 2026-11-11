import base64
import hashlib
import hmac
import json
import os
import stat
import tempfile
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
from urllib.parse import parse_qs, urlparse

import httpx
import jwt
from cryptography.hazmat.primitives import serialization
from jwt.algorithms import RSAAlgorithm

from squeakpose.services.openai_auth import (
    OIDC_CONFIGURATION_URL,
    ChatGPTAuthError,
    ChatGPTCredentialStore,
    ChatGPTProfile,
    create_authorization_attempt,
    usable_access_token,
    validate_id_token,
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
            if os.name == "posix":  # Windows has no POSIX permission bits
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


ISSUER = "https://auth.openai.com"
CLIENT_ID = "oaiapp_example"
NONCE = "nonce-value"
KEY_ID = "test-key"
JWKS_URI = "https://auth.openai.com/.well-known/jwks.json"


class IdTokenValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cryptography.hazmat.primitives.asymmetric import rsa

        cls.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        jwk = json.loads(RSAAlgorithm.to_jwk(cls.private_key.public_key()))
        jwk.update({"kid": KEY_ID, "use": "sig", "alg": "RS256"})
        cls.jwks = {"keys": [jwk]}

    def _client(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url) == OIDC_CONFIGURATION_URL:
                return httpx.Response(200, json={"issuer": ISSUER, "jwks_uri": JWKS_URI})
            if str(request.url) == JWKS_URI:
                return httpx.Response(200, json=self.jwks)
            return httpx.Response(404)

        return httpx.Client(transport=httpx.MockTransport(handler))

    def _claims(self, **overrides):
        now = int(time.time())
        claims = {
            "iss": ISSUER,
            "aud": CLIENT_ID,
            "sub": "subject-1",
            "nonce": NONCE,
            "iat": now,
            "exp": now + 3600,
            "email": "researcher@example.org",
        }
        claims.update(overrides)
        return claims

    def _validate(self, token):
        with self._client() as client:
            return validate_id_token(token, client_id=CLIENT_ID, nonce=NONCE, client=client)

    def _rs256(self, claims):
        return jwt.encode(claims, self.private_key, algorithm="RS256", headers={"kid": KEY_ID})

    def test_valid_token_is_accepted(self):
        self.assertEqual(self._validate(self._rs256(self._claims()))["sub"], "subject-1")

    def test_small_clock_skew_on_iat_is_tolerated(self):
        token = self._rs256(self._claims(iat=int(time.time()) + 2))
        self.assertEqual(self._validate(token)["sub"], "subject-1")

    def test_issued_at_far_in_future_is_rejected(self):
        token = self._rs256(self._claims(iat=int(time.time()) + 600))
        with self.assertRaisesRegex(ChatGPTAuthError, "invalid identity token"):
            self._validate(token)

    def test_unsigned_and_symmetric_header_algorithms_are_rejected(self):
        unsigned = jwt.encode(self._claims(), None, algorithm="none", headers={"kid": KEY_ID})
        # A classic confusion attack: HMAC keyed with the public key material.
        public_pem = self.private_key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        hmac_token = _hs256_token(self._claims(), public_pem, KEY_ID)
        for token in (unsigned, hmac_token):
            with self.subTest(alg=jwt.get_unverified_header(token)["alg"]):
                with self.assertRaises(ChatGPTAuthError):
                    self._validate(token)

    def test_jwk_with_unexpected_algorithm_is_refused(self):
        jwks = json.loads(json.dumps(self.jwks))
        jwks["keys"][0]["alg"] = "RS512"
        self.jwks, original = jwks, self.jwks
        try:
            token = jwt.encode(
                self._claims(), self.private_key, algorithm="RS512", headers={"kid": KEY_ID}
            )
            with self.assertRaisesRegex(ChatGPTAuthError, "unsupported algorithm"):
                self._validate(token)
        finally:
            self.jwks = original


def _hs256_token(claims, secret: bytes, key_id: str) -> str:
    # PyJWT refuses to HMAC-sign with PEM keys, so build the token by hand.
    def encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")

    header = encode(json.dumps({"alg": "HS256", "typ": "JWT", "kid": key_id}).encode())
    payload = encode(json.dumps(claims).encode())
    signature = hmac.new(secret, f"{header}.{payload}".encode(), hashlib.sha256).digest()
    return f"{header}.{payload}.{encode(signature)}"


def _profile(store, *, access_token="old-access", expires_at=0.0):
    return ChatGPTProfile(
        email="researcher@example.org",
        issuer=ISSUER,
        subject="subject-1",
        client_id=CLIENT_ID,
        host_id=store.host_id(),
        id_token="id-token",
        access_token=access_token,
        refresh_token="refresh-1",
        expires_at=expires_at,
        scopes=["chatgpt.tokens.use.direct"],
    )


class CredentialStoreConcurrencyTests(unittest.TestCase):
    def test_second_process_reuses_token_refreshed_while_it_waited(self):
        with TemporaryDirectory() as tmp:
            first = ChatGPTCredentialStore(tmp)
            second = ChatGPTCredentialStore(tmp)  # separate instance: separate lock fd
            first.save_profile(_profile(first))

            refreshes = []
            refresh_entered = threading.Event()
            release_refresh = threading.Event()

            def token_endpoint(request: httpx.Request) -> httpx.Response:
                refreshes.append(parse_qs(request.content.decode())["refresh_token"][0])
                refresh_entered.set()
                self.assertTrue(release_refresh.wait(10))
                return httpx.Response(
                    200,
                    json={
                        "access_token": "new-access",
                        "refresh_token": "refresh-2",
                        "expires_in": 3600,
                    },
                )

            results = {}

            def run(name, store):
                with httpx.Client(transport=httpx.MockTransport(token_endpoint)) as client:
                    results[name] = usable_access_token(store, client=client)[1]

            second_loaded = threading.Event()
            original_load = second.load_profile

            def tracking_load():
                profile = original_load()
                second_loaded.set()
                return profile

            second.load_profile = tracking_load

            first_thread = threading.Thread(target=run, args=("first", first))
            first_thread.start()
            self.assertTrue(refresh_entered.wait(10))  # first holds the credential lock
            second_thread = threading.Thread(target=run, args=("second", second))
            second_thread.start()
            self.assertTrue(second_loaded.wait(10))  # second saw the expired profile
            release_refresh.set()
            first_thread.join(10)
            second_thread.join(10)

            self.assertEqual(refreshes, ["refresh-1"])
            self.assertEqual(results, {"first": "new-access", "second": "new-access"})
            saved = ChatGPTCredentialStore(tmp).load_profile()
            self.assertEqual(saved.refresh_token, "refresh-2")

    def test_credential_lock_is_reentrant_and_excludes_other_instances(self):
        with TemporaryDirectory() as tmp:
            first = ChatGPTCredentialStore(tmp)
            second = ChatGPTCredentialStore(tmp)
            acquired = threading.Event()
            with first.credential_lock():
                with first.credential_lock():  # nested use must not deadlock
                    first.save_profile(_profile(first, expires_at=time.time() + 3600))

                def take():
                    with second.credential_lock():
                        acquired.set()

                thread = threading.Thread(target=take)
                thread.start()
                self.assertFalse(acquired.wait(0.3))
            self.assertTrue(acquired.wait(10))
            thread.join(10)

    def test_writes_use_unique_owner_only_temporary_files(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp) / "openai"
            store = ChatGPTCredentialStore(directory)
            temporaries = []
            modes = []
            real_mkstemp = tempfile.mkstemp
            real_replace = os.replace

            def recording_mkstemp(*args, **kwargs):
                descriptor, name = real_mkstemp(*args, **kwargs)
                temporaries.append(name)
                return descriptor, name

            def recording_replace(source, destination):
                modes.append(stat.S_IMODE(os.stat(source).st_mode))
                return real_replace(source, destination)

            with (
                mock.patch.object(tempfile, "mkstemp", recording_mkstemp),
                mock.patch.object(os, "replace", recording_replace),
            ):
                store.save_profile(_profile(store, expires_at=1.0))
                store.save_profile(_profile(store, expires_at=2.0))

            self.assertEqual(len(temporaries), 3)  # host.json + two profile writes
            self.assertEqual(len(set(temporaries)), 3)
            for name in temporaries:
                self.assertEqual(Path(name).parent, directory)
                self.assertFalse(Path(name).exists())
            self.assertEqual(
                sorted(path.name for path in directory.iterdir()),
                ["credentials.lock", "host.json", "profile.json"],
            )
            self.assertEqual(store.load_profile().expires_at, 2.0)
            if os.name != "nt":
                self.assertEqual(modes, [0o600, 0o600, 0o600])
                self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
                for name in ("credentials.lock", "host.json", "profile.json"):
                    self.assertEqual(stat.S_IMODE((directory / name).stat().st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
