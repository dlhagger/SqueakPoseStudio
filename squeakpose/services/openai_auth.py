"""Sign in with ChatGPT for the local open-source desktop client.

Credentials are deliberately app-global rather than project-local: projects may be
shared, archived, or committed, while OAuth credentials belong to one installation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode

import httpx
import jwt

AUTHORIZE_URL = "https://auth.openai.com/api/accounts/authorize"
TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"
OIDC_CONFIGURATION_URL = "https://auth.openai.com/.well-known/openid-configuration"
RESOURCE = "https://api.openai.com/v1"
SCOPES = (
    "openid",
    "profile",
    "email",
    "offline_access",
    "resource.invoke",
    "chatgpt.tokens.use.direct",
)
REQUIRED_PLAN_SCOPE = "chatgpt.tokens.use.direct"


class ChatGPTAuthError(RuntimeError):
    """A safe, user-presentable sign-in or credential error."""


@dataclass(frozen=True, slots=True)
class AuthorizationAttempt:
    state: str
    nonce: str
    code_verifier: str
    code_challenge: str
    redirect_uri: str
    client_id: str
    host_id: str
    authorization_url: str
    returning_subject: str = ""


@dataclass(slots=True)
class ChatGPTProfile:
    email: str
    issuer: str
    subject: str
    client_id: str
    host_id: str
    id_token: str
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_at: float = 0.0
    scopes: list[str] = field(default_factory=list)

    @property
    def plan_usage_enabled(self) -> bool:
        return REQUIRED_PLAN_SCOPE in self.scopes

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ChatGPTProfile:
        return cls(
            email=str(value.get("email") or ""),
            issuer=str(value.get("issuer") or ""),
            subject=str(value.get("subject") or ""),
            client_id=str(value.get("client_id") or ""),
            host_id=str(value.get("host_id") or value.get("ext_agent_host_id") or ""),
            id_token=str(value.get("id_token") or ""),
            access_token=str(value.get("access_token") or ""),
            refresh_token=str(value.get("refresh_token") or ""),
            token_type=str(value.get("token_type") or "Bearer"),
            expires_at=float(value.get("expires_at") or 0.0),
            scopes=[str(scope) for scope in value.get("scopes", ())],
        )


def default_openai_config_dir() -> Path:
    override = os.environ.get("SQUEAKPOSE_OPENAI_CONFIG_DIR")
    if override:
        return Path(override).expanduser().absolute()
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "SqueakPoseStudio" / "openai"


class ChatGPTCredentialStore:
    """Atomic owner-only storage for one MVP ChatGPT registration."""

    def __init__(self, directory: str | os.PathLike[str] | None = None) -> None:
        self.directory = Path(directory) if directory is not None else default_openai_config_dir()
        self.host_path = self.directory / "host.json"
        self.profile_path = self.directory / "profile.json"
        self._lock = threading.RLock()

    def host_id(self) -> str:
        with self._lock:
            if self.host_path.is_file():
                try:
                    value = json.loads(self.host_path.read_text(encoding="utf-8"))
                    host_id = str(value.get("ext_agent_host_id") or "")
                    if host_id.startswith("urn:uuid:"):
                        return host_id
                except (OSError, ValueError, TypeError):
                    pass
            host_id = f"urn:uuid:{uuid.uuid4()}"
            self._write_json(self.host_path, {"ext_agent_host_id": host_id})
            return host_id

    def load_profile(self) -> ChatGPTProfile | None:
        with self._lock:
            if not self.profile_path.is_file():
                return None
            try:
                payload = json.loads(self.profile_path.read_text(encoding="utf-8"))
                profile = ChatGPTProfile.from_mapping(payload)
            except (OSError, ValueError, TypeError):
                return None
            if not profile.client_id or not profile.subject or not profile.host_id:
                return None
            return profile

    def save_profile(self, profile: ChatGPTProfile) -> None:
        with self._lock:
            self._write_json(self.profile_path, asdict(profile))

    def clear_tokens(self) -> ChatGPTProfile | None:
        with self._lock:
            profile = self.load_profile()
            if profile is None:
                return None
            profile.id_token = ""
            profile.access_token = ""
            profile.refresh_token = ""
            profile.expires_at = 0.0
            profile.scopes = []
            self.save_profile(profile)
            return profile

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = path.with_suffix(path.suffix + ".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(dict(payload), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
            os.chmod(path, 0o600)
        except Exception:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise


def create_authorization_attempt(
    *,
    redirect_uri: str,
    store: ChatGPTCredentialStore,
    app_name: str = "SqueakPose Studio",
) -> AuthorizationAttempt:
    profile = store.load_profile()
    returning = profile is not None and bool(profile.client_id)
    client_id = profile.client_id if profile is not None and returning else "dynamic_agent_client"
    verifier = _base64url(secrets.token_bytes(64))
    challenge = _base64url(hashlib.sha256(verifier.encode("ascii")).digest())
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    host_id = store.host_id()
    parameters = {
        "client_id": client_id,
        "ext_agent_host_id": host_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": " ".join(SCOPES),
        "resource": RESOURCE,
        "state": state,
        "nonce": nonce,
        "code_challenge_method": "S256",
        "code_challenge": challenge,
    }
    if returning and profile is not None:
        if profile.id_token:
            parameters["id_token_hint"] = profile.id_token
        if profile.email:
            parameters["login_hint"] = profile.email
    else:
        parameters["agent_name_hint"] = app_name
    return AuthorizationAttempt(
        state=state,
        nonce=nonce,
        code_verifier=verifier,
        code_challenge=challenge,
        redirect_uri=redirect_uri,
        client_id=client_id,
        host_id=host_id,
        authorization_url=f"{AUTHORIZE_URL}?{urlencode(parameters)}",
        returning_subject=profile.subject if returning and profile is not None else "",
    )


def complete_authorization(
    attempt: AuthorizationAttempt,
    callback: Mapping[str, str],
    *,
    store: ChatGPTCredentialStore,
    client: httpx.Client | None = None,
) -> ChatGPTProfile:
    if not secrets.compare_digest(str(callback.get("state") or ""), attempt.state):
        raise ChatGPTAuthError("The ChatGPT sign-in response did not match this request.")
    if callback.get("error"):
        description = callback.get("error_description") or callback.get("error")
        raise ChatGPTAuthError(f"ChatGPT sign-in was not completed: {description}")
    code = str(callback.get("code") or "")
    if not code:
        raise ChatGPTAuthError("ChatGPT sign-in returned no authorization code.")
    issued_client_id = str(callback.get("client_id") or attempt.client_id)
    if attempt.client_id == "dynamic_agent_client":
        if not issued_client_id or issued_client_id == "dynamic_agent_client":
            raise ChatGPTAuthError("ChatGPT did not finish registering this installation.")
    elif issued_client_id != attempt.client_id:
        raise ChatGPTAuthError("ChatGPT returned credentials for a different registration.")

    owns_client = client is None
    http = client or httpx.Client(timeout=30.0)
    try:
        response = http.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": issued_client_id,
                "code": code,
                "code_verifier": attempt.code_verifier,
                "redirect_uri": attempt.redirect_uri,
                "resource": RESOURCE,
            },
        )
        response.raise_for_status()
        token_payload = response.json()
        identity = validate_id_token(
            str(token_payload.get("id_token") or ""),
            client_id=issued_client_id,
            nonce=attempt.nonce,
            client=http,
        )
    except ChatGPTAuthError:
        raise
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        raise ChatGPTAuthError("Could not exchange or validate ChatGPT credentials.") from exc
    finally:
        if owns_client:
            http.close()

    subject = str(identity.get("sub") or "")
    if attempt.returning_subject and subject != attempt.returning_subject:
        raise ChatGPTAuthError("The selected ChatGPT account does not match this registration.")
    scopes = _normalize_scopes(token_payload.get("scope"))
    profile = ChatGPTProfile(
        email=str(identity.get("email") or ""),
        issuer=str(identity.get("iss") or ""),
        subject=subject,
        client_id=issued_client_id,
        host_id=attempt.host_id,
        id_token=str(token_payload.get("id_token") or ""),
        access_token=str(token_payload.get("access_token") or ""),
        refresh_token=str(token_payload.get("refresh_token") or ""),
        token_type=str(token_payload.get("token_type") or "Bearer"),
        expires_at=time.time() + float(token_payload.get("expires_in") or 3600),
        scopes=scopes,
    )
    if not profile.access_token:
        raise ChatGPTAuthError("ChatGPT sign-in returned no access token.")
    store.save_profile(profile)
    return profile


def validate_id_token(
    token: str,
    *,
    client_id: str,
    nonce: str,
    client: httpx.Client,
) -> dict[str, Any]:
    if not token:
        raise ChatGPTAuthError("ChatGPT sign-in returned no identity token.")
    try:
        configuration = client.get(OIDC_CONFIGURATION_URL)
        configuration.raise_for_status()
        metadata = configuration.json()
        issuer = str(metadata.get("issuer") or "")
        jwks_uri = str(metadata.get("jwks_uri") or "")
        if not issuer or not jwks_uri:
            raise ChatGPTAuthError("OpenAI identity metadata was incomplete.")
        jwks_response = client.get(jwks_uri)
        jwks_response.raise_for_status()
        key_set = jwt.PyJWKSet.from_dict(jwks_response.json())
        header = jwt.get_unverified_header(token)
        key_id = str(header.get("kid") or "")
        signing_key = next((key for key in key_set.keys if key.key_id == key_id), None)
        if signing_key is None:
            raise ChatGPTAuthError("OpenAI identity signing key was not found.")
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=[str(header.get("alg") or "RS256")],
            audience=client_id,
            issuer=issuer,
            options={"require": ["exp", "iss", "aud", "sub", "nonce"]},
        )
    except ChatGPTAuthError:
        raise
    except jwt.PyJWTError as exc:
        raise ChatGPTAuthError("ChatGPT returned an invalid identity token.") from exc
    if not secrets.compare_digest(str(claims.get("nonce") or ""), nonce):
        raise ChatGPTAuthError("The ChatGPT identity token nonce did not match.")
    return dict(claims)


def refresh_profile(
    profile: ChatGPTProfile,
    *,
    store: ChatGPTCredentialStore,
    client: httpx.Client | None = None,
) -> ChatGPTProfile:
    if not profile.refresh_token:
        raise ChatGPTAuthError("Reconnect ChatGPT to renew access.")
    owns_client = client is None
    http = client or httpx.Client(timeout=30.0)
    try:
        response = http.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "client_id": profile.client_id,
                "refresh_token": profile.refresh_token,
                "resource": RESOURCE,
            },
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        raise ChatGPTAuthError("ChatGPT access expired and could not be renewed.") from exc
    finally:
        if owns_client:
            http.close()
    profile.access_token = str(payload.get("access_token") or "")
    profile.refresh_token = str(payload.get("refresh_token") or profile.refresh_token)
    profile.id_token = str(payload.get("id_token") or profile.id_token)
    profile.token_type = str(payload.get("token_type") or profile.token_type)
    profile.expires_at = time.time() + float(payload.get("expires_in") or 3600)
    if payload.get("scope") is not None:
        profile.scopes = _normalize_scopes(payload.get("scope"))
    if not profile.access_token:
        raise ChatGPTAuthError("ChatGPT token renewal returned no access token.")
    store.save_profile(profile)
    return profile


def usable_access_token(
    store: ChatGPTCredentialStore,
    *,
    client: httpx.Client | None = None,
) -> tuple[ChatGPTProfile, str]:
    profile = store.load_profile()
    if profile is None or not profile.access_token:
        raise ChatGPTAuthError("Connect ChatGPT before using OpenAI auto-labeling.")
    if not profile.plan_usage_enabled:
        raise ChatGPTAuthError("Enable ChatGPT plan usage when reconnecting this account.")
    if profile.expires_at <= time.time() + 60:
        profile = refresh_profile(profile, store=store, client=client)
    return profile, profile.access_token


def revoke_profile(
    store: ChatGPTCredentialStore,
    *,
    client: httpx.Client | None = None,
) -> None:
    profile = store.load_profile()
    if profile is None:
        return
    owns_client = client is None
    http = client or httpx.Client(timeout=30.0)
    try:
        if profile.refresh_token:
            configuration = http.get(OIDC_CONFIGURATION_URL)
            configuration.raise_for_status()
            endpoint = str(configuration.json().get("revocation_endpoint") or "")
            if endpoint:
                response = http.post(
                    endpoint,
                    data={
                        "token": profile.refresh_token,
                        "token_type_hint": "refresh_token",
                        "client_id": profile.client_id,
                    },
                )
                response.raise_for_status()
    except (httpx.HTTPError, ValueError, TypeError):
        # Local sign-out must still stop use of a credential. The UI tells the user
        # that remote revocation could not be confirmed through the raised error.
        store.clear_tokens()
        raise ChatGPTAuthError(
            "Signed out locally, but remote revocation could not be confirmed. "
            "You can disconnect SqueakPose Studio in ChatGPT Settings."
        )
    finally:
        if owns_client:
            http.close()
    store.clear_tokens()


def _normalize_scopes(value: Any) -> list[str]:
    if isinstance(value, str):
        return sorted(set(value.split()))
    if isinstance(value, (list, tuple, set)):
        return sorted({str(scope) for scope in value if str(scope)})
    return []


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


__all__ = [
    "AuthorizationAttempt",
    "ChatGPTAuthError",
    "ChatGPTCredentialStore",
    "ChatGPTProfile",
    "complete_authorization",
    "create_authorization_attempt",
    "default_openai_config_dir",
    "refresh_profile",
    "revoke_profile",
    "usable_access_token",
    "validate_id_token",
]
