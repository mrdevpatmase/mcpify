"""
Regression coverage for the security-hardening pieces pulled in from
GitHub (commit 1440ff6) and then wired up/fixed here: encrypt_secret/
decrypt_secret were defined but never called anywhere (dead code -
secrets were still stored in plaintext despite the "AES-256
encryption" claim), verify_admin_key was defined but not attached to
any endpoint, and a bare `proxy_id` reference in the audit-log line
that same commit added was an undefined-name NameError crashing every
successful call_api/graphql_query - found live, not by inspection, by
actually calling those tools locally and getting a 500.
"""
import asyncio

import pytest

from app.proxy import ProxyMCPManager, _decrypt_proxy_secrets, _encrypt_proxy_secrets
from app.security import decrypt_secret, encrypt_secret


def test_encrypt_decrypt_round_trip():
    original = "super-secret-api-key-123"
    encrypted = encrypt_secret(original)
    assert encrypted != original
    assert encrypted.startswith("enc:")
    assert decrypt_secret(encrypted) == original


def test_decrypt_passes_through_unencrypted_values():
    # Anything not produced by encrypt_secret (no "enc:" prefix) must be
    # handed back unchanged - covers values written before encryption
    # existed, or when it silently no-ops (cryptography missing).
    assert decrypt_secret("plain-value") == "plain-value"
    assert decrypt_secret(None) is None
    assert decrypt_secret("") == ""


def test_encrypt_decrypt_handles_none_and_empty():
    assert encrypt_secret(None) is None
    assert encrypt_secret("") == ""


def test_encrypt_proxy_secrets_covers_api_key_and_oauth_fields():
    proxy = {
        "proxy_id": "p1",
        "target_url": "https://example.com",
        "api_key": "plain-api-key",
        "oauth_config": {
            "grant_type": "authorization_code",
            "client_id": "cid",
            "client_secret": "plain-client-secret",
            "cached_token": "plain-access-token",
            "refresh_token": "plain-refresh-token",
        },
    }
    encrypted = _encrypt_proxy_secrets(proxy)

    # Original dict must be untouched - other code may still hold a
    # reference to it expecting plaintext.
    assert proxy["api_key"] == "plain-api-key"

    assert encrypted["api_key"].startswith("enc:")
    assert encrypted["oauth_config"]["client_secret"].startswith("enc:")
    assert encrypted["oauth_config"]["cached_token"].startswith("enc:")
    assert encrypted["oauth_config"]["refresh_token"].startswith("enc:")
    assert encrypted["oauth_config"]["client_id"] == "cid"  # not a secret, left alone

    decrypted = _decrypt_proxy_secrets(encrypted)
    assert decrypted["api_key"] == "plain-api-key"
    assert decrypted["oauth_config"]["client_secret"] == "plain-client-secret"
    assert decrypted["oauth_config"]["cached_token"] == "plain-access-token"
    assert decrypted["oauth_config"]["refresh_token"] == "plain-refresh-token"


def test_encrypt_proxy_secrets_handles_missing_fields():
    proxy = {"proxy_id": "p2", "target_url": "https://example.com"}
    encrypted = _encrypt_proxy_secrets(proxy)
    assert "api_key" not in encrypted
    decrypted = _decrypt_proxy_secrets(encrypted)
    assert decrypted == proxy


def test_call_api_success_path_does_not_crash_on_audit_log():
    """
    The concrete regression: a bare `proxy_id` name (not proxy["proxy_id"])
    in the audit-log line inside handle_jsonrpc_request's call_api
    success path raised NameError on every successful call, turning
    into an opaque 500 for the single most-used tool in this app.
    """
    async def run():
        manager = ProxyMCPManager()
        proxy = await manager.create_proxy(target_url="https://httpbin.org", has_mcp=False)
        result = await manager.handle_jsonrpc_request(proxy, {
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "call_api", "arguments": {"endpoint": "/get", "method": "GET"}},
        })
        assert "error" not in result
        assert result["result"]["content"][0]["text"]
        # The fixed line should have recorded a real audit entry too.
        log = await manager.get_audit_log()
        assert len(log) == 1
        assert log[0]["tool"] == "call_api"
        assert log[0]["proxy_id"] == proxy["proxy_id"]

    asyncio.run(run())


def test_audit_log_in_memory_caps_and_orders_newest_first():
    async def run():
        manager = ProxyMCPManager()
        for i in range(5):
            await manager.record_audit_event({"tool": "call_api", "seq": i})
        log = await manager.get_audit_log(limit=3)
        assert len(log) == 3
        # Newest first.
        assert [e["seq"] for e in log] == [4, 3, 2]

    asyncio.run(run())


def test_verify_admin_key_open_when_unset(monkeypatch):
    from fastapi import Request
    from app.security import verify_admin_key

    monkeypatch.delenv("ADMIN_API_KEY", raising=False)

    async def run():
        scope = {"type": "http", "headers": []}
        req = Request(scope)
        result = await verify_admin_key(req, api_key=None)
        assert result is None

    asyncio.run(run())


def test_verify_admin_key_rejects_missing_or_wrong_key(monkeypatch):
    from fastapi import HTTPException, Request
    from app.security import verify_admin_key

    monkeypatch.setenv("ADMIN_API_KEY", "correct-key")

    async def run():
        scope = {"type": "http", "headers": []}
        req = Request(scope)
        with pytest.raises(HTTPException) as exc_info:
            await verify_admin_key(req, api_key="wrong-key")
        assert exc_info.value.status_code == 401

        with pytest.raises(HTTPException):
            await verify_admin_key(req, api_key=None)

    asyncio.run(run())


def test_verify_admin_key_accepts_correct_key(monkeypatch):
    from fastapi import Request
    from app.security import verify_admin_key

    monkeypatch.setenv("ADMIN_API_KEY", "correct-key")

    async def run():
        scope = {"type": "http", "headers": []}
        req = Request(scope)
        result = await verify_admin_key(req, api_key="correct-key")
        assert result == "correct-key"

    asyncio.run(run())
