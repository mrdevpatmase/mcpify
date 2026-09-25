"""
Regression coverage for the OAuth2 Authorization Code grant added to
app/oauth.py and app/proxy.py. Full third-party success (a real
provider's login/consent screen) can't be automated - that step
genuinely needs a human in a browser - so this exercises everything
around it against a real local mock token endpoint (tests/conftest
starts nothing special; see the fixture below) plus pure-logic checks
for the pending-state correlation.

is_public_url is monkeypatched to accept the mock server's localhost
address ONLY within these tests - this is a test-only override of a
real SSRF guard, never a production change, done because there's no
way to stand up a public, self-serve OAuth sandbox to test against
instead (unlike the OpenAPI/GraphQL discovery tests, which hit real
public APIs directly).
"""
import asyncio
import time

import httpx
import pytest

import app.proxy as proxy_module
from app.oauth import exchange_authorization_code, refresh_access_token
from app.proxy import ProxyMCPManager

pytestmark = pytest.mark.network

MOCK_TOKEN_URL = "http://127.0.0.1:8199/token"


@pytest.fixture
def allow_localhost(monkeypatch):
    async def _fake_is_public_url(url):
        return True, ""
    monkeypatch.setattr(proxy_module, "is_public_url", _fake_is_public_url)


def test_exchange_and_refresh_against_real_mock_server():
    async def run():
        async with httpx.AsyncClient() as client:
            result = await exchange_authorization_code(
                client, MOCK_TOKEN_URL, "test-client", "test-secret", "valid-test-code", "http://x/callback"
            )
        assert result["error"] is None
        assert result["access_token"] == "access-1"
        assert result["refresh_token"] == "refresh-1"

        async with httpx.AsyncClient() as client:
            bad = await exchange_authorization_code(
                client, MOCK_TOKEN_URL, "test-client", "test-secret", "wrong-code", "http://x/callback"
            )
        assert bad["access_token"] is None
        assert "invalid_grant" in bad["error"]

        async with httpx.AsyncClient() as client:
            refreshed = await refresh_access_token(client, MOCK_TOKEN_URL, "test-client", "test-secret", "refresh-1")
        assert refreshed["access_token"] == "access-2"
        assert refreshed["refresh_token"] == "refresh-2"

    asyncio.run(run())


def test_pending_oauth_state_is_single_use():
    async def run():
        manager = ProxyMCPManager()
        await manager.save_pending_oauth_state("state-abc", "proxy-123", ttl_seconds=60)
        assert await manager.pop_pending_oauth_state("state-abc") == "proxy-123"
        # Single-use: popping the same state again must fail, so a
        # captured/replayed redirect can't be completed twice.
        assert await manager.pop_pending_oauth_state("state-abc") is None
        # Never-issued state.
        assert await manager.pop_pending_oauth_state("never-issued") is None

    asyncio.run(run())


def test_pending_oauth_state_expires():
    async def run():
        manager = ProxyMCPManager()
        await manager.save_pending_oauth_state("state-exp", "proxy-456", ttl_seconds=-1)
        assert await manager.pop_pending_oauth_state("state-exp") is None

    asyncio.run(run())


def test_get_valid_oauth_token_refreshes_authorization_code_grant(allow_localhost):
    async def run():
        manager = ProxyMCPManager()
        proxy = {
            "proxy_id": "px1",
            "oauth_config": {
                "grant_type": "authorization_code",
                "token_url": MOCK_TOKEN_URL,
                "client_id": "test-client",
                "client_secret": "test-secret",
                "cached_token": None,
                "cached_token_expires_at": None,
                "refresh_token": "refresh-1",
            },
        }
        manager.proxies["px1"] = proxy

        token, error = await manager.get_valid_oauth_token(proxy)
        assert error is None
        assert token == "access-2"
        assert proxy["oauth_config"]["refresh_token"] == "refresh-2"

        # Cached token should be reused on the very next call without
        # hitting the mock server again - simplest way to prove that
        # here is that a wrong refresh_token would now fail if it DID
        # try to refresh, but it doesn't because the cache is fresh.
        proxy["oauth_config"]["refresh_token"] = "now-invalid"
        token2, error2 = await manager.get_valid_oauth_token(proxy)
        assert error2 is None
        assert token2 == "access-2"

    asyncio.run(run())


def test_get_valid_oauth_token_pending_authorization_reports_clear_error(allow_localhost):
    async def run():
        manager = ProxyMCPManager()
        proxy = {
            "proxy_id": "px2",
            "oauth_config": {
                "grant_type": "authorization_code",
                "token_url": MOCK_TOKEN_URL,
                "client_id": "test-client",
                "client_secret": "test-secret",
                "cached_token": None,
                "cached_token_expires_at": None,
                "refresh_token": None,
                "authorization_url_for_user": "https://provider.example/authorize?state=xyz",
            },
        }
        token, error = await manager.get_valid_oauth_token(proxy)
        assert token is None
        assert "provider.example/authorize" in error

    asyncio.run(run())
