"""
OAuth2 support: Client Credentials (RFC 6749 section 4.4) and
Authorization Code (section 4.1) grants.

Client Credentials is the "machine-to-machine" grant: exchange a
client_id/client_secret directly for an access token, no browser/user
redirect involved - a user can just supply it at proxy-creation time
the same way an api_key already works.

Authorization Code is the "login with Google/Slack" kind everyone
thinks of first with OAuth - it DOES need a human in the loop (the
target's own login/consent screen), so unlike client_credentials it
can't be a single synchronous step at proxy-creation time. main.py's
/proxy/create returns an authorization_url for the user to visit once;
/oauth/callback receives the redirect back and completes the exchange.
Both grants end up producing the same shape of result here, and the
resulting token is cached/refreshed identically by
ProxyMCPManager.get_valid_oauth_token - only how the FIRST token is
obtained differs.
"""
import time
from typing import Any, Dict, Optional

import httpx


async def _post_token_request(client: httpx.AsyncClient, token_url: str, data: Dict[str, str]) -> Dict[str, Any]:
    """
    Shared POST + response-parsing for every grant type below. Returns
    {"access_token": str, "refresh_token": Optional[str], "expires_at":
    epoch_seconds, "error": None} on success, or {"access_token": None,
    "refresh_token": None, "expires_at": None, "error": str} on failure.
    expires_at is computed a little early (30s before the provider's own
    expiry, or a conservative 5-minute default when a provider omits
    expires_in) so a call already in flight right at the boundary
    doesn't get handed a token that's gone stale mid-request.
    """
    try:
        resp = await client.post(token_url, data=data, timeout=10.0)
    except Exception as e:
        return {"access_token": None, "refresh_token": None, "expires_at": None, "error": f"Could not reach token endpoint: {e}"}

    if resp.status_code >= 400:
        return {
            "access_token": None, "refresh_token": None, "expires_at": None,
            "error": f"Token endpoint refused the request ({resp.status_code}): {resp.text[:300]}",
        }

    try:
        body = resp.json()
    except Exception:
        return {"access_token": None, "refresh_token": None, "expires_at": None, "error": "Token endpoint did not return valid JSON"}

    access_token = body.get("access_token")
    if not access_token:
        return {"access_token": None, "refresh_token": None, "expires_at": None, "error": "Token endpoint response had no access_token field"}

    expires_in = body.get("expires_in", 300)
    try:
        expires_in = int(expires_in)
    except (TypeError, ValueError):
        expires_in = 300

    expires_at = time.time() + max(expires_in - 30, 30)
    return {"access_token": access_token, "refresh_token": body.get("refresh_token"), "expires_at": expires_at, "error": None}


async def fetch_client_credentials_token(
    client: httpx.AsyncClient, token_url: str, client_id: str, client_secret: str, scope: Optional[str] = None
) -> Dict[str, Any]:
    """Performs the standard OAuth2 client_credentials grant."""
    data = {"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret}
    if scope:
        data["scope"] = scope
    return await _post_token_request(client, token_url, data)


async def exchange_authorization_code(
    client: httpx.AsyncClient, token_url: str, client_id: str, client_secret: str, code: str, redirect_uri: str
) -> Dict[str, Any]:
    """
    Exchanges an authorization code (received at the redirect_uri after
    the user logged in and consented on the target's own site) for an
    access token. redirect_uri must be sent again here and must match
    the one used to build the authorization_url exactly - the spec
    requires this so a stolen code can't be replayed from a different
    redirect target.
    """
    data = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": redirect_uri,
    }
    return await _post_token_request(client, token_url, data)


async def refresh_access_token(
    client: httpx.AsyncClient, token_url: str, client_id: str, client_secret: str, refresh_token: str
) -> Dict[str, Any]:
    """Exchanges a refresh_token for a new access token, once the cached
    one from either grant above has expired."""
    data = {
        "grant_type": "refresh_token",
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
    }
    return await _post_token_request(client, token_url, data)
