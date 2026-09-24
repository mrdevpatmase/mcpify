"""
Client Credentials OAuth2 support (RFC 6749 section 4.4) - the
"machine-to-machine" grant: exchange a client_id/client_secret directly
for an access token at a token endpoint, no browser/user redirect
involved.

Deliberately NOT the Authorization Code flow (the "login with Google/
Slack" kind everyone thinks of first with OAuth): that requires
registering a real OAuth app with EVERY target platform ahead of time
(redirect URI allowlisting, consent screens, per-provider app
credentials issued by that platform's own developer console) - it can't
be something a user just supplies at proxy-creation time the way an
api_key already works, so it doesn't fit this app's existing "give us
your credential, we handle the rest" pattern. Client Credentials does:
plenty of real B2B/enterprise APIs use exactly this grant for
server-to-server access with no human in the loop.
"""
import time
from typing import Any, Dict, Optional

import httpx


async def fetch_client_credentials_token(
    client: httpx.AsyncClient, token_url: str, client_id: str, client_secret: str, scope: Optional[str] = None
) -> Dict[str, Any]:
    """
    Performs the standard OAuth2 client_credentials grant. Returns
    {"access_token": str, "expires_at": epoch_seconds, "error": None} on
    success, or {"access_token": None, "expires_at": None, "error": str}
    on failure. expires_at is computed a little early (30s before the
    provider's own expiry, or a conservative 5-minute default when a
    provider omits expires_in) so a call already in flight right at the
    boundary doesn't get handed a token that's gone stale mid-request.
    """
    data = {"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret}
    if scope:
        data["scope"] = scope

    try:
        resp = await client.post(token_url, data=data, timeout=10.0)
    except Exception as e:
        return {"access_token": None, "expires_at": None, "error": f"Could not reach token endpoint: {e}"}

    if resp.status_code >= 400:
        return {
            "access_token": None, "expires_at": None,
            "error": f"Token endpoint refused the request ({resp.status_code}): {resp.text[:300]}",
        }

    try:
        body = resp.json()
    except Exception:
        return {"access_token": None, "expires_at": None, "error": "Token endpoint did not return valid JSON"}

    access_token = body.get("access_token")
    if not access_token:
        return {"access_token": None, "expires_at": None, "error": "Token endpoint response had no access_token field"}

    expires_in = body.get("expires_in", 300)
    try:
        expires_in = int(expires_in)
    except (TypeError, ValueError):
        expires_in = 300

    expires_at = time.time() + max(expires_in - 30, 30)
    return {"access_token": access_token, "expires_at": expires_at, "error": None}
