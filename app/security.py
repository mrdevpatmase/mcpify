import asyncio
import ipaddress
import socket
import os
import base64
import hashlib
import logging
from typing import Optional, Dict, Any, Tuple
from urllib.parse import urlparse, urljoin

import httpx
from fastapi import Request, HTTPException, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger("mcpify")

# Admin API Key Security Header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_admin_key(request: Request, api_key: str = Security(api_key_header)) -> Optional[str]:
    """
    FastAPI dependency: Validates request against ADMIN_API_KEY environment variable.
    If ADMIN_API_KEY is configured in .env/environment, requests MUST provide
    a matching X-API-Key header or Bearer token.
    If ADMIN_API_KEY is not set, admin endpoints remain open (for local dev mode).
    """
    expected_key = os.getenv("ADMIN_API_KEY")
    if not expected_key:
        return None  # Unprotected mode (local development)

    # Check X-API-Key header or Authorization: Bearer <key>
    auth_header = request.headers.get("Authorization", "")
    bearer_key = auth_header[7:].strip() if auth_header.startswith("Bearer ") else None

    provided_key = api_key or bearer_key
    if not provided_key or provided_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or missing X-API-Key / Authorization Bearer token."
        )
    return provided_key


_warned_missing_encryption_key = False


def _get_encryption_key() -> bytes:
    """
    Derive a 32-byte key for Fernet (AES-128-CBC + HMAC, despite the
    "AES-256" shorthand this is usually described by) encryption of
    secrets at rest.

    The hardcoded fallback below is a real value sitting in this repo's
    git history forever - anyone who can read the source (which is
    everyone; this repo isn't private infra) can derive the exact same
    key from it. That's not "weaker encryption", it's no encryption at
    all: it stops a casual glance at a Redis dump, nothing else. Warn
    loudly (once, not on every single call) rather than silently
    degrading to security theater with no visible signal that
    ENCRYPTION_KEY was never actually configured.
    """
    global _warned_missing_encryption_key
    secret = os.getenv("ENCRYPTION_KEY")
    if not secret:
        if not _warned_missing_encryption_key:
            logger.warning(
                "[Security] ENCRYPTION_KEY is not set - falling back to a hardcoded default that provides "
                "NO real protection (it's public, in this repo's source). Set ENCRYPTION_KEY in the environment "
                "to a long random value to actually encrypt secrets at rest."
            )
            _warned_missing_encryption_key = True
        secret = "mcpify_default_secret_key_change_in_prod"
    key_bytes = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(key_bytes)


def encrypt_secret(plain_text: Optional[str]) -> Optional[str]:
    """Encrypt sensitive credentials (API keys, OAuth client secrets) at rest."""
    if not plain_text:
        return plain_text
    try:
        from cryptography.fernet import Fernet
        f = Fernet(_get_encryption_key())
        return "enc:" + f.encrypt(plain_text.encode("utf-8")).decode("utf-8")
    except Exception as e:
        # A silent fallback here means secrets get written to Redis in
        # PLAINTEXT with no error and no visible sign anything went
        # wrong - the one failure mode a "protect secrets at rest"
        # feature can least afford to have pass unnoticed.
        logger.warning("[Security] encrypt_secret failed, storing value unencrypted: %s", e)
        return plain_text


def decrypt_secret(cipher_text: Optional[str]) -> Optional[str]:
    """Decrypt sensitive credentials at rest."""
    if not cipher_text or not cipher_text.startswith("enc:"):
        return cipher_text
    try:
        from cryptography.fernet import Fernet
        f = Fernet(_get_encryption_key())
        raw_cipher = cipher_text[4:]
        return f.decrypt(raw_cipher.encode("utf-8")).decode("utf-8")
    except Exception as e:
        logger.warning(f"Failed to decrypt secret: {e}")
        return cipher_text



def normalize_url(url: str) -> str:
    """
    Ensure URL has a scheme and strip query string/fragment - those are
    never meaningful as part of a base URL. The path IS kept: plenty of
    real APIs are deliberately hosted under a path on a shared domain
    (e.g. "https://dog.ceo/api", "https://api.stripe.com/v1") and every
    endpoint call this app makes is built by appending onto this base, so
    dropping that path silently breaks those APIs - verified live against
    dog.ceo/api, where every call_api call 404'd once the /api prefix was
    lost. This used to strip the path too (to handle someone pasting a
    full marketing-page URL from their browser, e.g.
    "https://example.com/landing/page?x=1", rather than the bare domain -
    see git history), but that traded a silent wrong-data bug (a real API
    quietly returning 404s from the wrong base) for a softer, recoverable
    one (a mis-scoped page still yields a working proxy, just possibly
    probed under the wrong sub-path - the user can retry with the
    corrected URL if so). Silent wrong data is worse, so path wins.

    Shared by analyzer.py, proxy.py, and main.py so a URL is normalized
    the same way no matter which entry point receives it.
    """
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"https://{url}"

    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        # e.g. "javascript:alert(1)" parses with a non-numeric "port" -
        # urlparse defers that error to attribute access. Not a URL we can
        # make sense of; let is_public_url's hostname check reject it
        # cleanly instead of raising here.
        return url.rstrip("/")

    if not hostname:
        return url.rstrip("/")

    base = f"{parsed.scheme}://{hostname}"
    if port:
        base += f":{port}"
    path = parsed.path.rstrip("/")
    if path:
        base += path
    return base


_NAT64_WELL_KNOWN_PREFIX = ipaddress.ip_network("64:ff9b::/96")


def _is_unsafe_ip(ip_str: str) -> bool:
    """
    Returns True if the IP is private, loopback, link-local, reserved, or
    otherwise internal.

    64:ff9b::/96 (RFC 6052's NAT64 "Well-Known Prefix") is a special case:
    it's just an IPv6 wrapper around a real IPv4 address in its last 32
    bits, synthesized by DNS64 on IPv6-only/NAT64 networks for IPv4-only
    hosts - verified live (stripe.com and badssl.com resolved to this on
    this network). Python's ipaddress module marks the WHOLE prefix
    is_reserved=True (it's a special-purpose IANA block), which would
    reject every ordinary public IPv4-only site whenever DNS64
    synthesizes this instead of returning a plain A record - unwrap it
    and check the REAL embedded destination instead of the wrapper.
    Recurses so a NAT64 address wrapping an actually-private IPv4 (e.g.
    64:ff9b::7f00:1 embeds 127.0.0.1) still correctly gets rejected.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True

    if isinstance(ip, ipaddress.IPv6Address) and ip in _NAT64_WELL_KNOWN_PREFIX:
        embedded_ipv4 = ipaddress.IPv4Address(ip.packed[-4:])
        return _is_unsafe_ip(str(embedded_ipv4))

    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


async def is_public_url(url: str) -> tuple[bool, str]:
    """
    Validates that a target URL's scheme is HTTP(S) and its hostname resolves
    only to public IP addresses, to prevent SSRF against internal/cloud-metadata
    infrastructure via the URL-probing and proxy-creation endpoints.

    Returns (is_safe, reason). reason is empty when is_safe is True.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Malformed URL."

    if parsed.scheme not in ("http", "https"):
        return False, "Only http/https URLs are allowed."

    hostname = parsed.hostname
    if not hostname:
        return False, "URL has no hostname."

    if hostname.lower() in ("localhost", "0.0.0.0", "::1"):
        return False, "Requests to localhost are not allowed."

    try:
        loop = asyncio.get_event_loop()
        infos = await loop.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False, f"Could not resolve hostname: {hostname}"
    except Exception as e:
        return False, f"DNS resolution error: {e}"

    if not infos:
        return False, f"Could not resolve hostname: {hostname}"

    for info in infos:
        ip_str = info[4][0]
        if _is_unsafe_ip(ip_str):
            return False, f"Target resolves to a non-public address ({ip_str}); not allowed."

    return True, ""


_MULTI_PART_TLDS = {
    "co.uk", "org.uk", "gov.uk", "ac.uk", "co.in", "co.jp", "co.kr", "co.nz",
    "com.au", "com.br", "com.cn", "com.mx", "com.sg", "com.tw", "co.za",
}


def _registrable_domain(hostname: str) -> str:
    """
    Approximates "the real site" a hostname belongs to, for deciding
    whether a redirect target is still the same site
    (api.example.com -> www.example.com, following it is the whole
    point of resolve_canonical_base) or a genuinely different one -
    verified live: HubSpot's real API domain (api.hubapi.com) has no
    root page and 302s the bare root to developers.hubspot.com, a
    completely unrelated docs site on a different registrable domain,
    not "the API moved here". Following that redirect resolved every
    later probe against the wrong target entirely, losing the real API.

    Not a full public-suffix-list implementation (no new dependency for
    this) - handles the common two-label case plus a short list of
    frequently-seen multi-label ccTLDs. An unrecognized multi-label TLD
    occasionally being one label too coarse is an acceptable miss here:
    this is already a best-effort convenience, not a security boundary -
    is_public_url is what actually guards against unsafe redirect
    targets, unaffected by this.
    """
    labels = hostname.lower().split(".")
    if len(labels) < 2:
        return hostname.lower()
    last_two = ".".join(labels[-2:])
    if len(labels) >= 3:
        last_three = ".".join(labels[-3:])
        if last_two in _MULTI_PART_TLDS:
            return last_three
    return last_two


async def resolve_canonical_base(url: str) -> str:
    """
    Many real sites blanket-redirect at the domain level (apex -> www,
    http -> https, bare-domain -> a trailing slash). Every other outbound
    call in this app deliberately does NOT follow redirects (SSRF
    hardening - see is_public_url), so such a site would otherwise look
    completely dead: every probe just bounces off a 3xx and the proxy's
    actual tool calls would too.

    This resolves that ONE-TIME, at proxy-creation time - but validates
    EACH hop's destination against is_public_url BEFORE connecting to it,
    not just the final one. httpx's own follow_redirects=True would
    connect to every hop first and only let us inspect the destination
    afterward - a malicious target's first hop could point straight at an
    internal address and that request would already have happened by the
    time anything checked it (a "blind" SSRF: no response body comes
    back to the caller, but the request itself still reaches the internal
    target). Falls back to the original URL on any failure (timeout,
    malformed response, too many hops, or an unsafe hop) rather than
    raising, since this is a best-effort convenience, not a required step.
    """
    current = url
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            for _ in range(5):
                resp = await client.get(current)
                if resp.status_code not in (301, 302, 303, 307, 308):
                    break
                location = resp.headers.get("location")
                if not location:
                    break
                next_url = urljoin(current, location)
                is_safe, _ = await is_public_url(next_url)
                if not is_safe:
                    return url
                next_host = urlparse(next_url).hostname or ""
                original_host = urlparse(url).hostname or ""
                if _registrable_domain(next_host) != _registrable_domain(original_host):
                    # A genuinely different site, not "this same API
                    # moved" - stop here and keep probing the ORIGINAL
                    # target rather than silently switching to whatever
                    # unrelated site the redirect happened to point at.
                    break
                if original_host.lower().startswith("api.") and not next_host.lower().startswith("api."):
                    # Same registrable domain isn't always enough either -
                    # verified live: api.trello.com's bare root 301s to
                    # trello.com (the login/marketing page), same
                    # registrable domain but a completely different
                    # purpose. The real API answers fine at its actual
                    # paths (GET /1/members/me returns a real API error,
                    # not HTML) - following this redirect would have
                    # dropped the "api." host and probed the marketing
                    # site instead.
                    break
                current = next_url
            else:
                # Exhausted the hop budget without landing on a final page.
                return url
    except Exception:
        return url

    if current.rstrip("/") == url.rstrip("/"):
        return url
    return normalize_url(current)
