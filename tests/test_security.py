"""
Regression coverage for app/security.py's redirect-resolution logic -
found live testing HubSpot's real API: resolve_canonical_base followed a
redirect off the API domain entirely onto an unrelated docs site.
"""
import asyncio

import pytest

from app.security import _registrable_domain, resolve_canonical_base

pytestmark = pytest.mark.network


def test_registrable_domain_distinguishes_unrelated_sites():
    # The concrete bug: HubSpot's real API domain redirects its bare root
    # to a completely different, unrelated docs domain.
    assert _registrable_domain("api.hubapi.com") != _registrable_domain("developers.hubspot.com")
    # Same-site subdomain variants must still match (this is what
    # resolve_canonical_base's apex/www following depends on).
    assert _registrable_domain("github.com") == _registrable_domain("www.github.com")
    assert _registrable_domain("api.example.com") == _registrable_domain("www.example.com")
    # A short list of common two-label ccTLDs is handled specially, not
    # just "last two labels".
    assert _registrable_domain("api.example.co.uk") == _registrable_domain("www.example.co.uk")


def test_cross_domain_redirect_is_not_followed():
    """
    The regression this test locks in: a bare-root redirect to a
    genuinely different site must NOT replace the original target -
    verified live against api.hubapi.com, which 302s its root to
    developers.hubspot.com (HubSpot's real API still answers correctly at
    its actual sub-paths on the ORIGINAL domain - following the redirect
    would have pointed every later probe at the wrong site entirely).
    """
    resolved = asyncio.run(resolve_canonical_base("https://api.hubapi.com"))
    assert resolved == "https://api.hubapi.com"


def test_same_domain_redirect_is_still_followed():
    """Not a blanket ban on following redirects - same-site apex/http
    upgrades (what this function exists for) must keep working."""
    resolved = asyncio.run(resolve_canonical_base("http://github.com"))
    assert "github.com" in resolved
    assert resolved.startswith("https://")
