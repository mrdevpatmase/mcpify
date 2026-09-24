"""
Regression coverage for app/openapi_tools.py against real, stable public
specs - this feature had zero automated coverage despite several real bugs
found live (YAML specs, servers-declared base paths, apiKey-in-header auth,
unencoded path params) before this file existed. Marked @pytest.mark.network
like the rest of the real-world corpus - see test_real_world_corpus.py's
module docstring for why.
"""
import asyncio

import httpx
import pytest

from app.openapi_tools import (
    build_request,
    discover_openapi_spec,
    operations_to_mcp_tools,
    parse_operations,
    resolve_api_base,
    resolve_auth_header_name,
)

pytestmark = pytest.mark.network

PETSTORE = "https://petstore3.swagger.io/api/v3"


def test_discovers_and_calls_a_real_spec():
    async def run():
        async with httpx.AsyncClient(timeout=10.0) as client:
            spec = await discover_openapi_spec(client, PETSTORE)
        assert spec is not None
        ops = parse_operations(spec, PETSTORE)
        assert len(ops) > 0
        tools = operations_to_mcp_tools(ops)
        names = {t["name"] for t in tools}
        assert "get_pet_by_id" in names

        op = next(o for o in ops if o["tool_name"] == "get_pet_by_id")
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{op['base_url']}{build_request(op, {'petId': 1})['path']}")
        assert resp.status_code == 200
        assert resp.json()["id"] == 1

    asyncio.run(run())


def test_path_param_injection_is_encoded_not_executed():
    """The concrete bug this test locks in: an unencoded path param let a
    caller inject extra path segments or a query string into the URL."""
    op = {
        "path_template": "/pet/{petId}",
        "path_params": [{"name": "petId", "required": True, "type": "string"}],
        "query_params": [],
        "has_body": False,
    }
    result = build_request(op, {"petId": "../admin/secret"})
    assert result["error"] is None
    assert "/" not in result["path"].split("/pet/", 1)[1]
    assert result["path"] == "/pet/..%2Fadmin%2Fsecret"

    result2 = build_request(op, {"petId": "x?evil=1"})
    assert "?" not in result2["path"]


def test_resolve_api_base_handles_servers_variants():
    # Relative server path, distinct from the discovery domain (n8n's real shape)
    assert resolve_api_base({"servers": [{"url": "/api/v1"}]}, "https://host.example") == "https://host.example/api/v1"
    # Already part of the discovery URL - must not double up (petstore's real shape)
    assert resolve_api_base({"servers": [{"url": "/api/v3"}]}, "https://host.example/api/v3") == "https://host.example/api/v3"
    # Absolute server URL
    assert resolve_api_base({"servers": [{"url": "https://api.example.com"}]}, "https://host.example") == "https://api.example.com"
    # Template server - keep the static suffix only
    assert resolve_api_base({"servers": [{"url": "{url}/api/v1"}]}, "https://host.example") == "https://host.example/api/v1"
    # No servers at all - falls back to the discovery URL
    assert resolve_api_base({}, "https://host.example") == "https://host.example"


def test_resolve_auth_header_name_prefers_declared_apikey_header():
    spec = {
        "security": [{"ApiKeyAuth": []}, {"BearerAuth": []}],
        "components": {
            "securitySchemes": {
                "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-N8N-API-KEY"},
                "BearerAuth": {"type": "http", "scheme": "bearer"},
            }
        },
    }
    assert resolve_auth_header_name(spec) == "X-N8N-API-KEY"
    # No security declared at all - default (None -> caller falls back to Bearer)
    assert resolve_auth_header_name({}) is None
