"""
Regression coverage for app/graphql_tools.py against a real, stable
public GraphQL API - same pattern as test_openapi_discovery.py. Marked
@pytest.mark.network like the rest of the real-world corpus.
"""
import asyncio

import httpx
import pytest

from app.graphql_tools import _render_type_ref, discover_graphql_schema, graphql_tools_schema

pytestmark = pytest.mark.network

COUNTRIES_API = "https://countries.trevorblades.com"


def test_discovers_a_real_graphql_schema():
    async def run():
        async with httpx.AsyncClient(timeout=10.0) as client:
            config = await discover_graphql_schema(client, COUNTRIES_API)
        assert config is not None
        assert config["endpoint"] == f"{COUNTRIES_API}/graphql"
        names = {op["name"] for op in config["operations"]}
        assert "countries" in names
        assert "country" in names
        country_op = next(op for op in config["operations"] if op["name"] == "country")
        assert any(a["name"] == "code" for a in country_op["args"])

    asyncio.run(run())


def test_discovery_returns_none_for_non_graphql_target():
    async def run():
        async with httpx.AsyncClient(timeout=10.0) as client:
            config = await discover_graphql_schema(client, "https://httpbin.org")
        assert config is None

    asyncio.run(run())


def test_render_type_ref_handles_wrapped_types():
    assert _render_type_ref({"kind": "SCALAR", "name": "String"}) == "String"
    assert _render_type_ref({"kind": "NON_NULL", "ofType": {"kind": "SCALAR", "name": "ID"}}) == "ID!"
    assert _render_type_ref({
        "kind": "NON_NULL",
        "ofType": {"kind": "LIST", "ofType": {"kind": "NON_NULL", "ofType": {"kind": "SCALAR", "name": "Country"}}}
    }) == "[Country!]!"
    assert _render_type_ref(None) == "Unknown"


def test_graphql_tools_schema_shape():
    tools = graphql_tools_schema({"endpoint": "https://x.test/graphql", "operations": [{}, {}]})
    names = {t["name"] for t in tools}
    assert names == {"graphql_schema", "graphql_query"}
    query_tool = next(t for t in tools if t["name"] == "graphql_query")
    assert "query" in query_tool["inputSchema"]["required"]
