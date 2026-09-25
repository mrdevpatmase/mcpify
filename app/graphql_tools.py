from typing import Any, Dict, List, Optional

import httpx

GRAPHQL_PATHS = ["/graphql", "/api/graphql", "/v1/graphql", "/query"]

MAX_OPERATIONS = 50

_INTROSPECTION_QUERY = """
query IntrospectionQuery {
  __schema {
    queryType { name }
    mutationType { name }
    types {
      name
      kind
      fields {
        name
        description
        args {
          name
          type { kind name ofType { kind name ofType { kind name ofType { kind name } } } }
        }
        type { kind name ofType { kind name ofType { kind name ofType { kind name } } } }
      }
    }
  }
}
"""


def _render_type_ref(type_ref: Optional[Dict[str, Any]]) -> str:
    """
    Renders a GraphQL introspection type reference into a compact,
    human/LLM-readable string like "[String!]!" - introspection wraps
    NON_NULL/LIST modifiers as nested "ofType" objects rather than a flat
    string, so this has to walk that nesting itself.
    """
    if not type_ref:
        return "Unknown"
    kind = type_ref.get("kind")
    if kind == "NON_NULL":
        return f"{_render_type_ref(type_ref.get('ofType'))}!"
    if kind == "LIST":
        return f"[{_render_type_ref(type_ref.get('ofType'))}]"
    return type_ref.get("name") or "Unknown"


async def discover_graphql_schema(client: httpx.AsyncClient, base_url: str) -> Optional[Dict[str, Any]]:
    """
    Probes common GraphQL endpoint paths with a real introspection query
    and, if one answers, returns a compact summary of the root
    Query/Mutation operations (name, args, return type) - enough for an
    LLM to compose valid queries against without walking the entire type
    system itself (real schemas - GitHub's, Shopify's - can have
    thousands of types; only the root-level entry points are surfaced,
    capped at MAX_OPERATIONS same as openapi_tools.py's operation cap).

    Returns None if no path answers a real introspection query. That
    includes servers that deliberately disable introspection in
    production (a common security practice) - not a bug to route around,
    just a target this feature can't help with; callers fall back to the
    generic call_api tool exactly as they already do when OpenAPI
    discovery fails.
    """
    for path in GRAPHQL_PATHS:
        url = f"{base_url}{path}"
        try:
            resp = await client.post(
                url,
                json={"query": _INTROSPECTION_QUERY},
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                timeout=8.0,
            )
        except Exception:
            continue

        if resp.status_code >= 400:
            continue
        if "json" not in (resp.headers.get("content-type") or "").lower():
            continue
        try:
            body = resp.json()
        except Exception:
            continue

        schema = (body or {}).get("data", {}).get("__schema") if isinstance(body, dict) else None
        if not schema:
            continue

        types_by_name = {t["name"]: t for t in (schema.get("types") or []) if t.get("name")}
        operations: List[Dict[str, Any]] = []
        for op_type, root_name in (
            ("query", (schema.get("queryType") or {}).get("name")),
            ("mutation", (schema.get("mutationType") or {}).get("name")),
        ):
            root_type = types_by_name.get(root_name) if root_name else None
            if not root_type:
                continue
            for field in (root_type.get("fields") or []):
                operations.append({
                    "operation_type": op_type,
                    "name": field["name"],
                    "description": field.get("description"),
                    "args": [
                        {"name": a["name"], "type": _render_type_ref(a.get("type"))}
                        for a in (field.get("args") or [])
                    ],
                    "returns": _render_type_ref(field.get("type")),
                })
                if len(operations) >= MAX_OPERATIONS:
                    break
            if len(operations) >= MAX_OPERATIONS:
                break

        if not operations:
            # A real GraphQL server that happens to expose no root fields
            # at all isn't a realistic case - more likely this "endpoint"
            # answered with a schema-shaped JSON coincidentally. Treat as
            # not found rather than handing back an empty, useless tool.
            continue

        return {"endpoint": url, "operations": operations}

    return None


def graphql_tools_schema(graphql_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """MCP tool defs exposed when GraphQL discovery succeeds: a schema
    listing tool plus a generic query/mutation executor. One generic
    executor (not one tool per operation, unlike OpenAPI's named tools)
    because GraphQL queries compose fields freely - an LLM that can see
    the schema summary can write a correct query string directly, and
    forcing every field combination into separate fixed tools would
    either explode in count or lose that composability."""
    op_count = len(graphql_config.get("operations") or [])
    return [
        {
            "name": "graphql_schema",
            "description": f"List the {op_count} discovered GraphQL queries/mutations (name, args, return type) available on this API. Call this first.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "graphql_query",
            "description": "Execute a raw GraphQL query or mutation string against this API, with optional variables.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The GraphQL query/mutation document."},
                    "variables": {"type": "object", "description": "Variables referenced by the query, if any."},
                },
                "required": ["query"],
            },
        },
    ]
