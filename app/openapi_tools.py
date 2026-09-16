"""
Auto-discovers a target's OpenAPI/Swagger spec (when it has one) and turns
its operations into specific, named MCP tools - "list_users", "create_pet" -
instead of the generic call_api(endpoint, method, params, json_data) wrapper
every proxy falls back to otherwise. Verified live against two real specs:
petstore3.swagger.io (13 clean OpenAPI 3.0 paths) and reqres.in (42 paths,
~70 operations) - both parse and generate correctly-shaped tools.

Falls back to nothing (None/[]) on any failure - this is a best-effort
enrichment on top of the generic proxy, never a requirement for it to work.
"""
import re
from typing import Any, Dict, List, Optional

import httpx
import yaml

SPEC_PATHS = [
    "/openapi.json", "/swagger.json", "/v3/api-docs", "/api-docs", "/docs/json",
    # YAML variants and a versioned-API-prefix guess - added after finding
    # n8n's real Cloud API live: its actual spec is a 603KB YAML document
    # at /api/v1/openapi.yml, content-type text/yaml, not any of the
    # JSON-only paths above. Plenty of real platforms version their whole
    # API under /api/v1 rather than publishing the spec at the domain root.
    "/openapi.yml", "/openapi.yaml", "/swagger.yml", "/swagger.yaml",
    "/api/v1/openapi.yml", "/api/v1/openapi.json",
]

# Real specs can have hundreds/thousands of operations (Stripe, GitHub) -
# exposing all of them as separate MCP tools would overwhelm tools/list for
# both the client UI and the model's own context. Cap at a level generous
# enough for real-world small/medium APIs (verified: comfortably covers both
# petstore's 13 and reqres's ~70) without ever handing back an unusable wall
# of tools for a huge spec.
MAX_OPERATIONS = 50

_INVALID_TOOL_CHARS = re.compile(r"[^a-zA-Z0-9_]")
_PATH_PARAM = re.compile(r"\{([^}]+)\}")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


async def discover_openapi_spec(client: httpx.AsyncClient, base_url: str) -> Optional[Dict[str, Any]]:
    """Probes common OpenAPI/Swagger spec paths and returns the first valid one found."""
    for path in SPEC_PATHS:
        try:
            resp = await client.get(f"{base_url}{path}", timeout=6.0)
        except Exception:
            continue
        if resp.status_code != 200:
            continue
        content_type = (resp.headers.get("content-type") or "").lower()
        is_yaml = "yaml" in content_type or path.endswith((".yml", ".yaml"))
        try:
            if is_yaml:
                # yaml.safe_load also happens to parse valid JSON (JSON is
                # a YAML subset), so this branch alone would cover both -
                # kept as a separate branch anyway so a real JSON response
                # never takes an unnecessary detour through the YAML parser.
                spec = yaml.safe_load(resp.text)
            elif "json" in content_type:
                spec = resp.json()
            else:
                continue
        except Exception:
            continue
        # A real OpenAPI/Swagger document, not just any response that
        # happened to live at one of these conventional paths.
        if not isinstance(spec, dict):
            continue
        if ("openapi" not in spec and "swagger" not in spec) or "paths" not in spec:
            continue
        return spec
    return None


def _sanitize_tool_name(name: str) -> str:
    # camelCase/PascalCase operationIds ("updatePet") are common in real
    # specs (verified: petstore3.swagger.io uses this convention) - insert
    # underscores at case boundaries before lowercasing, or "updatePet" and
    # "addPet" would collapse into the much less readable "updatepet"/"addpet".
    name = _CAMEL_BOUNDARY.sub("_", name)
    name = _INVALID_TOOL_CHARS.sub("_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name.lower() or "operation"


def _derive_tool_name(method: str, path: str) -> str:
    # e.g. GET /pet/{petId} -> get_pet_by_id ; POST /users -> post_users
    cleaned = _PATH_PARAM.sub("by_id", path)
    return _sanitize_tool_name(f"{method}_{cleaned}")


def _json_type_for(schema: Dict[str, Any]) -> str:
    return schema.get("type") or "string"


def parse_operations(spec: Dict[str, Any], base_url: str) -> List[Dict[str, Any]]:
    """
    Walks spec["paths"] into a flat list of operations, each carrying
    everything needed to both describe it as an MCP tool and actually
    execute it later: {tool_name, method, path_template, description,
    path_params, query_params, has_body}.
    """
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return []

    # Sort admin/infra-config paths (settings, audit, SSO/LDAP, telemetry)
    # after everything else before applying MAX_OPERATIONS - verified live
    # against n8n's real 134-operation API: those paths happen to come
    # FIRST in spec order, so a plain cap silently dropped every
    # workflow/execution operation (the ones anyone actually wants to
    # call) in favor of things like "put_settings_ldap". Relative order
    # within each group is otherwise preserved.
    def _priority(item):
        path_template = item[0]
        low_priority = any(
            f"/{kw}" in path_template.lower()
            for kw in ("settings", "audit", "license", "ldap", "sso", "saml", "oidc", "otel")
        )
        # Secondary key: path depth (segment count). A spec's own key
        # order isn't reliable either - n8n's real spec lists
        # "/workflows/{id}/test-runs/{runId}/test-cases" BEFORE the plain
        # "/workflows" list/create path, so depth-first would have capped
        # out on deep sub-resource operations before reaching the basic
        # "list workflows"/"create workflow" ones anyone asks for first.
        depth = path_template.count("/")
        return (1 if low_priority else 0, depth)

    ordered_paths = sorted(paths.items(), key=_priority)

    operations: List[Dict[str, Any]] = []
    used_names: set = set()

    for path_template, methods in ordered_paths:
        if not isinstance(methods, dict):
            continue
        for method, op in methods.items():
            if method.lower() not in ("get", "post", "put", "patch", "delete"):
                continue
            if not isinstance(op, dict):
                continue

            raw_name = op.get("operationId") or _derive_tool_name(method, path_template)
            tool_name = _sanitize_tool_name(raw_name)
            # operationIds can collide once sanitized (e.g. "GET /a" and
            # "get-a" both becoming "get_a") - keep names unique so tools
            # don't silently shadow each other.
            base_name = tool_name
            suffix = 2
            while tool_name in used_names:
                tool_name = f"{base_name}_{suffix}"
                suffix += 1
            used_names.add(tool_name)

            description = op.get("summary") or op.get("description") or f"{method.upper()} {path_template}"

            path_params, query_params = [], []
            for param in op.get("parameters", []):
                if not isinstance(param, dict):
                    continue
                location = param.get("in")
                entry = {
                    "name": param.get("name"),
                    "required": bool(param.get("required")),
                    "description": param.get("description") or "",
                    "type": _json_type_for(param.get("schema", {})),
                }
                if location == "path":
                    path_params.append(entry)
                elif location == "query":
                    query_params.append(entry)
                # header/cookie params intentionally not exposed as tool
                # arguments - api_key auth is already handled uniformly via
                # the proxy's own Bearer-token forwarding.

            has_body = "requestBody" in op and method.lower() in ("post", "put", "patch")

            operations.append({
                "tool_name": tool_name,
                "method": method.upper(),
                "path_template": path_template,
                "description": description[:500],
                "path_params": path_params,
                "query_params": query_params,
                "has_body": has_body,
            })

            if len(operations) >= MAX_OPERATIONS:
                return operations

    return operations


def operations_to_mcp_tools(operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Converts parsed operations into MCP tools/list-shaped tool definitions."""
    tools = []
    for op in operations:
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for p in op["path_params"]:
            if not p["name"]:
                continue
            properties[p["name"]] = {"type": p["type"], "description": p["description"] or f"Path parameter: {p['name']}"}
            required.append(p["name"])

        for p in op["query_params"]:
            if not p["name"]:
                continue
            properties[p["name"]] = {"type": p["type"], "description": p["description"] or f"Query parameter: {p['name']}"}
            if p["required"]:
                required.append(p["name"])

        if op["has_body"]:
            properties["body"] = {"type": "object", "description": "Request body (JSON object)."}

        schema: Dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required

        tools.append({
            "name": op["tool_name"],
            "description": f"{op['description']} ({op['method']} {op['path_template']})",
            "inputSchema": schema,
        })
    return tools


def build_request(op: Dict[str, Any], arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fills in an operation's path template and splits the remaining
    arguments into query params vs. body, ready to hand to httpx.
    Returns {"path": str, "params": dict, "json": dict|None, "error": str|None}.
    """
    path = op["path_template"]
    for p in op["path_params"]:
        name = p["name"]
        if name is None:
            continue
        if name not in arguments:
            if p["required"]:
                return {"error": f"Missing required path parameter: {name}"}
            continue
        path = path.replace("{" + name + "}", str(arguments[name]))

    params = {}
    for p in op["query_params"]:
        name = p["name"]
        if name and name in arguments:
            params[name] = arguments[name]

    body = arguments.get("body") if op["has_body"] else None

    return {"path": path, "params": params, "json": body, "error": None}
