import os
import time
import uuid
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import httpx

from app.security import normalize_url, is_public_url
from app.openapi_tools import operations_to_mcp_tools, build_request
from app.oauth import fetch_client_credentials_token

logger = logging.getLogger("mcpify")


async def _revalidate_target_safety(target_url: str) -> Optional[str]:
    """
    Returns an error reason if target_url is unsafe to connect to right
    now, or None if it's still safe.

    target_url's safety was previously only checked ONCE, at
    proxy-creation time (main.py's /proxy/create) - but DNS isn't static.
    A malicious target's domain can resolve to a public IP at creation
    time (passing that check and getting a proxy created), then be
    repointed to an internal/cloud-metadata address by the time any
    actual tool call is made - a classic DNS-rebinding SSRF bypass this
    proxy was otherwise completely unprotected against, since
    forward_request/call_api/health_check all connected to target_url
    directly with no re-check at all. Re-validating on every outbound
    call closes that gap, at the cost of one extra DNS lookup per call.
    """
    is_safe, reason = await is_public_url(target_url)
    return None if is_safe else reason


def _parse_mcp_response_body(text: str, content_type: str) -> Optional[Dict[str, Any]]:
    """
    A real streamable-HTTP MCP server can legitimately answer a single
    POST request either as plain JSON or SSE-framed
    ("event: message\\ndata: {...}\\n\\n") - verified live that DeepWiki's
    real MCP server uses SSE framing even for an ordinary tools/call, not
    just a long-lived stream. A plain resp.json() silently fails on the
    SSE-framed form (it isn't valid JSON on its own), which used to make
    every call to such a server look like it failed and fall back to the
    wrong generic REST-wrapper tools - has_mcp was detected correctly,
    but the actual forwarding never worked for exactly the servers this
    was meant to support.
    """
    try:
        return json.loads(text)
    except Exception:
        pass
    if "text/event-stream" in (content_type or "").lower():
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                try:
                    return json.loads(line[len("data:"):].strip())
                except Exception:
                    continue
    return None

REDIS_KEY_PREFIX = "mcpify:proxy:"
REDIS_INDEX_PREFIX = "mcpify:proxy_by_url:"


class ProxyMCPManager:
    """Manages proxy sessions for agents lacking native MCP endpoints.

    Proxy records persist to Redis (Upstash) when UPSTASH_REDIS_URL is set,
    so they survive redeploys/restarts instead of vanishing with every
    process restart. Falls back to an in-memory dict when no Redis URL is
    configured, so local dev and the test suite need no Redis at all.
    SSE sessions (self.sessions) are always in-memory - an asyncio.Queue
    is tied to one open connection in one process and can't be persisted
    meaningfully; a redeploy naturally drops open SSE connections anyway.
    """

    def __init__(self):
        self.redis_url = os.getenv("UPSTASH_REDIS_URL") or os.getenv("REDIS_URL")
        self._redis = None
        self.proxies: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, asyncio.Queue] = {}

    def _get_redis(self):
        if not self.redis_url:
            return None
        if self._redis is None:
            import redis.asyncio as redis_asyncio
            self._redis = redis_asyncio.from_url(self.redis_url, decode_responses=True)
        return self._redis

    def get_base_app_url(self, request: Optional[Any] = None) -> str:
        if request:
            try:
                proto = request.headers.get("x-forwarded-proto") or getattr(request.url, "scheme", "http")
                host = request.headers.get("x-forwarded-host") or request.headers.get("host") or getattr(request.url, "netloc", "")
                if host:
                    return f"{proto}://{host}".rstrip("/")
            except Exception:
                pass
        return os.getenv("APP_URL", "http://127.0.0.1:10000").rstrip("/")

    async def create_proxy(
        self, target_url: str, has_mcp: bool = False, api_key: Optional[str] = None,
        request: Optional[Any] = None, openapi_operations: Optional[List[Dict[str, Any]]] = None,
        oauth_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new proxy session or return existing active proxy."""
        target_url = normalize_url(target_url)
        current_base_url = self.get_base_app_url(request)

        redis = self._get_redis()
        now_str = datetime.now(timezone.utc).isoformat()

        if redis is not None:
            existing_id = await redis.get(f"{REDIS_INDEX_PREFIX}{target_url}")
            if existing_id:
                existing = await self._redis_get_proxy(redis, existing_id)
                if existing:
                    existing["last_used"] = now_str
                    if api_key:
                        existing["api_key"] = api_key
                    if oauth_config:
                        existing["oauth_config"] = oauth_config
                    # Refresh has_mcp too, not just last_used/api_key: the
                    # caller (main.py) always re-runs the real handshake
                    # check fresh before calling create_proxy, but that
                    # freshly-computed value was being silently discarded
                    # here in favor of whatever was cached from whenever
                    # this record was first created - stale in exactly the
                    # same way proxy_url was (verified live: a deepwiki
                    # proxy created before the OAuth/406/streaming
                    # detection fixes landed kept reporting has_mcp=False
                    # forever after, even once the code correctly detects
                    # it as True, because reuse never looked at the fresh
                    # value at all).
                    existing["has_mcp"] = has_mcp
                    existing["openapi_operations"] = openapi_operations
                    # Refresh proxy_url to the CURRENT app base, not
                    # whatever it was when this record was first created -
                    # a record made under an old domain (e.g. this app
                    # redeployed to a new Render URL, or moved to a new
                    # account entirely, while sharing the same Redis) would
                    # otherwise keep handing out a dead URL forever on
                    # every reuse. Verified live: exactly this happened
                    # migrating off a suspended Render service - every
                    # cached proxy kept returning the old, now-dead host.
                    existing["proxy_url"] = f"{current_base_url}/proxy/{existing['proxy_id']}/mcp"
                    await self._redis_save_proxy(redis, existing)
                    return existing

            proxy_id = str(uuid.uuid4())[:8]
            proxy_url = f"{current_base_url}/proxy/{proxy_id}/mcp"
            proxy_data = {
                "proxy_id": proxy_id,
                "proxy_url": proxy_url,
                "target_url": target_url,
                "has_mcp": has_mcp,
                "api_key": api_key,
                "openapi_operations": openapi_operations,
                "oauth_config": oauth_config,
                "created_at": now_str,
                "last_used": now_str,
                "status": "active"
            }

            # Atomic claim (SET ... NX), not a plain set after the GET
            # above: two concurrent create_proxy calls for the same
            # brand-new target_url would otherwise both see no existing
            # index entry, both create their own proxy_id, and both write
            # the index - whichever write lands last silently orphans the
            # other's proxy record (still valid, just unreachable via
            # dedup lookup). NX makes only one of them actually win the
            # index slot; the loser reuses the winner's record instead of
            # leaving its own orphaned.
            claimed = await redis.set(f"{REDIS_INDEX_PREFIX}{target_url}", proxy_id, nx=True)
            if not claimed:
                winner_id = await redis.get(f"{REDIS_INDEX_PREFIX}{target_url}")
                winner = await self._redis_get_proxy(redis, winner_id) if winner_id else None
                if winner:
                    winner["last_used"] = now_str
                    if api_key:
                        winner["api_key"] = api_key
                    if oauth_config:
                        winner["oauth_config"] = oauth_config
                    winner["has_mcp"] = has_mcp
                    winner["openapi_operations"] = openapi_operations
                    winner["proxy_url"] = f"{current_base_url}/proxy/{winner['proxy_id']}/mcp"
                    await self._redis_save_proxy(redis, winner)
                    return winner
                # Winner's own record vanished somehow - fall through and
                # save ours anyway rather than returning nothing usable.

            await self._redis_save_proxy(redis, proxy_data)
            return proxy_data

        # In-memory fallback
        for proxy_id, proxy in self.proxies.items():
            if proxy["target_url"] == target_url:
                proxy["last_used"] = now_str
                if api_key:
                    proxy["api_key"] = api_key
                if oauth_config:
                    proxy["oauth_config"] = oauth_config
                proxy["has_mcp"] = has_mcp
                proxy["openapi_operations"] = openapi_operations
                proxy["proxy_url"] = f"{current_base_url}/proxy/{proxy_id}/mcp"
                return proxy

        proxy_id = str(uuid.uuid4())[:8]
        proxy_url = f"{current_base_url}/proxy/{proxy_id}/mcp"
        proxy_data = {
            "proxy_id": proxy_id,
            "proxy_url": proxy_url,
            "target_url": target_url,
            "has_mcp": has_mcp,
            "api_key": api_key,
            "openapi_operations": openapi_operations,
            "oauth_config": oauth_config,
            "created_at": now_str,
            "last_used": now_str,
            "status": "active"
        }
        self.proxies[proxy_id] = proxy_data
        return proxy_data

    async def _redis_get_proxy(self, redis, proxy_id: str) -> Optional[Dict[str, Any]]:
        raw = await redis.get(f"{REDIS_KEY_PREFIX}{proxy_id}")
        return json.loads(raw) if raw else None

    async def _redis_save_proxy(self, redis, proxy: Dict[str, Any]) -> None:
        await redis.set(f"{REDIS_KEY_PREFIX}{proxy['proxy_id']}", json.dumps(proxy))

    async def get_proxy(self, proxy_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve proxy configuration by proxy ID, or None if it doesn't exist."""
        redis = self._get_redis()
        now_str = datetime.now(timezone.utc).isoformat()

        if redis is not None:
            proxy = await self._redis_get_proxy(redis, proxy_id)
            if proxy:
                proxy["last_used"] = now_str
                await self._redis_save_proxy(redis, proxy)
            return proxy

        proxy = self.proxies.get(proxy_id)
        if proxy:
            proxy["last_used"] = now_str
        return proxy

    async def list_proxies(self) -> List[Dict[str, Any]]:
        """Return list of all active proxies, with credentials masked."""
        redis = self._get_redis()
        proxies: List[Dict[str, Any]] = []

        if redis is not None:
            keys = [k async for k in redis.scan_iter(match=f"{REDIS_KEY_PREFIX}*")]
            if keys:
                raw_values = await redis.mget(keys)
                proxies = [json.loads(v) for v in raw_values if v]
        else:
            proxies = list(self.proxies.values())

        result = []
        for proxy in proxies:
            sanitized = {k: v for k, v in proxy.items() if k not in ("api_key", "oauth_config")}
            sanitized["has_api_key"] = bool(proxy.get("api_key"))
            sanitized["has_oauth"] = bool(proxy.get("oauth_config"))
            result.append(sanitized)
        return result

    async def record_ping_status(self, proxy_id: str, status_code: Optional[int]) -> None:
        """Record the last health-ping status against the live proxy record.

        list_proxies() returns sanitized copies (to keep api_key out of
        responses), so callers must write ping results back through this
        method rather than mutating list_proxies()'s output directly.
        """
        redis = self._get_redis()
        if redis is not None:
            proxy = await self._redis_get_proxy(redis, proxy_id)
            if proxy:
                proxy["last_ping_status"] = status_code
                await self._redis_save_proxy(redis, proxy)
            return

        proxy = self.proxies.get(proxy_id)
        if proxy:
            proxy["last_ping_status"] = status_code

    async def get_valid_oauth_token(self, proxy: Dict[str, Any]) -> tuple:
        """
        Returns (access_token, error) for a proxy configured with OAuth2
        client_credentials. Reuses the cached token while it's still
        valid; transparently fetches and persists a fresh one otherwise,
        so a tool call never has to know whether this is the first call
        or the hundredth. (None, None) means no OAuth is configured for
        this proxy at all - callers fall back to plain api_key/Bearer.
        """
        oauth_config = proxy.get("oauth_config")
        if not oauth_config:
            return None, None

        if oauth_config.get("cached_token") and oauth_config.get("cached_token_expires_at", 0) > time.time():
            return oauth_config["cached_token"], None

        # token_url's safety was only checked ONCE, at proxy-creation time
        # - same DNS-rebinding gap as target_url itself (see
        # _revalidate_target_safety's docstring), closed the same way:
        # re-check right before every actual outbound call, not just the
        # first one, since a refresh can happen much later than creation.
        is_safe, reason = await is_public_url(oauth_config["token_url"])
        if not is_safe:
            return None, f"Refusing to call OAuth token endpoint: {reason}"

        async with httpx.AsyncClient() as client:
            result = await fetch_client_credentials_token(
                client, oauth_config["token_url"], oauth_config["client_id"],
                oauth_config["client_secret"], oauth_config.get("scope")
            )
        if result["error"]:
            return None, result["error"]

        oauth_config["cached_token"] = result["access_token"]
        oauth_config["cached_token_expires_at"] = result["expires_at"]
        proxy["oauth_config"] = oauth_config

        redis = self._get_redis()
        if redis is not None:
            await self._redis_save_proxy(redis, proxy)
        # In-memory proxies are mutated in place above (proxy is the same
        # dict object stored in self.proxies) - nothing further to persist.

        return result["access_token"], None

    async def forward_request(self, proxy_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Forward request to target or process MCP wrapper tool calls."""
        proxy = await self.get_proxy(proxy_id)
        if not proxy:
            return {
                "jsonrpc": "2.0",
                "id": request_data.get("id"),
                "error": {
                    "code": -32600,
                    "message": f"Proxy ID '{proxy_id}' not found or inactive."
                }
            }

        target_url = proxy["target_url"]
        has_mcp = proxy.get("has_mcp", False)

        # If target has native /mcp, attempt forwarding direct MCP JSON-RPC call
        if has_mcp and await _revalidate_target_safety(target_url) is None:
            target_mcp_url = f"{target_url}/mcp"
            # Accept must include text/event-stream, not just Content-Type:
            # a spec-compliant streamable-HTTP server can refuse a request
            # missing it (verified live: DeepWiki 406s a POST without this
            # exact Accept header, same as its GET behavior).
            headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
            oauth_token, oauth_error = await self.get_valid_oauth_token(proxy)
            if oauth_token:
                headers["Authorization"] = f"Bearer {oauth_token}"
            elif proxy.get("api_key"):
                headers["Authorization"] = f"Bearer {proxy['api_key']}"
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.post(
                        target_mcp_url,
                        json=request_data,
                        headers=headers
                    )
                    content_type = resp.headers.get("content-type", "")
                    if resp.status_code < 400 and "text/html" not in content_type:
                        parsed = _parse_mcp_response_body(resp.text, content_type)
                        if parsed is not None:
                            return parsed
            except Exception as e:
                logger.warning("[ProxyMCP] Forwarding to native target /mcp failed: %s", e)

        # Otherwise, serve as MCPify REST wrapper server
        return await self.handle_jsonrpc_request(proxy, request_data)

    async def handle_jsonrpc_request(self, proxy: Dict[str, Any], req: Dict[str, Any]) -> Dict[str, Any]:
        """Process JSON-RPC 2.0 requests for REST API wrapper tools."""
        method = req.get("method")
        req_id = req.get("id")
        params = req.get("params", {})
        if not isinstance(params, dict):
            # A client sending "params" as something other than an object
            # (a string, a list, null) used to crash with an unhandled
            # AttributeError on the params.get(...) calls below, returning
            # a raw HTTP 500 instead of a proper JSON-RPC error - callers
            # expect JSON-RPC-shaped errors, not an opaque server crash.
            params = {}
        target_url = proxy["target_url"]

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "MCPify-Proxy",
                        "version": "1.0.0"
                    }
                }
            }

        elif method == "notifications/initialized":
            return {}

        elif method == "ping":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {}
            }

        elif method == "tools/list":
            discovered_tools = operations_to_mcp_tools(proxy["openapi_operations"]) if proxy.get("openapi_operations") else []
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": discovered_tools + [
                        {
                            "name": "call_api",
                            "description": "Forward HTTP requests to the target agent API endpoints.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "endpoint": {
                                        "type": "string",
                                        "description": "Endpoint path (e.g. /health, /api/v1/query)"
                                    },
                                    "method": {
                                        "type": "string",
                                        "enum": ["GET", "POST", "PUT", "DELETE"],
                                        "default": "GET",
                                        "description": "HTTP method to execute"
                                    },
                                    "params": {
                                        "type": "object",
                                        "description": "Query parameters dictionary"
                                    },
                                    "json_data": {
                                        "type": "object",
                                        "description": "JSON request body dictionary"
                                    }
                                },
                                "required": ["endpoint"]
                            }
                        },
                        {
                            "name": "get_info",
                            "description": "Get basic proxy configuration and target agent metadata.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {}
                            }
                        },
                        {
                            "name": "health_check",
                            "description": "Ping the target agent health endpoint to check availability.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {}
                            }
                        }
                    ]
                }
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments", {})
            if not isinstance(args, dict):
                # Same reasoning as the params guard above: "arguments" as
                # a string/list/null used to crash on args.get(...) below.
                args = {}

            discovered_ops = {op["tool_name"]: op for op in (proxy.get("openapi_operations") or [])}

            if tool_name == "call_api" or tool_name in discovered_ops:
                effective_base = target_url
                auth_header_name = None
                if tool_name == "call_api":
                    endpoint = args.get("endpoint", "")
                    if not endpoint.startswith("/"):
                        endpoint = f"/{endpoint}"
                    http_method = args.get("method", "GET").upper()
                    query_params = args.get("params")
                    json_payload = args.get("json_data")
                else:
                    # A specific tool discovered from the target's own
                    # OpenAPI spec (e.g. "get_pet_by_id") - same execution
                    # path as call_api from here on, just with the request
                    # built from the operation's template instead of raw
                    # user-supplied endpoint/method/params.
                    op = discovered_ops[tool_name]
                    built = build_request(op, args)
                    if built["error"]:
                        return {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {
                                "content": [{"type": "text", "text": built["error"]}],
                                "isError": True
                            }
                        }
                    endpoint = built["path"]
                    http_method = op["method"]
                    query_params = built["params"] or None
                    json_payload = built["json"]
                    # The spec's own "servers" entry can declare a base
                    # path (or even a different host) that its paths are
                    # actually rooted at, distinct from the proxy's stored
                    # target_url (see resolve_api_base's docstring - real
                    # example: n8n's spec says "/api/v1", and every
                    # discovered path is relative to THAT, not the bare
                    # domain root). Use it as the real destination when
                    # present.
                    effective_base = op.get("base_url") or target_url
                    auth_header_name = op.get("auth_header_name")

                full_target_url = f"{effective_base}{endpoint}"

                unsafe_reason = await _revalidate_target_safety(effective_base)
                if unsafe_reason:
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [
                                {"type": "text", "text": f"Refusing to call target: {unsafe_reason}"}
                            ],
                            "isError": True
                        }
                    }

                headers = {}
                oauth_token, oauth_error = await self.get_valid_oauth_token(proxy)
                if oauth_error:
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": f"OAuth2 token refresh failed: {oauth_error}"}],
                            "isError": True
                        }
                    }
                if oauth_token:
                    # OAuth2 client_credentials takes priority over a
                    # plain api_key when both are somehow set - it's the
                    # more specific, more recently-issued credential.
                    headers["Authorization"] = f"Bearer {oauth_token}"
                elif proxy.get("api_key"):
                    # Not every real API wants "Authorization: Bearer" -
                    # verified live: n8n's spec declares its key goes in
                    # an "X-N8N-API-KEY" header instead. Use whatever the
                    # discovered operation's spec says when it says
                    # anything; Bearer stays the default for call_api and
                    # for specs with no declared auth scheme.
                    if auth_header_name:
                        headers[auth_header_name] = proxy["api_key"]
                    else:
                        headers["Authorization"] = f"Bearer {proxy['api_key']}"

                try:
                    async with httpx.AsyncClient(timeout=12.0) as client:
                        resp = await client.request(
                            method=http_method,
                            url=full_target_url,
                            params=query_params,
                            json=json_payload,
                            headers=headers
                        )
                        content_type = resp.headers.get("content-type", "")
                        if "json" in content_type:
                            try:
                                body_res = resp.json()
                            except Exception:
                                body_res = resp.text
                        else:
                            body_res = resp.text

                        output = {
                            "status_code": resp.status_code,
                            "url": str(resp.url),
                            "response": body_res
                        }
                        return {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": json.dumps(output, indent=2)
                                    }
                                ]
                            }
                        }
                except Exception as e:
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Error connecting to target endpoint {full_target_url}: {str(e)}"
                                }
                            ],
                            "isError": True
                        }
                    }

            elif tool_name == "get_info":
                info = {
                    "proxy_id": proxy["proxy_id"],
                    "target_url": target_url,
                    "proxy_url": proxy["proxy_url"],
                    "has_mcp": proxy.get("has_mcp", False),
                    "has_api_key": bool(proxy.get("api_key")),
                    "created_at": proxy.get("created_at"),
                    "last_used": proxy.get("last_used"),
                    "status": proxy.get("status", "active")
                }
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(info, indent=2)
                            }
                        ]
                    }
                }

            elif tool_name == "health_check":
                unsafe_reason = await _revalidate_target_safety(target_url)
                if unsafe_reason:
                    res = {
                        "target_url": target_url,
                        "reachable": False,
                        "error": f"Refusing to call target: {unsafe_reason}"
                    }
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": json.dumps(res, indent=2)}],
                            "isError": True
                        }
                    }
                try:
                    health_url = f"{target_url}/health"
                    async with httpx.AsyncClient(timeout=8.0) as client:
                        # No follow_redirects: target_url was only SSRF-checked
                        # once, at proxy-creation time - a redirect here could
                        # otherwise point anywhere without being re-validated.
                        resp = await client.get(health_url)
                        res = {
                            "target_url": target_url,
                            "health_url": health_url,
                            "status_code": resp.status_code,
                            "reachable": resp.status_code < 400
                        }
                except Exception as e:
                    res = {
                        "target_url": target_url,
                        "reachable": False,
                        "error": str(e)
                    }

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(res, indent=2)
                            }
                        ]
                    }
                }

            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: '{tool_name}'"
                    }
                }

        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method '{method}' not supported by MCPify proxy."
                }
            }


proxy_manager = ProxyMCPManager()
