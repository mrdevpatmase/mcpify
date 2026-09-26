import os
import uuid
import json
import secrets
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from urllib.parse import urlencode

from fastapi import FastAPI, Request, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
import httpx
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi_mcp import FastApiMCP
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded


from app.mcp_handler import router as mcp_router, mcp
from app.proxy import proxy_manager
from app.generator import generate_proxy_config
from app.security import is_public_url, resolve_canonical_base, normalize_url
from app.analyzer import verify_mcp_handshake
from app.openapi_tools import discover_openapi_spec, parse_operations
from app.graphql_tools import discover_graphql_schema
from app.oauth import fetch_client_credentials_token, exchange_authorization_code
from app.rate_limit import limiter

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mcpify")

# Pre-initialize FastMCP ASGI apps (SSE transport & Streamable HTTP transport)
sse_app = mcp.sse_app()
http_app = mcp.streamable_http_app()

scheduler = AsyncIOScheduler()


async def ping_proxy_targets_job():
    """APScheduler job: Ping proxy targets every 10 minutes."""
    proxies = await proxy_manager.list_proxies()
    if not proxies:
        return

    async with httpx.AsyncClient(timeout=10.0) as client:
        for p in proxies:
            proxy_id = p["proxy_id"]
            target_url = p["target_url"]

            # Same DNS-rebinding concern as every other outbound call to
            # target_url (see _revalidate_target_safety in app/proxy.py)
            # but worse here: this runs unattended on a timer, forever,
            # with no user request or attacker action needed to trigger
            # it once a domain's DNS has been repointed since creation.
            is_safe, reason = await is_public_url(target_url)
            if not is_safe:
                logger.warning("[APScheduler] Skipping ping for proxy %s - target no longer safe: %s", proxy_id, reason)
                continue

            health_target = f"{target_url}/health"
            try:
                resp = await client.get(health_target)
                await proxy_manager.record_ping_status(proxy_id, resp.status_code)
                logger.info("[APScheduler] Pinged proxy %s target (%s) - Status: %s", proxy_id, health_target, resp.status_code)
            except Exception as e:
                logger.warning("[APScheduler] Error pinging proxy %s (%s): %s", proxy_id, health_target, e)


async def ping_keep_alive():
    """Ping /health every 10 minutes to prevent Render free-tier idling."""
    app_url = os.getenv("APP_URL", "http://127.0.0.1:10000").rstrip("/")
    health_url = f"{app_url}/health"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(health_url)
            logger.info("[Keep-Alive] Pinged %s - Status: %s", health_url, response.status_code)
    except Exception as e:
        logger.warning("[Keep-Alive] Keep-alive ping error: %s", e)


async def keep_alive_worker():
    """Background loop that periodically triggers the keep-alive ping."""
    while True:
        try:
            await asyncio.sleep(600)  # Wait 10 minutes
            await ping_keep_alive()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning("[Keep-Alive] Background loop exception: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: launch keep-alive background worker
    worker_task = asyncio.create_task(keep_alive_worker())
    logger.info("[MCPify] Keep-alive background worker initialized (interval: 10 mins).")

    # Startup: launch APScheduler for proxy targets
    try:
        scheduler.add_job(
            ping_proxy_targets_job,
            'interval',
            minutes=10,
            id='ping_proxy_targets',
            replace_existing=True
        )
        scheduler.start()
        logger.info("[MCPify] APScheduler initialized (proxy target ping interval: 10 mins).")
    except Exception as e:
        logger.warning("[MCPify] APScheduler initialization notice: %s", e)

    # Startup: enter FastMCP streamable HTTP session manager
    async with mcp.session_manager.run():
        logger.info("[MCPify] FastMCP Streamable HTTP session manager active.")
        yield

    # Shutdown: stop scheduler & worker gracefully
    try:
        scheduler.shutdown()
    except Exception:
        pass
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    logger.info("[MCPify] Keep-alive worker, scheduler, and MCP session manager stopped.")


# Initialize FastAPI application
app = FastAPI(
    title="MCPify Agent & Bridge",
    description=(
        "Analyze any AI agent URL, detect framework signatures, "
        "and generate ready-to-use Model Context Protocol (MCP) configs. "
        "Exposes dual-mode MCP endpoints and dynamic MCP proxy middleware."
    ),
    version="1.1.0",
    lifespan=lifespan
)

# Rate limiting for endpoints that trigger outbound probes / create state.
# Note: SlowAPIMiddleware is deliberately NOT used here - it inspects every
# route's handler for __name__/__module__, which crashes on the raw ASGI
# apps mounted below (app.mount("/mcp", ...)). The @limiter.limit(...)
# decorators on individual routes work standalone without the middleware.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS enabled for all origins and headers - this API is stateless
# (no cookies/browser sessions anywhere in this app; the MCP SSE
# "session_id" is just a query-param connection handle, unrelated), so
# allow_credentials stays False. Wildcard origins + allow_credentials=True
# is a well-known CORS misconfiguration: browsers only permit it by
# reflecting the caller's exact Origin back (Starlette does this
# automatically) rather than honoring "*" literally, which would let any
# website make credentialed requests on a visiting user's behalf the
# moment this app ever adds cookie-based auth - not needed today, but not
# worth leaving armed for that day either.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request schemas
class CreateProxyRequest(BaseModel):
    url: str = Field(..., description="Target AI agent URL to proxy")
    api_key: Optional[str] = Field(
        None,
        description="Bearer token to forward to the target agent's protected endpoints, if it requires auth."
    )
    oauth_token_url: Optional[str] = Field(
        None,
        description="OAuth2 token endpoint. Required for either OAuth grant below. Takes priority over api_key."
    )
    oauth_client_id: Optional[str] = Field(None, description="OAuth2 client ID.")
    oauth_client_secret: Optional[str] = Field(None, description="OAuth2 client secret.")
    oauth_scope: Optional[str] = Field(None, description="Optional OAuth2 scope to request.")
    oauth_authorization_url: Optional[str] = Field(
        None,
        description="OAuth2 authorization endpoint (the target's login/consent page URL). Setting this switches "
                    "to the Authorization Code grant (interactive browser login) instead of client_credentials - "
                    "the response will include an authorization_url to visit once to complete it."
    )


# 1. Health check & Web UI / Root endpoints
@app.get("/health", summary="Health Check")
async def health_check():
    """Health check endpoint returning service status."""
    return {"status": "ok"}


def render_web_page() -> HTMLResponse:
    """Renders the web frontend, dynamically including the navbar component if present."""
    web_dir = os.path.join(os.path.dirname(__file__), "app", "web")
    html_path = os.path.join(web_dir, "index.html")
    if not os.path.exists(html_path):
        return HTMLResponse("<h1>MCPify API Running</h1>")

    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    navbar_path = os.path.join(web_dir, "navbar.html")
    if os.path.exists(navbar_path) and "<!-- NAVBAR_COMPONENT -->" in content:
        with open(navbar_path, "r", encoding="utf-8") as nf:
            navbar_content = nf.read()
        content = content.replace("<!-- NAVBAR_COMPONENT -->", navbar_content)

    return HTMLResponse(content)


@app.get("/", summary="MCPify Web Interface", response_class=HTMLResponse)
async def root():
    """Serves the MCPify Web UI frontend."""
    return render_web_page()


@app.get("/api", summary="Root Discovery Index")
async def api_info():
    """Landing endpoint with API discovery details."""
    return {
        "service": "MCPify",
        "description": "AI Agent URL Analyzer & Proxy MCP Server",
        "version": "1.1.0",
        "mcp_endpoints": {
            "streamable_http": {
                "method": "POST",
                "path": "/mcp",
                "transport": "HTTP Streamable (MCP 2024-11 standard)"
            },
            "sse_transport": {
                "method": "GET",
                "path": "/mcp/sse",
                "messages_path": "/mcp/messages",
                "transport": "Server-Sent Events (SSE)"
            }
        },
        "proxy_endpoints": {
            "create_proxy": "POST /proxy/create",
            "proxy_mcp": "GET /proxy/{proxy_id}/mcp",
            "proxy_health": "GET /proxy/{proxy_id}/health",
            "list_proxies": "GET /proxy/list"
        },
        "rest_endpoints": {
            "health": "/health",
            "analyze_agent": "/analyze",
            "generate_config": "/generate",
            "integration_guide": "/guide",
            "openapi_docs": "/docs"
        }
    }


# 2. PROXY ENDPOINTS

@app.post("/proxy/create", summary="Create Proxy MCP Endpoint")
@limiter.limit("10/minute")
async def create_proxy_endpoint(request: Request, payload: CreateProxyRequest):
    """Creates a proxy MCP endpoint for a target URL."""
    target_url = payload.url.strip()
    if not target_url:
        raise HTTPException(status_code=400, detail="Missing required 'url' parameter.")

    normalized_url = normalize_url(target_url)

    is_safe, reason = await is_public_url(normalized_url)
    if not is_safe:
        raise HTTPException(status_code=400, detail=f"Refusing to proxy target: {reason}")

    # Resolve apex->www / http->https style domain-level redirects once,
    # so a site that blanket-redirects every path doesn't look dead.
    normalized_url = await resolve_canonical_base(normalized_url)

    # Confirm via a real MCP JSON-RPC handshake rather than trusting a GET
    # probe's status code alone - a route coincidentally living at /mcp for
    # unrelated reasons (seen in the wild) can otherwise look like a hit.
    # Passes the user's api_key through: several real MCP servers are
    # OAuth-protected (Sentry, Supermetrics, Explorium's Vibe Prospecting
    # all verified live) and only answer once authenticated.
    async with httpx.AsyncClient(timeout=5.0) as client:
        has_mcp = await verify_mcp_handshake(client, normalized_url, api_key=payload.api_key)

    # No native MCP - see if the target publishes an OpenAPI/Swagger spec
    # so the proxy can expose specific, named tools (e.g. "get_pet_by_id")
    # instead of only the generic call_api(endpoint, method, ...) wrapper.
    # Best-effort: falls back to None on any failure, same as has_mcp above
    # falling back to the generic proxy.
    openapi_operations = None
    if not has_mcp:
        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                spec = await discover_openapi_spec(client, normalized_url)
                if spec:
                    openapi_operations = parse_operations(spec, normalized_url) or None
        except Exception:
            openapi_operations = None

    # If it's not a REST API with an OpenAPI spec either, see if it's a
    # GraphQL API - same best-effort, falls-back-to-generic-call_api
    # pattern as the OpenAPI check above. Only probed when OpenAPI
    # discovery came up empty since a target is realistically one or the
    # other, not both.
    graphql_config = None
    if not has_mcp and not openapi_operations:
        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                graphql_config = await discover_graphql_schema(client, normalized_url)
        except Exception:
            graphql_config = None

    # OAuth2 client_credentials (machine-to-machine, no browser login) -
    # an alternative to api_key for targets that need it. Validated and
    # test-fetched once here, at creation time, the same way api_key
    # itself isn't verified until first use but has_mcp/openapi discovery
    # already do their own checks eagerly - failing fast with a clear
    # error beats a proxy that looks created but can never authenticate.
    oauth_config = None
    authorization_url_for_user = None
    if payload.oauth_token_url:
        if not payload.oauth_client_id or not payload.oauth_client_secret:
            raise HTTPException(status_code=400, detail="oauth_client_id and oauth_client_secret are required when oauth_token_url is set.")
        token_url_safe, token_url_reason = await is_public_url(payload.oauth_token_url)
        if not token_url_safe:
            raise HTTPException(status_code=400, detail=f"Refusing oauth_token_url: {token_url_reason}")

        if payload.oauth_authorization_url:
            # Authorization Code grant - can't fetch a token synchronously
            # here the way client_credentials does below: a human has to
            # log into the TARGET's own site first. Build a pending
            # oauth_config and hand back a URL for the user to visit once;
            # /oauth/callback completes the exchange when they get
            # redirected back.
            auth_url_safe, auth_url_reason = await is_public_url(payload.oauth_authorization_url)
            if not auth_url_safe:
                raise HTTPException(status_code=400, detail=f"Refusing oauth_authorization_url: {auth_url_reason}")

            # Prefer the deployment's own configured APP_URL over the
            # incoming request's Host header for this one - unlike
            # proxy_url (a convenience link handed back to the same
            # caller who already knows what they sent), redirect_uri is
            # embedded into a request sent to a THIRD PARTY (the OAuth
            # provider), which will send the user's authorization code
            # back to it. A spoofed Host header on this request must not
            # be able to redirect that code somewhere else.
            app_base_url = os.getenv("APP_URL", "").rstrip("/") or proxy_manager.get_base_app_url(request)
            redirect_uri = f"{app_base_url}/oauth/callback"
            state = secrets.token_urlsafe(32)
            query = {
                "response_type": "code",
                "client_id": payload.oauth_client_id,
                "redirect_uri": redirect_uri,
                "state": state,
            }
            if payload.oauth_scope:
                query["scope"] = payload.oauth_scope
            authorization_url_for_user = f"{payload.oauth_authorization_url}?{urlencode(query)}"

            oauth_config = {
                "grant_type": "authorization_code",
                "authorization_url": payload.oauth_authorization_url,
                "token_url": payload.oauth_token_url,
                "client_id": payload.oauth_client_id,
                "client_secret": payload.oauth_client_secret,
                "scope": payload.oauth_scope,
                "redirect_uri": redirect_uri,
                "cached_token": None,
                "cached_token_expires_at": None,
                "refresh_token": None,
                "authorization_url_for_user": authorization_url_for_user,
                "oauth_state": state,
            }
        else:
            async with httpx.AsyncClient() as client:
                token_result = await fetch_client_credentials_token(
                    client, payload.oauth_token_url, payload.oauth_client_id, payload.oauth_client_secret, payload.oauth_scope
                )
            if token_result["error"]:
                raise HTTPException(status_code=400, detail=f"OAuth2 client_credentials setup failed: {token_result['error']}")
            oauth_config = {
                "grant_type": "client_credentials",
                "token_url": payload.oauth_token_url,
                "client_id": payload.oauth_client_id,
                "client_secret": payload.oauth_client_secret,
                "scope": payload.oauth_scope,
                "cached_token": token_result["access_token"],
                "cached_token_expires_at": token_result["expires_at"],
            }

    proxy_data = await proxy_manager.create_proxy(
        target_url=normalized_url, has_mcp=has_mcp, api_key=payload.api_key, request=request,
        openapi_operations=openapi_operations, oauth_config=oauth_config, graphql_config=graphql_config
    )
    proxy_id = proxy_data["proxy_id"]
    proxy_url = proxy_data["proxy_url"]

    if authorization_url_for_user:
        await proxy_manager.save_pending_oauth_state(oauth_config["oauth_state"], proxy_id)

    configs = generate_proxy_config(proxy_url, normalized_url)

    return {
        "proxy_id": proxy_id,
        "proxy_url": proxy_url,
        "authorization_url": authorization_url_for_user,
        "authorization_required": bool(authorization_url_for_user),
        "target_url": proxy_data["target_url"],
        "claude_desktop_config": configs["claude_desktop"],
        "cursor_config": configs["cursor_vscode"],
        "windsurf_config": configs["windsurf"],
        "cline_config": configs["cline"],
        "vscode_config": configs["vscode"],
        "claude_code_cli_config": configs["claude_code_cli"],
        "status": proxy_data["status"]
    }


@app.get("/oauth/callback", summary="OAuth2 Authorization Code Callback")
async def oauth_callback(
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    error_description: Optional[str] = Query(None),
):
    """
    Where a target's OAuth login/consent page redirects the user's
    browser back to after they approve (or deny) access. state
    correlates this callback to the proxy that started the flow (see
    save_pending_oauth_state's docstring for why it must be unguessable
    and single-use) - it is NOT the same thing as a client_id, and
    nothing here trusts the request beyond what that lookup returns.
    """
    if error:
        return HTMLResponse(
            f"<h2>Authorization failed</h2><p>{error}: {error_description or 'No further details provided.'}</p>",
            status_code=400,
        )
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing 'code' or 'state' parameter.")

    proxy_id = await proxy_manager.pop_pending_oauth_state(state)
    if not proxy_id:
        raise HTTPException(
            status_code=400,
            detail="This authorization link has expired or was already used. Please start the connection again."
        )

    proxy = await proxy_manager.get_proxy(proxy_id)
    if not proxy or not proxy.get("oauth_config"):
        raise HTTPException(status_code=404, detail="Proxy configuration not found.")

    oauth_config = proxy["oauth_config"]

    # token_url was only validated once, at /proxy/create time - re-check
    # now for the same DNS-rebinding reason every other outbound call in
    # this app does (see _revalidate_target_safety's docstring).
    token_url_safe, reason = await is_public_url(oauth_config["token_url"])
    if not token_url_safe:
        raise HTTPException(status_code=400, detail=f"Refusing to call token endpoint: {reason}")

    async with httpx.AsyncClient() as client:
        result = await exchange_authorization_code(
            client, oauth_config["token_url"], oauth_config["client_id"], oauth_config["client_secret"],
            code, oauth_config["redirect_uri"]
        )
    if result["error"]:
        return HTMLResponse(f"<h2>Token exchange failed</h2><p>{result['error']}</p>", status_code=400)

    oauth_config["cached_token"] = result["access_token"]
    oauth_config["cached_token_expires_at"] = result["expires_at"]
    if result.get("refresh_token"):
        oauth_config["refresh_token"] = result["refresh_token"]
    proxy["oauth_config"] = oauth_config
    await proxy_manager.save_proxy(proxy)

    if not oauth_config.get("refresh_token"):
        return HTMLResponse(
            "<h2>Connected (no long-term refresh)</h2>"
            "<p>You're connected now, but this provider didn't issue a refresh token, so this connection will "
            "need a fresh login again once the current session expires.</p>"
        )
    return HTMLResponse("<h2>Connected!</h2><p>You can close this tab and return to Claude.</p>")


@app.get("/proxy/list", summary="List Active Proxies")
async def list_proxies_endpoint():
    """List all active proxy sessions."""
    proxies = await proxy_manager.list_proxies()
    return {
        "proxies": proxies,
        "total": len(proxies)
    }


@app.get("/proxy/{proxy_id}/health", summary="Check Proxy & Target Health")
async def proxy_health_endpoint(proxy_id: str):
    """Check if proxy is active and target URL is reachable."""
    proxy = await proxy_manager.get_proxy(proxy_id)
    if not proxy:
        raise HTTPException(status_code=404, detail=f"Proxy ID '{proxy_id}' not found.")

    target_url = proxy["target_url"]
    target_reachable = False
    status_code = None
    latency_ms = None

    # target_url's safety was only checked ONCE, at proxy-creation time -
    # DNS isn't static, so re-check now too (see _revalidate_target_safety
    # in app/proxy.py for the full DNS-rebinding rationale: a domain could
    # resolve to a public IP at creation and be repointed to an internal
    # one by the time this endpoint is actually hit).
    is_safe, reason = await is_public_url(target_url)
    if not is_safe:
        return {
            "proxy_id": proxy_id,
            "proxy_status": "active",
            "target_url": target_url,
            "target_reachable": False,
            "target_status_code": None,
            "latency_ms": None,
            "error": f"Refusing to check target: {reason}",
            "has_mcp": proxy.get("has_mcp", False),
            "created_at": proxy.get("created_at"),
            "last_used": proxy.get("last_used")
        }

    start_time = asyncio.get_event_loop().time()
    try:
        # No follow_redirects: this proxy's target_url was only SSRF-checked
        # once, at creation time - a redirect here isn't re-validated.
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(target_url)
            status_code = resp.status_code
            target_reachable = resp.status_code < 500
            latency_ms = round((asyncio.get_event_loop().time() - start_time) * 1000, 2)
    except Exception as e:
        pass

    return {
        "proxy_id": proxy_id,
        "proxy_status": "active",
        "target_url": target_url,
        "target_reachable": target_reachable,
        "target_status_code": status_code,
        "latency_ms": latency_ms,
        "has_mcp": proxy.get("has_mcp", False),
        "created_at": proxy.get("created_at"),
        "last_used": proxy.get("last_used")
    }


async def _handle_proxy_sse(proxy_id: str, request: Request):
    proxy = await proxy_manager.get_proxy(proxy_id)
    if not proxy:
        raise HTTPException(status_code=404, detail=f"Proxy ID '{proxy_id}' not found.")

    session_id = str(uuid.uuid4())
    queue = asyncio.Queue()
    proxy_manager.sessions[session_id] = queue

    app_url = proxy_manager.get_base_app_url(request)
    messages_url = f"{app_url}/proxy/{proxy_id}/mcp/messages/?session_id={session_id}"

    async def event_generator():
        try:
            # Send initial endpoint event per MCP SSE spec
            yield {
                "event": "endpoint",
                "data": messages_url
            }

            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=20.0)
                    if msg is not None:
                        yield {
                            "event": "message",
                            "data": json.dumps(msg)
                        }
                except asyncio.TimeoutError:
                    # Keep connection alive with silent ping
                    yield {
                        "event": "ping",
                        "data": ""
                    }
        finally:
            proxy_manager.sessions.pop(session_id, None)

    return EventSourceResponse(event_generator())


@app.get("/proxy/{proxy_id}/mcp", summary="Proxy MCP SSE Endpoint")
async def proxy_mcp_sse_1(proxy_id: str, request: Request):
    return await _handle_proxy_sse(proxy_id, request)


@app.get("/proxy/{proxy_id}/mcp/", summary="Proxy MCP SSE Endpoint (Trailing Slash)")
async def proxy_mcp_sse_2(proxy_id: str, request: Request):
    return await _handle_proxy_sse(proxy_id, request)


async def _handle_proxy_http(proxy_id: str, payload: Dict[str, Any]):
    proxy = await proxy_manager.get_proxy(proxy_id)
    if not proxy:
        raise HTTPException(status_code=404, detail=f"Proxy ID '{proxy_id}' not found.")
    return await proxy_manager.forward_request(proxy_id, payload)


@app.post("/proxy/{proxy_id}/mcp", summary="Proxy MCP Streamable HTTP Endpoint")
async def proxy_mcp_http_1(proxy_id: str, payload: Dict[str, Any] = Body(...)):
    return await _handle_proxy_http(proxy_id, payload)


@app.post("/proxy/{proxy_id}/mcp/", summary="Proxy MCP Streamable HTTP Endpoint (Trailing Slash)")
async def proxy_mcp_http_2(proxy_id: str, payload: Dict[str, Any] = Body(...)):
    return await _handle_proxy_http(proxy_id, payload)


async def _handle_proxy_messages(proxy_id: str, payload: Dict[str, Any], session_id: Optional[str] = None):
    proxy = await proxy_manager.get_proxy(proxy_id)
    if not proxy:
        raise HTTPException(status_code=404, detail=f"Proxy ID '{proxy_id}' not found.")

    response_data = await proxy_manager.forward_request(proxy_id, payload)

    if session_id and session_id in proxy_manager.sessions:
        session_queue = proxy_manager.sessions[session_id]
        if response_data:
            await session_queue.put(response_data)
        return {"status": "accepted"}
    else:
        return response_data


@app.post("/proxy/{proxy_id}/mcp/messages", summary="Proxy MCP Message Handler")
async def proxy_mcp_msg_1(proxy_id: str, payload: Dict[str, Any] = Body(...), session_id: Optional[str] = Query(None)):
    return await _handle_proxy_messages(proxy_id, payload, session_id)


@app.post("/proxy/{proxy_id}/mcp/messages/", summary="Proxy MCP Message Handler (Trailing Slash)")
async def proxy_mcp_msg_2(proxy_id: str, payload: Dict[str, Any] = Body(...), session_id: Optional[str] = Query(None)):
    return await _handle_proxy_messages(proxy_id, payload, session_id)


@app.post("/proxy/{proxy_id}/messages", summary="Proxy MCP Message Handler (Legacy)")
async def proxy_mcp_msg_3(proxy_id: str, payload: Dict[str, Any] = Body(...), session_id: Optional[str] = Query(None)):
    return await _handle_proxy_messages(proxy_id, payload, session_id)


@app.post("/proxy/{proxy_id}/messages/", summary="Proxy MCP Message Handler (Legacy Trailing Slash)")
async def proxy_mcp_msg_4(proxy_id: str, payload: Dict[str, Any] = Body(...), session_id: Optional[str] = Query(None)):
    return await _handle_proxy_messages(proxy_id, payload, session_id)



# 3. Include REST routes (/analyze, /generate, /guide)
app.include_router(mcp_router)

# 4. Expose all FastAPI /api/ & REST endpoints as MCP tools via FastApiMCP at /mcp-server
fastapi_mcp = FastApiMCP(app, name="DataHub Talk to Data")
fastapi_mcp.mount_sse(mount_path="/mcp-server/sse")
fastapi_mcp.mount_http(mount_path="/mcp-server/mcp")
fastapi_mcp.mount(mount_path="/mcp-server")

# 5. Mount Dual MCP Transports at /mcp
# Streamable HTTP transport (POST /mcp & POST /mcp/ for MCP 2024-11 spec)
app.add_route("/mcp", http_app, methods=["POST"])
app.add_route("/mcp/", http_app, methods=["POST"])

# SSE transport (GET /mcp/sse & POST /mcp/messages for legacy SSE connectors)
app.mount("/mcp", sse_app)

# 6. Frontend Catch-All Route (MUST be defined AFTER all API & MCP routes/mounts)
@app.get("/{full_path:path}", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend_catch_all(full_path: str):
    """
    Catch-all route serving frontend Web UI.
    Excludes API, MCP, Proxy, Docs, and Health endpoints from returning HTML SPA page.
    """
    clean_path = full_path.lstrip("/")
    if (
        clean_path.startswith("mcp-server") or
        clean_path.startswith("api") or
        clean_path.startswith("health") or
        clean_path.startswith("mcp") or
        clean_path.startswith("proxy") or
        clean_path.startswith("docs") or
        clean_path.startswith("openapi.json") or
        clean_path.startswith("redoc")
    ):
        raise HTTPException(status_code=404, detail="Endpoint not found.")

    web_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "app", "web"))
    requested_file = os.path.abspath(os.path.join(web_dir, clean_path))

    # Safely serve static files from app/web/ (e.g. navbar.html, navbar.css, images)
    if requested_file.startswith(web_dir) and os.path.isfile(requested_file) and clean_path != "index.html":
        return FileResponse(requested_file)

    return render_web_page()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

