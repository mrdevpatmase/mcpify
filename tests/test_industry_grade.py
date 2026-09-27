"""
Regression coverage for the 6 "industry grade" additions built after
the security-hardening pass: proxy TTL/expiry, structured tool-error
types, per-proxy rate limiting, outbound payload size caps, the admin
proxy-delete endpoint, and structured JSON logging.
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

from app.logging_config import JSONFormatter
from app.proxy import (
    MAX_OUTBOUND_PAYLOAD_BYTES,
    PER_PROXY_RATE_LIMIT,
    ProxyMCPManager,
    _is_proxy_expired,
    _payload_too_large,
)


def test_is_proxy_expired_true_past_ttl():
    stale = (datetime.now(timezone.utc) - timedelta(days=91)).isoformat()
    assert _is_proxy_expired({"last_used": stale}) is True


def test_is_proxy_expired_false_within_ttl():
    fresh = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    assert _is_proxy_expired({"last_used": fresh}) is False


def test_is_proxy_expired_false_when_missing_last_used():
    assert _is_proxy_expired({}) is False


def test_get_proxy_evicts_expired_in_memory_entry():
    async def run():
        manager = ProxyMCPManager()
        stale = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        manager.proxies["old1"] = {"proxy_id": "old1", "target_url": "https://x.test", "last_used": stale}
        assert await manager.get_proxy("old1") is None
        assert "old1" not in manager.proxies

    asyncio.run(run())


def test_list_proxies_excludes_expired_in_memory_entries():
    async def run():
        manager = ProxyMCPManager()
        stale = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        fresh = datetime.now(timezone.utc).isoformat()
        manager.proxies["old1"] = {"proxy_id": "old1", "target_url": "https://x.test", "last_used": stale}
        manager.proxies["new1"] = {"proxy_id": "new1", "target_url": "https://y.test", "last_used": fresh}
        listed = await manager.list_proxies()
        ids = {p["proxy_id"] for p in listed}
        assert ids == {"new1"}

    asyncio.run(run())


def test_delete_proxy_removes_in_memory_record():
    async def run():
        manager = ProxyMCPManager()
        proxy = await manager.create_proxy(target_url="https://httpbin.org")
        assert await manager.delete_proxy(proxy["proxy_id"]) is True
        assert await manager.get_proxy(proxy["proxy_id"]) is None
        # Deleting again (already gone) reports False, not an error.
        assert await manager.delete_proxy(proxy["proxy_id"]) is False

    asyncio.run(run())


def test_per_proxy_rate_limit_blocks_after_threshold():
    manager = ProxyMCPManager()
    for _ in range(PER_PROXY_RATE_LIMIT):
        assert manager._check_per_proxy_rate_limit("px1") is True
    # One over the limit within the same window must be rejected.
    assert manager._check_per_proxy_rate_limit("px1") is False
    # A different proxy_id has its own independent budget.
    assert manager._check_per_proxy_rate_limit("px2") is True


def test_call_api_tool_call_respects_per_proxy_rate_limit():
    async def run():
        manager = ProxyMCPManager()
        proxy = await manager.create_proxy(target_url="https://httpbin.org")
        for _ in range(PER_PROXY_RATE_LIMIT):
            manager._check_per_proxy_rate_limit(proxy["proxy_id"])
        result = await manager.handle_jsonrpc_request(proxy, {
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "call_api", "arguments": {"endpoint": "/get"}},
        })
        assert result["result"]["isError"] is True
        assert result["result"]["errorType"] == "rate_limited"

    asyncio.run(run())


def test_payload_too_large_detects_oversized_json():
    small = {"a": 1}
    assert _payload_too_large(small) is False
    big = {"data": "x" * (MAX_OUTBOUND_PAYLOAD_BYTES + 1000)}
    assert _payload_too_large(big) is True
    assert _payload_too_large(None) is False


def test_call_api_rejects_oversized_json_data():
    async def run():
        manager = ProxyMCPManager()
        proxy = await manager.create_proxy(target_url="https://httpbin.org")
        oversized = "x" * (MAX_OUTBOUND_PAYLOAD_BYTES + 1000)
        result = await manager.handle_jsonrpc_request(proxy, {
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "call_api", "arguments": {
                "endpoint": "/post", "method": "POST", "json_data": {"blob": oversized}
            }},
        })
        assert result["result"]["isError"] is True
        assert result["result"]["errorType"] == "payload_too_large"

    asyncio.run(run())


def test_call_api_unsafe_target_reports_target_blocked_error_type():
    async def run():
        manager = ProxyMCPManager()
        # A proxy record can point at a target that's since become
        # unsafe (DNS-rebinding) - simulate that directly rather than
        # via a real create_proxy call, which would itself reject an
        # unsafe target up front.
        proxy = {
            "proxy_id": "pxlocal", "target_url": "http://127.0.0.1:9",
            "has_mcp": False, "api_key": None, "oauth_config": None, "openapi_operations": None,
        }
        result = await manager.handle_jsonrpc_request(proxy, {
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "call_api", "arguments": {"endpoint": "/x"}},
        })
        assert result["result"]["isError"] is True
        assert result["result"]["errorType"] == "target_blocked"

    asyncio.run(run())


def test_json_formatter_produces_valid_parseable_json():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="mcpify", level=logging.INFO, pathname="x.py", lineno=1,
        msg="hello %s", args=("world",), exc_info=None,
    )
    line = formatter.format(record)
    parsed = json.loads(line)
    assert parsed["message"] == "hello world"
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "mcpify"
    assert "timestamp" in parsed
