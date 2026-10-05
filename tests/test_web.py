import os
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root_serves_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "MCPify" in response.text
    assert "MCP Server" in response.text

def test_pricing_page_serves_html():
    response = client.get("/pricing")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Pro" in response.text
    assert "Enterprise" in response.text

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_api_info():
    response = client.get("/api")
    assert response.status_code == 200
    assert response.json()["service"] == "MCPify"

def test_generate_requires_auth():
    # /generate (like /analyze, /guide, /proxy/create) creates a proxy
    # record tied to the caller's account for a non-native-MCP target -
    # unauthenticated callers must be rejected before any DB/network
    # work happens, not just discouraged.
    response = client.post("/generate", json={"url": "https://fastapi.tiangolo.com"})
    assert response.status_code == 401


@pytest.mark.network
def test_generate_config():
    # Needs both a real network target and a logged-in user - run
    # manually against an environment with DATABASE_URL configured, not
    # part of the default (network-excluded) suite.
    signup = client.post(
        "/auth/signup",
        json={"email": "test-generate@example.com", "password": "testpass123",
              "first_name": "Test", "last_name": "User"},
    )
    token = signup.json()["access_token"]
    response = client.post(
        "/generate", json={"url": "https://fastapi.tiangolo.com"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "detected_framework" in data["analysis"]
    assert "claude_desktop" in data["configs"]
    assert "cursor_vscode" in data["configs"]
    assert "windsurf" in data["configs"]
    assert "cline" in data["configs"]
    assert "vscode" in data["configs"]
    assert "claude_code_cli" in data["configs"]

def test_sitemap_xml():
    response = client.get("/sitemap.xml")
    assert response.status_code == 200
    assert "xml" in response.headers["content-type"]
    assert "https://mcpify.aikart.co/" in response.text

def test_robots_txt():
    response = client.get("/robots.txt")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "User-agent: *" in response.text
    assert "Sitemap: https://mcpify.aikart.co/sitemap.xml" in response.text

def test_llms_txt():
    response = client.get("/llms.txt")
    assert response.status_code == 200
    assert "text/markdown" in response.headers["content-type"]
    assert "https://mcpify.aikart.co" in response.text

    well_known = client.get("/.well-known/llms.txt")
    assert well_known.status_code == 200
    assert "https://mcpify.aikart.co" in well_known.text


