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
    assert "Generate MCP links" in response.text

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_api_info():
    response = client.get("/api")
    assert response.status_code == 200
    assert response.json()["service"] == "MCPify"

def test_generate_config():
    response = client.post("/generate", json={"url": "https://fastapi.tiangolo.com"})
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


