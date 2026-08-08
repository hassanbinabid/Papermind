import pytest
from httpx import ASGITransport, AsyncClient

from app.api import app
from tests.conftest import TEST_API_KEY

pytestmark = pytest.mark.asyncio


async def test_ingest_requires_api_key(async_client):
    files = {"file": ("test.md", b"# hello", "text/markdown")}
    response = await async_client.post("/ingest", files=files)
    assert response.status_code == 422  # missing required X-API-Key header


async def test_ingest_rejects_bad_api_key(async_client):
    files = {"file": ("test.md", b"# hello", "text/markdown")}
    response = await async_client.post(
        "/ingest", files=files, headers={"X-API-Key": "wrong-key"}
    )
    assert response.status_code == 401


async def test_ingest_succeeds_with_valid_key(ingest_overrides):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        files = {"file": ("test.md", b"# hello world", "text/markdown")}
        response = await ac.post(
            "/ingest", files=files, headers={"X-API-Key": TEST_API_KEY}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["filename"] == "test.md"


async def test_ingest_rejects_unsupported_file_type(async_client):
    files = {"file": ("test.txt", b"hello", "text/plain")}
    response = await async_client.post(
        "/ingest", files=files, headers={"X-API-Key": TEST_API_KEY}
    )
    assert response.status_code == 400


async def test_list_documents_no_auth_required(async_client):
    response = await async_client.get("/documents")
    assert response.status_code == 200


async def test_delete_requires_api_key(async_client):
    response = await async_client.delete("/documents/somefile.pdf")
    assert response.status_code == 422  # missing required X-API-Key header


async def test_delete_succeeds_with_valid_key(async_client):
    response = await async_client.delete(
        "/documents/somefile.pdf", headers={"X-API-Key": TEST_API_KEY}
    )
    assert response.status_code == 200
    assert response.json()["deleted"] is True
