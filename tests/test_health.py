import pytest

pytestmark = pytest.mark.asyncio


async def test_health_ok(async_client):
    response = await async_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("ok", "degraded")
    assert "total_chunks" in body
