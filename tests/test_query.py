import pytest

pytestmark = pytest.mark.asyncio


async def test_query_returns_answer(async_client):
    response = await async_client.post("/query", json={"question": "What is PaperMind?"})
    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "What is PaperMind?"
    assert "Fake answer" in body["answer"]


async def test_query_rejects_empty_question(async_client):
    response = await async_client.post("/query", json={"question": "   "})
    assert response.status_code == 422
