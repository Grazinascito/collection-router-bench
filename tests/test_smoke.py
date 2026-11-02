"""Smoke: ok on the fake server; swap order follows the plan."""

from __future__ import annotations

import asyncio

import httpx

from crb.fake import FakeTransport, FakeTruth, make_client
from crb.prompts import state_text
from crb.smoke import run_smoke, swap_order


def test_fallback_order():
    order = swap_order("primary/model", "fallback/model")
    assert order == [
        ("primary/model", "systemone"),
        ("fallback/model", "systemone"),
        ("fallback/model", "chat_logprobs"),
    ]


def test_smoke_ok_fake_server(splits, collections):
    items = splits["train"][:5]
    truth = FakeTruth({state_text(b): list(b.labels) for b in splits["train"]})

    async def _run():
        client = make_client(truth)

        async def client_for(model_id: str) -> httpx.AsyncClient:
            return client

        try:
            run_meta: dict = {}
            model_id, impl = await run_smoke(
                client_for, items, collections, "fake/model", "fake/model", "smoke-run", run_meta
            )
            return model_id, impl, run_meta
        finally:
            await client.aclose()

    model_id, impl, run_meta = asyncio.run(_run())
    assert model_id == "fake/model"
    assert impl == "systemone"
    assert run_meta["smoke_attempts"][0]["ok"] is True


def test_smoke_falls_back_to_chat_logprobs(splits, collections):
    """A server where systemone answers 501 must push the smoke to B'."""
    import json as _json

    items = splits["train"][:5]
    inner = FakeTransport(FakeTruth({state_text(b): list(b.labels) for b in splits["train"]}))

    class NoSystemOne(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/systemone":
                return httpx.Response(501, json={"error": {"message": "unsupported"}})
            response = await inner.handle_async_request(request)
            body = _json.loads(response.read())
            return httpx.Response(response.status_code, json=body)

    async def _run():
        client = httpx.AsyncClient(transport=NoSystemOne(), base_url="http://t")

        async def client_for(model_id: str) -> httpx.AsyncClient:
            return client

        try:
            run_meta: dict = {}
            return await run_smoke(
                client_for, items, collections, "m1", "m1", "run", run_meta
            ), run_meta
        finally:
            await client.aclose()

    (model_id, impl), run_meta = asyncio.run(_run())
    assert impl == "chat_logprobs"
    attempts = run_meta["smoke_attempts"]
    assert attempts[0]["ok"] is False and "501" in attempts[0]["error"]
