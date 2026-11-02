"""Arm B request body follows the vLLM 0.32.0 systemone contract (api_check.md)."""

from __future__ import annotations

import json

import httpx
import pytest

from crb.arms.systemone import SystemOneArm, SystemOneUnavailable

MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def _arm(collections):
    return SystemOneArm(MODEL, collections)


def test_body_has_11_noul_in_yaml_order(splits, collections):
    request = _arm(collections).build_request(splits["test"][0], "run-9", 3)
    questions = request["questions"]
    assert list(questions) == [c.id for c in collections]
    for spec in questions.values():
        assert spec["type"] == "noul"
    assert request["cache_salt"] == "run-9-r3-B"
    assert request["chat_template_kwargs"] == {"enable_thinking": False}
    assert isinstance(request["state"], str)


def test_only_allowed_fields(splits, collections):
    request = _arm(collections).build_request(splits["test"][0], "run", 1)
    assert set(request) <= {
        "model",
        "state",
        "questions",
        "instructions",
        "chat_template_kwargs",
        "cache_salt",
    }
    for spec in request["questions"].values():
        assert set(spec) == {"type", "instructions", "criteria"}
        assert set(spec["criteria"]) == {"true", "false"}


def test_criteria_come_from_yaml(splits, collections):
    request = _arm(collections).build_request(splits["test"][0], "run", 1)
    first = collections[0]
    assert request["questions"][first.id]["criteria"]["true"] == " ".join(first.includes.split())
    assert request["questions"][first.id]["criteria"]["false"] == " ".join(first.excludes.split())


def _response(probs: dict[str, float]) -> httpx.Response:
    answers = {
        qid: {
            "type": "noul",
            "noul": p,
            "probabilities": {"yes": p, "no": 1 - p},
            "confidence": max(p, 1 - p),
        }
        for qid, p in probs.items()
    }
    diagnostics = {qid: {"label_mass": 0.9, "argmax_is_label": True} for qid in probs}
    return httpx.Response(
        200,
        json={
            "answers": answers,
            "diagnostics": diagnostics,
            "usage": {"input_tokens": 800, "output_tokens": 11},
        },
    )


def test_threshold_05(splits, collections):
    arm = _arm(collections)
    ids = [c.id for c in collections]
    probs = {qid: 0.5 for qid in ids}
    out = arm.parse_response(json.loads(_response(probs).read()))
    assert out["pred"] == ids  # >= 0.5 counts
    probs[ids[0]] = 0.4999
    out = arm.parse_response(json.loads(_response(probs).read()))
    assert ids[0] not in out["pred"]


def test_records_label_mass(splits, collections):
    arm = _arm(collections)
    probs = {c.id: 0.9 for c in collections}
    body = json.loads(_response(probs).read())
    out = arm.parse_response(body)
    assert out["label_mass"] == 0.9
    assert set(out["probs"]) == {c.id for c in collections}


def test_400_501_raise_unavailable(splits, collections):
    import asyncio

    from crb.fake import FakeTruth, make_client

    class BadTransport(httpx.AsyncBaseTransport):
        def __init__(self, status: int) -> None:
            super().__init__()
            self.status = status

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(self.status, json={"error": {"message": "nope"}})

    async def _check(status: int) -> None:
        client = httpx.AsyncClient(transport=BadTransport(status), base_url="http://t")
        try:
            with pytest.raises(SystemOneUnavailable):
                await _arm(collections).decide(client, splits["test"][0], "run", 1)
        finally:
            await client.aclose()

    for status in (400, 501):
        asyncio.run(_check(status))

    async def _good() -> None:
        client = make_client(FakeTruth({}))
        try:
            out = await _arm(collections).decide(client, splits["test"][0], "run", 1)
            assert len(out["probs"]) == 11
        finally:
            await client.aclose()

    asyncio.run(_good())
