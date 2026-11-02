"""Arm A has no hidden fallback: schema, parsing, format errors, dedupe."""

from __future__ import annotations

import json

from crb.arms.generation import GenerationArm

MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def _arm(collections):
    return GenerationArm(MODEL, collections)


def _body(content: str, finish_reason: str = "stop") -> dict:
    return {
        "choices": [
            {"finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}
        ],
        "usage": {"prompt_tokens": 900, "completion_tokens": 20},
    }


def test_request_uses_json_schema_enum(splits, collections):
    request = _arm(collections).build_request(splits["test"][0], "run-1", 2)
    schema = request["response_format"]["json_schema"]["schema"]
    enum = schema["properties"]["collections"]["items"]["enum"]
    assert enum == [c.id for c in collections]
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["temperature"] == 0
    assert request["max_tokens"] == 128
    assert request["cache_salt"] == "run-1-r2-A"
    assert request["chat_template_kwargs"] == {"enable_thinking": False}


def test_no_guided_fields(splits, collections):
    request = _arm(collections).build_request(splits["test"][0], "run-1", 1)
    for key in list(request):
        assert not key.startswith("guided_"), f"guided field leaked: {key}"


def test_parse_valid(collections):
    out = _arm(collections).parse_completion(
        _body(json.dumps({"collections": ["web", "templates"]}))
    )
    assert out["pred"] == ["web", "templates"]
    assert out["format_error"] is False
    assert out["error"] is False


def test_invalid_json_is_empty_and_flagged(collections):
    out = _arm(collections).parse_completion(_body("NOT JSON"))
    assert out["pred"] == []
    assert out["format_error"] is True


def test_length_finish_is_format_error(collections):
    out = _arm(collections).parse_completion(
        _body(json.dumps({"collections": ["web"]}), finish_reason="length")
    )
    assert out["pred"] == []
    assert out["format_error"] is True


def test_dedupe_ids(collections):
    out = _arm(collections).parse_completion(
        _body(json.dumps({"collections": ["web", "web", "templates"]}))
    )
    assert out["pred"] == ["web", "templates"]


def test_unknown_ids_dropped(collections):
    out = _arm(collections).parse_completion(_body(json.dumps({"collections": ["web", "banana"]})))
    assert out["pred"] == ["web"]
