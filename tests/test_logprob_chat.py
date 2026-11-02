"""Arm B fallback (B'): yes/no logprob normalization over top-20 variants."""

from __future__ import annotations

import math

from crb.arms.logprob_chat import LogprobChatArm

MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def _lp(p: float) -> float:
    return math.log(p)


def test_yes_no_normalization():
    entry = {"yes": _lp(0.5), "Yes": _lp(0.2), "no": _lp(0.2), "No": _lp(0.1)}
    out = LogprobChatArm.parse_top_logprobs(entry)
    # masses: yes 0.7, no 0.3 -> p_yes = 0.7
    assert abs(out["p_yes"] - 0.7) < 1e-9
    assert abs(out["label_mass"] - 1.0) < 1e-9
    assert out["error"] is False


def test_leading_space_variants_count():
    entry = {" yes": _lp(0.4), " no": _lp(0.6)}
    out = LogprobChatArm.parse_top_logprobs(entry)
    assert abs(out["p_yes"] - 0.4) < 1e-9


def test_missing_labels_is_error():
    assert LogprobChatArm.parse_top_logprobs({"maybe": _lp(0.9)})["error"] is True
    assert LogprobChatArm.parse_top_logprobs(None)["error"] is True
    assert LogprobChatArm.parse_top_logprobs({})["error"] is True


def test_label_mass():
    entry = {"yes": _lp(0.2), "no": _lp(0.3), "banana": _lp(0.5)}
    out = LogprobChatArm.parse_top_logprobs(entry)
    assert abs(out["label_mass"] - 0.5) < 1e-9  # non-label tokens excluded


def test_build_requests_shape(splits, collections):
    arm = LogprobChatArm(MODEL, collections)
    requests = arm.build_requests(splits["test"][0], "run-2", 1)
    assert len(requests) == 11
    for r in requests:
        assert r["max_tokens"] == 1
        assert r["temperature"] == 0
        assert r["logprobs"] is True
        assert r["top_logprobs"] == 20
        assert r["cache_salt"] == "run-2-r1-B"
        assert r["messages"][0]["role"] == "system"
