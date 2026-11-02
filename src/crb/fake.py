"""A deterministic fake vLLM server (httpx.MockTransport) for tests and --fake runs.

The runner injects a truth map (state_text -> gold labels). Arm A answers with
the truth minus a rare omission; arm B probabilities sit near 0.9/0.1 with a
rare systematic flip, so the fake pipeline produces non-degenerate metrics.
A rare invalid-JSON completion exercises arm A's format_error path.
"""

from __future__ import annotations

import hashlib
import json
import math

import httpx


class FakeTruth:
    def __init__(self, truth_map: dict[str, list[str]] | None = None) -> None:
        self.truth_map = truth_map or {}

    def labels(self, state: str) -> list[str]:
        if state in self.truth_map:
            return list(self.truth_map[state])
        return []

    def _hash(self, *parts: str) -> int:
        return hashlib.sha1("::".join(parts).encode()).digest()[0]

    def p_yes(self, state: str, collection_id: str) -> float:
        truth = collection_id in self.labels(state)
        jitter = (self._hash("p", state, collection_id) % 9) / 100.0
        return (0.9 if truth else 0.1) + (jitter if truth else -jitter)

    def arm_a_pred(self, state: str) -> list[str]:
        labels = self.labels(state)
        # drop one label when the bucket says so (imperfect but realistic)
        if labels and self._hash("drop", state) % 13 == 0:
            labels = labels[:-1]
        # rare wrong addition
        if self._hash("add", state) % 29 == 0:
            labels = labels + ["web"]
        return labels

    def arm_b_flip(self, state: str) -> bool:
        return self._hash("flip", state) % 17 == 3

    def invalid_json(self, state: str) -> bool:
        return self._hash("bad", state) == 199


def _state_of_chat_request(payload: dict) -> str:
    for message in payload.get("messages", []):
        content = message.get("content", "")
        if isinstance(content, str) and "\nBlock:\n" in content:
            return content.split("\nBlock:\n", 1)[1]
    return ""


def _collection_from_question(payload: dict) -> str:
    text = payload.get("messages", [{}])[-1].get("content", "")
    marker = 'collection "'
    if marker in text:
        return text.split(marker, 1)[1].split('"', 1)[0]
    return ""


def _name_error_note() -> str:
    return ""


class FakeTransport(httpx.AsyncBaseTransport):
    """Routes /health, /v1/chat/completions and /v1/systemone deterministically."""

    def __init__(self, truth: FakeTruth) -> None:
        self.truth = truth
        self.requests: list[dict] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode()) if request.content else {}
        self.requests.append({"path": request.url.path})
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json=self._chat(payload))
        if request.url.path == "/v1/systemone":
            return httpx.Response(200, json=self._systemone(payload))
        return httpx.Response(404, json={"error": {"message": f"no route {request.url.path}"}})

    # ------------------------------------------------------------------
    def _chat(self, payload: dict) -> dict:
        state = _state_of_chat_request(payload)
        if payload.get("max_tokens") == 1 and payload.get("logprobs"):
            cid = _collection_from_question(payload)
            p_yes = self.truth.p_yes(state, cid)
            yes, no = p_yes, 1.0 - p_yes

            def _lp(p: float) -> float:
                return math.log(max(p, 1e-9))

            entry = {
                "yes": _lp(yes * 0.7),
                "Yes": _lp(yes * 0.2),
                "no": _lp(no * 0.7),
                "No": _lp(no * 0.2),
            }
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "yes" if yes >= no else "no"},
                        "logprobs": {"top_logprobs": [entry]},
                    }
                ],
                "usage": {"prompt_tokens": 700, "completion_tokens": 1},
            }
        if self.truth.invalid_json(state):
            content = "NOT JSON AT ALL"
        else:
            content = json.dumps({"collections": self.truth.arm_a_pred(state)})
        return {
            "choices": [{"finish_reason": "stop", "message": {"content": content}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 24},
        }

    def _systemone(self, payload: dict) -> dict:
        state = payload.get("state", "")
        labels = set(self.truth.labels(state))
        flip = self.truth.arm_b_flip(state)
        flip_done = False
        answers: dict = {}
        diagnostics: dict = {}
        for qid in payload.get("questions", {}):
            p = self.truth.p_yes(state, qid)
            if flip and not flip_done and qid not in labels:
                p = 0.83  # systematic wrong addition on ~1/17 of items
                flip_done = True
            answers[qid] = {
                "type": "noul",
                "noul": p,
                "probabilities": {"yes": p, "no": 1.0 - p},
                "confidence": max(p, 1.0 - p) * 0.9,
            }
            diagnostics[qid] = {"label_mass": 0.9, "argmax_is_label": True}
        return {
            "id": "fake",
            "model": payload.get("model"),
            "answers": answers,
            "diagnostics": diagnostics,
            "usage": {"input_tokens": 800, "output_tokens": 11},
        }


def make_client(truth: FakeTruth, **kwargs) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="http://fake",
        transport=FakeTransport(truth),
        timeout=httpx.Timeout(60.0),
        **kwargs,
    )
