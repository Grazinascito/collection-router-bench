"""Arms A and B receive the same information; prompts are frozen by the lock."""

from __future__ import annotations

from crb.arms.generation import GenerationArm
from crb.arms.systemone import SystemOneArm
from crb.prompts import (
    collections_block,
    generation_message,
    load_lock_file,
    prompt_lock,
    state_text,
)
from crb.schema import Block, Preview


def _long_block() -> Block:
    return Block.model_validate(
        {
            "id": "x-0001",
            "split": "test",
            "kind": "link",
            "user_title": "t",
            "user_description": None,
            "url": "https://example.com/synthetic-0001",
            "preview": Preview(site_name="s", title="p", description="d"),
            "content": "w " * 5000,
            "content_status": "read",
            "lang": "en",
            "labels": [],
            "difficulty": "easy",
            "group_id": "g-x",
            "rationale": "r",
        }
    )


def _not_read_block() -> Block:
    return Block.model_validate(
        {
            "id": "x-0002",
            "split": "test",
            "kind": "link",
            "user_title": "t",
            "user_description": None,
            "url": "https://example.com/synthetic-0002",
            "preview": Preview(site_name="s", title="p", description="d"),
            "content": None,
            "content_status": "not_read",
            "lang": "en",
            "labels": [],
            "difficulty": "easy",
            "group_id": "g-y",
            "rationale": "r",
        }
    )


def test_same_state_text_for_all_arms(splits, collections):
    block = splits["test"][0]
    gen = GenerationArm("m", collections)
    sysarm = SystemOneArm("m", collections)
    a_content = gen.build_request(block, "run", 1)["messages"][0]["content"]
    assert state_text(block) in a_content
    assert sysarm.build_request(block, "run", 1)["state"] == state_text(block)


def test_content_cut_2000():
    text = state_text(_long_block())
    content_part = text.split("Content:\n", 1)[1]
    assert len(content_part) <= 2000


def test_not_read_has_no_content():
    text = state_text(_not_read_block())
    assert "Content status: not_read" in text
    assert text.endswith("Content:\n(empty)")


def test_collections_block_has_11(collections):
    block = collections_block(collections)
    for c in collections:
        assert f"- id: {c.id}" in block
    assert block.count("- id: ") == 11


def test_generation_message_layout(splits, collections):
    message = generation_message(splits["test"][0], collections)
    assert message.index("## Collections") < message.index("Block:\n")
    assert state_text(splits["test"][0]) in message


def test_prompt_lock(collections):
    assert prompt_lock(collections) == load_lock_file()
