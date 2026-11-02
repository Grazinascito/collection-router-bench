"""Shared prompt material: state_text, task instructions, collections block, lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from crb.schema import Block, Collection

CONTENT_CHAR_LIMIT = 2000

TASK_INSTRUCTIONS = """You are the routing assistant of a second-brain app. Read the saved block \
and decide which of the collections below it belongs to. Rules: a block may belong to zero or \
more collections; decide using each collection's name, includes and excludes; use the block's \
content, the user's title and description, and the preview data; if the content was not read, \
use only the preview data; if no collection fits, the block stays uncollected (an empty set)."""

GENERATION_SUFFIX = """Return JSON with the ids that fit. Return an empty list if none fits.

Block:
"""


def _clean(text: str | None) -> str:
    text = (text or "").strip()
    return text if text else "(empty)"


def state_text(block: Block, char_limit: int = CONTENT_CHAR_LIMIT) -> str:
    """The block as one string, fields in a fixed order; empty fields become '(empty)'."""
    preview = block.preview
    lines = [
        f"Type: {block.kind}",
        f"User title: {_clean(block.user_title)}",
        f"User description: {_clean(block.user_description)}",
        f"URL: {_clean(block.url)}",
        f"Site: {_clean(preview.site_name if preview else None)}",
        f"Preview title: {_clean(preview.title if preview else None)}",
        f"Preview description: {_clean(preview.description if preview else None)}",
        f"Content status: {block.content_status}",
        f"Content:\n{_clean(block.content)[:char_limit]}",
    ]
    return "\n".join(lines)


def collections_block(collections: list[Collection]) -> str:
    parts = ["## Collections"]
    for c in collections:
        parts.append(
            f"- id: {c.id} | name: {c.name}\n"
            f"  includes: {' '.join(c.includes.split())}\n"
            f"  excludes: {' '.join(c.excludes.split())}"
        )
    return "\n".join(parts)


def noul_question(c: Collection) -> str:
    return f'Does this block belong in the collection "{c.name}"?'


def generation_message(block: Block, collections: list[Collection]) -> str:
    return (
        f"{TASK_INSTRUCTIONS}\n\n{collections_block(collections)}\n\n"
        f"{GENERATION_SUFFIX}{state_text(block)}"
    )


def systemone_instructions(collections: list[Collection]) -> str:
    return f"{TASK_INSTRUCTIONS}\n\n{collections_block(collections)}"


# ---------------------------------------------------------------------------
# Prompt lock: sha256 of the rendered templates, so the prompts are frozen
# before the test run (tests fail if a template changes without updating the
# lock file).
# ---------------------------------------------------------------------------


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_lock(collections: list[Collection]) -> dict[str, str]:
    return {
        "task_instructions": _sha(TASK_INSTRUCTIONS),
        "generation_suffix": _sha(GENERATION_SUFFIX),
        "collections_block": _sha(collections_block(collections)),
        "noul_question": _sha(noul_question(collections[0])),
    }


def load_lock_file() -> dict[str, str]:
    path = Path(__file__).with_name("prompts.lock")
    return json.loads(path.read_text(encoding="utf-8"))
