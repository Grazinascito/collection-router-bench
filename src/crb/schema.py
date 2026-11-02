"""Pydantic models for collections, dataset blocks, predictions and bench records."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

CollectionId = Literal[
    "cognitive_psychology",
    "dev_engineering",
    "design_ui",
    "ui_inspiration",
    "app_patterns",
    "app_ideas",
    "templates",
    "web",
    "wellbeing",
    "career_work",
    "learning_study",
]

COLLECTION_IDS: tuple[str, ...] = (
    "cognitive_psychology",
    "dev_engineering",
    "design_ui",
    "ui_inspiration",
    "app_patterns",
    "app_ideas",
    "templates",
    "web",
    "wellbeing",
    "career_work",
    "learning_study",
)

SplitName = Literal["train", "validation", "test"]
BlockKind = Literal["note", "link"]
ContentStatus = Literal["note", "read", "not_read"]
Language = Literal["pt", "en"]
Difficulty = Literal["easy", "hard"]


class Collection(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: CollectionId
    name: str
    includes: str
    excludes: str


class CollectionsFile(BaseModel):
    version: int
    collections: list[Collection]


class Preview(BaseModel):
    model_config = ConfigDict(frozen=True)

    site_name: str | None = None
    title: str | None = None
    description: str | None = None


class Block(BaseModel):
    """One saved item of the second-brain app (one JSONL line of the dataset)."""

    model_config = ConfigDict(frozen=True)

    id: str
    split: SplitName
    kind: BlockKind
    user_title: str | None
    user_description: str | None
    url: str | None
    preview: Preview | None
    content: str | None
    content_status: ContentStatus
    lang: Language
    labels: list[CollectionId]
    difficulty: Difficulty
    group_id: str
    rationale: str

    @field_validator("labels")
    @classmethod
    def _labels_cardinality(cls, v: list[str]) -> list[str]:
        if len(v) > 2:
            raise ValueError(f"at most 2 labels per block, got {len(v)}")
        return v

    @model_validator(mode="after")
    def _kind_consistency(self) -> Block:
        if self.kind == "note":
            if self.url is not None or self.preview is not None:
                raise ValueError("a note must have url=null and preview=null")
            if self.content_status != "note":
                raise ValueError("a note must have content_status='note'")
        else:
            if not self.url:
                raise ValueError("a link must have a url")
            if self.content_status == "note":
                raise ValueError("a link cannot have content_status='note'")
        if self.content_status == "not_read" and self.content is not None:
            raise ValueError("content_status='not_read' requires content=null")
        if self.content_status in ("note", "read") and not self.content:
            raise ValueError(f"content_status='{self.content_status}' requires content")
        return self


class Prediction(BaseModel):
    labels: list[str] = []
    probs: dict[str, float] | None = None


class Record(BaseModel):
    """One measured request, appended to results/<run_id>/raw/<arm>.jsonl."""

    model_config = ConfigDict(extra="allow")

    run_id: str
    server_id: str | None
    model_id: str
    model_sha: str | None
    arm: Literal["A", "B", "C"]
    arm_b_impl: str | None
    rep: int
    item_id: str
    latency_ms: float
    pred: list[str]
    probs: dict[str, float] | None
    label_mass: float | None
    usage: dict[str, int] | None
    finish_reason: str | None
    format_error: bool
    error: bool


def record_from_raw(raw: dict[str, Any]) -> Record:
    """Coerce one parsed JSONL line into a Record, tolerating extra fields."""
    return Record.model_validate(raw)
