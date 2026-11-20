"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_QUESTION_CHARS = 500


class AskRequest(BaseModel):
    parcel: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)

    @field_validator("question", "parcel")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v.strip()


class ReviewRequest(BaseModel):
    action: Literal["approve", "edit", "reject"]
    text: str | None = Field(default=None, max_length=5000)

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str | None) -> str | None:
        return v.strip() if v else v


class Health(BaseModel):
    status: str
    documents: int
    chunks: int
    gemini: bool


class Property(BaseModel):
    parcel: str
    address: str | None
    town: str | None = None
    current_owner: str | None
    documents: int


class ChainEntry(BaseModel):
    doc_id: str
    title: str
    year: int | None
    type: str | None
    date_recorded: str | None = None
    num_pages: int | None = None
    sellers: list[str]
    buyers: list[str]
    price_in_deed: float | None
    price_reported: float | None
    transfer_tax: float | None = None
    nominal: bool
    previous: str | None = None
