"""Pydantic schemas and parsing utilities for SLM-Router decision contracts."""

import json
import re
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class InvalidModelOutputError(ValueError):
    """Raised when the SLM produces malformed, incomplete, or invalid routing output."""
    pass


class MemoryRequest(BaseModel):
    keys: List[str] = Field(
        ...,
        min_length=1,
        description="List of memory keys required to answer the query",
    )

    @field_validator("keys")
    @classmethod
    def validate_keys(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("keys list cannot be empty")
        cleaned_keys = [k.strip() for k in v if isinstance(k, str) and k.strip()]
        if not cleaned_keys:
            raise ValueError("keys list must contain at least one non-empty string key")
        return cleaned_keys


class RouteDecision(BaseModel):
    processing: Literal["LOCAL", "CLOUD"] = Field(
        ...,
        description="Target destination for query execution ('LOCAL' or 'CLOUD')",
    )
    memory_required: bool = Field(
        ...,
        description="Whether answering the query requires stored personal user/child memory",
    )
    memory_request: Optional[MemoryRequest] = Field(
        default=None,
        description="Memory keys requested if memory_required is true; null otherwise",
    )

    @model_validator(mode="after")
    def validate_memory_consistency(self) -> "RouteDecision":
        if self.memory_required:
            if self.memory_request is None or not self.memory_request.keys:
                raise ValueError(
                    "memory_request with at least one key is required when memory_required is True"
                )
        else:
            if self.memory_request is not None:
                raise ValueError(
                    "memory_request must be null/None when memory_required is False"
                )
        return self

    def __str__(self) -> str:
        return self.processing

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.processing == other
        return super().__eq__(other)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


def extract_json_block(raw_text: str) -> str:
    """Extract a JSON object substring from raw text, handling markdown blocks or leading/trailing text."""
    cleaned = raw_text.strip()
    if not cleaned:
        raise InvalidModelOutputError("Model returned empty output")

    # Check for markdown code fences (e.g. ```json { ... } ```)
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    # Find the outermost JSON curly braces
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return cleaned[first_brace : last_brace + 1].strip()

    return cleaned


def parse_decision(raw_output: str) -> RouteDecision:
    """Parse and strictly validate raw model output against RouteDecision schema."""
    if not raw_output or not raw_output.strip():
        raise InvalidModelOutputError("Model returned empty or whitespace-only output")

    json_str = extract_json_block(raw_output)
    try:
        data = json.loads(json_str)
    except Exception as e:
        raise InvalidModelOutputError(
            f"Failed to decode JSON from model output: {e}. Raw text: {raw_output!r}"
        ) from e

    if not isinstance(data, dict):
        raise InvalidModelOutputError(
            f"Model output JSON must be an object/dict, got {type(data).__name__}: {raw_output!r}"
        )

    try:
        return RouteDecision.model_validate(data)
    except Exception as e:
        raise InvalidModelOutputError(
            f"Model output violates RouteDecision schema: {e}. Parsed payload: {data}"
        ) from e
