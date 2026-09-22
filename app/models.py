"""Request/response schemas with strict, reject-the-whole-request validation."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

MAX_OPERATIONS = 24

_WRITE_READ_FIELDS = ("value",)
_CAS_FIELDS = ("expected", "update", "success")


class OperationIn(BaseModel):
    """One completed operation submitted by the auditor.

    The set of payload fields must match ``type`` exactly; any inconsistency
    (missing field for the type, or a field belonging to another type)
    rejects the whole request.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    type: Literal["write", "read", "cas", "compare-and-swap"]
    invoke: StrictInt
    respond: StrictInt
    # write / read payload
    value: Optional[StrictInt] = None
    # compare-and-swap payload
    expected: Optional[StrictInt] = None
    update: Optional[StrictInt] = None
    success: Optional[StrictBool] = None

    @model_validator(mode="after")
    def _check_interval_and_fields(self) -> "OperationIn":
        if self.invoke > self.respond:
            raise ValueError(
                f"operation '{self.id}': invalid interval, "
                f"invoke ({self.invoke}) must be <= respond ({self.respond})"
            )
        if self.type in ("write", "read"):
            if self.value is None:
                raise ValueError(
                    f"operation '{self.id}': type '{self.type}' requires field 'value'"
                )
            stray = [f for f in _CAS_FIELDS if getattr(self, f) is not None]
            if stray:
                raise ValueError(
                    f"operation '{self.id}': type '{self.type}' must not carry "
                    f"compare-and-swap field(s) {stray}"
                )
        else:  # cas / compare-and-swap
            missing = [f for f in _CAS_FIELDS if getattr(self, f) is None]
            if missing:
                raise ValueError(
                    f"operation '{self.id}': compare-and-swap requires "
                    f"field(s) {missing}"
                )
            if self.value is not None:
                raise ValueError(
                    f"operation '{self.id}': compare-and-swap must not carry "
                    f"field 'value'"
                )
        return self


class CheckRequest(BaseModel):
    """Audit request: initial register value plus 1..24 completed operations."""

    model_config = ConfigDict(extra="forbid")

    initial_value: StrictInt
    operations: list[OperationIn] = Field(min_length=1, max_length=MAX_OPERATIONS)

    @model_validator(mode="after")
    def _check_unique_ids(self) -> "CheckRequest":
        ids = [op.id for op in self.operations]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate operation identifier(s): {duplicates}")
        return self


class StepOut(BaseModel):
    """Register value before and after one executed operation."""

    id: str
    before: int
    after: int


class CheckResponse(BaseModel):
    """Audit verdict.

    When ``linearizable`` is false, ``order``/``steps``/``unique`` are null:
    no partial order is ever fabricated.
    """

    linearizable: bool
    unique: Optional[bool] = None
    order: Optional[list[str]] = None
    steps: Optional[list[StepOut]] = None
    message: str
