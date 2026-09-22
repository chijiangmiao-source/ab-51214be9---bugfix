"""FastAPI application exposing the linearizability audit endpoints."""

from __future__ import annotations

from fastapi import FastAPI

from . import __version__
from .models import CheckRequest, CheckResponse, StepOut
from .solver import KIND_CAS, Operation, solve

app = FastAPI(
    title="Register Linearizability Audit Service",
    version=__version__,
    description=(
        "Audits whether a set of completed register operations admits a total "
        "order that respects real-time precedence and single-copy register "
        "semantics."
    ),
)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    """Liveness/readiness probe."""
    return {"status": "ok"}


@app.post(
    "/api/v1/linearizability/check",
    response_model=CheckResponse,
    tags=["audit"],
)
def check_linearizability(request: CheckRequest) -> CheckResponse:
    """Decide whether the submitted history is linearizable."""
    operations = [
        Operation(
            identifier=op.id,
            invoke=op.invoke,
            respond=op.respond,
            kind=KIND_CAS if op.type == "compare-and-swap" else op.type,
            value=op.value,
            expected=op.expected,
            update=op.update,
            success=op.success,
        )
        for op in request.operations
    ]
    result = solve(request.initial_value, operations)

    if not result.linearizable:
        return CheckResponse(
            linearizable=False,
            unique=None,
            order=None,
            steps=None,
            message=(
                "not linearizable: no total order respects both the real-time "
                "precedence constraints and the register semantics"
            ),
        )

    assert result.order is not None and result.steps is not None
    return CheckResponse(
        linearizable=True,
        unique=result.unique,
        order=list(result.order),
        steps=[StepOut(id=s.identifier, before=s.before, after=s.after) for s in result.steps],
        message=(
            "linearizable: order is the lexicographically smallest valid "
            "total order (by operation identifier)"
            + (" and it is unique" if result.unique else " and it is not unique")
        ),
    )
