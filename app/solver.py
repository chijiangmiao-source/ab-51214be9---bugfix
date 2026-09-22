"""Linearizability engine for a single integer register.

Given an initial register value and a set of completed operations (each with
an invocation time and a response time), this module searches for a total
order of the operations such that:

* real-time precedence is respected: if ``respond(A) <= invoke(B)`` then A
  must appear before B, and
* the sequential register semantics hold: writes always take effect, reads
  must return the current value, and a compare-and-swap success flag must
  agree with the current value.

The engine reports the lexicographically smallest valid total order (ordered
by operation identifier), the register value before/after every step, and
whether the valid total order is unique.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

KIND_WRITE = "write"
KIND_READ = "read"
KIND_CAS = "cas"


@dataclass(frozen=True)
class Operation:
    """A completed register operation with its real-time interval."""

    identifier: str
    invoke: int
    respond: int
    kind: str  # KIND_WRITE | KIND_READ | KIND_CAS
    value: Optional[int] = None  # write: written value / read: returned value
    expected: Optional[int] = None  # cas: expected current value
    update: Optional[int] = None  # cas: value installed on success
    success: Optional[bool] = None  # cas: reported success flag


@dataclass(frozen=True)
class Step:
    """One executed step of a valid total order."""

    identifier: str
    before: int
    after: int


@dataclass(frozen=True)
class SolveResult:
    """Outcome of the linearizability search."""

    linearizable: bool
    order: Optional[tuple[str, ...]]  # lexicographically smallest valid order
    steps: Optional[tuple[Step, ...]]  # per-step before/after register values
    unique: Optional[bool]  # True iff exactly one valid total order exists


def _apply(op: Operation, value: int) -> tuple[bool, int]:
    """Try to execute ``op`` against the current register ``value``.

    Returns ``(ok, new_value)``. ``ok`` is False when the operation's
    observed result (read value or CAS success flag) contradicts the
    current register value, in which case ``new_value`` is meaningless.
    """
    if op.kind == KIND_WRITE:
        return True, op.value  # type: ignore[return-value]
    if op.kind == KIND_READ:
        return value == op.value, value
    # compare-and-swap
    if value == op.expected:
        return op.success is True, op.update  # type: ignore[return-value]
    return op.success is False, value


def _predecessor_masks(ops: Sequence[Operation]) -> tuple[list[int], bool]:
    """Compute transitive real-time predecessor bitmasks.

    ``pred[j]`` has bit ``i`` set iff operation ``i`` must precede operation
    ``j`` (directly or transitively) because ``respond(i) <= invoke(j)``.
    Also reports whether the precedence relation is cyclic, which makes any
    total order impossible.
    """
    n = len(ops)
    pred = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and ops[i].respond <= ops[j].invoke:
                pred[j] |= 1 << i
    # Transitive closure (Floyd-Warshall over bitmasks).
    for k in range(n):
        bit_k = 1 << k
        for i in range(n):
            if pred[i] & bit_k:
                pred[i] |= pred[k]
    cyclic = any((pred[i] >> i) & 1 for i in range(n))
    return pred, cyclic


def solve(initial_value: int, operations: Sequence[Operation]) -> SolveResult:
    """Search for a valid linearization of ``operations``.

    Returns the lexicographically smallest valid total order (comparing
    operation identifiers as strings, position by position) together with
    the per-step register values, and whether that order is the only valid
    one. When no valid order exists, no partial order is produced.
    """
    n = len(operations)
    pred, cyclic = _predecessor_masks(operations)
    if cyclic:
        return SolveResult(False, None, None, None)

    # The register only ever holds the initial value, a written value, or a
    # CAS update value, so compress the value domain to small indices; memo
    # keys then fit in a single integer ``mask * width + value_index``.
    domain = {initial_value}
    for op in operations:
        if op.kind == KIND_WRITE:
            domain.add(op.value)  # type: ignore[arg-type]
        elif op.kind == KIND_CAS:
            domain.add(op.update)  # type: ignore[arg-type]
    index_of = {v: i for i, v in enumerate(sorted(domain))}
    width = len(index_of)

    # Precompute per-operation transitions against value indices.
    # Each entry is (kind, a, b):
    #   write -> (KIND_WRITE, new_value_index, None)
    #   read  -> (KIND_READ, required_value_index, None)
    #   cas   -> (KIND_CAS, expected_value_index, new_value_index)
    # A required/expected index of -1 never matches any register state.
    transitions: list[tuple[str, int, int]] = []
    for op in operations:
        if op.kind == KIND_WRITE:
            transitions.append((KIND_WRITE, index_of[op.value], 0))  # type: ignore[index]
        elif op.kind == KIND_READ:
            required = index_of.get(op.value, -1)
            if required == -1:
                # Reads a value the register can never hold: no linearization.
                return SolveResult(False, None, None, None)
            transitions.append((KIND_READ, required, 0))
        else:
            expected = index_of.get(op.expected, -1)
            if op.success and expected == -1:
                # A successful CAS whose expected value can never occur.
                return SolveResult(False, None, None, None)
            transitions.append((KIND_CAS, expected, index_of[op.update]))  # type: ignore[index]

    # Candidates are tried in ascending identifier order at every position,
    # so the first complete order found is the lexicographically smallest.
    index_order = sorted(range(n), key=lambda i: operations[i].identifier)

    infeasible: set[int] = set()  # dead-end states: mask * width + value index
    first: list[int] = []  # first solution found (lexicographically smallest)
    found = 0  # number of solutions seen, capped at 2
    path: list[int] = []

    def enabled_batches(remaining: int, value_idx: int) -> list[tuple[tuple[int, ...], int]]:
        """Return executable operations, compacting equal writes into one batch.

        A run of enabled writes that installs the same value has the same net
        register effect in every permutation.  Treating that run as one search
        edge avoids visiting every subset of a large collection of redundant
        writes while retaining the identifier order used for the witness.
        """
        write_groups: dict[int, list[int]] = {}
        batches: list[tuple[tuple[int, ...], int]] = []

        for i in index_order:
            bit = 1 << i
            if not remaining & bit or pred[i] & remaining:
                continue
            kind, a, b = transitions[i]
            if kind == KIND_WRITE:
                write_groups.setdefault(a, []).append(i)
                continue
            if kind == KIND_READ:
                if value_idx == a:
                    batches.append(((i,), value_idx))
                continue
            if value_idx == a:
                if operations[i].success is True:
                    batches.append(((i,), b))
            elif operations[i].success is False:
                batches.append(((i,), value_idx))

        for target, members in write_groups.items():
            batches.append((tuple(members), target))
        batches.sort(key=lambda item: operations[item[0][0]].identifier)
        return batches

    def dfs(remaining: int, value_idx: int, has_permutation: bool = False) -> None:
        nonlocal found
        if remaining == 0:
            if not first:
                first.extend(path)
            found = min(2, found + (2 if has_permutation else 1))
            return
        key = remaining * width + value_idx
        if key in infeasible:
            return
        feasible_child = False
        for batch, nxt in enabled_batches(remaining, value_idx):
            batch_mask = 0
            for i in batch:
                batch_mask |= 1 << i
            path.extend(batch)
            before = found
            dfs(
                remaining ^ batch_mask,
                nxt,
                has_permutation or len(batch) > 1,
            )
            del path[-len(batch):]
            if found > before:
                feasible_child = True
            if found >= 2:
                # Two distinct solutions suffice to decide uniqueness; do not
                # mark this state infeasible since it was not fully explored.
                return
        if not feasible_child:
            infeasible.add(key)

    dfs((1 << n) - 1, index_of[initial_value])

    if found == 0:
        return SolveResult(False, None, None, None)

    steps = []
    value = initial_value
    for i in first:
        ok, nxt = _apply(operations[i], value)
        assert ok  # the recorded path was validated during the search
        steps.append(Step(operations[i].identifier, value, nxt))
        value = nxt

    return SolveResult(
        linearizable=True,
        order=tuple(operations[i].identifier for i in first),
        steps=tuple(steps),
        unique=found == 1,
    )
