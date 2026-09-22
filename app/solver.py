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


# Memoized verdict for a state; zero doubles as the bytearray "unknown".
_UNKNOWN = 0
_INFEASIBLE = 1  # zero valid total orders
_ONE = 2  # exactly one valid total order
_MANY = 3  # two or more valid total orders

# Largest contiguous memo table to allocate (32 MiB, one byte per state);
# larger state spaces (many distinct symmetry classes) fall back to a sparse
# dict. The sparse case stays within a few percent of the table performance.
_MEMO_TABLE_LIMIT = 1 << 25


def solve(initial_value: int, operations: Sequence[Operation]) -> SolveResult:
    """Search for a valid linearization of ``operations``.

    Returns the lexicographically smallest valid total order (comparing
    operation identifiers as strings, position by position) together with
    the per-step register values, and whether that order is the only valid
    one. When no valid order exists, no partial order is produced.
    """
    n = len(operations)
    if n == 0:
        return SolveResult(True, (), (), True)

    pred, cyclic = _predecessor_masks(operations)
    if cyclic:
        return SolveResult(False, None, None, None)

    # Work in ascending identifier rank order: trying candidates rank by
    # rank makes the first complete order the lexicographically smallest.
    ranks = sorted(range(n), key=lambda i: operations[i].identifier)
    rank_of = [0] * n
    for rank, original in enumerate(ranks):
        rank_of[original] = rank
    ops = [operations[ranks[r]] for r in range(n)]

    rank_pred = [0] * n
    rank_succ = [0] * n
    for j in range(n):
        bits = pred[ranks[j]]
        while bits:
            lsb = bits & -bits
            i = lsb.bit_length() - 1
            bits ^= lsb
            rank_pred[j] |= 1 << rank_of[i]
            rank_succ[rank_of[i]] |= 1 << j

    # The register only ever holds the initial value, a written value, or a
    # CAS update value, so compress the value domain to small indices.
    domain = {initial_value}
    for op in ops:
        if op.kind == KIND_WRITE:
            domain.add(op.value)  # type: ignore[arg-type]
        elif op.kind == KIND_CAS:
            domain.add(op.update)  # type: ignore[arg-type]
    index_of = {v: i for i, v in enumerate(sorted(domain))}
    width = len(index_of)

    # transitions[i][v] is the register value index after executing op i at
    # value v, plus one; zero means the observed result contradicts v.
    transitions: list[list[int]] = []
    impossible = False
    for op in ops:
        row = [0] * width
        if op.kind == KIND_WRITE:
            target = index_of[op.value]  # type: ignore[index]
            for v in range(width):
                row[v] = target + 1
        elif op.kind == KIND_READ:
            required = index_of.get(op.value, -1)
            if required == -1:
                impossible = True
            else:
                row[required] = required + 1
        else:
            expected = index_of.get(op.expected, -1)
            target = index_of[op.update]  # type: ignore[index]
            if op.success and expected == -1:
                impossible = True
            for v in range(width):
                if op.success:
                    if v == expected:
                        row[v] = target + 1
                elif v != expected:
                    row[v] = v + 1
        transitions.append(row)
    if impossible:
        return SolveResult(False, None, None, None)

    # Symmetry classes. Two operations are interchangeable exactly when they
    # have the same transition table, the same observed result and the same
    # real-time relationships to every other operation. Because predecessor
    # and successor masks are part of the signature, a dependency on one
    # member of another class is always a dependency on all of them, so a
    # state is fully described by how many members of each class are still
    # unplaced (plus the current value): which particular labels remain never
    # affects feasibility or the number of completions.
    class_keys: dict[tuple, int] = {}
    class_of = [0] * n
    for i, op in enumerate(ops):
        if op.kind == KIND_WRITE:
            signature: tuple = ("w", tuple(transitions[i]))
        elif op.kind == KIND_READ:
            signature = ("r", tuple(transitions[i]))
        else:
            signature = ("c", tuple(transitions[i]), op.success is True)
        key = (signature, rank_pred[i], rank_succ[i])
        cid = class_keys.get(key)
        if cid is None:
            cid = len(class_keys)
            class_keys[key] = cid
        class_of[i] = cid

    class_count = len(class_keys)
    members = [0] * class_count
    for i in range(n):
        members[class_of[i]] |= 1 << i
    member_totals = [mask.bit_count() for mask in members]
    class_bit = [1 << class_of[i] for i in range(n)]

    # Class-level predecessors: class c is blocked until every predecessor
    # class is fully placed (membership is all-or-nothing, see above).
    class_pred = [0] * class_count
    for c in range(class_count):
        rep = (members[c] & -members[c]).bit_length() - 1
        bits = rank_pred[rep]
        while bits:
            lsb = bits & -bits
            i = lsb.bit_length() - 1
            bits ^= lsb
            class_pred[c] |= 1 << class_of[i]

    # Mixed-radix key of the per-class placed counts. Every removal of a
    # member adds ``stride[c]``; states differing only by a permutation of
    # interchangeable labels therefore share one memo entry.
    strides = [0] * class_count
    combos = 1
    for c in range(class_count):
        strides[c] = combos
        combos *= member_totals[c] + 1

    applicable = [0] * width
    for v in range(width):
        mask = 0
        for i in range(n):
            if transitions[i][v]:
                mask |= 1 << i
        applicable[v] = mask

    # Necessary-condition prune. A value-dependent operation (read or
    # successful CAS) that is already enabled while the register holds a
    # different value can only be rescued by some still-unplaced operation
    # that installs its required value and that real-time ordering allows in
    # front of it. ``rescuers[i]`` lists (required value index, such
    # producers); with none left the current state is a dead end.
    producers = [0] * width
    for i, op in enumerate(ops):
        if op.kind == KIND_WRITE:
            producers[index_of[op.value]] |= 1 << i  # type: ignore[index]
        elif op.kind == KIND_CAS and op.success:
            producers[index_of[op.update]] |= 1 << i  # type: ignore[index]
    rescuers: list[Optional[tuple[int, int]]] = [None] * n
    for i, op in enumerate(ops):
        forced_after = 0
        for p in range(n):
            if rank_pred[p] & (1 << i):
                forced_after |= 1 << p
        if op.kind == KIND_READ:
            needed = next(v for v in range(width) if transitions[i][v])
            rescuers[i] = (needed, producers[needed] & ~forced_after)
        elif op.kind == KIND_CAS and op.success:
            needed = next(
                (v for v in range(width)
                 if transitions[i][v] and transitions[i][v] - 1 != v),
                -1,
            )
            if needed != -1:
                rescuers[i] = (needed, producers[needed] & ~forced_after)

    table_size = combos * width
    memo: bytearray | dict[int, int]
    if table_size <= _MEMO_TABLE_LIMIT:
        memo = bytearray(table_size)

        def memo_get(key: int) -> int:
            return memo[key]  # type: ignore[index]

        def memo_set(key: int, verdict: int) -> None:
            memo[key] = verdict  # type: ignore[index]
    else:
        memo = {}

        def memo_get(key: int) -> int:
            return memo.get(key, _UNKNOWN)  # type: ignore[union-attr]

        def memo_set(key: int, verdict: int) -> None:
            memo[key] = verdict  # type: ignore[index]

    initial_idx = index_of[initial_value]
    full_mask = (1 << n) - 1
    path: list[int] = []
    first: Optional[list[int]] = None

    def dfs(placed_key: int, value_idx: int, remaining: int,
            classes_done: int) -> int:
        """Count valid completions, capped at two.

        Returns _INFEASIBLE / _ONE / _MANY. As a side effect the first leaf
        reached (the lexicographically smallest valid order) is recorded.
        """
        nonlocal first
        if remaining == 0:
            if first is None:
                first = path.copy()
            return _ONE

        key = placed_key * width + value_idx
        cached = memo_get(key)
        if cached:
            return cached

        enabled = remaining
        while enabled:
            lsb = enabled & -enabled
            i = lsb.bit_length() - 1
            enabled ^= lsb
            rescue = rescuers[i]
            if rescue is not None:
                needed, available = rescue
                if needed != value_idx and not (
                    available & remaining
                ) and not (rank_pred[i] & remaining):
                    memo_set(key, _INFEASIBLE)
                    return _INFEASIBLE

        total = 0
        seen_classes = 0
        candidates = remaining & applicable[value_idx]
        while candidates:
            lsb = candidates & -candidates
            i = lsb.bit_length() - 1
            candidates ^= lsb

            cid = class_of[i]
            cbit = class_bit[i]
            if seen_classes & cbit:
                # Smallest remaining member already offered this class.
                continue
            seen_classes |= cbit
            if class_pred[cid] & ~classes_done:
                continue

            remaining_in_class = members[cid] & remaining
            rep_bit = remaining_in_class & -remaining_in_class
            rep = rep_bit.bit_length() - 1
            multiplicity = remaining_in_class.bit_count()
            next_value = transitions[rep][value_idx] - 1
            done_now = classes_done | (
                cbit if multiplicity == 1 else 0
            )

            path.append(rep)
            child = dfs(
                placed_key + strides[cid],
                next_value,
                remaining ^ rep_bit,
                done_now,
            )
            path.pop()

            if child != _INFEASIBLE:
                if child == _MANY or multiplicity > 1:
                    # Multiple interchangeable labels may occupy this slot,
                    # or the continuation itself already branches.
                    total = 2
                    break
                total += 1
                if total >= 2:
                    total = 2
                    break

        verdict = _ONE if total == 1 else (
            _MANY if total >= 2 else _INFEASIBLE
        )
        memo_set(key, verdict)
        return verdict

    root = dfs(0, initial_idx, full_mask, 0)
    if root == _INFEASIBLE or first is None:
        return SolveResult(False, None, None, None)

    steps = []
    value = initial_value
    for i in first:
        ok, nxt = _apply(ops[i], value)
        assert ok  # the recorded path was validated during the search
        steps.append(Step(ops[i].identifier, value, nxt))
        value = nxt

    return SolveResult(
        linearizable=True,
        order=tuple(ops[i].identifier for i in first),
        steps=tuple(steps),
        unique=root == _ONE,
    )
