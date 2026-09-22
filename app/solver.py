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

Search strategy
---------------

Every operation is its own search edge: concurrent writes of the same value
may be needed at different positions (one can "restore" a value after a
later CAS), so operations are never collapsed into a single forced batch.

Two sound prunings keep the enumeration of 24-operation histories fast:

1. *Automorphic operations.* Operations with identical register behaviour
   (writes of the same value; reads of the same value; CAS with the same
   expected/update/success triple) *and* identical real-time predecessor and
   successor sets are interchangeable: swapping their labels turns one valid
   order into another. Among the currently enabled members of such a class
   only the smallest identifier is branched on.
2. *Idle no-ops.* An enabled operation that is valid at the current value and
   leaves it unchanged -- a read observing the current value, a failed CAS,
   or a successful CAS whose update equals the current value.  Given any
   solution that starts with another such no-op q, moving the smallest one p
   to the front yields another valid solution: p is enabled now and the swap
   changes no observed value, so while p remains it is the only idle no-op
   that needs a branch here.  A write is never treated this way: it changes
   the value even when the written value happens to equal the current one.

Whenever a branch stands in for one or more skipped siblings, any solution
reached through it has distinct companion orders (label swaps or no-op
repositionings); this multiplicity is propagated to the leaf so the
uniqueness verdict stays exact. The first complete order visited is the
lexicographically smallest; dead-end ``(remaining operations, register
value)`` states are memoized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

KIND_WRITE = "write"
KIND_READ = "read"
KIND_CAS = "cas"

# Internal transition codes.
_T_WRITE = 0
_T_READ = 1
_T_CAS = 2


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


def _predecessor_masks(ops: Sequence[Operation]) -> tuple[list[int], list[int], bool]:
    """Compute transitive real-time predecessor and successor bitmasks.

    ``pred[j]`` has bit ``i`` set iff operation ``i`` must precede operation
    ``j`` (directly or transitively) because ``respond(i) <= invoke(j)``;
    ``succ`` is the inverse relation. Also reports whether the precedence
    relation is cyclic, which makes any total order impossible.
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
    succ = [0] * n
    for i in range(n):
        m = pred[i]
        while m:
            low = m & -m
            j = low.bit_length() - 1
            succ[j] |= 1 << i
            m ^= low
    return pred, succ, cyclic


def solve(initial_value: int, operations: Sequence[Operation]) -> SolveResult:
    """Search for a valid linearization of ``operations``.

    Returns the lexicographically smallest valid total order (comparing
    operation identifiers as strings, position by position) together with
    the per-step register values, and whether that order is the only valid
    one. When no valid order exists, no partial order is produced.
    """
    n = len(operations)
    pred, succ, cyclic = _predecessor_masks(operations)
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

    # Precompute per-operation transitions as parallel arrays:
    #   write -> (_T_WRITE, new value index, 0)
    #   read  -> (_T_READ, required value index, 0)
    #   cas   -> (_T_CAS, expected value index, update value index)
    # A required/expected index of -1 never matches any register state.
    kinds = [0] * n
    param_a = [0] * n
    param_b = [0] * n
    for i, op in enumerate(operations):
        if op.kind == KIND_WRITE:
            kinds[i] = _T_WRITE
            param_a[i] = index_of[op.value]  # type: ignore[index]
        elif op.kind == KIND_READ:
            required = index_of.get(op.value, -1)
            if required == -1:
                # Reads a value the register can never hold: no linearization.
                return SolveResult(False, None, None, None)
            kinds[i] = _T_READ
            param_a[i] = required
        else:
            expected = index_of.get(op.expected, -1)
            if op.success and expected == -1:
                # A successful CAS whose expected value can never occur.
                return SolveResult(False, None, None, None)
            kinds[i] = _T_CAS
            param_a[i] = expected
            param_b[i] = index_of[op.update]  # type: ignore[index]

    # Automorphism classes: identical register behaviour *and* identical
    # real-time neighbourhood. Swapping two members preserves both the
    # register semantics and every precedence constraint, so they are fully
    # interchangeable in any total order.
    class_of = [0] * n
    class_keys: dict[tuple, int] = {}
    for i, op in enumerate(operations):
        if op.kind == KIND_WRITE:
            behaviour = (KIND_WRITE, op.value)
        elif op.kind == KIND_READ:
            behaviour = (KIND_READ, op.value)
        else:
            behaviour = (KIND_CAS, op.expected, op.update, op.success)
        key = (behaviour, pred[i], succ[i])
        if key not in class_keys:
            class_keys[key] = len(class_keys)
        class_of[i] = class_keys[key]

    # Candidates are tried in ascending identifier order at every position,
    # so the first complete order found is the lexicographically smallest.
    index_order = sorted(range(n), key=lambda i: operations[i].identifier)
    bits = [1 << i for i in range(n)]
    full_mask = (1 << n) - 1

    # Per-operation value masks (bits are compressed value indices):
    #   install_mask[i] -- values executing i may leave behind (writes and
    #     successful CAS; reads/failed CAS install nothing), and
    #   accept_mask[i]  -- values i may observe (reads: the required value;
    #     successful CAS: its expected value; failed CAS: anything else).
    all_values = (1 << width) - 1
    install_mask = [0] * n
    accept_mask = [0] * n
    for i, op in enumerate(operations):
        kind = kinds[i]
        if kind == _T_WRITE:
            install_mask[i] = 1 << param_a[i]
        elif kind == _T_READ:
            accept_mask[i] = 1 << param_a[i]
        elif op.success:
            install_mask[i] = 1 << param_b[i]
            accept_mask[i] = all_values if param_a[i] == -1 else 1 << param_a[i]
        else:
            accept_mask[i] = all_values if param_a[i] == -1 else all_values ^ (1 << param_a[i])

    # Bitmasks of all installers (writes and successful CAS) and of all
    # value observers (reads and CAS).
    installer_bits = 0
    observer_bits = 0
    for i in range(n):
        if install_mask[i]:
            installer_bits |= bits[i]
        if accept_mask[i]:
            observer_bits |= bits[i]

    # Epoch accounting (a second relaxation, complementary to the observable
    # check below).  Visiting a value v starts an "epoch": the register holds
    # v from an entry (a write of v, or a successful CAS into v from another
    # value) until an exit (a successful CAS out of v).  Every destructive
    # CAS consumes a distinct epoch, and a read (or a successful v->v CAS)
    # needs at least one epoch.  Comparing those demands with the remaining
    # entries (plus the epoch the register is currently in) is necessary for
    # any continuation, regardless of real-time constraints.
    entry_target = [-1] * n  # value entered by executing i
    exit_source = [-1] * n  # value left by executing i (destructive CAS)
    epoch_observer = [-1] * n  # value whose epoch must exist (read / v->v CAS)
    for i, op in enumerate(operations):
        kind = kinds[i]
        if kind == _T_WRITE:
            entry_target[i] = param_a[i]
        elif kind == _T_READ:
            epoch_observer[i] = param_a[i]
        elif op.success:
            if param_a[i] == param_b[i]:
                epoch_observer[i] = param_a[i]  # successful CAS v -> v
            else:
                entry_target[i] = param_b[i]
                exit_source[i] = param_a[i]  # -1 when v can never occur

    def state_forward_feasible(remaining: int, value_idx: int) -> bool:
        """Relaxed checks that a satisfying continuation is still possible.

        1. Every pending observer can still observe an accepted value: the
           current value (if no installer is forced before it) plus the
           installed value of every remaining installer j that could be the
           last installer before it -- j not forced after the observer and
           with no forced predecessor installer of the observer after j.
        2. For every value, the remaining destructive CAS exits cannot
           outnumber the available epochs (entries into the value, plus the
           epoch the register currently occupies), and a value with pending
           observers but no destructive exit still needs one epoch.
        """
        current_bit = 1 << value_idx
        pending_obs = observer_bits & remaining
        while pending_obs:
            low = pending_obs & -pending_obs
            i = low.bit_length() - 1
            pending_obs ^= low

            blockers = pred[i] & remaining & installer_bits
            observable = current_bit if not blockers else 0
            candidates = (
                installer_bits & remaining & ~low & ~succ[i]
            )
            while candidates:
                clow = candidates & -candidates
                j = clow.bit_length() - 1
                candidates ^= clow
                if not (succ[j] & blockers):
                    observable |= install_mask[j]

            if not (observable & accept_mask[i]):
                return False

        entries = [0] * width
        exits = [0] * width
        observed = [False] * width
        m = remaining
        while m:
            low = m & -m
            i = low.bit_length() - 1
            m ^= low
            tgt = entry_target[i]
            if tgt != -1:
                entries[tgt] += 1
            src = exit_source[i]
            if src != -1:
                exits[src] += 1
            obs = epoch_observer[i]
            if obs != -1:
                observed[obs] = True

        for v in range(width):
            needed = exits[v] + (1 if observed[v] and exits[v] == 0 else 0)
            supply = entries[v] + (1 if v == value_idx else 0)
            if needed > supply:
                return False
        return True

    infeasible: set[int] = set()  # dead-end states: mask * width + value index
    first: list[int] = []  # first solution found (lexicographically smallest)
    first_ambiguous = False  # first path stood in for a skipped sibling
    found = 0  # distinct valid orders accounted for, capped at 2
    path: list[int] = []

    def dfs(remaining: int, value_idx: int, ambiguous: bool = False) -> None:
        nonlocal found, first_ambiguous
        if remaining == 0:
            if not first:
                first.extend(path)
                first_ambiguous = ambiguous
            found = 2 if (ambiguous or found >= 1) else 1
            return

        key = remaining * width + value_idx
        if key in infeasible:
            return
        if not state_forward_feasible(remaining, value_idx):
            infeasible.add(key)
            return

        # Build the branches in ascending identifier order.  Two sound ways
        # of collapsing skipped siblings into a representative branch:
        #
        # * Automorphism twins: enabled members of the same class (identical
        #   register behaviour and precedence neighbourhood) of any kind are
        #   label swaps of one another; only the smallest is branched on.
        # * Idle no-ops: enabled operations valid at the current value that
        #   leave it unchanged -- reads observing the current value, failed
        #   CAS operations, and successful CAS whose update equals the
        #   current value.  In any solution beginning with another such
        #   no-op q, swapping the smallest one p ahead of q keeps both valid
        #   (neither observes a different value), so while p remains it is
        #   the only idle no-op needing a branch here.  Writes are excluded:
        #   they change the value even when it equals the written one.
        #
        # Either collapse marks the branch ambiguous: its companion order is
        # distinct, so a leaf reached through it proves non-uniqueness alone.
        branches: list[tuple[int, int, bool]] = []
        class_rep: dict[int, int] = {}
        class_count: dict[int, int] = {}
        idle_rep: int = -1
        idle_count = 0

        for i in index_order:
            bit = bits[i]
            if not remaining & bit:
                continue
            if pred[i] & remaining:
                continue  # a real-time predecessor is not placed yet

            kind = kinds[i]
            if kind == _T_READ:
                if value_idx != param_a[i]:
                    continue  # the read would observe a different value
                target = value_idx
                idle = True
            elif kind == _T_WRITE:
                target = param_a[i]
                idle = False
            else:
                expected = param_a[i]
                if value_idx == expected:
                    if operations[i].success is not True:
                        continue
                    target = param_b[i]
                    idle = target == value_idx
                else:
                    if operations[i].success is not False:
                        continue
                    target = value_idx
                    idle = True

            cls = class_of[i]
            class_count[cls] = class_count.get(cls, 0) + 1
            if cls in class_rep:
                continue  # represented by the smaller class twin
            if idle:
                idle_count += 1
                if idle_rep != -1:
                    continue  # represented by the smallest idle no-op
                idle_rep = i
            class_rep[cls] = i
            branches.append((i, target, False))

        # Propagate the skipped-sibling multiplicity onto representatives.
        for pos, (i, target, _multi) in enumerate(branches):
            multi = class_count[class_of[i]] > 1 or (
                i == idle_rep and idle_count > 1
            )
            if multi:
                branches[pos] = (i, target, True)

        feasible_child = False
        for i, next_value, multi in branches:
            path.append(i)
            before = found
            dfs(remaining ^ bits[i], next_value, ambiguous or multi)
            path.pop()
            if found > before:
                feasible_child = True
            if found >= 2:
                # Distinct orders exist; uniqueness is settled. Do not mark
                # this state infeasible: remaining branches were not explored.
                return

        if not feasible_child:
            infeasible.add(key)

    dfs(full_mask, index_of[initial_value])

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
        unique=found == 1 and not first_ambiguous,
    )
