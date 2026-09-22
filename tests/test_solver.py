"""Unit tests for the linearizability engine."""

import time

import pytest

from app.solver import KIND_CAS, KIND_READ, KIND_WRITE, Operation, solve


def op(identifier, kind, invoke, respond, **payload):
    return Operation(
        identifier=identifier, invoke=invoke, respond=respond, kind=kind, **payload
    )


def write(identifier, value, invoke, respond):
    return op(identifier, KIND_WRITE, invoke, respond, value=value)


def read(identifier, value, invoke, respond):
    return op(identifier, KIND_READ, invoke, respond, value=value)


def cas(identifier, expected, update, success, invoke, respond):
    return op(
        identifier,
        KIND_CAS,
        invoke,
        respond,
        expected=expected,
        update=update,
        success=success,
    )


def replay(initial_value, operations, result):
    """Independently re-execute ``result.order`` and return its steps.

    Asserts that the engine's verdict is internally reproducible: every id
    occurs once, the before/after values form a consistent chain starting at
    the initial value, reads observe the current value, and every CAS
    success flag agrees with the value it would see.
    """
    by_id = {o.identifier: o for o in operations}
    assert result.order is not None and result.steps is not None
    assert len(result.order) == len(operations)
    assert sorted(result.order) == sorted(by_id)
    assert [s.identifier for s in result.steps] == list(result.order)

    value = initial_value
    replayed = []
    for step in result.steps:
        operation = by_id[step.identifier]
        assert step.before == value
        if operation.kind == KIND_WRITE:
            value = operation.value
        elif operation.kind == KIND_READ:
            assert operation.value == value
        else:
            succeeds = value == operation.expected
            assert succeeds is operation.success
            if succeeds:
                value = operation.update
        assert step.after == value
        replayed.append((step.before, step.after))
    return replayed


class TestBasicHistories:
    def test_single_write_is_linearizable_and_unique(self):
        result = solve(0, [write("w", 5, 0, 1)])
        assert result.linearizable
        assert result.order == ("w",)
        assert result.unique is True
        assert [(s.before, s.after) for s in result.steps] == [(0, 5)]

    def test_sequential_write_then_read(self):
        result = solve(0, [write("w", 7, 0, 2), read("r", 7, 3, 4)])
        assert result.linearizable
        assert result.order == ("w", "r")
        assert result.unique is True
        assert [(s.before, s.after) for s in result.steps] == [(0, 7), (7, 7)]

    def test_read_of_never_written_value_is_not_linearizable(self):
        result = solve(0, [read("r", 3, 0, 1)])
        assert not result.linearizable
        assert result.order is None and result.steps is None and result.unique is None

    def test_read_must_observe_the_latest_write(self):
        # w(1) finishes before r starts, and r returns the initial value 0.
        result = solve(0, [write("w", 1, 0, 2), read("r", 0, 3, 4)])
        assert not result.linearizable


class TestRealTimePrecedence:
    def test_response_equal_to_invoke_still_orders(self):
        # respond(w) == invoke(r): w must precede r, so r must see the write.
        result = solve(0, [write("w", 9, 0, 5), read("r", 9, 5, 8)])
        assert result.linearizable
        assert result.order == ("w", "r")

    def test_simultaneous_instantaneous_ops_form_a_cycle(self):
        # Both intervals are the same instant, so each must precede the other.
        result = solve(0, [write("a", 1, 4, 4), read("b", 0, 4, 4)])
        assert not result.linearizable

    def test_overlapping_ops_may_be_reordered(self):
        # r overlaps w, so r may be placed before w and still return 0.
        result = solve(0, [write("w", 1, 0, 10), read("r", 0, 2, 3)])
        assert result.linearizable
        assert result.order == ("r", "w")


class TestCompareAndSwap:
    def test_successful_cas_installs_update(self):
        result = solve(0, [cas("c", 0, 1, True, 0, 1), read("r", 1, 2, 3)])
        assert result.linearizable
        assert result.order == ("c", "r")
        assert [(s.before, s.after) for s in result.steps] == [(0, 1), (1, 1)]

    def test_failed_cas_keeps_value(self):
        result = solve(5, [cas("c", 0, 1, False, 0, 1), read("r", 5, 2, 3)])
        assert result.linearizable
        assert [(s.before, s.after) for s in result.steps] == [(5, 5), (5, 5)]

    def test_success_flag_contradicting_value_is_rejected_by_semantics(self):
        # Current value equals expected, so the CAS cannot report failure.
        result = solve(0, [cas("c", 0, 1, False, 0, 1)])
        assert not result.linearizable
        # Current value differs from expected, so the CAS cannot succeed.
        result = solve(0, [cas("c", 1, 2, True, 0, 1)])
        assert not result.linearizable

    def test_cas_success_with_update_equal_to_expected(self):
        result = solve(2, [cas("c", 2, 2, True, 0, 1)])
        assert result.linearizable
        assert [(s.before, s.after) for s in result.steps] == [(2, 2)]


class TestOrderSelectionAndUniqueness:
    def test_lexicographically_smallest_order_is_returned(self):
        # Two overlapping reads of the same value: both orders valid.
        ops = [read("b", 0, 0, 10), read("a", 0, 0, 10)]
        result = solve(0, ops)
        assert result.linearizable
        assert result.order == ("a", "b")
        assert result.unique is False

    def test_lexicographic_choice_respects_feasibility(self):
        # "a" (read 1) can only go after the write, so the smallest feasible
        # order starts with "b" even though "a" < "b".
        ops = [read("a", 1, 0, 10), write("b", 1, 0, 10)]
        result = solve(0, ops)
        assert result.linearizable
        assert result.order == ("b", "a")
        assert result.unique is True

    def test_unique_order_detected(self):
        ops = [
            write("w1", 1, 0, 2),
            read("r1", 1, 3, 4),
            cas("c1", 1, 2, True, 5, 6),
            read("r2", 2, 7, 8),
        ]
        result = solve(0, ops)
        assert result.linearizable
        assert result.order == ("w1", "r1", "c1", "r2")
        assert result.unique is True

    def test_non_unique_order_detected(self):
        # Two overlapping writes: either order is valid.
        ops = [write("a", 1, 0, 10), write("b", 2, 0, 10)]
        result = solve(0, ops)
        assert result.linearizable
        assert result.order == ("a", "b")
        assert result.unique is False

    def test_steps_cover_every_operation_once(self):
        ops = [
            write("w", 4, 0, 10),
            read("x", 0, 0, 10),
            cas("c", 4, 9, True, 0, 10),
            read("r", 9, 0, 10),
        ]
        result = solve(0, ops)
        assert result.linearizable
        assert sorted(result.order) == ["c", "r", "w", "x"]
        assert [s.identifier for s in result.steps] == list(result.order)
        # Steps form a consistent value chain starting at the initial value.
        assert result.steps[0].before == 0
        for left, right in zip(result.steps, result.steps[1:]):
            assert left.after == right.before


class TestReportedAuditErrors:
    """The two concrete regressions reported against the audit endpoint."""

    def test_long_write_restores_value_after_cas_for_later_read(self):
        # b-seed must precede c-advance, which must precede d-check; the
        # long-running a-restore slots in after the CAS so d-check reads 1.
        operations = [
            write("a-restore", 1, 0, 10),
            write("b-seed", 1, 0, 1),
            cas("c-advance", 1, 2, True, 1, 2),
            read("d-check", 1, 2, 3),
        ]
        result = solve(0, operations)
        assert result.linearizable is True
        assert result.order == (
            "b-seed",
            "c-advance",
            "a-restore",
            "d-check",
        )
        assert result.unique is True
        assert replay(0, operations, result) == [(0, 1), (1, 2), (2, 1), (1, 1)]

    def test_three_fully_overlapping_ops_have_four_valid_orders(self):
        # a-write/b-read/z-write all overlap; the read needs some write
        # before it, giving exactly four legal orders; the lexicographic
        # verdict of the complete set is a-write, b-read, z-write.
        operations = [
            write("a-write", 1, 0, 10),
            read("b-read", 1, 0, 10),
            write("z-write", 1, 0, 10),
        ]
        result = solve(0, operations)
        assert result.linearizable is True
        assert result.order == ("a-write", "b-read", "z-write")
        assert result.unique is False
        assert replay(0, operations, result) == [(0, 1), (1, 1), (1, 1)]

    def test_same_value_concurrent_writes_are_not_forced_adjacent(self):
        # The second write must be available *after* the CAS as a reset even
        # though an identical write seeded the CAS before it.  The seed is a
        # forced predecessor (respond 0 <= every other invoke); the first
        # reset may precede or follow the CAS, so the history is not unique,
        # but the lexicographically smallest order seeds and advances first.
        operations = [
            write("seed", 1, 0, 0),
            cas("advance", 1, 2, True, 1, 2),
            write("reset-1", 1, 1, 10),
            write("reset-2", 1, 3, 10),
            read("observe", 1, 4, 5),
        ]
        result = solve(0, operations)
        assert result.linearizable is True
        # After the CAS a same-value reset restores 1 so the read can run;
        # the remaining identical write is free ("observe" < "reset-2").
        assert result.order == (
            "seed",
            "advance",
            "reset-1",
            "observe",
            "reset-2",
        )
        assert result.unique is False
        assert replay(0, operations, result) == [
            (0, 1), (1, 2), (2, 1), (1, 1), (1, 1)
        ]


# ---------------------------------------------------------------------------
# Boundary histories: 20..24 operations mixing equal-value concurrent writes
# with later reads/CAS.  Every entry is (family, size, initial, ops,
# expected_linearizable, expected_unique).  The solver must finish well within
# SOLVER_DEADLINE_S without shrinking the supported operation count.
# ---------------------------------------------------------------------------

SOLVER_DEADLINE_S = 5.0


def _tail_reads(size, prefix, start=10):
    """Sequential reads of 1 appended after the interleaved core."""
    return prefix + [
        read(f"tail{k:02d}", 1, start + 2 * k, start + 2 * k + 1)
        for k in range(size - len(prefix))
    ]


def family_a(size):
    # The reported unique order, extended with trailing sequential reads.
    prefix = [
        write("b-seed", 1, 0, 1),
        cas("c-advance", 1, 2, True, 1, 2),
        write("a-restore", 1, 0, 10),
        read("d-check", 1, 2, 3),
    ]
    return 0, _tail_reads(size, prefix), True, True


def family_b(size):
    # The reported four-order history, extended with trailing reads.
    prefix = [
        write("a-write", 1, 0, 10),
        read("b-read", 1, 0, 10),
        write("z-write", 1, 0, 10),
    ]
    return 0, _tail_reads(size, prefix), True, False


def family_c(size):
    # Reads of 2 must all run inside the single CAS epoch, so the
    # lexicographically smaller a-restore has to be postponed until after
    # them even though it is enabled right after the CAS.
    ops = [
        write("b-seed", 1, 0, 1),
        cas("c-advance", 1, 2, True, 1, 2),
        write("a-restore", 1, 0, 10),
        read("d-check", 1, 2, 3),
    ]
    ops += [read(f"e-r2-{k:02d}", 2, 0, 100) for k in range(size - 4)]
    return 0, ops, True, False


def family_d(size):
    # Epoch accounting on the edge: destructive CAS 1->2 versus entries into 1.
    counts = {20: (6, 7, 7), 22: (7, 7, 8), 23: (7, 8, 8), 24: (8, 9, 7)}
    writes, exits, reads2 = counts[size]
    ops = [write(f"w1-{k:02d}", 1, 0, 100) for k in range(writes)]
    ops += [cas(f"x-{k:02d}", 1, 2, True, 0, 100) for k in range(exits)]
    ops += [read(f"r2-{k:02d}", 2, 0, 100) for k in range(reads2)]
    return 0, ops, size == 22, None if size != 22 else False


def family_e(size):
    # Equal-value writes plus failed CAS (a no-op wherever the value differs)
    # and reads of the written value, all fully overlapping.
    writers = size // 3
    failed = (size - writers) // 2
    readers = size - writers - failed
    ops = [write(f"w-{k:02d}", 1, 0, 100) for k in range(writers)]
    ops += [cas(f"f-{k:02d}", 2, 9, False, 0, 100) for k in range(failed)]
    ops += [read(f"r-{k:02d}", 1, 0, 100) for k in range(readers)]
    return 1, ops, True, False


def family_f(size):
    # Two identical long-running restore writes around the CAS.
    prefix = [
        write("b-seed", 1, 0, 1),
        cas("c-advance", 1, 2, True, 1, 2),
        write("a1-restore", 1, 0, 10),
        write("a2-restore", 1, 0, 10),
        read("d-check", 1, 2, 3),
    ]
    return 0, _tail_reads(size, prefix), True, False


def family_g(size):
    # Successful v->v CAS (an epoch observer, not an installer) mixed with
    # ordinary writes, failed CAS and reads.
    writers, failed, self_cas = 8, 6, 4
    ops = [write(f"w-{k:02d}", 1, 0, 100) for k in range(writers)]
    ops += [cas(f"f-{k:02d}", 2, 9, False, 0, 100) for k in range(failed)]
    ops += [cas(f"s-{k:02d}", 1, 1, True, 0, 100) for k in range(self_cas)]
    ops += [read(f"r-{k:02d}", 1, 0, 100)
            for k in range(size - writers - failed - self_cas)]
    return 1, ops, True, False


def family_h(size):
    # respond(d-check) == invoke(a-restore) forces d-check before the
    # restore, so after the CAS nothing can make d-check read 1: no total
    # order exists, and no partial order may be returned.
    ops = [
        write("b-seed", 1, 0, 1),
        cas("c-advance", 1, 2, True, 1, 2),
        write("a-restore", 1, 3, 10),
        read("d-check", 1, 2, 3),
    ]
    ops += [read(f"e-r2-{k:02d}", 2, 0, 100) for k in range(size - 4)]
    return 0, ops, False, None


BOUNDARY_CASES = (
    [("A", size, family_a(size)) for size in range(20, 25)]
    + [("B", size, family_b(size)) for size in range(20, 25)]
    + [("C", size, family_c(size)) for size in range(20, 25)]
    + [("D", size, family_d(size)) for size in (20, 22, 23, 24)]
    + [("E", size, family_e(size)) for size in (22, 24)]
    + [("F", size, family_f(size)) for size in (20, 21, 23)]
    + [("G", size, family_g(size)) for size in (23, 24)]
    + [("H", size, family_h(size)) for size in (20, 22, 24)]
)


class TestBoundaryHistories:
    @pytest.mark.parametrize(
        "family,size,case",
        BOUNDARY_CASES,
        ids=[f"family-{f}-n{s}" for f, s, _ in BOUNDARY_CASES],
    )
    def test_boundary_history(self, family, size, case):
        assert 20 <= size <= 24
        initial, operations, linearizable, unique = case
        assert len(operations) == size

        started = time.perf_counter()
        result = solve(initial, operations)
        elapsed = time.perf_counter() - started
        assert elapsed < SOLVER_DEADLINE_S, f"solver took {elapsed:.3f}s"

        assert result.linearizable is linearizable
        if not linearizable:
            # Explicit "no solution": never fabricate a partial order.
            assert result.order is None
            assert result.steps is None
            assert result.unique is None
            return

        assert result.unique is unique
        replay(initial, operations, result)

        if family == "A":
            assert result.order[:4] == (
                "b-seed",
                "c-advance",
                "a-restore",
                "d-check",
            )
        elif family == "B":
            assert result.order[:3] == ("a-write", "b-read", "z-write")
        elif family == "C":
            assert result.order[:2] == ("b-seed", "c-advance")
            assert result.order[-2:] == ("a-restore", "d-check")
        elif family == "H":  # pragma: no cover - guarded by the branch above
            pytest.fail("family H must be non-linearizable")


class TestSolverVsBruteForceOracle:
    """Differential checks against an exhaustive permutation oracle.

    These guard every pruning rule: the optimized search must keep the exact
    linearizability / uniqueness / lexicographic-minimum verdicts.
    """

    @staticmethod
    def _brute_force(initial_value, operations):
        """Enumerate every precedence-respecting permutation."""
        import itertools

        size = len(operations)
        pred = [0] * size
        for i in range(size):
            for j in range(size):
                if i != j and operations[i].respond <= operations[j].invoke:
                    pred[j] |= 1 << i

        count = 0
        lex_min = None
        for perm in itertools.permutations(range(size)):
            placed = 0
            value = initial_value
            ok = True
            for i in perm:
                if pred[i] & ~placed:
                    ok = False
                    break
                operation = operations[i]
                if operation.kind == KIND_WRITE:
                    value = operation.value
                elif operation.kind == KIND_READ:
                    if value != operation.value:
                        ok = False
                        break
                else:
                    if value == operation.expected:
                        if not operation.success:
                            ok = False
                            break
                        value = operation.update
                    elif operation.success:
                        ok = False
                        break
                placed |= 1 << i
            if ok:
                count += 1
                order = tuple(operations[i].identifier for i in perm)
                lex_min = order if lex_min is None else min(lex_min, order)
        return count, lex_min

    @pytest.mark.parametrize("seed", range(40))
    def test_random_small_histories_match_oracle(self, seed):
        import random

        rng = random.Random(9000 + seed)
        size = rng.randint(1, 7)
        identifiers = rng.sample([f"id{value:04d}" for value in range(200)], size)
        if seed % 2 == 0:
            invokes = [0] * size
            responds = [10] * size
        else:
            invokes = [rng.randint(0, 3) for _ in range(size)]
            responds = [t + rng.randint(0, 3) for t in invokes]

        operations = []
        for k in range(size):
            choice = rng.choice("wwrrcc")
            if choice == "w":
                operations.append(write(
                    identifiers[k], rng.randint(0, 2), invokes[k], responds[k]))
            elif choice == "r":
                operations.append(read(
                    identifiers[k], rng.randint(0, 2), invokes[k], responds[k]))
            else:
                operations.append(cas(
                    identifiers[k], rng.randint(0, 2), rng.randint(0, 2),
                    rng.choice([True, False]), invokes[k], responds[k]))

        result = solve(0, operations)
        count, lex_min = self._brute_force(0, operations)
        assert result.linearizable is (count > 0)
        if count:
            assert result.order == lex_min
            assert result.unique is (count == 1)


class TestLargerHistories:
    def test_twenty_four_operations(self):
        # A forced chain of 24 alternating write/read pairs.
        ops = []
        for i in range(12):
            ops.append(write(f"w{i:02d}", i + 1, 2 * i, 2 * i))
            ops.append(read(f"r{i:02d}", i + 1, 2 * i + 1, 2 * i + 1))
        result = solve(0, ops)
        assert result.linearizable
        assert result.unique is True
        assert len(result.steps) == 24

    def test_many_overlapping_reads_resolve_quickly(self):
        ops = [read(f"r{i:02d}", 3, 0, 100) for i in range(24)]
        result = solve(3, ops)
        assert result.linearizable
        assert result.unique is False
        assert result.order == tuple(f"r{i:02d}" for i in range(24))

    def test_twenty_four_same_value_writes_complete_within_deadline(self):
        ops = [write(f"w{i:02d}", 1, 0, 100) for i in range(24)]
        started = time.perf_counter()
        result = solve(0, ops)
        assert time.perf_counter() - started < SOLVER_DEADLINE_S
        assert result.linearizable
        assert result.unique is False
        assert result.steps[0].before == 0
        assert all(step.after == 1 for step in result.steps)
