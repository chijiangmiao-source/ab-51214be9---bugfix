"""Unit tests for the linearizability engine."""

import itertools
import time

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


# Generous ceiling: the 24-operation adversarial cases finish in ~1s on
# commodity hardware; the acceptance suite must stay comfortably below any
# reasonable test timeout without reducing the supported history size.
BOUNDARY_TIME_BUDGET_SECONDS = 10.0


def brute_force_verdict(initial_value, operations):
    """Independent reference: enumerate every permutation.

    Returns (linearizable, lexicographically_smallest_order, unique).
    """
    n = len(operations)
    pred = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and operations[i].respond <= operations[j].invoke:
                pred[j] |= 1 << i
    valid = []
    for perm in itertools.permutations(range(n)):
        position = [0] * n
        for k, i in enumerate(perm):
            position[i] = k
        if any(
            position[i] >= position[j]
            for j in range(n)
            for i in range(n)
            if pred[j] >> i & 1
        ):
            continue
        value = initial_value
        ok = True
        for i in perm:
            operation = operations[i]
            if operation.kind == KIND_WRITE:
                value = operation.value
            elif operation.kind == KIND_READ:
                if value != operation.value:
                    ok = False
                    break
            elif value == operation.expected:
                if operation.success is not True:
                    ok = False
                    break
                value = operation.update
            else:
                if operation.success is not False:
                    ok = False
                    break
        if ok:
            valid.append(tuple(operations[i].identifier for i in perm))
    if not valid:
        return False, None, None
    return True, min(valid), len(valid) == 1


def assert_result_recomputes(initial_value, operations, result):
    """Independently replay the reported order and steps."""
    assert result.linearizable
    assert result.order is not None and result.steps is not None
    by_id = {operation.identifier: operation for operation in operations}
    # The order covers every operation exactly once.
    assert sorted(result.order) == sorted(by_id)
    assert [step.identifier for step in result.steps] == list(result.order)

    value = initial_value
    for step in result.steps:
        operation = by_id[step.identifier]
        assert step.before == value
        if operation.kind == KIND_WRITE:
            value = operation.value
        elif operation.kind == KIND_READ:
            assert operation.value == value
        elif operation.success:
            assert operation.expected == step.before
            value = operation.update
        else:
            assert operation.expected != step.before
        assert step.after == value

    # Every real-time precedence edge is honored by the order.
    position = {identifier: k for k, identifier in enumerate(result.order)}
    for a in operations:
        for b in operations:
            if a is not b and a.respond <= b.invoke:
                assert position[a.identifier] < position[b.identifier]


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


class TestReportedAuditBugs:
    """The two concrete histories from the audit bug report."""

    def test_long_write_restores_value_after_cas(self):
        # b-seed must precede c-advance; c-advance must precede d-check;
        # the long-running a-restore is free to land after the CAS and
        # restore the value read by d-check.
        operations = [
            write("a-restore", 1, 0, 10),
            write("b-seed", 1, 0, 1),
            cas("c-advance", 1, 2, True, 1, 2),
            read("d-check", 1, 2, 3),
        ]
        result = solve(0, operations)
        assert result.linearizable is True
        assert result.order == (
            "b-seed", "c-advance", "a-restore", "d-check",
        )
        assert result.unique is True
        assert [(s.identifier, s.before, s.after) for s in result.steps] == [
            ("b-seed", 0, 1),
            ("c-advance", 1, 2),
            ("a-restore", 2, 1),
            ("d-check", 1, 1),
        ]
        assert_result_recomputes(0, operations, result)

    def test_three_fully_overlapping_same_value_ops(self):
        operations = [
            write("a-write", 1, 0, 10),
            write("z-write", 1, 0, 10),
            read("b-read", 1, 0, 10),
        ]
        result = solve(0, operations)
        assert result.linearizable is True
        # Four legal total orders; the lexicographically smallest places
        # the read between the two writes, never adjacently batches writes.
        assert result.order == ("a-write", "b-read", "z-write")
        assert result.unique is False
        assert_result_recomputes(0, operations, result)


class TestBruteForceAgreement:
    def test_random_small_histories_match_permutation_enumeration(self):
        import random

        rng = random.Random(20240922)
        for _ in range(150):
            n = rng.randint(1, 8)
            operations = []
            for k in range(n):
                lo, hi = sorted((rng.randint(0, 5), rng.randint(0, 5)))
                identifier = f"{rng.choice('abcdefgh')}{k}"
                kind = rng.choice([KIND_WRITE, KIND_READ, KIND_CAS])
                if kind == KIND_WRITE:
                    operations.append(write(identifier, rng.randint(0, 3), lo, hi))
                elif kind == KIND_READ:
                    operations.append(read(identifier, rng.randint(0, 3), lo, hi))
                else:
                    operations.append(cas(
                        identifier,
                        rng.randint(0, 3),
                        rng.randint(0, 3),
                        rng.choice([True, False]),
                        lo,
                        hi,
                    ))
            initial = rng.randint(0, 3)
            result = solve(initial, operations)
            linearizable, order, unique = brute_force_verdict(initial, operations)
            assert result.linearizable is linearizable
            if linearizable:
                assert result.order == order
                assert result.unique is unique
                assert_result_recomputes(initial, operations, result)
            else:
                # No solution: never fabricate a partial order.
                assert result.order is None
                assert result.steps is None
                assert result.unique is None


def _same_value_writes(m, invoke=0, respond=10, value=1, prefix="w"):
    return [write(f"{prefix}{i:02d}", value, invoke, respond) for i in range(m)]


class TestBoundaryHistories20to24:
    """20-24 operation histories: same-value concurrent writes mixed with
    later reads / CAS. Every verdict is independently replayed, and each
    request finishes well inside the test timeout at the full 24-op scale.
    """

    def test_forced_tail_after_concurrent_writes(self):
        for n in range(20, 25):
            m = n - 4
            operations = _same_value_writes(m)
            operations += [
                cas("c1-advance", 1, 2, True, 10, 11),
                read("r2-value", 2, 11, 12),
                cas("c2-misses", 1, 9, False, 12, 13),
                read("r2-still", 2, 13, 14),
            ]
            started = time.perf_counter()
            result = solve(0, operations)
            elapsed = time.perf_counter() - started

            assert result.linearizable is True
            assert result.unique is False  # the m concurrent writes permute
            assert result.order == (
                *(f"w{i:02d}" for i in range(m)),
                "c1-advance", "r2-value", "c2-misses", "r2-still",
            )
            assert_result_recomputes(0, operations, result)
            assert elapsed < BOUNDARY_TIME_BUDGET_SECONDS

    def test_long_write_restores_after_cas_scaled(self):
        for n in range(20, 25):
            m = n - 3
            operations = _same_value_writes(m, respond=100, prefix="a")
            operations += [
                write("b-seed", 1, 0, 1),
                cas("c-advance", 1, 2, True, 1, 2),
                read("d-check", 1, 2, 3),
            ]
            started = time.perf_counter()
            result = solve(0, operations)
            elapsed = time.perf_counter() - started

            assert result.linearizable is True
            assert result.unique is False
            # The lex-smallest feasible order delays only the greatest
            # restore write until after the CAS.
            assert result.order == (
                *(f"a{i:02d}" for i in range(m - 1)),
                "b-seed", "c-advance", f"a{m - 1:02d}", "d-check",
            )
            assert_result_recomputes(0, operations, result)
            assert elapsed < BOUNDARY_TIME_BUDGET_SECONDS

    def test_impossible_read_after_concurrent_writes(self):
        for n in range(20, 25):
            m = n - 2
            operations = _same_value_writes(m)
            operations += [
                read("r-sees-one", 1, 10, 11),
                read("r-demands-two", 2, 11, 12),
            ]
            started = time.perf_counter()
            result = solve(0, operations)
            elapsed = time.perf_counter() - started

            assert result.linearizable is False
            assert result.order is None
            assert result.steps is None
            assert result.unique is None
            assert elapsed < BOUNDARY_TIME_BUDGET_SECONDS

    def test_impossible_second_successful_cas(self):
        for n in range(20, 25):
            m = n - 3
            operations = _same_value_writes(m)
            operations += [
                cas("c1-one-to-two", 1, 2, True, 10, 11),
                read("r-two", 2, 11, 12),
                cas("c2-demands-one", 1, 3, True, 12, 13),
            ]
            started = time.perf_counter()
            result = solve(0, operations)
            elapsed = time.perf_counter() - started

            assert result.linearizable is False
            assert result.order is None
            assert result.steps is None
            assert result.unique is None
            assert elapsed < BOUNDARY_TIME_BUDGET_SECONDS

    def test_failed_cas_and_reads_around_concurrent_writes(self):
        for n in range(20, 25):
            m = n - 4
            operations = _same_value_writes(m)
            operations += [
                cas("c1-one-to-two", 1, 2, True, 10, 11),
                cas("c2-misses-one", 1, 9, False, 11, 12),
                read("r2-first", 2, 12, 13),
                read("r2-again", 2, 13, 14),
            ]
            started = time.perf_counter()
            result = solve(0, operations)
            elapsed = time.perf_counter() - started

            assert result.linearizable is True
            assert result.unique is False
            assert_result_recomputes(0, operations, result)
            assert result.steps[-1].after == 2
            assert elapsed < BOUNDARY_TIME_BUDGET_SECONDS

    def test_overlapping_same_value_writes_and_reads(self):
        for n in range(20, 25):
            m = n // 2
            operations = _same_value_writes(m, value=5)
            operations += [
                read(f"r{i:02d}", 5, 0, 10) for i in range(n - m)
            ]
            started = time.perf_counter()
            result = solve(5, operations)
            elapsed = time.perf_counter() - started

            assert result.linearizable is True
            assert result.unique is False
            assert result.order[0] == "r00"
            assert_result_recomputes(5, operations, result)
            assert elapsed < BOUNDARY_TIME_BUDGET_SECONDS
