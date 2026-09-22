"""Unit tests for the linearizability engine."""

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
