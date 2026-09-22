"""API-level tests: validation, verdicts, and the health endpoint."""

import time

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

CHECK_URL = "/api/v1/linearizability/check"


def write(identifier, value, invoke, respond):
    return {"id": identifier, "type": "write", "value": value,
            "invoke": invoke, "respond": respond}


def read(identifier, value, invoke, respond):
    return {"id": identifier, "type": "read", "value": value,
            "invoke": invoke, "respond": respond}


def cas(identifier, expected, update, success, invoke, respond, type_name="cas"):
    return {"id": identifier, "type": type_name, "expected": expected,
            "update": update, "success": success,
            "invoke": invoke, "respond": respond}


def post(payload):
    return client.post(CHECK_URL, json=payload)


class TestHealth:
    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestLinearizableHistories:
    def test_simple_linearizable_history(self):
        response = post({
            "initial_value": 0,
            "operations": [write("w", 1, 0, 2), read("r", 1, 3, 4)],
        })
        assert response.status_code == 200
        body = response.json()
        assert body["linearizable"] is True
        assert body["order"] == ["w", "r"]
        assert body["unique"] is True
        assert body["steps"] == [
            {"id": "w", "before": 0, "after": 1},
            {"id": "r", "before": 1, "after": 1},
        ]

    def test_non_unique_order_reported(self):
        response = post({
            "initial_value": 0,
            "operations": [read("b", 0, 0, 9), read("a", 0, 0, 9)],
        })
        body = response.json()
        assert body["linearizable"] is True
        assert body["order"] == ["a", "b"]
        assert body["unique"] is False

    def test_compare_and_swap_alias_accepted(self):
        response = post({
            "initial_value": 0,
            "operations": [cas("c", 0, 1, True, 0, 1, type_name="compare-and-swap")],
        })
        assert response.status_code == 200
        body = response.json()
        assert body["linearizable"] is True
        assert body["steps"] == [{"id": "c", "before": 0, "after": 1}]

    def test_negative_times_and_values_allowed(self):
        response = post({
            "initial_value": -7,
            "operations": [write("w", -3, -10, -5), read("r", -3, -5, -1)],
        })
        assert response.status_code == 200
        assert response.json()["linearizable"] is True


class TestNonLinearizableHistories:
    def test_impossible_read(self):
        response = post({
            "initial_value": 0,
            "operations": [write("w", 1, 0, 2), read("r", 0, 3, 4)],
        })
        assert response.status_code == 200
        body = response.json()
        assert body["linearizable"] is False
        assert body["order"] is None
        assert body["steps"] is None
        assert body["unique"] is None
        assert "not linearizable" in body["message"]

    def test_contradictory_cas_flag(self):
        response = post({
            "initial_value": 0,
            "operations": [cas("c", 0, 1, False, 0, 1)],
        })
        assert response.status_code == 200
        assert response.json()["linearizable"] is False


class TestValidation:
    def test_duplicate_identifiers_rejected(self):
        response = post({
            "initial_value": 0,
            "operations": [write("a", 1, 0, 1), read("a", 1, 2, 3)],
        })
        assert response.status_code == 422

    def test_invalid_interval_rejected(self):
        response = post({
            "initial_value": 0,
            "operations": [write("a", 1, 5, 2)],
        })
        assert response.status_code == 422

    def test_read_with_cas_fields_rejected(self):
        bad = read("r", 0, 0, 1)
        bad["expected"] = 0
        response = post({"initial_value": 0, "operations": [bad]})
        assert response.status_code == 422

    def test_write_without_value_rejected(self):
        bad = {"id": "w", "type": "write", "invoke": 0, "respond": 1}
        response = post({"initial_value": 0, "operations": [bad]})
        assert response.status_code == 422

    def test_cas_missing_success_rejected(self):
        bad = {"id": "c", "type": "cas", "expected": 0, "update": 1,
               "invoke": 0, "respond": 1}
        response = post({"initial_value": 0, "operations": [bad]})
        assert response.status_code == 422

    def test_cas_with_value_field_rejected(self):
        bad = cas("c", 0, 1, True, 0, 1)
        bad["value"] = 1
        response = post({"initial_value": 0, "operations": [bad]})
        assert response.status_code == 422

    def test_unknown_type_rejected(self):
        bad = {"id": "x", "type": "increment", "value": 1,
               "invoke": 0, "respond": 1}
        response = post({"initial_value": 0, "operations": [bad]})
        assert response.status_code == 422

    def test_operation_count_bounds(self):
        response = post({"initial_value": 0, "operations": []})
        assert response.status_code == 422
        response = post({
            "initial_value": 0,
            "operations": [write(f"w{i}", i, 0, 1) for i in range(25)],
        })
        assert response.status_code == 422
        response = post({
            "initial_value": 0,
            "operations": [write(f"w{i}", i, 0, 1) for i in range(24)],
        })
        assert response.status_code == 200

    @pytest.mark.parametrize("field,value", [
        ("invoke", "0"), ("respond", 1.5), ("invoke", True),
    ])
    def test_non_integer_times_rejected(self, field, value):
        req = write("w", 1, 0, 1)
        req[field] = value
        response = post({"initial_value": 0, "operations": [req]})
        assert response.status_code == 422

    def test_non_integer_value_rejected(self):
        response = post({"initial_value": 0,
                         "operations": [write("w", "1", 0, 1)]})
        assert response.status_code == 422

    def test_non_boolean_success_rejected(self):
        bad = cas("c", 0, 1, 1, 0, 1)
        response = post({"initial_value": 0, "operations": [bad]})
        assert response.status_code == 422

    def test_extra_top_level_field_rejected(self):
        response = post({
            "initial_value": 0,
            "operations": [write("w", 1, 0, 1)],
            "debug": True,
        })
        assert response.status_code == 422

    def test_empty_identifier_rejected(self):
        response = post({"initial_value": 0,
                         "operations": [write("", 1, 0, 1)]})
        assert response.status_code == 422


class TestReportedHistories:
    """Public HTTP smoke tests for the two audited histories."""

    HISTORY_ONE = {
        "initial_value": 0,
        "operations": [
            {"id": "a-restore", "type": "write", "value": 1,
             "invoke": 0, "respond": 10},
            {"id": "b-seed", "type": "write", "value": 1,
             "invoke": 0, "respond": 1},
            # c-advance exercises the CAS payload; its "type" string is
            # parametrized over both accepted names below.
            {"id": "c-advance", "expected": 1, "update": 2,
             "success": True, "invoke": 1, "respond": 2},
            {"id": "d-check", "type": "read", "value": 1,
             "invoke": 2, "respond": 3},
        ],
    }

    EXPECTED_STEPS_ONE = [
        {"id": "b-seed", "before": 0, "after": 1},
        {"id": "c-advance", "before": 1, "after": 2},
        {"id": "a-restore", "before": 2, "after": 1},
        {"id": "d-check", "before": 1, "after": 1},
    ]

    @pytest.mark.parametrize("cas_type", ["cas", "compare-and-swap"])
    def test_history_one_unique_order_and_value_chain(self, cas_type):
        payload = {
            "initial_value": 0,
            "operations": [
                dict(op, type=cas_type) if op["id"] == "c-advance" else op
                for op in self.HISTORY_ONE["operations"]
            ],
        }
        response = post(payload)
        assert response.status_code == 200
        body = response.json()
        assert body["linearizable"] is True
        assert body["unique"] is True
        assert body["order"] == [
            "b-seed", "c-advance", "a-restore", "d-check",
        ]
        assert body["steps"] == self.EXPECTED_STEPS_ONE

    def test_history_two_canonical_order_and_non_unique(self):
        response = post({
            "initial_value": 0,
            "operations": [
                write("a-write", 1, 0, 10),
                write("z-write", 1, 0, 10),
                read("b-read", 1, 0, 10),
            ],
        })
        assert response.status_code == 200
        body = response.json()
        assert body["linearizable"] is True
        assert body["unique"] is False
        # Four legal total orders; the canonical (lexicographically
        # smallest) one interleaves the read instead of pairing the writes.
        assert body["order"] == ["a-write", "b-read", "z-write"]
        assert body["steps"] == [
            {"id": "a-write", "before": 0, "after": 1},
            {"id": "b-read", "before": 1, "after": 1},
            {"id": "z-write", "before": 1, "after": 1},
        ]


class TestBoundaryHistoriesOverHttp:
    """The 20-24 operation scale must keep working over the public API."""

    def _twenty_four_op_payload(self):
        operations = [write(f"w{i:02d}", 1, 0, 10) for i in range(20)]
        operations += [
            cas("c1-advance", 1, 2, True, 10, 11),
            read("r2-value", 2, 11, 12),
            cas("c2-misses", 1, 9, False, 12, 13),
            read("r2-still", 2, 13, 14),
        ]
        return {"initial_value": 0, "operations": operations}

    def test_twenty_four_ops_solved_within_timeout(self):
        payload = self._twenty_four_op_payload()
        started = time.perf_counter()
        response = post(payload)
        elapsed = time.perf_counter() - started
        assert response.status_code == 200
        assert elapsed < 10.0
        body = response.json()
        assert body["linearizable"] is True
        assert body["unique"] is False
        assert body["order"] == [
            *(f"w{i:02d}" for i in range(20)),
            "c1-advance", "r2-value", "c2-misses", "r2-still",
        ]
        assert len(body["steps"]) == 24
        # Independent replay of the returned value chain.
        value = 0
        kinds = {op["id"]: op for op in payload["operations"]}
        for step in body["steps"]:
            assert step["before"] == value
            operation = kinds[step["id"]]
            if operation["type"] == "write":
                value = operation["value"]
            elif operation["type"] == "read":
                assert operation["value"] == value
            elif operation["success"]:
                assert operation["expected"] == value
                value = operation["update"]
            else:
                assert operation["expected"] != value
            assert step["after"] == value

    def test_twenty_four_ops_unsat_returns_no_partial_order(self):
        operations = [write(f"w{i:02d}", 1, 0, 10) for i in range(21)]
        operations += [
            cas("c1-one-to-two", 1, 2, True, 10, 11),
            read("r-two", 2, 11, 12),
            cas("c2-demands-one", 1, 3, True, 12, 13),
        ]
        started = time.perf_counter()
        response = post({"initial_value": 0, "operations": operations})
        elapsed = time.perf_counter() - started
        assert elapsed < 10.0
        assert response.status_code == 200
        body = response.json()
        assert body["linearizable"] is False
        assert body["order"] is None
        assert body["steps"] is None
        assert body["unique"] is None

    @pytest.mark.parametrize("count", [20, 21, 22, 23, 24])
    def test_same_value_concurrent_writes_with_reads(self, count):
        write_count = count // 2
        operations = [write(f"w{i:02d}", 4, 0, 10) for i in range(write_count)]
        operations += [
            read(f"r{i:02d}", 4, 0, 10) for i in range(count - write_count)
        ]
        started = time.perf_counter()
        response = post({"initial_value": 4, "operations": operations})
        elapsed = time.perf_counter() - started
        assert response.status_code == 200
        assert elapsed < 10.0
        body = response.json()
        assert body["linearizable"] is True
        assert body["unique"] is False
        assert len(body["steps"]) == count
        assert body["order"][0].startswith("r")
