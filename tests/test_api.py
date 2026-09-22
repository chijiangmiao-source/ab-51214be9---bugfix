"""API-level tests: validation, verdicts, and the health endpoint."""

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
