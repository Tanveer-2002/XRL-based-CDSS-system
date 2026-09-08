"""
Integration Tests for FastAPI REST API Endpoints.
"""

from pathlib import Path
import pytest
from starlette.testclient import TestClient

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from api.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_root(client):
    # Test Root
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "online"

    # Test Health
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"


def test_patients_endpoints(client):
    res_patients = client.get("/api/patients")
    assert res_patients.status_code == 200
    patients = res_patients.json()
    assert len(patients) > 0
    test_sid = patients[0]["subject_id"]

    # Test timeline
    res_timeline = client.get(f"/api/patients/{test_sid}/timeline")
    assert res_timeline.status_code == 200
    timeline = res_timeline.json()
    assert len(timeline) > 0

    # Test latest
    res_latest = client.get(f"/api/patients/{test_sid}/latest")
    assert res_latest.status_code == 200
    assert "hba1c" in res_latest.json()


def test_predict_and_explain_endpoints(client):
    res_patients = client.get("/api/patients")
    test_sid = res_patients.json()[0]["subject_id"]

    # Test action prediction
    res_action = client.post("/api/predict/action", json={"subject_id": test_sid})
    assert res_action.status_code == 200
    action_data = res_action.json()
    assert "action_code" in action_data
    assert "q_values" in action_data
    assert action_data["action_code"] in [0, 1, 2, 3]

    # Test SHAP explanation
    res_explain = client.post("/api/predict/explain", json={"subject_id": test_sid})
    assert res_explain.status_code == 200
    explain_data = res_explain.json()
    assert "clinical_explanation" in explain_data
    assert "summary_statement" in explain_data["clinical_explanation"]


def test_report_endpoints(client):
    res_patients = client.get("/api/patients")
    test_sid = res_patients.json()[0]["subject_id"]

    # Test JSON report
    res_json = client.get(f"/api/reports/{test_sid}")
    assert res_json.status_code == 200
    assert "trajectory_metrics" in res_json.json()

    # Test HTML dashboard rendering
    res_html = client.get(f"/api/reports/{test_sid}/html")
    assert res_html.status_code == 200
    assert "<html" in res_html.text.lower()
    assert "Clinical Decision Support Report" in res_html.text
