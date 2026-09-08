"""
Unit & Integration Tests for Continual RL Optimization (Strategies 1 & 2).
"""

from pathlib import Path
import pytest
from starlette.testclient import TestClient
import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from rl.environment import MonitoringAction, StateConstructor
from rl.feedback import FeedbackManager
from api.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def feedback_mgr(tmp_path):
    mgr = FeedbackManager(feedback_dir=tmp_path / "test_feedback")
    return mgr


def test_clinician_feedback_logging(feedback_mgr):
    # 1. Test Agreement
    res_agree = feedback_mgr.record_clinician_feedback(
        subject_id="P_TEST_1",
        model_action=MonitoringAction.ROUTINE_MONITORING,
        clinician_action=MonitoringAction.ROUTINE_MONITORING,
        notes="Patient is stable.",
    )
    assert res_agree["status"] == "success"
    assert res_agree["is_override"] is False

    # 2. Test Override (Doctor escalates to Early Risk Alert)
    res_override = feedback_mgr.record_clinician_feedback(
        subject_id="P_TEST_1",
        model_action=MonitoringAction.ROUTINE_MONITORING,
        clinician_action=MonitoringAction.EARLY_RISK_ALERT,
        notes="Patient exhibits rising BP and microalbuminuria.",
    )
    assert res_override["status"] == "success"
    assert res_override["is_override"] is True
    assert res_override["reward_assigned"] == 3.5

    # Check stats
    stats = feedback_mgr.get_feedback_statistics()
    assert stats["total_clinician_feedback"] == 2
    assert stats["clinician_overrides"] == 1
    assert stats["clinician_accepts"] == 1


def test_pending_recommendation_and_encounter_completion(feedback_mgr):
    sc = StateConstructor()
    state_vec = np.zeros(sc.state_dim, dtype=np.float32)
    obs = {"subject_id": "P_TEST_2", "hba1c": 7.5, "chartdate": "2023-01-01", "overall_trajectory_status": "stable"}

    # 1. Log pending recommendation (Strategy 1 initial step)
    pend_id = feedback_mgr.log_pending_recommendation(
        subject_id="P_TEST_2",
        action=MonitoringAction.INCREASED_MONITORING,
        state_vector=state_vec,
        obs_data=obs,
    )
    assert pend_id.startswith("PEND-P_TEST_2")

    stats = feedback_mgr.get_feedback_statistics()
    assert stats["pending_followups"] >= 1

    # 2. Register next follow-up encounter (Strategy 1 closing the loop)
    followup_data = {
        "chartdate": "2023-04-01",
        "hba1c": 7.1,  # Improved
        "glucose": 115.0,
        "sbp": 120.0,
        "dbp": 78.0,
        "bmi": 26.5,
        "creatinine": 0.95,
    }
    res_enc = feedback_mgr.register_patient_encounter(
        subject_id="P_TEST_2",
        encounter_data=followup_data,
    )
    assert res_enc["status"] == "success"
    assert len(res_enc["closed_transitions"]) == 1
    assert res_enc["closed_transitions"][0]["action"] == MonitoringAction.INCREASED_MONITORING


def test_export_retraining_dataset(feedback_mgr):
    # Add a feedback override
    feedback_mgr.record_clinician_feedback(
        subject_id="P_TEST_3",
        model_action=MonitoringAction.ROUTINE_MONITORING,
        clinician_action=MonitoringAction.CLINICAL_EVALUATION,
        notes="Acute glycemic destabilization.",
    )

    states, actions, rewards, next_states, dones = feedback_mgr.export_retraining_dataset()
    assert len(states) > 0
    assert states.ndim == 2
    assert states.shape[1] == 56
    assert len(actions) == len(states)
    assert len(rewards) == len(states)


def test_api_feedback_endpoints(client):
    # 1. Post feedback
    res_fb = client.post(
        "/api/feedback",
        json={
            "subject_id": "P1001",
            "model_action": 0,
            "clinician_action": 2,
            "notes": "Doctor escalated to Early Risk Alert due to rising trend.",
        },
    )
    assert res_fb.status_code == 200
    assert res_fb.json()["status"] == "success"
    assert res_fb.json()["is_override"] is True

    # 2. Get feedback stats
    res_stats = client.get("/api/feedback/stats")
    assert res_stats.status_code == 200
    stats = res_stats.json()
    assert "total_clinician_feedback" in stats
    assert stats["total_clinician_feedback"] >= 1

    # 3. Post encounter
    res_enc = client.post(
        "/api/patients/P1001/encounter",
        json={
            "chartdate": "2024-05-01",
            "hba1c": 8.0,
            "glucose": 135.0,
            "sbp": 130.0,
            "dbp": 82.0,
        },
    )
    assert res_enc.status_code == 200
    assert res_enc.json()["status"] == "success"
