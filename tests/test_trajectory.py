"""
Unit Tests for Health Trajectory Analysis Module.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from trajectory.changes import compute_biomarker_changes
from trajectory.moving_average import compute_moving_averages
from trajectory.pipeline import TrajectoryPipeline, evaluate_trajectory_status
from trajectory.timeline import build_patient_timeline
from trajectory.trends import classify_biomarker_trends


@pytest.fixture
def sample_patient_history():
    """Generates synthetic longitudinal history for a deteriorating patient."""
    dates = ["2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01"]
    hba1c_vals = [6.5, 7.0, 7.8, 8.7]       # Clear deterioration: +0.5, +0.8, +0.9
    glucose_vals = [120.0, 140.0, 175.0, 210.0]
    sbp_vals = [122.0, 128.0, 138.0, 146.0]
    bmi_vals = [26.0, 26.5, 27.2, 28.1]

    rows = []
    for i in range(len(dates)):
        rows.append({
            "subject_id": "P_DET",
            "chartdate": dates[i],
            "hba1c": hba1c_vals[i],
            "glucose": glucose_vals[i],
            "sbp": sbp_vals[i],
            "dbp": 80.0,
            "bmi": bmi_vals[i],
            "cholesterol": 190.0,
            "ldl": 110.0,
            "hdl": 45.0,
            "triglycerides": 150.0,
            "creatinine": 0.9,
            "age": 58,
            "gender": "Male",
            "smoking_status": "never",
            "physical_activity": "moderate",
        })
    return pd.DataFrame(rows)


def test_timeline_construction(sample_patient_history):
    timeline = build_patient_timeline(sample_patient_history)
    assert list(timeline["visit_number"]) == [1, 2, 3, 4]
    assert timeline["days_since_last_visit"].iloc[0] == 0.0
    # Between 2020-01-01 and 2020-04-01 is ~91 days
    assert 90 <= timeline["days_since_last_visit"].iloc[1] <= 92


def test_biomarker_changes_and_rates(sample_patient_history):
    timeline = build_patient_timeline(sample_patient_history)
    changes = compute_biomarker_changes(timeline, biomarkers=["hba1c", "glucose", "sbp"])

    # First visit has delta = 0
    assert changes["hba1c_change"].iloc[0] == 0.0
    assert changes["hba1c_rate"].iloc[0] == 0.0

    # Second visit: 7.0 - 6.5 = +0.5
    assert round(changes["hba1c_change"].iloc[1], 2) == 0.50
    # Monthly rate should be positive
    assert changes["hba1c_rate"].iloc[1] > 0.10


def test_moving_averages(sample_patient_history):
    timeline = build_patient_timeline(sample_patient_history)
    ma_df = compute_moving_averages(timeline, biomarkers=["hba1c"], window=3)

    # First visit rolling MA equals first value
    assert ma_df["hba1c_moving_avg"].iloc[0] == 6.5
    # Second visit rolling MA: (6.5 + 7.0) / 2 = 6.75
    assert ma_df["hba1c_moving_avg"].iloc[1] == 6.75
    # Third visit: (6.5 + 7.0 + 7.8) / 3 = 7.10
    assert ma_df["hba1c_moving_avg"].iloc[2] == 7.10


def test_trend_classification(sample_patient_history):
    timeline = build_patient_timeline(sample_patient_history)
    changes = compute_biomarker_changes(timeline, biomarkers=["hba1c"])
    trends = classify_biomarker_trends(changes, biomarkers=["hba1c"])

    # First visit: "stable" / 0
    assert trends["hba1c_trend"].iloc[0] == "stable"
    assert trends["hba1c_trend_num"].iloc[0] == 0

    # Second visit: change is +0.50 (threshold is 0.3) -> "increasing" / 1
    assert trends["hba1c_trend"].iloc[1] == "increasing"
    assert trends["hba1c_trend_num"].iloc[1] == 1


def test_trajectory_pipeline_overall_status(sample_patient_history):
    engine = TrajectoryPipeline()
    traj_df = engine.run(sample_patient_history)

    # Check trajectory columns created
    assert "hba1c_change" in traj_df.columns
    assert "hba1c_rate" in traj_df.columns
    assert "hba1c_moving_avg" in traj_df.columns
    assert "hba1c_trend_num" in traj_df.columns
    assert "overall_trajectory_status" in traj_df.columns

    # Check that deteriorating patient's later visits are flagged as deteriorating or rapidly_deteriorating
    latest_status = traj_df["overall_trajectory_status"].iloc[-1]
    assert latest_status in ["deteriorating", "rapidly_deteriorating"]
