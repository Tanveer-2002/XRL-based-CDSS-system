"""
Patient Routes for Type 2 Diabetes CDSS API.

Endpoints:
- GET /api/patients: List all patients in the system with visit counts and latest status.
- GET /api/patients/{subject_id}/timeline: Retrieve full chronological trajectory history.
- GET /api/patients/{subject_id}/latest: Retrieve the most recent observation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

router = APIRouter(prefix="/api/patients", tags=["Patients"])

_trajectory_cache: Optional[pd.DataFrame] = None


def get_trajectory_data() -> pd.DataFrame:
    """Loads and caches the processed trajectory dataset."""
    global _trajectory_cache
    if _trajectory_cache is not None:
        return _trajectory_cache

    path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"
    if not path.exists():
        # Fallback to interim or empty
        path = config.INTERIM_DATA_DIR / "t2d_cohort_raw.csv"

    if path.exists():
        _trajectory_cache = pd.read_csv(path)
        return _trajectory_cache
    else:
        raise HTTPException(
            status_code=404,
            detail="Trajectory dataset not found. Please run the data preprocessing pipeline first.",
        )


@router.get("", response_model=List[Dict[str, Any]])
def list_patients():
    """Lists all patients currently registered in the CDSS with summary metadata."""
    df = get_trajectory_data()
    summary = []
    for sid, group in df.groupby("subject_id"):
        sorted_g = group.sort_values(by="chartdate")
        latest = sorted_g.iloc[-1]
        summary.append({
            "subject_id": str(sid),
            "age": int(latest.get("age", 60)),
            "gender": str(latest.get("gender", "Unknown")),
            "total_visits": len(group),
            "first_visit": str(sorted_g.iloc[0]["chartdate"]),
            "latest_visit": str(latest["chartdate"]),
            "latest_hba1c": float(latest.get("hba1c", 0.0)),
            "overall_trajectory_status": str(latest.get("overall_trajectory_status", "stable")),
        })
    return summary


@router.get("/{subject_id}/timeline")
def get_patient_timeline(subject_id: str):
    """Retrieves full chronological trajectory timeline for a specific patient."""
    df = get_trajectory_data()
    patient_df = df[df["subject_id"].astype(str) == str(subject_id)].sort_values(by="chartdate")
    if patient_df.empty:
        raise HTTPException(status_code=404, detail=f"Patient {subject_id} not found.")

    return patient_df.fillna("").to_dict(orient="records")


@router.get("/{subject_id}/latest")
def get_patient_latest(subject_id: str):
    """Retrieves the latest clinical observation and trajectory status for a patient."""
    df = get_trajectory_data()
    patient_df = df[df["subject_id"].astype(str) == str(subject_id)].sort_values(by="chartdate")
    if patient_df.empty:
        raise HTTPException(status_code=404, detail=f"Patient {subject_id} not found.")

    latest = patient_df.iloc[-1]
    return latest.fillna("").to_dict()
