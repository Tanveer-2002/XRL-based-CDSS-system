"""
Report Routes for Type 2 Diabetes CDSS API.

Endpoints:
- GET /api/reports/{subject_id}: Retrieve full cumulative health report in JSON.
- GET /api/reports/{subject_id}/markdown: Retrieve cumulative report in Markdown.
- GET /api/reports/{subject_id}/html: Render full interactive HTML dashboard in browser.
"""

from pathlib import Path
from typing import Any, Dict
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from .prediction_routes import explain_monitoring_action, ActionPredictionRequest
from report.report_generator import (
    generate_cumulative_report,
    render_html_dashboard,
    render_markdown_report,
)

router = APIRouter(prefix="/api/reports", tags=["Cumulative Reports"])


def _build_report_for_patient(subject_id: str):
    """Helper to assemble cumulative report for a patient."""
    traj_path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"
    if not traj_path.exists():
        raise HTTPException(status_code=404, detail="Trajectory data not found.")

    all_df = pd.read_csv(traj_path)
    patient_df = all_df[all_df["subject_id"].astype(str) == str(subject_id)].sort_values(by="chartdate")
    if patient_df.empty:
        raise HTTPException(status_code=404, detail=f"Patient {subject_id} not found.")

    # Get RL recommendation and XAI explanation
    exp_res = explain_monitoring_action(ActionPredictionRequest(subject_id=str(subject_id)))

    rl_rec = {
        "action_code": exp_res["action_code"],
        "action_name": exp_res["action_name"],
        "recommended_interval": exp_res["recommended_interval"],
        "urgency": exp_res["urgency"],
        "q_values": exp_res["q_values"],
        "description": exp_res["clinical_explanation"]["clinical_narrative"],
    }

    report = generate_cumulative_report(
        patient_history_df=patient_df,
        rl_recommendation_dict=rl_rec,
        xai_explanation_dict=exp_res["clinical_explanation"],
    )
    return report


@router.get("/{subject_id}")
def get_json_report(subject_id: str):
    """Retrieves full cumulative health report as JSON."""
    report = _build_report_for_patient(subject_id)
    return report.to_dict()


@router.get("/{subject_id}/markdown", response_class=PlainTextResponse)
def get_markdown_report(subject_id: str):
    """Retrieves cumulative health report formatted as Markdown."""
    report = _build_report_for_patient(subject_id)
    return render_markdown_report(report)


@router.get("/{subject_id}/html", response_class=HTMLResponse)
def get_html_report(subject_id: str):
    """Renders interactive HTML cumulative dashboard for web browser viewing."""
    report = _build_report_for_patient(subject_id)
    return render_html_dashboard(report)
