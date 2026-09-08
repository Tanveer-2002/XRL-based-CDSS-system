"""
Consolidated FastAPI REST API Server for Type 2 Diabetes CDSS.

Combines:
1. Patient Queries & Timeline Endpoints (/api/patients)
2. Offline RL (CQL) Action Predictions (/api/predict/action)
3. Explainable AI (SHAP) Clinical Rationales (/api/predict/explain)
4. Cumulative Healthcare Reporting & Interactive HTML Dashboards (/api/reports)
5. Root & Health Check Endpoints
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
import numpy as np
import pandas as pd
from pydantic import BaseModel
import torch

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from preprocessing.normalization import ClinicalScaler
from rl.environment import MonitoringAction, StateConstructor, get_action_metadata
from rl.cql import DiscreteCQLNetwork, fine_tune_cql_with_feedback
from rl.feedback import FeedbackManager
from xai.explainer import RLShapExplainer, generate_clinical_explanation
from report.generator import (
    generate_cumulative_report,
    render_html_dashboard,
    render_markdown_report,
)

# Global preloaded inference artifacts
_model: Optional[DiscreteCQLNetwork] = None
_scaler: Optional[ClinicalScaler] = None
_state_constructor: Optional[StateConstructor] = None
_explainer: Optional[RLShapExplainer] = None
_feedback_manager: Optional[FeedbackManager] = None


def load_inference_artifacts():
    """Preloads trained model, scaler, and background data for real-time inference."""
    global _model, _scaler, _state_constructor, _explainer, _feedback_manager

    _state_constructor = StateConstructor()
    _feedback_manager = FeedbackManager()

    # Load ClinicalScaler
    scaler_path = config.SCALERS_DIR / "scaler.pkl"
    if scaler_path.exists():
        _scaler = ClinicalScaler()
        _scaler.load(scaler_path)

    # Load Discrete CQL Model
    model_path = config.RL_AGENT_DIR / "cql_model.pt"
    if model_path.exists():
        _model = DiscreteCQLNetwork(state_dim=_state_constructor.state_dim, num_actions=config.NUM_ACTIONS)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        _model.load_state_dict(checkpoint["model_state_dict"])
        _model.eval()

    # Load Background Data for SHAP Explainer
    traj_path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"
    if traj_path.exists() and _model is not None:
        traj_df = pd.read_csv(traj_path)
        sample_df = traj_df.sample(min(40, len(traj_df)), random_state=42)
        bg_matrix = _state_constructor.extract_state_matrix(sample_df)
        _explainer = RLShapExplainer(
            model=_model,
            background_data=bg_matrix,
            state_constructor=_state_constructor,
            nsamples=50,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI application."""
    print("[*] Starting Type 2 Diabetes CDSS API...")
    try:
        load_inference_artifacts()
        print("[OK] Model and inference artifacts preloaded successfully.")
    except Exception as e:
        print(f"[!] Artifact preloading deferred: {e}")
    yield
    print("[*] Shutting down Type 2 Diabetes CDSS API...")


app = FastAPI(
    title=config.API_CONFIG["title"],
    version=config.API_CONFIG["version"],
    description="Clinical Decision Support System for Type 2 Diabetes based on Longitudinal Health Trajectory Analysis, Offline RL (CQL), and Explainable AI (SHAP).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================================
# REQUEST & RESPONSE SCHEMAS
# =========================================================================

class ActionPredictionRequest(BaseModel):
    subject_id: Optional[str] = None
    observations: Optional[List[Dict[str, Any]]] = None


class ActionPredictionResponse(BaseModel):
    subject_id: str
    action_code: int
    action_name: str
    urgency: str
    recommended_interval: str
    clinical_guidance: str
    q_values: Dict[str, float]
    overall_trajectory_status: str


class ClinicianFeedbackRequest(BaseModel):
    subject_id: str
    model_action: int
    clinician_action: int
    notes: Optional[str] = ""


class PatientEncounterRequest(BaseModel):
    chartdate: Optional[str] = None
    hba1c: Optional[float] = None
    glucose: Optional[float] = None
    sbp: Optional[float] = None
    dbp: Optional[float] = None
    bmi: Optional[float] = None
    creatinine: Optional[float] = None
    ldl: Optional[float] = None
    hdl: Optional[float] = None
    triglycerides: Optional[float] = None
    cholesterol: Optional[float] = None


class RetrainRequest(BaseModel):
    epochs: int = 5
    lr: float = 1e-5
    safety_threshold_sensitivity: float = 0.90
    safety_threshold_specificity: float = 0.85


from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

STATIC_INDEX = Path(__file__).resolve().parent.parent / "static" / "index.html"


# =========================================================================
# 1. CORE & HEALTH ENDPOINTS
# =========================================================================

@app.get("/")
def root(request: Request):
    accept = request.headers.get("accept", "")
    if "text/html" in accept and STATIC_INDEX.exists():
        return FileResponse(STATIC_INDEX)

    return {
        "system": config.API_CONFIG["title"],
        "version": config.API_CONFIG["version"],
        "status": "online",
        "description": "Longitudinal Health Trajectory & Offline RL Monitoring CDSS",
        "endpoints": {
            "ui": "/dashboard",
            "docs": "/docs",
            "health": "/health",
            "patients": "/api/patients",
            "predict_action": "/api/predict/action",
            "explain_action": "/api/predict/explain",
            "reports": "/api/reports/{subject_id}",
            "html_dashboard": "/api/reports/{subject_id}/html",
        },
    }


@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
def get_dashboard():
    """Renders clinical dashboard frontend."""
    if STATIC_INDEX.exists():
        return FileResponse(STATIC_INDEX)
    raise HTTPException(status_code=404, detail="Dashboard UI not found.")


@app.get("/health")
def health():
    return {"status": "healthy", "service": "diabetes-cdss-api"}


# =========================================================================
# 2. PATIENT ENDPOINTS
# =========================================================================

@app.get("/api/patients")
def list_patients(limit: int = 50, skip: int = 0):
    traj_path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"
    if not traj_path.exists():
        raise HTTPException(status_code=404, detail="Trajectory dataset not found. Run pipeline first.")

    df = pd.read_csv(traj_path)
    grouped = df.groupby("subject_id")

    patient_summaries = []
    for sid, group in grouped:
        latest = group.sort_values(by="chartdate").iloc[-1]
        patient_summaries.append({
            "subject_id": str(sid),
            "age": int(latest.get("age", 60)),
            "gender": str(latest.get("gender", "Unknown")),
            "total_visits": int(len(group)),
            "monitoring_duration_days": float(latest.get("cumulative_days", 0.0)),
            "latest_hba1c": float(latest.get("hba1c", 7.0)),
            "latest_sbp": float(latest.get("sbp", 125.0)),
            "overall_trajectory_status": str(latest.get("overall_trajectory_status", "stable")),
        })

    return patient_summaries[skip: skip + limit]


@app.get("/api/patients/{subject_id}/timeline")
def get_patient_timeline(subject_id: str):
    traj_path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"
    if not traj_path.exists():
        raise HTTPException(status_code=404, detail="Trajectory data not found.")

    df = pd.read_csv(traj_path)
    patient_records = df[df["subject_id"].astype(str) == str(subject_id)].sort_values(by="chartdate")
    if patient_records.empty:
        raise HTTPException(status_code=404, detail=f"Patient {subject_id} not found.")

    return patient_records.fillna(0.0).to_dict(orient="records")


@app.get("/api/patients/{subject_id}/latest")
def get_patient_latest(subject_id: str):
    traj_path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"
    if not traj_path.exists():
        raise HTTPException(status_code=404, detail="Trajectory data not found.")

    df = pd.read_csv(traj_path)
    patient_records = df[df["subject_id"].astype(str) == str(subject_id)].sort_values(by="chartdate")
    if patient_records.empty:
        raise HTTPException(status_code=404, detail=f"Patient {subject_id} not found.")

    return patient_records.iloc[-1].fillna(0.0).to_dict()


# =========================================================================
# 3. PREDICTION & EXPLANATION ENDPOINTS
# =========================================================================

def _get_or_compute_patient_df(request: ActionPredictionRequest) -> Tuple[pd.DataFrame, str]:
    if request.observations and len(request.observations) > 0:
        raw_df = pd.DataFrame(request.observations)
        subject_id = str(raw_df.iloc[0].get("subject_id", "CUSTOM_PATIENT"))
        from trajectory.pipeline import build_trajectory_features, evaluate_trajectory_status
        traj_df, _ = build_trajectory_features(raw_df)
        traj_df["overall_trajectory_status"] = traj_df.apply(evaluate_trajectory_status, axis=1)
        return traj_df, subject_id

    if request.subject_id:
        traj_path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"
        if not traj_path.exists():
            raise HTTPException(status_code=404, detail="Trajectory database not found.")
        all_df = pd.read_csv(traj_path)
        p_df = all_df[all_df["subject_id"].astype(str) == str(request.subject_id)].copy()
        if p_df.empty:
            raise HTTPException(status_code=404, detail=f"Patient {request.subject_id} not found.")
        return p_df, str(request.subject_id)

    raise HTTPException(status_code=400, detail="Must provide either subject_id or observations list.")


@app.post("/api/predict/action", response_model=ActionPredictionResponse)
def predict_monitoring_action(request: ActionPredictionRequest):
    if _model is None:
        load_inference_artifacts()
        if _model is None:
            raise HTTPException(status_code=503, detail="CQL Model artifact is not ready.")

    patient_df, subject_id = _get_or_compute_patient_df(request)

    if _scaler is not None and "hba1c_scaled" not in patient_df.columns:
        patient_df = _scaler.transform(patient_df)

    latest_row = patient_df.sort_values(by="chartdate").iloc[-1]
    state_vector = _state_constructor.extract_state_vector(latest_row)

    state_tensor = torch.tensor(state_vector, dtype=torch.float32)
    q_vals = _model.get_q_values(state_tensor).cpu().numpy()
    action_idx = int(np.argmax(q_vals))

    meta = get_action_metadata(action_idx)
    q_dict = {config.ACTION_NAMES[i]: round(float(q_vals[i]), 3) for i in range(len(q_vals))}

    # Log pending recommendation for future longitudinal outcome tracking (Strategy 1)
    if _feedback_manager is not None and not str(subject_id).startswith("CUSTOM_"):
        try:
            _feedback_manager.log_pending_recommendation(
                subject_id=subject_id,
                action=action_idx,
                state_vector=state_vector,
                obs_data=latest_row.to_dict(),
            )
        except Exception as e:
            print(f"[!] Pending recommendation logging deferred: {e}")

    return ActionPredictionResponse(
        subject_id=subject_id,
        action_code=action_idx,
        action_name=meta["name"],
        urgency=meta["urgency"],
        recommended_interval=meta["interval"],
        clinical_guidance=meta["description"],
        q_values=q_dict,
        overall_trajectory_status=str(latest_row.get("overall_trajectory_status", "stable")),
    )


@app.post("/api/predict/explain")
def explain_monitoring_action(request: ActionPredictionRequest):
    if _model is None or _explainer is None:
        load_inference_artifacts()
        if _model is None or _explainer is None:
            raise HTTPException(status_code=503, detail="Inference/XAI models not loaded.")

    pred_res = predict_monitoring_action(request)
    patient_df, subject_id = _get_or_compute_patient_df(request)

    if _scaler is not None and "hba1c_scaled" not in patient_df.columns:
        patient_df = _scaler.transform(patient_df)

    latest_row = patient_df.sort_values(by="chartdate").iloc[-1]
    state_vector = _state_constructor.extract_state_vector(latest_row)

    action_idx = pred_res.action_code
    shap_data = _explainer.explain_action(state_vector, action=action_idx)

    clinical_exp = generate_clinical_explanation(
        action=action_idx,
        obs_data=latest_row,
        attributions=shap_data["attributions"],
        top_k=5,
    )

    return {
        "subject_id": subject_id,
        "action_code": action_idx,
        "action_name": pred_res.action_name,
        "urgency": pred_res.urgency,
        "recommended_interval": pred_res.recommended_interval,
        "q_values": pred_res.q_values,
        "overall_trajectory_status": pred_res.overall_trajectory_status,
        "shap_attributions": shap_data["attributions"],
        "clinical_explanation": clinical_exp,
    }


# =========================================================================
# 4. REPORT ENDPOINTS
# =========================================================================

def _build_report_for_patient(subject_id: str):
    traj_path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"
    if not traj_path.exists():
        raise HTTPException(status_code=404, detail="Trajectory data not found.")

    all_df = pd.read_csv(traj_path)
    patient_df = all_df[all_df["subject_id"].astype(str) == str(subject_id)].sort_values(by="chartdate")
    if patient_df.empty:
        raise HTTPException(status_code=404, detail=f"Patient {subject_id} not found.")

    exp_res = explain_monitoring_action(ActionPredictionRequest(subject_id=str(subject_id)))

    rl_rec = {
        "action_code": exp_res["action_code"],
        "action_name": exp_res["action_name"],
        "recommended_interval": exp_res["recommended_interval"],
        "urgency": exp_res["urgency"],
        "q_values": exp_res["q_values"],
        "clinical_guidance": exp_res["clinical_explanation"]["clinical_narrative"],
    }

    report = generate_cumulative_report(
        patient_history_df=patient_df,
        rl_recommendation_dict=rl_rec,
        xai_explanation_dict=exp_res["clinical_explanation"],
    )
    return report


@app.get("/api/reports/{subject_id}")
def get_json_report(subject_id: str):
    report = _build_report_for_patient(subject_id)
    return report.to_dict()


@app.get("/api/reports/{subject_id}/markdown", response_class=PlainTextResponse)
def get_markdown_report(subject_id: str):
    report = _build_report_for_patient(subject_id)
    return render_markdown_report(report)


@app.get("/api/reports/{subject_id}/html", response_class=HTMLResponse)
def get_html_report(subject_id: str):
    report = _build_report_for_patient(subject_id)
    return render_html_dashboard(report)


# =========================================================================
# 5. CONTINUAL RL OPTIMIZATION ENDPOINTS (STRATEGIES 1 & 2)
# =========================================================================

@app.post("/api/feedback")
def submit_clinician_feedback(req: ClinicianFeedbackRequest):
    """
    Ingests clinician agreement or action override (Strategy 2 - RLHF).
    """
    if _feedback_manager is None:
        load_inference_artifacts()

    res = _feedback_manager.record_clinician_feedback(
        subject_id=req.subject_id,
        model_action=req.model_action,
        clinician_action=req.clinician_action,
        notes=req.notes or "",
    )
    return res


@app.post("/api/patients/{subject_id}/encounter")
def register_patient_encounter(subject_id: str, req: PatientEncounterRequest):
    """
    Registers a new follow-up encounter for a patient (Strategy 1 - Longitudinal Outcomes).
    Closes the loop on previous pending recommendations and records completed transitions.
    """
    if _feedback_manager is None:
        load_inference_artifacts()

    encounter_data = {k: v for k, v in req.dict().items() if v is not None}
    res = _feedback_manager.register_patient_encounter(
        subject_id=subject_id,
        encounter_data=encounter_data,
    )
    return res


@app.get("/api/feedback/stats")
def get_feedback_stats():
    """
    Returns counts of clinician feedback, overrides, and completed longitudinal transitions.
    """
    if _feedback_manager is None:
        load_inference_artifacts()
    return _feedback_manager.get_feedback_statistics()


@app.post("/api/retrain")
def trigger_model_retraining(req: Optional[RetrainRequest] = None):
    """
    Triggers incremental warm-start fine-tuning on feedback and outcome transitions.
    Applies Experience Replay and Champion vs. Challenger Safety Gate.
    """
    if req is None:
        req = RetrainRequest()

    res = fine_tune_cql_with_feedback(
        epochs=req.epochs,
        lr=req.lr,
        safety_threshold_sensitivity=req.safety_threshold_sensitivity,
        safety_threshold_specificity=req.safety_threshold_specificity,
    )

    # If new challenger model was promoted, reload artifacts into memory live!
    if res.get("status") == "promoted":
        load_inference_artifacts()

    return res

