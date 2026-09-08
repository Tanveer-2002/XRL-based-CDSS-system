"""
Prediction and Explanation Routes for Type 2 Diabetes CDSS API.

Endpoints:
- POST /api/predict/action: Predicts optimal RL monitoring action and Q-values.
- POST /api/predict/explain: Generates SHAP attributions and natural language explanation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import torch

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from preprocessing.normalization import ClinicalScaler
from rl.actions import get_action_metadata
from rl.model import DiscreteCQLNetwork
from rl.state import StateConstructor
from trajectory.pipeline import TrajectoryPipeline
from xai.explanation_generator import generate_clinical_explanation
from xai.shap_explainer import RLShapExplainer

router = APIRouter(prefix="/api/predict", tags=["Predictions & Explanations"])

# Model artifact state holders
_model: Optional[DiscreteCQLNetwork] = None
_scaler: Optional[ClinicalScaler] = None
_explainer: Optional[RLShapExplainer] = None
_state_constructor: Optional[StateConstructor] = None
_trajectory_pipeline: Optional[TrajectoryPipeline] = None


def load_inference_artifacts():
    """Initializes and caches model, scaler, and explainer artifacts."""
    global _model, _scaler, _explainer, _state_constructor, _trajectory_pipeline

    if _state_constructor is None:
        _state_constructor = StateConstructor()
    if _trajectory_pipeline is None:
        _trajectory_pipeline = TrajectoryPipeline()

    model_path = config.RL_AGENT_DIR / "cql_model.pt"
    scaler_path = config.SCALERS_DIR / "scaler.pkl"

    if _scaler is None and scaler_path.exists():
        _scaler = ClinicalScaler().load(scaler_path)

    if _model is None and model_path.exists():
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dim = checkpoint.get("state_dim", _state_constructor.state_dim)
        _model = DiscreteCQLNetwork(state_dim=state_dim, num_actions=config.NUM_ACTIONS)
        _model.load_state_dict(checkpoint["model_state_dict"])
        _model.eval()

        # Initialize SHAP explainer with synthetic baseline background
        bg_data = np.zeros((30, state_dim), dtype=np.float32)
        _explainer = RLShapExplainer(model=_model, background_data=bg_data, state_constructor=_state_constructor)


class ActionPredictionRequest(BaseModel):
    subject_id: Optional[str] = Field(None, description="Registered patient subject ID")
    observations: Optional[List[Dict[str, Any]]] = Field(
        None, description="Optional raw longitudinal observation records for new patient"
    )


class ActionPredictionResponse(BaseModel):
    subject_id: str
    action_code: int
    action_name: str
    urgency: str
    recommended_interval: str
    clinical_guidance: str
    q_values: Dict[str, float]
    overall_trajectory_status: str


@router.post("/action", response_model=ActionPredictionResponse)
def predict_monitoring_action(req: ActionPredictionRequest):
    """
    Computes the optimal Offline RL monitoring action based on the patient's
    longitudinal health trajectory and current clinical state.
    """
    load_inference_artifacts()
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Trained RL agent not loaded. Please train the CQL agent first.",
        )

    # 1. Obtain patient DataFrame
    if req.observations and len(req.observations) > 0:
        patient_df = pd.DataFrame(req.observations)
        subject_id = str(req.observations[0].get("subject_id", "NEW_PATIENT"))
    elif req.subject_id:
        traj_path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"
        if not traj_path.exists():
            raise HTTPException(status_code=404, detail="Processed trajectory dataset not found.")
        all_df = pd.read_csv(traj_path)
        patient_df = all_df[all_df["subject_id"].astype(str) == str(req.subject_id)]
        if patient_df.empty:
            raise HTTPException(status_code=404, detail=f"Patient {req.subject_id} not found.")
        subject_id = str(req.subject_id)
    else:
        raise HTTPException(status_code=400, detail="Must provide either subject_id or observations.")

    # 2. Compute trajectory if raw records provided
    if "overall_trajectory_status" not in patient_df.columns:
        patient_df = _trajectory_pipeline.run(patient_df)

    # 3. Apply normalization
    if _scaler is not None and "hba1c_scaled" not in patient_df.columns:
        patient_df = _scaler.transform(patient_df)

    latest_row = patient_df.sort_values(by="chartdate").iloc[-1]
    state_vector = _state_constructor.extract_state_vector(latest_row)

    # 4. Predict Q-values and action
    state_tensor = torch.tensor(state_vector, dtype=torch.float32)
    q_vals = _model.get_q_values(state_tensor).cpu().numpy()
    action_idx = int(np.argmax(q_vals))

    meta = get_action_metadata(action_idx)
    q_dict = {config.ACTION_NAMES[i]: round(float(q_vals[i]), 3) for i in range(len(q_vals))}

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


@router.post("/explain")
def explain_monitoring_action(req: ActionPredictionRequest):
    """
    Computes SHAP feature attributions and synthesizes a natural language
    clinical explanation justifying the RL monitoring decision.
    """
    load_inference_artifacts()
    if _model is None or _explainer is None:
        raise HTTPException(
            status_code=503,
            detail="Trained model or explainer not loaded.",
        )

    # Run action prediction first to obtain state and selected action
    pred = predict_monitoring_action(req)
    action_idx = pred.action_code

    # Retrieve patient latest row
    if req.observations and len(req.observations) > 0:
        patient_df = pd.DataFrame(req.observations)
        if "overall_trajectory_status" not in patient_df.columns:
            patient_df = _trajectory_pipeline.run(patient_df)
        if _scaler is not None and "hba1c_scaled" not in patient_df.columns:
            patient_df = _scaler.transform(patient_df)
    else:
        traj_path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"
        all_df = pd.read_csv(traj_path)
        patient_df = all_df[all_df["subject_id"].astype(str) == str(req.subject_id)]

    latest_row = patient_df.sort_values(by="chartdate").iloc[-1]
    state_vector = _state_constructor.extract_state_vector(latest_row)

    # Compute SHAP values
    shap_res = _explainer.explain_action(state_vector, action=action_idx)

    # Generate clinical natural language explanation
    explanation = generate_clinical_explanation(
        action=action_idx,
        obs_data=latest_row,
        attributions=shap_res["attributions"],
        top_k=5,
    )

    return {
        "subject_id": pred.subject_id,
        "action_code": pred.action_code,
        "action_name": pred.action_name,
        "urgency": pred.urgency,
        "recommended_interval": pred.recommended_interval,
        "q_values": pred.q_values,
        "shap_summary": shap_res,
        "clinical_explanation": explanation,
    }
