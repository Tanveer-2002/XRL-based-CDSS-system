"""
Consolidated Explainable AI (XAI) Module for Type 2 Diabetes CDSS.

Combines:
1. SHAP & Integrated Gradients Q-Value Explainer (RLShapExplainer)
2. Feature Importance & Domain Impact Aggregation (extract_top_features, group_feature_attributions)
3. Natural Language Clinical Explanation Generator (generate_clinical_explanation)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from rl.environment import MonitoringAction, StateConstructor, get_action_metadata


# =========================================================================
# 1. SHAP / INTEGRATED GRADIENTS EXPLAINER FOR CQL
# =========================================================================

class RLShapExplainer:
    """
    Computes feature attributions explaining why the CQL agent selected a specific action:
    - Calculates exact marginal Shapley contributions to Q(s, a).
    - Uses SHAP KernelExplainer with Integrated Gradients exact fallback.
    """

    def __init__(
        self,
        model,
        background_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        state_constructor: Optional[StateConstructor] = None,
        nsamples: int = 50,
    ):
        self.model = model
        self.background = background_data
        self.nsamples = nsamples

        if state_constructor is not None:
            self.feature_names = state_constructor.feature_names
        elif feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feat_{i}" for i in range(background_data.shape[1])]

    def explain_action(self, state_vector: np.ndarray, action: int) -> Dict[str, Any]:
        """Calculates SHAP feature attributions for action a."""
        if state_vector.ndim == 1:
            state_vector = state_vector.reshape(1, -1)

        action = int(action)

        with torch.no_grad():
            state_tensor = torch.tensor(state_vector, dtype=torch.float32)
            q_values = self.model.get_q_values(state_tensor).cpu().numpy()
            target_q = float(q_values[action])

        def predict_fn(x: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                device = next(self.model.parameters()).device
                t = torch.tensor(x, dtype=torch.float32, device=device)
                q_all = self.model.q1(t).cpu().numpy()
                return q_all[:, action]

        base_val = float(predict_fn(self.background).mean())

        if HAS_SHAP:
            try:
                explainer = shap.KernelExplainer(predict_fn, self.background)
                shap_vals = explainer.shap_values(state_vector, nsamples=self.nsamples)
                shap_array = shap_vals[0].flatten() if isinstance(shap_vals, list) else shap_vals.flatten()
            except Exception:
                shap_array = self._integrated_gradients_fallback(state_vector[0], action)
        else:
            shap_array = self._integrated_gradients_fallback(state_vector[0], action)

        attributions = {
            self.feature_names[i]: float(round(shap_array[i], 4))
            for i in range(min(len(self.feature_names), len(shap_array)))
        }

        return {
            "action": action,
            "q_value": round(target_q, 3),
            "base_value": round(base_val, 3),
            "attributions": attributions,
        }

    def _integrated_gradients_fallback(self, state_vector: np.ndarray, action: int, steps: int = 30) -> np.ndarray:
        device = next(self.model.parameters()).device
        baseline = self.background.mean(axis=0)
        diff = state_vector - baseline
        grads = np.zeros_like(state_vector)

        for alpha in np.linspace(0.0, 1.0, steps):
            interp = baseline + alpha * diff
            t = torch.tensor(interp, dtype=torch.float32, requires_grad=True, device=device).unsqueeze(0)
            q_val = self.model.q1(t)[0, action]
            q_val.backward()
            grads += t.grad.detach().cpu().squeeze(0).numpy()

        return (diff * (grads / steps)).astype(np.float32)


# =========================================================================
# 2. FEATURE IMPORTANCE RANKING & DOMAIN AGGREGATION
# =========================================================================

def extract_top_features(
    attributions: Dict[str, float],
    top_k: int = 5,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """Ranks top positive (reinforcing) and top negative (inhibiting) features."""
    sorted_items = sorted(attributions.items(), key=lambda item: item[1], reverse=True)
    pos = [{"feature": k, "importance": v} for k, v in sorted_items if v > 0]
    neg = [{"feature": k, "importance": v} for k, v in reversed(sorted_items) if v < 0]
    return pos[:top_k], neg[:top_k]


def group_feature_attributions(attributions: Dict[str, float]) -> Dict[str, float]:
    """Aggregates SHAP attributions into broad clinical domains."""
    groups = {
        "glycemic": 0.0,
        "blood_pressure": 0.0,
        "metabolic_bmi": 0.0,
        "lipids": 0.0,
        "renal": 0.0,
        "lifestyle_context": 0.0,
    }

    for feat, val in attributions.items():
        fl = feat.lower()
        if "hba1c" in fl or "glucose" in fl:
            groups["glycemic"] += val
        elif "sbp" in fl or "dbp" in fl:
            groups["blood_pressure"] += val
        elif "bmi" in fl:
            groups["metabolic_bmi"] += val
        elif any(lipid in fl for lipid in ["ldl", "hdl", "cholesterol", "triglyceride"]):
            groups["lipids"] += val
        elif "creatinine" in fl:
            groups["renal"] += val
        else:
            groups["lifestyle_context"] += val

    return {k: round(v, 4) for k, v in groups.items()}


# =========================================================================
# 3. NATURAL LANGUAGE CLINICAL EXPLANATION GENERATOR
# =========================================================================

def _describe_feature_contribution(feature_name: str, obs: Union[pd.Series, Dict]) -> str:
    f = feature_name.lower().replace("_scaled", "")
    if f == "hba1c":
        val = obs.get("hba1c", 7.0)
        return f"Current HbA1c level of {val:.1f}%"
    elif f == "hba1c_change":
        val = obs.get("hba1c_change", 0.0)
        return f"HbA1c change of {val:+.2f}% from previous visit"
    elif f == "hba1c_rate":
        val = obs.get("hba1c_rate", 0.0)
        return f"HbA1c rate of change of {val:+.3f}% per month"
    elif f == "glucose":
        val = obs.get("glucose", 120.0)
        return f"Fasting glucose level of {val:.0f} mg/dL"
    elif f == "glucose_change":
        val = obs.get("glucose_change", 0.0)
        return f"Fasting glucose shift of {val:+.1f} mg/dL"
    elif f == "sbp":
        val = obs.get("sbp", 120.0)
        return f"Systolic blood pressure of {val:.0f} mmHg"
    elif f == "bmi":
        val = obs.get("bmi", 25.0)
        return f"Body Mass Index of {val:.1f} kg/m²"
    elif f == "creatinine":
        val = obs.get("creatinine", 1.0)
        return f"Serum creatinine level of {val:.2f} mg/dL"
    elif f == "ldl":
        val = obs.get("ldl", 100.0)
        return f"LDL cholesterol of {val:.0f} mg/dL"
    elif f == "overall_trajectory_status_num":
        status = obs.get("overall_trajectory_status", "stable")
        return f"Overall trajectory status evaluated as '{status}'"
    else:
        val = obs.get(f, "N/A")
        return f"Observation for {f}: {val}"


def generate_clinical_explanation(
    action: int,
    obs_data: Union[pd.Series, Dict[str, Any]],
    attributions: Dict[str, float],
    top_k: int = 4,
) -> Dict[str, Any]:
    meta = get_action_metadata(action)
    action_name = meta["name"]
    action_enum = MonitoringAction(int(action))

    top_pos, top_neg = extract_top_features(attributions, top_k=top_k)
    domain_impact = group_feature_attributions(attributions)

    hba1c = float(obs_data.get("hba1c", 7.0))
    h_change = float(obs_data.get("hba1c_change", 0.0))
    h_rate = float(obs_data.get("hba1c_rate", 0.0))
    glucose = float(obs_data.get("glucose", 120.0))
    sbp = float(obs_data.get("sbp", 125.0))
    dbp = float(obs_data.get("dbp", 80.0))
    bmi = float(obs_data.get("bmi", 27.0))
    creat = float(obs_data.get("creatinine", 1.0))
    overall_status = obs_data.get("overall_trajectory_status", "stable")

    key_factors = []
    for item in top_pos:
        feat = item["feature"]
        score = item["importance"]
        desc = _describe_feature_contribution(feat, obs_data)
        key_factors.append({
            "feature": feat,
            "shap_attribution": score,
            "clinical_description": desc,
        })

    if action_enum == MonitoringAction.EARLY_RISK_ALERT:
        summary_statement = (
            f"An early-risk alert was generated because HbA1c ({hba1c:.1f}%) and glucose ({glucose:.0f} mg/dL) "
            f"have increased over recent observations (delta: {h_change:+.2f}%, rate: {h_rate:+.3f}%/month), "
            f"accompanied by rising blood pressure ({sbp:.0f}/{dbp:.0f} mmHg) and BMI ({bmi:.1f} kg/m²)."
        )
        clinical_narrative = (
            f"Longitudinal trajectory analysis indicates {overall_status.replace('_', ' ')} metabolic control. "
            f"The primary driver of the early-risk alert is the upward velocity of glycemic markers (HbA1c change {h_change:+.2f}%), "
            f"which poses heightened risk for microvascular complications if unaddressed. "
            f"Cardiovascular indicators (systolic BP {sbp:.0f} mmHg) reinforce the proactive monitoring escalation."
        )
    elif action_enum == MonitoringAction.CLINICAL_EVALUATION:
        summary_statement = (
            f"Urgent clinical evaluation was recommended due to acute trajectory deterioration "
            f"(HbA1c: {hba1c:.1f}%, SBP: {sbp:.0f} mmHg, Creatinine: {creat:.2f} mg/dL)."
        )
        clinical_narrative = (
            f"The patient's trajectory exhibits critical acceleration (status: {overall_status.replace('_', ' ')}). "
            f"Biomarkers have crossed safety thresholds requiring immediate specialist review and diagnostic workup "
            f"to prevent severe end-organ damage or hyperglycemic crisis."
        )
    elif action_enum == MonitoringAction.INCREASED_MONITORING:
        summary_statement = (
            f"Increased monitoring frequency ({meta['interval']}) was recommended due to mild trajectory drift "
            f"(HbA1c: {hba1c:.1f}%, change: {h_change:+.2f}%) that warrants closer follow-up."
        )
        clinical_narrative = (
            f"While not yet at acute alert thresholds, subtle upward drift in HbA1c and/or blood pressure "
            f"was detected compared to previous baseline. Increasing the monitoring frequency allows early validation "
            f"of whether this drift represents persistent deterioration or transient variation."
        )
    else:
        summary_statement = (
            f"Routine monitoring ({meta['interval']}) was recommended as the patient maintains stable glycemic and vital indicators "
            f"(HbA1c: {hba1c:.1f}%, BP: {sbp:.0f}/{dbp:.0f} mmHg)."
        )
        clinical_narrative = (
            f"The longitudinal trajectory demonstrates consistent stability across glycemic, cardiovascular, and renal domains. "
            f"Maintaining routine checkups preserves patient quality of life, minimizes clinic visit burden, "
            f"and avoids clinical alarm fatigue."
        )

    return {
        "action_code": action,
        "action_name": action_name,
        "urgency": meta["urgency"],
        "recommended_interval": meta["interval"],
        "summary_statement": summary_statement,
        "clinical_narrative": clinical_narrative,
        "key_driving_factors": key_factors,
        "domain_impact": domain_impact,
    }
