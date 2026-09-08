"""
Continual RL Feedback & Longitudinal Outcome Manager for Type 2 Diabetes CDSS.

Combines:
- Strategy 1: Longitudinal Follow-up Outcome Ingestion (closing the (s, a, r, s') loop)
- Strategy 2: Clinician-in-the-Loop Feedback & Expert Overrides (RLHF for CDSS)
- Multi-Source Experience Replay Buffer preparation for safe incremental fine-tuning.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from rl.environment import (
    MonitoringAction,
    StateConstructor,
    compute_monitoring_reward,
    get_action_metadata,
)
from trajectory.pipeline import (
    build_trajectory_features,
    evaluate_trajectory_status,
)


class FeedbackManager:
    """
    Manages dual-source continual learning data:
    1. Clinician reviews and action overrides (Strategy 2)
    2. Completed longitudinal follow-up outcomes (Strategy 1)
    """

    def __init__(self, feedback_dir: Optional[Path] = None):
        self.feedback_dir = feedback_dir or config.FEEDBACK_DIR
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

        self.feedback_file = self.feedback_dir / "clinician_feedback.csv"
        self.pending_file = self.feedback_dir / "pending_recommendations.csv"
        self.completed_file = self.feedback_dir / "completed_transitions.csv"

        self.state_constructor = StateConstructor()
        self._init_storage()

    def _init_storage(self):
        """Initializes empty storage CSV files with headers if they do not exist."""
        if not self.feedback_file.exists():
            df = pd.DataFrame(columns=[
                "feedback_id", "timestamp", "subject_id", "model_action",
                "clinician_action", "is_override", "notes", "state_vector_json"
            ])
            df.to_csv(self.feedback_file, index=False)

        if not self.pending_file.exists():
            df = pd.DataFrame(columns=[
                "pending_id", "timestamp", "subject_id", "chartdate",
                "recommended_action", "status", "state_vector_json", "obs_json"
            ])
            df.to_csv(self.pending_file, index=False)

        if not self.completed_file.exists():
            df = pd.DataFrame(columns=[
                "transition_id", "timestamp", "subject_id", "source",
                "action", "reward", "done", "state_vector_json", "next_state_vector_json"
            ])
            df.to_csv(self.completed_file, index=False)

    # =========================================================================
    # STRATEGY 2: CLINICIAN-IN-THE-LOOP FEEDBACK & OVERRIDES
    # =========================================================================

    def record_clinician_feedback(
        self,
        subject_id: str,
        model_action: int,
        clinician_action: int,
        notes: str = "",
        state_vector: Optional[np.ndarray] = None,
        obs_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Records physician review of a CDSS recommendation:
        - If clinician agrees: confirms model choice (reinforcement bonus).
        - If clinician overrides: treats doctor's decision as expert demonstration.
        """
        model_action = int(model_action)
        clinician_action = int(clinician_action)
        is_override = (model_action != clinician_action)
        now_str = datetime.now().isoformat()
        feedback_id = f"FB-{subject_id}-{datetime.now().strftime('%Y%m%d%H%M%S%f')[:18]}"

        if state_vector is None and obs_data is not None:
            state_vector = self.state_constructor.extract_state_vector(pd.Series(obs_data))
        elif state_vector is None:
            state_vector = np.zeros(self.state_constructor.state_dim, dtype=np.float32)

        state_json = json.dumps(state_vector.tolist())

        # 1. Log to clinician feedback table
        fb_row = {
            "feedback_id": feedback_id,
            "timestamp": now_str,
            "subject_id": str(subject_id),
            "model_action": model_action,
            "clinician_action": clinician_action,
            "is_override": int(is_override),
            "notes": str(notes),
            "state_vector_json": state_json,
        }
        fb_df = pd.DataFrame([fb_row])
        fb_df.to_csv(self.feedback_file, mode="a", header=not self.feedback_file.exists(), index=False)

        # 2. Generate training transition from expert demonstration
        # If doctor overrode, the doctor's action gets high reward (+3.5), model's action gets penalized
        reward = 3.5 if is_override else 2.5
        trans_id = f"TR-FB-{feedback_id}"

        completed_row = {
            "transition_id": trans_id,
            "timestamp": now_str,
            "subject_id": str(subject_id),
            "source": "clinician_override" if is_override else "clinician_agreement",
            "action": clinician_action,
            "reward": reward,
            "done": 0,
            "state_vector_json": state_json,
            "next_state_vector_json": state_json,  # Same state for static override evaluation
        }
        pd.DataFrame([completed_row]).to_csv(
            self.completed_file, mode="a", header=not self.completed_file.exists(), index=False
        )

        return {
            "status": "success",
            "feedback_id": feedback_id,
            "is_override": is_override,
            "reward_assigned": reward,
            "message": "Clinician override recorded as expert demonstration." if is_override else "Clinician agreement recorded.",
        }

    # =========================================================================
    # STRATEGY 1: LONGITUDINAL OUTCOME INGESTION (DELAYED REWARD COMPLETION)
    # =========================================================================

    def log_pending_recommendation(
        self,
        subject_id: str,
        action: int,
        state_vector: np.ndarray,
        obs_data: Dict[str, Any],
        chartdate: Optional[str] = None,
    ) -> str:
        """Logs an active recommendation waiting for the patient's future encounter."""
        pending_id = f"PEND-{subject_id}-{datetime.now().strftime('%Y%m%d%H%M%S%f')[:18]}"
        now_str = datetime.now().isoformat()
        cdate = chartdate or obs_data.get("chartdate", now_str[:10])

        # Deduplicate: if an identical recommendation is already pending for this patient visit, avoid duplicate disk write
        if self.pending_file.exists():
            try:
                p_df = pd.read_csv(self.pending_file)
                if not p_df.empty and "subject_id" in p_df.columns:
                    existing = p_df[
                        (p_df["subject_id"].astype(str) == str(subject_id)) &
                        (p_df["chartdate"].astype(str) == str(cdate)) &
                        (p_df["recommended_action"] == int(action)) &
                        (p_df["status"] == "pending")
                    ]
                    if not existing.empty:
                        return str(existing.iloc[-1]["pending_id"])
            except Exception:
                pass

        pending_row = {
            "pending_id": pending_id,
            "timestamp": now_str,
            "subject_id": str(subject_id),
            "chartdate": str(cdate),
            "recommended_action": int(action),
            "status": "pending",
            "state_vector_json": json.dumps(state_vector.tolist()),
            "obs_json": json.dumps(obs_data),
        }
        pd.DataFrame([pending_row]).to_csv(
            self.pending_file, mode="a", header=not self.pending_file.exists(), index=False
        )
        return pending_id

    def register_patient_encounter(
        self,
        subject_id: str,
        encounter_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Ingests a new encounter for a patient (e.g. 3 months later):
        1. Appends new visit to the patient's trajectory records.
        2. Checks for pending recommendations from previous visits.
        3. If found, calculates biological outcome reward and completes the (s, a, r, s', done) transition!
        """
        subject_id = str(subject_id)
        now_str = datetime.now().isoformat()
        chartdate = str(encounter_data.get("chartdate", now_str[:10]))

        # Load existing trajectory dataset
        traj_path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"
        if not traj_path.exists():
            all_df = pd.DataFrame()
        else:
            all_df = pd.read_csv(traj_path)

        patient_records = all_df[all_df["subject_id"].astype(str) == subject_id].sort_values(by="chartdate").copy()

        # Build combined history
        if not patient_records.empty:
            new_row = patient_records.iloc[-1].to_dict()
            new_row.update(encounter_data)
        else:
            new_row = dict(encounter_data)
            new_row["age"] = int(encounter_data.get("age", 60))
            new_row["gender"] = str(encounter_data.get("gender", "Unknown"))
            new_row["smoking_status"] = str(encounter_data.get("smoking_status", "never"))
            new_row["physical_activity"] = str(encounter_data.get("physical_activity", "moderate"))

        new_row["subject_id"] = subject_id
        new_row["chartdate"] = chartdate

        # Fill defaults for missing biomarkers
        for marker in config.BIOMARKERS:
            if marker not in new_row or pd.isna(new_row[marker]):
                new_row[marker] = 7.0 if marker == "hba1c" else 120.0 if marker == "glucose" else 125.0 if marker == "sbp" else 1.0

        for cat in config.CATEGORICAL_FEATURES:
            if cat not in new_row or pd.isna(new_row[cat]):
                new_row[cat] = "never" if "smoking" in cat else "moderate" if "activity" in cat else "Unknown"

        combined_patient_df = pd.concat([patient_records, pd.DataFrame([new_row])], ignore_index=True)
        # Drop previous calculated trajectory columns before recomputing
        drop_cols = [c for c in combined_patient_df.columns if c.endswith("_change") or c.endswith("_rate") or c.endswith("_moving_avg") or c.endswith("_trend") or c.endswith("_trend_num") or c in ["visit_number", "days_since_last_visit", "cumulative_days", "overall_trajectory_status"]]
        clean_history = combined_patient_df.drop(columns=drop_cols, errors="ignore")

        # Re-compute trajectory features
        traj_df, _ = build_trajectory_features(clean_history)
        traj_df["overall_trajectory_status"] = traj_df.apply(evaluate_trajectory_status, axis=1)

        # Update disk trajectory data
        if not all_df.empty:
            all_df = all_df[all_df["subject_id"].astype(str) != subject_id]
            updated_all = pd.concat([all_df, traj_df], ignore_index=True)
        else:
            updated_all = traj_df

        updated_all.to_csv(traj_path, index=False)

        # 2. Check for pending recommendations to close the loop
        closed_transitions = []
        if self.pending_file.exists():
            pending_df = pd.read_csv(self.pending_file)
            patient_pending = pending_df[(pending_df["subject_id"].astype(str) == subject_id) & (pending_df["status"] == "pending")]

            if not patient_pending.empty:
                # Latest pending observation
                last_pending = patient_pending.iloc[-1]
                prev_obs = json.loads(last_pending["obs_json"])
                action = int(last_pending["recommended_action"])
                prev_state_vec = np.array(json.loads(last_pending["state_vector_json"]), dtype=np.float32)

                # Next state vector
                latest_processed_row = traj_df.iloc[-1]
                next_state_vec = self.state_constructor.extract_state_vector(latest_processed_row)

                # Compute true longitudinal outcome reward!
                outcome_reward = compute_monitoring_reward(
                    curr_obs=prev_obs,
                    action=action,
                    next_obs=latest_processed_row.to_dict(),
                )

                # Record completed transition
                trans_id = f"TR-OUTCOME-{last_pending['pending_id']}"
                completed_row = {
                    "transition_id": trans_id,
                    "timestamp": now_str,
                    "subject_id": subject_id,
                    "source": "longitudinal_outcome",
                    "action": action,
                    "reward": outcome_reward,
                    "done": 0,
                    "state_vector_json": json.dumps(prev_state_vec.tolist()),
                    "next_state_vector_json": json.dumps(next_state_vec.tolist()),
                }
                pd.DataFrame([completed_row]).to_csv(
                    self.completed_file, mode="a", header=not self.completed_file.exists(), index=False
                )

                # Mark pending as fulfilled
                pending_df.loc[pending_df["pending_id"] == last_pending["pending_id"], "status"] = "fulfilled"
                pending_df.to_csv(self.pending_file, index=False)

                closed_transitions.append({
                    "transition_id": trans_id,
                    "action": action,
                    "outcome_reward": outcome_reward,
                })

        return {
            "status": "success",
            "subject_id": subject_id,
            "new_visit_number": len(traj_df),
            "overall_trajectory_status": traj_df.iloc[-1]["overall_trajectory_status"],
            "closed_transitions": closed_transitions,
            "message": f"Successfully registered encounter for patient {subject_id} and updated longitudinal trajectory.",
        }

    # =========================================================================
    # STATS & RETRAINING DATASET PREPARATION
    # =========================================================================

    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Returns statistics on clinician feedback and completed outcomes."""
        stats = {
            "total_clinician_feedback": 0,
            "clinician_accepts": 0,
            "clinician_overrides": 0,
            "override_rate_percent": 0.0,
            "pending_followups": 0,
            "completed_transitions": 0,
            "clinician_override_transitions": 0,
            "longitudinal_outcome_transitions": 0,
        }

        if self.feedback_file.exists():
            fb_df = pd.read_csv(self.feedback_file)
            if not fb_df.empty:
                stats["total_clinician_feedback"] = len(fb_df)
                stats["clinician_overrides"] = int(fb_df["is_override"].sum())
                stats["clinician_accepts"] = len(fb_df) - stats["clinician_overrides"]
                stats["override_rate_percent"] = round((stats["clinician_overrides"] / len(fb_df)) * 100, 1)

        if self.pending_file.exists():
            p_df = pd.read_csv(self.pending_file)
            if not p_df.empty:
                stats["pending_followups"] = int((p_df["status"] == "pending").sum())

        if self.completed_file.exists():
            c_df = pd.read_csv(self.completed_file)
            if not c_df.empty:
                stats["completed_transitions"] = len(c_df)
                stats["clinician_override_transitions"] = int((c_df["source"] == "clinician_override").sum())
                stats["longitudinal_outcome_transitions"] = int((c_df["source"] == "longitudinal_outcome").sum())

        return stats

    def export_retraining_dataset(
        self,
        base_dataset_path: Optional[Path] = None,
        feedback_ratio: float = 0.3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Builds a balanced retraining dataset using Experience Replay:
        - Incorporates base historical transitions (1 - feedback_ratio)
        - Ingests clinician feedback + longitudinal outcome transitions (feedback_ratio)
        Prevents catastrophic forgetting while prioritizing new clinical insights.
        """
        if base_dataset_path is None:
            base_dataset_path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"

        # 1. Base transitions
        from rl.environment import generate_offline_transitions
        traj_df = pd.read_csv(base_dataset_path)
        b_s, b_a, b_r, b_ns, b_d, _ = generate_offline_transitions(traj_df, self.state_constructor)

        # 2. Check for completed transitions from feedback & outcomes
        if not self.completed_file.exists():
            return b_s, b_a, b_r, b_ns, b_d

        c_df = pd.read_csv(self.completed_file)
        if c_df.empty:
            return b_s, b_a, b_r, b_ns, b_d

        f_s = np.array([json.loads(s) for s in c_df["state_vector_json"]], dtype=np.float32)
        f_a = c_df["action"].values.astype(np.int64)
        f_r = c_df["reward"].values.astype(np.float32)
        f_ns = np.array([json.loads(ns) for ns in c_df["next_state_vector_json"]], dtype=np.float32)
        f_d = c_df["done"].values.astype(np.float32)

        # 3. Incorporate feedback transitions with calibrated replay scaling
        max_ratio_count = int(len(b_s) * (feedback_ratio / (1.0 - feedback_ratio)))
        target_feedback_count = min(len(f_s) * 3, max_ratio_count)
        target_feedback_count = max(len(f_s), target_feedback_count)
        indices = np.random.choice(len(f_s), size=target_feedback_count, replace=True)

        merged_s = np.concatenate([b_s, f_s[indices]], axis=0)
        merged_a = np.concatenate([b_a, f_a[indices]], axis=0)
        merged_r = np.concatenate([b_r, f_r[indices]], axis=0)
        merged_ns = np.concatenate([b_ns, f_ns[indices]], axis=0)
        merged_d = np.concatenate([b_d, f_d[indices]], axis=0)

        # Shuffle
        perm = np.random.permutation(len(merged_s))
        return merged_s[perm], merged_a[perm], merged_r[perm], merged_ns[perm], merged_d[perm]
