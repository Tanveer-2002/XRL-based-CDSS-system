"""
Consolidated RL Environment Module for Type 2 Diabetes CDSS.

Combines:
1. Discrete Action Space & Metadata (MonitoringAction, get_action_metadata)
2. State Vector Construction (StateConstructor)
3. Grounded Clinical Monitoring Reward Function (compute_monitoring_reward)
4. Offline Transition Generation (infer_behavior_action, generate_offline_transitions)
5. PyTorch Dataset & DataLoader Builders (OfflineRLDataset, create_rl_dataloaders)
"""

from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from preprocessing.pipeline import split_by_patient


# =========================================================================
# 1. DISCRETE ACTION SPACE & CLINICAL METADATA
# =========================================================================

class MonitoringAction(IntEnum):
    ROUTINE_MONITORING = 0
    INCREASED_MONITORING = 1
    EARLY_RISK_ALERT = 2
    CLINICAL_EVALUATION = 3


ACTION_DETAILS = {
    MonitoringAction.ROUTINE_MONITORING: {
        "name": "Routine Monitoring",
        "code": "ROUTINE_MONITORING",
        "interval": "6 months",
        "description": "Maintain standard outpatient monitoring schedule; patient is stable or well-controlled.",
        "urgency": "Low",
        "color": "#28a745",
    },
    MonitoringAction.INCREASED_MONITORING: {
        "name": "Increased Monitoring Frequency",
        "code": "INCREASED_MONITORING",
        "interval": "1 - 3 months",
        "description": "Increase monitoring cadence with repeat HbA1c and BP; mild drift detected.",
        "urgency": "Moderate",
        "color": "#ffc107",
    },
    MonitoringAction.EARLY_RISK_ALERT: {
        "name": "Early-Risk Alert",
        "code": "EARLY_RISK_ALERT",
        "interval": "2 - 4 weeks",
        "description": "Flag early deterioration in trajectory to clinician for timely intervention prior to severe complications.",
        "urgency": "High",
        "color": "#fd7e14",
    },
    MonitoringAction.CLINICAL_EVALUATION: {
        "name": "Clinical Evaluation",
        "code": "CLINICAL_EVALUATION",
        "interval": "Immediate / < 1 week",
        "description": "Urgent comprehensive diagnostic assessment and specialist review due to acute instability or severe risk.",
        "urgency": "Critical",
        "color": "#dc3545",
    },
}


def get_action_metadata(action: int) -> Dict[str, Any]:
    """Returns clinical metadata for an action index."""
    action_enum = MonitoringAction(int(action))
    return ACTION_DETAILS.get(action_enum, ACTION_DETAILS[MonitoringAction.ROUTINE_MONITORING])


# =========================================================================
# 2. STATE VECTOR CONSTRUCTOR
# =========================================================================

class StateConstructor:
    """
    Constructs normalized state vectors combining:
    - Scaled clinical measurements
    - Trajectory features (deltas, rates, moving averages, trend codes)
    - Patient demographic and encounter context
    """

    def __init__(self, biomarkers: Optional[List[str]] = None):
        self.biomarkers = biomarkers or config.BIOMARKERS
        self.feature_names: List[str] = self._determine_feature_names()

    def _determine_feature_names(self) -> List[str]:
        names = []
        for marker in self.biomarkers:
            names.append(f"{marker}_scaled")
        for marker in self.biomarkers:
            names.append(f"{marker}_change_scaled")
            names.append(f"{marker}_rate_scaled")
            names.append(f"{marker}_moving_avg_scaled")
            names.append(f"{marker}_trend_num")
        names.extend([
            "age_scaled",
            "days_since_last_visit_scaled",
            "gender_encoded",
            "smoking_status_encoded",
            "physical_activity_encoded",
            "overall_trajectory_status_num",
        ])
        return names

    @property
    def state_dim(self) -> int:
        return len(self.feature_names)

    def extract_state_matrix(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for feat in self.feature_names:
            if feat in df.columns:
                cols.append(df[feat].fillna(0.0).values)
            else:
                unscaled_name = feat.replace("_scaled", "")
                if unscaled_name in df.columns:
                    cols.append(df[unscaled_name].fillna(0.0).values)
                else:
                    cols.append(np.zeros(len(df)))
        return np.column_stack(cols).astype(np.float32)

    def extract_state_vector(self, row: pd.Series) -> np.ndarray:
        vals = []
        for feat in self.feature_names:
            val = row.get(feat, None)
            if val is None or pd.isna(val):
                unscaled_name = feat.replace("_scaled", "")
                val = row.get(unscaled_name, 0.0)
            vals.append(float(val) if not pd.isna(val) else 0.0)
        return np.array(vals, dtype=np.float32)


# =========================================================================
# 3. GROUNDED CLINICAL REWARD FUNCTION
# =========================================================================

def compute_monitoring_reward(
    curr_obs: Union[pd.Series, Dict],
    action: int,
    next_obs: Optional[Union[pd.Series, Dict]] = None,
) -> float:
    """
    Computes grounded clinical reward for monitoring decisions:
    - Rewards early detection during deterioration (+3.5)
    - Penalizes under-monitoring during deterioration (-3.5 to -4.0)
    - Rewards routine monitoring when stable (+2.5) to avoid alarm fatigue
    - Penalizes false alerts on stable trajectories (-2.5 to -3.5)
    """
    action = int(action)
    status = curr_obs.get("overall_trajectory_status", "stable")
    hba1c = float(curr_obs.get("hba1c", 7.0))
    sbp = float(curr_obs.get("sbp", 125.0))
    creat = float(curr_obs.get("creatinine", 1.0))

    if status == "rapidly_deteriorating" or hba1c >= 9.5 or sbp >= 160.0 or creat >= 2.0:
        if action == MonitoringAction.CLINICAL_EVALUATION:
            reward = 3.0
        elif action == MonitoringAction.EARLY_RISK_ALERT:
            reward = 2.0
        elif action == MonitoringAction.INCREASED_MONITORING:
            reward = -1.0
        else:
            reward = -4.0

    elif status == "deteriorating":
        if action == MonitoringAction.EARLY_RISK_ALERT:
            reward = 3.5
        elif action == MonitoringAction.INCREASED_MONITORING:
            reward = 2.0
        elif action == MonitoringAction.CLINICAL_EVALUATION:
            reward = 1.5 if hba1c >= 8.5 else 0.0
        else:
            reward = -3.5

    elif status == "stable":
        if action == MonitoringAction.ROUTINE_MONITORING:
            reward = 2.5
        elif action == MonitoringAction.INCREASED_MONITORING:
            reward = -0.5
        elif action == MonitoringAction.EARLY_RISK_ALERT:
            reward = -2.5
        else:
            reward = -3.5

    else:  # improving
        if action == MonitoringAction.ROUTINE_MONITORING:
            reward = 2.5
        elif action == MonitoringAction.INCREASED_MONITORING:
            reward = 1.0
        elif action == MonitoringAction.EARLY_RISK_ALERT:
            reward = -2.0
        else:
            reward = -3.5

    if next_obs is not None:
        next_hba1c = float(next_obs.get("hba1c", hba1c))
        next_status = next_obs.get("overall_trajectory_status", "stable")

        if status in ["deteriorating", "rapidly_deteriorating"] and action in [MonitoringAction.EARLY_RISK_ALERT, MonitoringAction.CLINICAL_EVALUATION]:
            if next_status in ["improving", "stable"]:
                reward += 1.0
        elif action == MonitoringAction.ROUTINE_MONITORING and next_hba1c >= 9.0:
            reward -= 1.5

    return round(reward, 2)


# =========================================================================
# 4. OFFLINE TRANSITIONS BUILDER
# =========================================================================

def infer_behavior_action(curr_row: pd.Series, next_row: Optional[pd.Series] = None) -> int:
    """Infers historical monitoring action from follow-up interval and clinical status."""
    if next_row is None:
        status = curr_row.get("overall_trajectory_status", "stable")
        if status == "rapidly_deteriorating":
            return MonitoringAction.CLINICAL_EVALUATION
        elif status == "deteriorating":
            return MonitoringAction.EARLY_RISK_ALERT
        else:
            return MonitoringAction.ROUTINE_MONITORING

    days_to_next = float(next_row.get("days_since_last_visit", 120.0))
    status = curr_row.get("overall_trajectory_status", "stable")
    hba1c = float(curr_row.get("hba1c", 7.0))

    if days_to_next <= 21.0 or hba1c >= 10.0:
        return MonitoringAction.CLINICAL_EVALUATION
    elif days_to_next <= 60.0 or status in ["deteriorating", "rapidly_deteriorating"]:
        return MonitoringAction.EARLY_RISK_ALERT if status.startswith("deteriorating") else MonitoringAction.INCREASED_MONITORING
    elif days_to_next <= 110.0:
        return MonitoringAction.INCREASED_MONITORING
    else:
        return MonitoringAction.ROUTINE_MONITORING


def generate_offline_transitions(
    trajectory_df: pd.DataFrame,
    state_constructor: Optional[StateConstructor] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Extracts (states, actions, rewards, next_states, dones) from trajectories."""
    if state_constructor is None:
        state_constructor = StateConstructor()

    states_list, actions_list, rewards_list, next_states_list, dones_list, rows_list = [], [], [], [], [], []

    for sid, group in trajectory_df.groupby("subject_id"):
        patient_records = group.sort_values(by="chartdate").reset_index(drop=True)
        T = len(patient_records)

        for t in range(T):
            curr_row = patient_records.iloc[t]
            curr_state = state_constructor.extract_state_vector(curr_row)

            is_terminal = (t == T - 1)
            next_row = patient_records.iloc[t + 1] if not is_terminal else None
            next_state = state_constructor.extract_state_vector(next_row) if not is_terminal else curr_state.copy()

            action = infer_behavior_action(curr_row, next_row)
            reward = compute_monitoring_reward(curr_row, action, next_row)

            states_list.append(curr_state)
            actions_list.append(action)
            rewards_list.append(reward)
            next_states_list.append(next_state)
            dones_list.append(1.0 if is_terminal else 0.0)

            rows_list.append({
                "subject_id": sid,
                "visit_number": curr_row.get("visit_number", t + 1),
                "action": action,
                "reward": reward,
                "done": 1 if is_terminal else 0,
                "overall_trajectory_status": curr_row.get("overall_trajectory_status", "stable"),
                "hba1c": curr_row.get("hba1c", 7.0),
                "glucose": curr_row.get("glucose", 120.0),
            })

    states = np.array(states_list, dtype=np.float32)
    actions = np.array(actions_list, dtype=np.int64)
    rewards = np.array(rewards_list, dtype=np.float32)
    next_states = np.array(next_states_list, dtype=np.float32)
    dones = np.array(dones_list, dtype=np.float32)

    transitions_df = pd.DataFrame(rows_list)
    output_path = config.PROCESSED_DATA_DIR / "rl_dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transitions_df.to_csv(output_path, index=False)
    print(f"[OK] Generated {len(states)} Offline RL transitions across {trajectory_df['subject_id'].nunique()} patients at {output_path}")

    return states, actions, rewards, next_states, dones, transitions_df


# =========================================================================
# 5. PYTORCH DATASET & DATALOADERS
# =========================================================================

class OfflineRLDataset(Dataset):
    """PyTorch Dataset encapsulating (s, a, r, s', d) transitions."""

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ):
        self.states = torch.as_tensor(states, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.long)
        self.rewards = torch.as_tensor(rewards, dtype=torch.float32)
        self.next_states = torch.as_tensor(next_states, dtype=torch.float32)
        self.dones = torch.as_tensor(dones, dtype=torch.float32)
        self.length = len(states)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx]


def create_rl_dataloaders(
    trajectory_df: pd.DataFrame,
    state_constructor: Optional[StateConstructor] = None,
    val_split: float = 0.2,
    batch_size: int = 64,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, OfflineRLDataset, OfflineRLDataset]:
    """Splits patient trajectories by subject_id (zero leakage) and builds PyTorch DataLoaders."""
    if state_constructor is None:
        state_constructor = StateConstructor()

    train_traj_df, val_traj_df = split_by_patient(trajectory_df, test_size=val_split, seed=seed)

    print(f"[*] Constructing Offline RL training transitions for {train_traj_df['subject_id'].nunique()} patients...")
    s_tr, a_tr, r_tr, ns_tr, d_tr, _ = generate_offline_transitions(train_traj_df, state_constructor)

    print(f"[*] Constructing Offline RL validation transitions for {val_traj_df['subject_id'].nunique()} patients...")
    s_val, a_val, r_val, ns_val, d_val, _ = generate_offline_transitions(val_traj_df, state_constructor)

    train_dataset = OfflineRLDataset(s_tr, a_tr, r_tr, ns_tr, d_tr)
    val_dataset = OfflineRLDataset(s_val, a_val, r_val, ns_val, d_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=(len(train_dataset) > batch_size))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"[OK] DataLoaders created: Train={len(train_dataset)} transitions, Val={len(val_dataset)} transitions.")
    return train_loader, val_loader, train_dataset, val_dataset
