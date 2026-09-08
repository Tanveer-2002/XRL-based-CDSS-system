"""
Unit Tests for Offline Reinforcement Learning (RL) Module.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from rl.actions import MonitoringAction, get_action_metadata
from rl.model import DiscreteCQLNetwork
from rl.reward import compute_monitoring_reward
from rl.state import StateConstructor


def test_action_space():
    assert config.NUM_ACTIONS == 4
    for a in range(4):
        meta = get_action_metadata(a)
        assert "name" in meta
        assert "interval" in meta
        assert "urgency" in meta


def test_state_constructor():
    sc = StateConstructor()
    assert sc.state_dim > 20

    dummy_row = {
        "hba1c": 7.5,
        "hba1c_change": 0.4,
        "hba1c_rate": 0.15,
        "hba1c_moving_avg": 7.3,
        "hba1c_trend_num": 1,
        "glucose": 150.0,
        "age": 62,
        "gender_encoded": 1,
        "smoking_status_encoded": 0,
        "physical_activity_encoded": 1,
        "overall_trajectory_status_num": 1,
    }
    vec = sc.extract_state_vector(dummy_row)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (sc.state_dim,)
    assert vec.dtype == np.float32


def test_clinical_reward_function():
    # 1. Deteriorating patient scenarios
    det_obs = {"overall_trajectory_status": "deteriorating", "hba1c": 8.2, "sbp": 142.0}
    
    # Proactive Early Risk Alert should be rewarded
    r_alert = compute_monitoring_reward(det_obs, MonitoringAction.EARLY_RISK_ALERT)
    assert r_alert > 0, "Early-risk alert should be rewarded for deteriorating trajectory."

    # Inattentive Routine Monitoring should be penalised
    r_routine_det = compute_monitoring_reward(det_obs, MonitoringAction.ROUTINE_MONITORING)
    assert r_routine_det < 0, "Routine monitoring should be penalised for deteriorating trajectory."

    # 2. Stable patient scenarios
    stable_obs = {"overall_trajectory_status": "stable", "hba1c": 6.4, "sbp": 120.0}

    # Routine monitoring should be rewarded
    r_routine_stable = compute_monitoring_reward(stable_obs, MonitoringAction.ROUTINE_MONITORING)
    assert r_routine_stable > 0, "Routine monitoring should be rewarded when patient is stable."

    # False alarm should be penalised (avoid alarm fatigue)
    r_alert_stable = compute_monitoring_reward(stable_obs, MonitoringAction.EARLY_RISK_ALERT)
    assert r_alert_stable < 0, "False alarms on stable patients should receive penalty."


def test_discrete_cql_network():
    state_dim = 35
    model = DiscreteCQLNetwork(state_dim=state_dim, num_actions=4, hidden_dim=64)

    batch_size = 8
    dummy_states = torch.randn(batch_size, state_dim)

    # Forward pass
    q1, q2 = model(dummy_states)
    assert q1.shape == (batch_size, 4)
    assert q2.shape == (batch_size, 4)

    # Action selection
    action = model.select_action(dummy_states[0])
    assert action in [0, 1, 2, 3]

    # Target update
    model.soft_update_targets(tau=0.01)
    target_q1, target_q2 = model.target_q(dummy_states)
    assert target_q1.shape == (batch_size, 4)
