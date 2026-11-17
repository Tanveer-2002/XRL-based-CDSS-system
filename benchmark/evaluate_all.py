"""
Unified Multi-Model Benchmark Evaluator for Type 2 Diabetes CDSS.

Evaluates all models on the identical patient-level test split:
- Historical Clinician Practice
- XGBoost Supervised Baseline
- Behavioral Cloning (BC) Deep Imitation Baseline
- Discrete BCQ Offline RL Baseline
- Discrete CQL Primary RL Agent
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from preprocessing.pipeline import split_by_patient
from rl.cql import QNetwork, clinical_rule_policy, DiscreteCQLNetwork
from rl.environment import (
    MonitoringAction,
    StateConstructor,
    compute_monitoring_reward,
    infer_behavior_action,
)
from xgboost.model import XGBoostAgent
from bc.model import BehavioralCloningPolicy
from bcq.model import DiscreteBCQNetwork


def load_all_models(
    state_dim: int,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Loads all available model checkpoints."""
    models = {}

    # 1. XGBoost
    xgb_path = PROJECT_ROOT / "xgboost" / "xgboost_model.json"
    if xgb_path.exists():
        xgb_agent = XGBoostAgent(num_class=config.NUM_ACTIONS)
        xgb_agent.load_model(xgb_path)
        models["XGBoost"] = {
            "agent": xgb_agent,
            "type": "Supervised ML",
            "predict_fn": lambda s: xgb_agent.select_action(s, deterministic=True),
        }
    else:
        print(f"[!] XGBoost checkpoint not found at {xgb_path}")

    # 2. Behavioral Cloning (BC)
    bc_path = PROJECT_ROOT / "bc" / "bc_model.pt"
    if bc_path.exists():
        bc_model = BehavioralCloningPolicy(state_dim=state_dim, num_actions=config.NUM_ACTIONS)
        bc_model.load_checkpoint(bc_path, device=device)
        models["Behavioral Cloning (BC)"] = {
            "agent": bc_model,
            "type": "Imitation Learning",
            "predict_fn": lambda s: bc_model.select_action(s, deterministic=True),
        }
    else:
        print(f"[!] BC checkpoint not found at {bc_path}")

    # 3. Discrete BCQ
    bcq_path = PROJECT_ROOT / "bcq" / "bcq_model.pt"
    if bcq_path.exists():
        bcq_model = DiscreteBCQNetwork(state_dim=state_dim, num_actions=config.NUM_ACTIONS)
        bcq_model.load_checkpoint(bcq_path, device=device)
        models["Discrete BCQ"] = {
            "agent": bcq_model,
            "type": "Offline RL (Constraint)",
            "predict_fn": lambda s: bcq_model.select_action(s, deterministic=True),
        }
    else:
        print(f"[!] BCQ checkpoint not found at {bcq_path}")

    # 4. Discrete CQL (Ours)
    cql_path = config.RL_AGENT_DIR / "cql_model.pt"
    if cql_path.exists():
        checkpoint = torch.load(cql_path, map_location=device)
        cql_model = DiscreteCQLNetwork(state_dim=state_dim, num_actions=config.NUM_ACTIONS).to(device)
        cql_model.load_state_dict(checkpoint["model_state_dict"])
        cql_model.eval()
        models["Discrete CQL (Ours)"] = {
            "agent": cql_model,
            "type": "Offline RL (Conservative Q)",
            "predict_fn": lambda s: cql_model.select_action(
                torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0),
                deterministic=True,
            ),
        }
    else:
        print(f"[!] CQL checkpoint not found at {cql_path}")

    return models


def compute_fitted_q_evaluation(
    predict_fn: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sc: StateConstructor,
    state_dim: int,
    epochs: int = 20,
    gamma: float = 0.99,
    device: str = "cpu",
) -> float:
    """
    Fits an independent Q-network to evaluate the expected return of target policy pi
    using offline transition data (Fitted Q-Evaluation, FQE).
    """
    transitions = []
    for _, group in train_df.groupby("subject_id"):
        records = group.sort_values(by="chartdate").reset_index(drop=True)
        T = len(records)
        for t in range(T):
            curr_row = records.iloc[t]
            next_row = records.iloc[t + 1] if t < T - 1 else None
            s = sc.extract_state_vector(curr_row)
            a = infer_behavior_action(curr_row, next_row)
            r = compute_monitoring_reward(curr_row, a, next_row)
            done = 1.0 if next_row is None else 0.0
            s_next = sc.extract_state_vector(next_row) if next_row is not None else np.zeros_like(s)
            transitions.append((s, a, r, s_next, done))

    if not transitions:
        return 0.0

    states = torch.tensor(np.array([t[0] for t in transitions]), dtype=torch.float32, device=device)
    actions = torch.tensor(np.array([t[1] for t in transitions]), dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.tensor(np.array([t[2] for t in transitions]), dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.tensor(np.array([t[3] for t in transitions]), dtype=torch.float32, device=device)
    dones = torch.tensor(np.array([t[4] for t in transitions]), dtype=torch.float32, device=device).unsqueeze(1)

    # Initial states (baseline encounters of test patients)
    initial_states = []
    for _, group in test_df.groupby("subject_id"):
        records = group.sort_values(by="chartdate").reset_index(drop=True)
        initial_states.append(sc.extract_state_vector(records.iloc[0]))
    initial_states_tensor = torch.tensor(np.array(initial_states), dtype=torch.float32, device=device)

    q_net = QNetwork(state_dim, config.NUM_ACTIONS).to(device)
    target_q_net = QNetwork(state_dim, config.NUM_ACTIONS).to(device)
    target_q_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Precompute target policy actions on next states
    with torch.no_grad():
        next_acts_list = [int(predict_fn(ns.cpu().numpy())) for ns in next_states]
        next_actions = torch.tensor(next_acts_list, dtype=torch.long, device=device).unsqueeze(1)

    for _ in range(epochs):
        with torch.no_grad():
            next_q = target_q_net(next_states).gather(1, next_actions)
            target = rewards + gamma * (1.0 - dones) * next_q
        current_q = q_net(states).gather(1, actions)
        loss = loss_fn(current_q, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for tp, p in zip(target_q_net.parameters(), q_net.parameters()):
            tp.data.copy_(0.05 * p.data + 0.95 * tp.data)

    with torch.no_grad():
        init_acts_list = [int(predict_fn(s0.cpu().numpy())) for s0 in initial_states_tensor]
        init_actions = torch.tensor(init_acts_list, dtype=torch.long, device=device).unsqueeze(1)
        init_v = q_net(initial_states_tensor).gather(1, init_actions).mean().item()

    return float(init_v)


def evaluate_policy_on_split(
    name: str,
    predict_fn: Any,
    test_df: pd.DataFrame,
    sc: StateConstructor,
    gamma: float = 0.99,
) -> Dict[str, Any]:
    """Evaluates a single model or heuristic policy on patient trajectories."""
    y_true_clinical = []
    y_behavior = []
    y_pred = []
    trajectory_returns = []

    deteriorating_total = 0
    early_detected = 0
    stable_total = 0
    false_alarms = 0

    total_patient_years = 0.0
    lead_times = []

    grouped = test_df.groupby("subject_id")
    for _, group in grouped:
        records = group.sort_values(by="chartdate").reset_index(drop=True)
        T = len(records)
        traj_ret = 0.0

        # Calculate patient observation span in years
        min_days = float(records["cumulative_days"].min()) if "cumulative_days" in records else 0.0
        max_days = float(records["cumulative_days"].max()) if "cumulative_days" in records else 0.0
        span_days = max(max_days - min_days, float(records["days_since_last_visit"].sum()) if "days_since_last_visit" in records else 30.0)
        span_days = max(span_days, 30.0)
        total_patient_years += span_days / 365.25

        patient_preds = []
        patient_clin_rules = []
        patient_statuses = []

        for t in range(T):
            curr_row = records.iloc[t]
            next_row = records.iloc[t + 1] if t < T - 1 else None

            a_clinical = clinical_rule_policy(curr_row)
            a_behav = infer_behavior_action(curr_row, next_row)

            s_vec = sc.extract_state_vector(curr_row)
            a_pred = predict_fn(curr_row, s_vec, a_behav)

            r_step = compute_monitoring_reward(curr_row, a_pred, next_row)
            traj_ret += (gamma ** t) * r_step

            y_true_clinical.append(a_clinical)
            y_behavior.append(a_behav)
            y_pred.append(a_pred)
            patient_preds.append(a_pred)
            patient_clin_rules.append(a_clinical)

            status = curr_row.get("overall_trajectory_status", "stable")
            patient_statuses.append(status)

            # Safety metrics
            if status in ["deteriorating", "rapidly_deteriorating"]:
                deteriorating_total += 1
                if a_pred in [
                    MonitoringAction.INCREASED_MONITORING,
                    MonitoringAction.EARLY_RISK_ALERT,
                    MonitoringAction.CLINICAL_EVALUATION,
                ]:
                    early_detected += 1
            elif status == "stable":
                stable_total += 1
                if a_pred in [
                    MonitoringAction.EARLY_RISK_ALERT,
                    MonitoringAction.CLINICAL_EVALUATION,
                ]:
                    false_alarms += 1

        trajectory_returns.append(traj_ret)

        # Lead time calculation for acute deterioration onset episodes
        for t in range(T):
            status = patient_statuses[t]
            a_clin = patient_clin_rules[t]
            is_critical = (
                a_clin == MonitoringAction.CLINICAL_EVALUATION
                or status == "rapidly_deteriorating"
                or (status == "deteriorating" and a_clin >= MonitoringAction.EARLY_RISK_ALERT)
            )

            # Detect onset of critical episode (first visit or preceded by non-critical)
            prev_crit = False
            if t > 0:
                prev_status = patient_statuses[t - 1]
                prev_clin = patient_clin_rules[t - 1]
                prev_crit = (
                    prev_clin == MonitoringAction.CLINICAL_EVALUATION
                    or prev_status == "rapidly_deteriorating"
                    or (prev_status == "deteriorating" and prev_clin >= MonitoringAction.EARLY_RISK_ALERT)
                )

            if is_critical and not prev_crit:
                # Find earliest continuous active escalation at or before t
                t_alert = None
                for k in range(t, -1, -1):
                    if patient_preds[k] in [
                        MonitoringAction.INCREASED_MONITORING,
                        MonitoringAction.EARLY_RISK_ALERT,
                        MonitoringAction.CLINICAL_EVALUATION,
                    ]:
                        t_alert = k
                    else:
                        break
                if t_alert is not None:
                    lt = float(records.iloc[t]["cumulative_days"]) - float(records.iloc[t_alert]["cumulative_days"])
                else:
                    lt = 0.0
                lead_times.append(lt)

    acc_guideline = accuracy_score(y_true_clinical, y_pred) * 100.0
    acc_behavior = accuracy_score(y_behavior, y_pred) * 100.0
    bal_acc = balanced_accuracy_score(y_true_clinical, y_pred) * 100.0
    macro_precision = precision_score(y_true_clinical, y_pred, average="macro", zero_division=0) * 100.0
    macro_recall = recall_score(y_true_clinical, y_pred, average="macro", zero_division=0) * 100.0
    f1_macro = f1_score(y_true_clinical, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true_clinical, y_pred, average="weighted", zero_division=0)

    sensitivity = (early_detected / deteriorating_total * 100.0) if deteriorating_total > 0 else 0.0
    false_alarm_rate = (false_alarms / stable_total * 100.0) if stable_total > 0 else 0.0
    false_alerts_per_pt_year = false_alarms / total_patient_years if total_patient_years > 0 else 0.0
    median_lead_time = float(np.median(lead_times)) if lead_times else 0.0
    mean_lead_time = float(np.mean(lead_times)) if lead_times else 0.0
    adjacent_acc = float(np.mean(np.abs(np.array(y_true_clinical) - np.array(y_pred)) <= 1) * 100.0)
    mean_return = float(np.mean(trajectory_returns))

    cm = confusion_matrix(y_true_clinical, y_pred, labels=list(range(4)))
    report = classification_report(
        y_true_clinical,
        y_pred,
        target_names=[config.ACTION_NAMES[i] for i in range(4)],
        output_dict=True,
        zero_division=0,
    )

    return {
        "model_name": name,
        "accuracy_guideline": acc_guideline,
        "accuracy_behavior": acc_behavior,
        "balanced_accuracy": bal_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "adjacent_accuracy": adjacent_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "sensitivity": sensitivity,
        "false_alarm_rate": false_alarm_rate,
        "false_alerts_per_patient_year": false_alerts_per_pt_year,
        "median_lead_time": median_lead_time,
        "mean_lead_time": mean_lead_time,
        "mean_return": mean_return,
        "confusion_matrix": cm,
        "report": report,
        "y_true": y_true_clinical,
        "y_pred": y_pred,
    }


def run_full_benchmark(
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Runs end-to-end benchmark across all models and baseline policies."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sc = StateConstructor()

    data_path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Trajectory data not found at {data_path}")

    df = pd.read_csv(data_path)
    train_df, test_df = split_by_patient(df, test_size=test_size, seed=seed)

    models_dict = load_all_models(state_dim=sc.state_dim, device=device)

    all_results = {}
    table_rows = []

    # 1. Historical Clinician Policy Baseline
    clinician_res = evaluate_policy_on_split(
        name="Historical Clinician",
        predict_fn=lambda row, s, a_behav: a_behav,
        test_df=test_df,
        sc=sc,
    )
    all_results["Historical Clinician"] = clinician_res

    # FQE for Historical Clinician
    clinician_fqe = compute_fitted_q_evaluation(
        predict_fn=lambda s: 0, # baseline default
        train_df=train_df,
        test_df=test_df,
        sc=sc,
        state_dim=sc.state_dim,
        device=device,
    )

    table_rows.append({
        "Model": "Historical Clinician",
        "Category": "Empirical Baseline",
        "Guideline Acc (%)": round(clinician_res["accuracy_guideline"], 2),
        "Balanced Acc (%)": round(clinician_res["balanced_accuracy"], 2),
        "Clinician Match (%)": 100.0,
        "Macro F1": round(clinician_res["f1_macro"], 4),
        "Sensitivity (%)": round(clinician_res["sensitivity"], 2),
        "False Alarm (%)": round(clinician_res["false_alarm_rate"], 2),
        "False / Pt-Yr": round(clinician_res["false_alerts_per_patient_year"], 2),
        "Median Lead Time (d)": round(clinician_res["median_lead_time"], 1),
        "Adjacent Acc (%)": round(clinician_res["adjacent_accuracy"], 2),
        "Mean Return (V)": round(clinician_res["mean_return"], 2),
        "FQE Value": round(clinician_fqe, 2),
    })

    # 2. Machine Learning & RL Models
    for name, info in models_dict.items():
        res = evaluate_policy_on_split(
            name=name,
            predict_fn=lambda row, s, a_b, fn=info["predict_fn"]: fn(s),
            test_df=test_df,
            sc=sc,
        )
        all_results[name] = res

        # Compute FQE value
        model_fqe = compute_fitted_q_evaluation(
            predict_fn=info["predict_fn"],
            train_df=train_df,
            test_df=test_df,
            sc=sc,
            state_dim=sc.state_dim,
            device=device,
        )

        table_rows.append({
            "Model": name,
            "Category": info["type"],
            "Guideline Acc (%)": round(res["accuracy_guideline"], 2),
            "Balanced Acc (%)": round(res["balanced_accuracy"], 2),
            "Clinician Match (%)": round(res["accuracy_behavior"], 2),
            "Macro F1": round(res["f1_macro"], 4),
            "Sensitivity (%)": round(res["sensitivity"], 2),
            "False Alarm (%)": round(res["false_alarm_rate"], 2),
            "False / Pt-Yr": round(res["false_alerts_per_patient_year"], 2),
            "Median Lead Time (d)": round(res["median_lead_time"], 1),
            "Adjacent Acc (%)": round(res["adjacent_accuracy"], 2),
            "Mean Return (V)": round(res["mean_return"], 2),
            "FQE Value": round(model_fqe, 2),
        })

    summary_df = pd.DataFrame(table_rows)
    return summary_df, all_results
