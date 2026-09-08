"""
Consolidated Discrete Conservative Q-Learning (CQL) Module for Type 2 Diabetes CDSS.

Combines:
1. Twin Q-Network Architecture (QNetwork, DiscreteCQLNetwork)
2. Double Q-Learning & Conservative Penalty Loss Training Loop (train_cql_epoch, evaluate_cql_val, train_cql_agent)
3. Clinical Rule Policy Heuristic (clinical_rule_policy)
4. Offline Policy Evaluation Benchmarking (evaluate_policy)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from .environment import (
    MonitoringAction,
    StateConstructor,
    compute_monitoring_reward,
    get_action_metadata,
)


# =========================================================================
# 1. DISCRETE CQL TWIN Q-NETWORK ARCHITECTURE
# =========================================================================

class QNetwork(nn.Module):
    """Deep Q-Network mapping state s to Q-values for all discrete actions."""

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DiscreteCQLNetwork(nn.Module):
    """
    Twin Q-Network Discrete Conservative Q-Learning (CQL) architecture:
    - Q1, Q2: Primary twin estimators (mitigates overestimation bias).
    - target_q1, target_q2: Polyak delayed target networks.
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int = config.NUM_ACTIONS,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions

        self.q1 = QNetwork(state_dim, num_actions, hidden_dim)
        self.q2 = QNetwork(state_dim, num_actions, hidden_dim)

        self.target_q1 = QNetwork(state_dim, num_actions, hidden_dim)
        self.target_q2 = QNetwork(state_dim, num_actions, hidden_dim)

        self.hard_update_targets()

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state), self.q2(state)

    def target_q(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.target_q1(state), self.target_q2(state)

    def select_action(self, state: torch.Tensor, deterministic: bool = True) -> int:
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            state = state.to(device)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            q_vals = self.q1(state)
            if deterministic:
                action = torch.argmax(q_vals, dim=-1).item()
            else:
                probs = nn.functional.softmax(q_vals, dim=-1)
                action = torch.multinomial(probs, 1).item()
        return action

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            state = state.to(device)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            q1_vals = self.q1(state)
        return q1_vals.squeeze(0)

    def soft_update_targets(self, tau: float = 0.005):
        for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def hard_update_targets(self):
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())


# =========================================================================
# 2. DISCRETE CQL TRAINING LOOP
# =========================================================================

def train_cql_epoch(
    model: DiscreteCQLNetwork,
    dataloader,
    optimizer: optim.Optimizer,
    gamma: float = 0.99,
    cql_alpha: float = 1.0,
    tau: float = 0.005,
    device: str = "cpu",
) -> Dict[str, float]:
    model.train()
    total_loss, total_td_loss, total_cql_loss, total_q_val, batch_count = 0.0, 0.0, 0.0, 0.0, 0

    for states, actions, rewards, next_states, dones in dataloader:
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        # 1. Double Q Target
        with torch.no_grad():
            next_q1, _ = model(next_states)
            next_actions = torch.argmax(next_q1, dim=-1, keepdim=True)
            target_next_q1, target_next_q2 = model.target_q(next_states)
            min_target_next_q = torch.min(
                target_next_q1.gather(1, next_actions),
                target_next_q2.gather(1, next_actions),
            ).squeeze(1)
            target_q = rewards + (1.0 - dones) * gamma * min_target_next_q

        # 2. Predicted Q-values
        q1_all, q2_all = model(states)
        q1 = q1_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2 = q2_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 3. Bellman TD Loss
        td_loss_1 = nn.functional.mse_loss(q1, target_q)
        td_loss_2 = nn.functional.mse_loss(q2, target_q)
        td_loss = 0.5 * (td_loss_1 + td_loss_2)

        # 4. Discrete CQL Penalty: logsumexp(Q(s, a)) - Q(s, a_data)
        cql_loss_1 = torch.logsumexp(q1_all, dim=-1).mean() - q1.mean()
        cql_loss_2 = torch.logsumexp(q2_all, dim=-1).mean() - q2.mean()
        cql_loss = 0.5 * (cql_loss_1 + cql_loss_2)

        # 5. Total Loss
        loss = td_loss + cql_alpha * cql_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        model.soft_update_targets(tau)

        total_loss += loss.item()
        total_td_loss += td_loss.item()
        total_cql_loss += cql_loss.item()
        total_q_val += q1.mean().item()
        batch_count += 1

    return {
        "loss": total_loss / max(1, batch_count),
        "td_loss": total_td_loss / max(1, batch_count),
        "cql_loss": total_cql_loss / max(1, batch_count),
        "avg_q": total_q_val / max(1, batch_count),
    }


def evaluate_cql_val(
    model: DiscreteCQLNetwork,
    dataloader,
    gamma: float = 0.99,
    device: str = "cpu",
) -> float:
    model.eval()
    total_val_loss, batch_count = 0.0, 0

    with torch.no_grad():
        for states, actions, rewards, next_states, dones in dataloader:
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)

            next_q1, _ = model(next_states)
            next_actions = torch.argmax(next_q1, dim=-1, keepdim=True)
            target_next_q1, target_next_q2 = model.target_q(next_states)
            min_target_next_q = torch.min(
                target_next_q1.gather(1, next_actions),
                target_next_q2.gather(1, next_actions),
            ).squeeze(1)
            target_q = rewards + (1.0 - dones) * gamma * min_target_next_q

            q1_all, q2_all = model(states)
            q1 = q1_all.gather(1, actions.unsqueeze(1)).squeeze(1)
            q2 = q2_all.gather(1, actions.unsqueeze(1)).squeeze(1)

            val_td = 0.5 * (nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q))
            total_val_loss += val_td.item()
            batch_count += 1

    return total_val_loss / max(1, batch_count)


def train_cql_agent(
    train_loader,
    val_loader,
    state_dim: int,
    epochs: int = config.RL_CONFIG["num_epochs"],
    lr: float = config.RL_CONFIG["lr"],
    gamma: float = config.RL_CONFIG["gamma"],
    cql_alpha: float = config.RL_CONFIG["cql_alpha"],
    tau: float = config.RL_CONFIG["tau"],
    device: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Tuple[DiscreteCQLNetwork, Dict[str, List[float]]]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if save_path is None:
        save_path = config.RL_AGENT_DIR / "cql_model.pt"

    print(f"[*] Initializing Discrete CQL Model on device: {device.upper()} (state_dim={state_dim}, actions={config.NUM_ACTIONS})")
    model = DiscreteCQLNetwork(state_dim=state_dim, num_actions=config.NUM_ACTIONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    history = {"epoch": [], "loss": [], "td_loss": [], "cql_loss": [], "avg_q": [], "val_td_loss": []}
    best_val_loss = float("inf")

    print(f"[*] Beginning Discrete CQL training ({epochs} epochs, alpha={cql_alpha}, gamma={gamma})...")

    for epoch in range(1, epochs + 1):
        metrics = train_cql_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            gamma=gamma,
            cql_alpha=cql_alpha,
            tau=tau,
            device=device,
        )

        val_td_loss = evaluate_cql_val(model, val_loader, gamma=gamma, device=device)

        history["epoch"].append(epoch)
        history["loss"].append(metrics["loss"])
        history["td_loss"].append(metrics["td_loss"])
        history["cql_loss"].append(metrics["cql_loss"])
        history["avg_q"].append(metrics["avg_q"])
        history["val_td_loss"].append(val_td_loss)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"  Epoch [{epoch:2d}/{epochs:2d}] - "
                f"Total Loss: {metrics['loss']:.4f} | "
                f"TD Loss: {metrics['td_loss']:.4f} | "
                f"CQL Penalty: {metrics['cql_loss']:.4f} | "
                f"Avg Q: {metrics['avg_q']:.3f} | "
                f"Val TD: {val_td_loss:.4f}"
            )

        if val_td_loss < best_val_loss:
            best_val_loss = val_td_loss
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "state_dim": state_dim,
                    "num_actions": config.NUM_ACTIONS,
                    "cql_alpha": cql_alpha,
                    "best_val_loss": best_val_loss,
                    "history": history,
                },
                save_path,
            )

    print(f"[OK] Discrete CQL training complete. Best model checkpoint saved to: {save_path}")
    return model, history

# Alias for backward compatibility
train_cql = train_cql_agent


def fine_tune_cql_with_feedback(
    model_path: Optional[Path] = None,
    epochs: int = 5,
    lr: float = 1e-5,
    cql_alpha: float = 1.0,
    device: Optional[str] = None,
    safety_threshold_sensitivity: float = 0.90,
    safety_threshold_specificity: float = 0.85,
) -> Dict[str, Any]:
    """
    Incrementally fine-tunes the active CQL model using:
    - Clinician feedback & overrides (Strategy 2)
    - Completed longitudinal follow-up outcomes (Strategy 1)
    - Experience Replay to prevent catastrophic forgetting.
    - Automated Champion vs. Challenger Safety Gate.
    """
    from datetime import datetime
    import shutil
    from torch.utils.data import DataLoader
    from .environment import OfflineRLDataset
    from .feedback import FeedbackManager

    if model_path is None:
        model_path = config.RL_AGENT_DIR / "cql_model.pt"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not model_path.exists():
        raise FileNotFoundError(f"Base model checkpoint not found at: {model_path}")

    fb_manager = FeedbackManager()
    stats = fb_manager.get_feedback_statistics()
    print(f"[*] Feedback Stats: {stats['total_clinician_feedback']} reviews, {stats['completed_transitions']} completed outcomes.")

    # 1. Export balanced replay dataset
    states, actions, rewards, next_states, dones = fb_manager.export_retraining_dataset()
    total_samples = len(states)

    # 80/20 train/val split
    indices = np.random.permutation(total_samples)
    split_idx = int(0.8 * total_samples)
    tr_idx, val_idx = indices[:split_idx], indices[split_idx:]

    train_ds = OfflineRLDataset(states[tr_idx], actions[tr_idx], rewards[tr_idx], next_states[tr_idx], dones[tr_idx])
    val_ds = OfflineRLDataset(states[val_idx], actions[val_idx], rewards[val_idx], next_states[val_idx], dones[val_idx])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # 2. Warm-start load model
    state_dim = states.shape[1]
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = DiscreteCQLNetwork(state_dim=state_dim, num_actions=config.NUM_ACTIONS).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    print(f"[*] Commencing warm-start fine-tuning ({epochs} epochs, lr={lr}) on device: {device.upper()}...")
    history = {"epoch": [], "loss": [], "td_loss": [], "cql_loss": [], "val_td_loss": []}

    for ep in range(1, epochs + 1):
        m = train_cql_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            gamma=0.99,
            cql_alpha=cql_alpha,
            tau=0.005,
            device=device,
        )
        val_td = evaluate_cql_val(model, val_loader, gamma=0.99, device=device)
        history["epoch"].append(ep)
        history["loss"].append(m["loss"])
        history["td_loss"].append(m["td_loss"])
        history["cql_loss"].append(m["cql_loss"])
        history["val_td_loss"].append(val_td)

        print(f"  Fine-tune Epoch [{ep}/{epochs}] - Loss: {m['loss']:.4f} | TD Loss: {m['td_loss']:.4f} | CQL: {m['cql_loss']:.4f} | Val TD: {val_td:.4f}")

    # 3. Champion vs. Challenger Safety Audit
    print("[*] Running Champion vs. Challenger Safety Gate Audit...")
    traj_path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"
    bench_df = pd.read_csv(traj_path)
    eval_res = evaluate_policy(model, bench_df, device=device)
    cql_metrics = eval_res["CQL_Policy"]

    det_sens = cql_metrics["early_detection_sensitivity"]
    spec = 1.0 - cql_metrics["false_alert_rate"]

    passed_safety = (det_sens >= safety_threshold_sensitivity) and (spec >= safety_threshold_specificity)
    print(f"[*] Safety Gate Results: Deterioration Sensitivity={det_sens*100:.1f}% (min {safety_threshold_sensitivity*100:.0f}%), Specificity={spec*100:.1f}% (min {safety_threshold_specificity*100:.0f}%)")

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = model_path.parent / f"cql_model_backup_{timestamp_str}.pt"

    if passed_safety:
        # Backup existing champion
        shutil.copyfile(model_path, backup_path)
        # Promote challenger
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "state_dim": state_dim,
                "num_actions": config.NUM_ACTIONS,
                "cql_alpha": cql_alpha,
                "history": history,
                "fine_tuned_at": datetime.now().isoformat(),
                "feedback_stats": stats,
            },
            model_path,
        )
        print(f"[OK] Challenger passed safety gate and promoted to production! Backup created at {backup_path.name}")
        promotion_status = "promoted"
    else:
        print(f"[!] Challenger failed safety gate. Preserving current production model.")
        promotion_status = "rejected_safety_gate"

    return {
        "status": promotion_status,
        "epochs": epochs,
        "final_loss": history["loss"][-1],
        "final_val_td": history["val_td_loss"][-1],
        "deterioration_sensitivity": det_sens,
        "specificity": spec,
        "passed_safety": passed_safety,
        "backup_path": str(backup_path) if passed_safety else None,
        "feedback_samples_used": len(states) - len(b_s) if 'b_s' in locals() else 0,
    }


# =========================================================================
# 3. CLINICAL RULE POLICY & POLICY EVALUATION BENCHMARK
# =========================================================================

def clinical_rule_policy(row: pd.Series) -> int:
    """Standard clinical guideline rule-based policy baseline."""
    status = row.get("overall_trajectory_status", "stable")
    hba1c = float(row.get("hba1c", 7.0))
    h_change = float(row.get("hba1c_change", 0.0))
    sbp = float(row.get("sbp", 125.0))
    creat = float(row.get("creatinine", 1.0))

    if hba1c >= 9.5 or sbp >= 160.0 or creat >= 2.0 or status == "rapidly_deteriorating":
        return MonitoringAction.CLINICAL_EVALUATION
    elif status == "deteriorating" or h_change >= 0.4:
        return MonitoringAction.EARLY_RISK_ALERT
    elif hba1c >= 7.5 or sbp >= 135.0 or abs(h_change) >= 0.2:
        return MonitoringAction.INCREASED_MONITORING
    else:
        return MonitoringAction.ROUTINE_MONITORING


def evaluate_policy(
    model: DiscreteCQLNetwork,
    trajectory_df: pd.DataFrame,
    state_constructor: Optional[StateConstructor] = None,
    gamma: float = 0.99,
    device: Optional[str] = None,
) -> Dict[str, Dict]:
    """Runs policy comparison across patient trajectories."""
    if state_constructor is None:
        state_constructor = StateConstructor()

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    cql_returns, behav_returns, rule_returns, rand_returns = [], [], [], []
    cql_actions, behav_actions, rule_actions, rand_actions = [], [], [], []

    deteriorating_total, cql_early_detected, behav_early_detected, rule_early_detected = 0, 0, 0, 0
    stable_total, cql_false_alerts, behav_false_alerts, rule_false_alerts = 0, 0, 0, 0

    for _, group in trajectory_df.groupby("subject_id"):
        patient_records = group.sort_values(by="chartdate").reset_index(drop=True)
        T = len(patient_records)

        cql_ret, behav_ret, rule_ret, rand_ret = 0.0, 0.0, 0.0, 0.0

        for t in range(T):
            curr_row = patient_records.iloc[t]
            next_row = patient_records.iloc[t + 1] if t < T - 1 else None
            state_vec = state_constructor.extract_state_vector(curr_row)
            state_tensor = torch.tensor(state_vec, dtype=torch.float32).to(device)

            # CQL Action
            a_cql = model.select_action(state_tensor, deterministic=True)
            r_cql = compute_monitoring_reward(curr_row, a_cql, next_row)
            cql_ret += (gamma ** t) * r_cql
            cql_actions.append(a_cql)

            # Behavior Action
            a_behav = int(curr_row.get("action", 0))
            r_behav = compute_monitoring_reward(curr_row, a_behav, next_row)
            behav_ret += (gamma ** t) * r_behav
            behav_actions.append(a_behav)

            # Rule Action
            a_rule = clinical_rule_policy(curr_row)
            r_rule = compute_monitoring_reward(curr_row, a_rule, next_row)
            rule_ret += (gamma ** t) * r_rule
            rule_actions.append(a_rule)

            # Random Action
            a_rand = np.random.randint(0, config.NUM_ACTIONS)
            r_rand = compute_monitoring_reward(curr_row, a_rand, next_row)
            rand_ret += (gamma ** t) * r_rand
            rand_actions.append(a_rand)

            # Sensitivity checks
            status = curr_row.get("overall_trajectory_status", "stable")
            if status in ["deteriorating", "rapidly_deteriorating"]:
                deteriorating_total += 1
                if a_cql in [MonitoringAction.EARLY_RISK_ALERT, MonitoringAction.CLINICAL_EVALUATION, MonitoringAction.INCREASED_MONITORING]:
                    cql_early_detected += 1
                if a_behav in [MonitoringAction.EARLY_RISK_ALERT, MonitoringAction.CLINICAL_EVALUATION, MonitoringAction.INCREASED_MONITORING]:
                    behav_early_detected += 1
                if a_rule in [MonitoringAction.EARLY_RISK_ALERT, MonitoringAction.CLINICAL_EVALUATION, MonitoringAction.INCREASED_MONITORING]:
                    rule_early_detected += 1
            elif status == "stable":
                stable_total += 1
                if a_cql in [MonitoringAction.EARLY_RISK_ALERT, MonitoringAction.CLINICAL_EVALUATION]:
                    cql_false_alerts += 1
                if a_behav in [MonitoringAction.EARLY_RISK_ALERT, MonitoringAction.CLINICAL_EVALUATION]:
                    behav_false_alerts += 1
                if a_rule in [MonitoringAction.EARLY_RISK_ALERT, MonitoringAction.CLINICAL_EVALUATION]:
                    rule_false_alerts += 1

        cql_returns.append(cql_ret)
        behav_returns.append(behav_ret)
        rule_returns.append(rule_ret)
        rand_returns.append(rand_ret)

    def summarize_policy(returns, actions, early_det, false_alerts):
        act_counts = pd.Series(actions).value_counts(normalize=True).to_dict()
        return {
            "mean_discounted_return": float(np.mean(returns)),
            "std_discounted_return": float(np.std(returns)),
            "action_distribution": {int(k): round(float(v), 3) for k, v in act_counts.items()},
            "early_detection_sensitivity": round(float(early_det / max(1, deteriorating_total)), 3),
            "false_alert_rate": round(float(false_alerts / max(1, stable_total)), 3),
        }

    results = {
        "CQL_Policy": summarize_policy(cql_returns, cql_actions, cql_early_detected, cql_false_alerts),
        "Clinical_Rule_Policy": summarize_policy(rule_returns, rule_actions, rule_early_detected, rule_false_alerts),
        "Historical_Behavior_Policy": summarize_policy(behav_returns, behav_actions, behav_early_detected, behav_false_alerts),
        "Random_Policy": summarize_policy(rand_returns, rand_actions, 0, 0),
    }

    print("\n=================== POLICY EVALUATION BENCHMARK ===================")
    for pol_name, metrics in results.items():
        print(f"Policy: {pol_name}")
        print(f"  Mean Discounted Return: {metrics['mean_discounted_return']:.3f} (+/- {metrics['std_discounted_return']:.3f})")
        print(f"  Early Detection Rate:   {metrics['early_detection_sensitivity']*100:.1f}%")
        print(f"  False Alert Rate:       {metrics['false_alert_rate']*100:.1f}%")
        print(f"  Action Distribution:    {metrics['action_distribution']}")
        print("-------------------------------------------------------------------")

    return results
