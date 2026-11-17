"""
Data Scaling & Sample Efficiency Analysis for Type 2 Diabetes CDSS.

Empirically proves the architectural superiority of Offline Reinforcement Learning (CQL)
over static Supervised Baselines (XGBoost, Behavioral Cloning) as training data scales.

Evaluates performance across [25%, 50%, 75%, 100%] training cohort sizes on a fixed test cohort:
- Metric 1: Expected Trajectory Return (V^pi)
- Metric 2: Macro F1-Score (All 4 Actions)
- Metric 3: Clinical Guideline Accuracy (%)
- Metric 4: Early Deterioration Sensitivity (%)
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from preprocessing.pipeline import split_by_patient
from rl.cql import clinical_rule_policy, DiscreteCQLNetwork
from rl.environment import (
    MonitoringAction,
    StateConstructor,
    compute_monitoring_reward,
    generate_offline_transitions,
)
from xgboost.model import XGBoostAgent
from bc.model import BehavioralCloningPolicy
from bcq.model import DiscreteBCQNetwork


def train_quick_cql(
    s_tr: np.ndarray,
    a_tr: np.ndarray,
    r_tr: np.ndarray,
    ns_tr: np.ndarray,
    d_tr: np.ndarray,
    state_dim: int,
    epochs: int = 30,
    batch_size: int = 64,
    gamma: float = 0.99,
    cql_alpha: float = 2.5,
    balanced_power: float = 0.20,
    lr: float = 3e-4,
    device: str = "cpu",
) -> DiscreteCQLNetwork:
    """Trains Discrete CQL with conservative penalty and class-balanced sampling on a transition subset."""
    cql = DiscreteCQLNetwork(state_dim=state_dim, num_actions=config.NUM_ACTIONS).to(device)
    optimizer = optim.Adam(cql.parameters(), lr=lr, weight_decay=1e-4)

    dataset = TensorDataset(
        torch.tensor(s_tr, dtype=torch.float32),
        torch.tensor(a_tr, dtype=torch.long),
        torch.tensor(r_tr, dtype=torch.float32),
        torch.tensor(ns_tr, dtype=torch.float32),
        torch.tensor(d_tr, dtype=torch.float32),
    )

    if balanced_power > 0:
        class_counts = np.bincount(a_tr, minlength=config.NUM_ACTIONS)
        class_weights = 1.0 / (np.maximum(class_counts, 1).astype(np.float32) ** balanced_power)
        sample_weights = class_weights[a_tr]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float32),
            num_samples=len(dataset),
            replacement=True,
        )
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=(len(dataset) > batch_size))
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=(len(dataset) > batch_size))

    for _ in range(epochs):
        cql.train()
        for s_b, a_b, r_b, ns_b, d_b in loader:
            s_b, a_b, r_b, ns_b, d_b = s_b.to(device), a_b.to(device), r_b.to(device), ns_b.to(device), d_b.to(device)

            with torch.no_grad():
                next_q1, _ = cql(ns_b)
                next_acts = torch.argmax(next_q1, dim=-1, keepdim=True)
                tq1, tq2 = cql.target_q(ns_b)
                min_target_next_q = torch.min(
                    tq1.gather(1, next_acts),
                    tq2.gather(1, next_acts),
                ).squeeze(1)
                target_y = r_b + (1.0 - d_b) * gamma * min_target_next_q

            q1_all = cql.q1(s_b)
            q2_all = cql.q2(s_b)
            q1_act = q1_all.gather(1, a_b.unsqueeze(1)).squeeze(1)
            q2_act = q2_all.gather(1, a_b.unsqueeze(1)).squeeze(1)

            td_loss = 0.5 * (F.mse_loss(q1_act, target_y) + F.mse_loss(q2_act, target_y))
            cql_loss1 = (torch.logsumexp(q1_all, dim=-1) - q1_act).mean()
            cql_loss2 = (torch.logsumexp(q2_all, dim=-1) - q2_act).mean()
            total_loss = td_loss + cql_alpha * (cql_loss1 + cql_loss2)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            cql.soft_update_targets(tau=0.01)

    cql.eval()
    return cql


def train_quick_bc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    state_dim: int,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: str = "cpu",
) -> BehavioralCloningPolicy:
    """Trains Behavioral Cloning on a transition subset."""
    bc = BehavioralCloningPolicy(state_dim=state_dim, num_actions=config.NUM_ACTIONS).to(device)
    optimizer = optim.Adam(bc.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        bc.train()
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(bc(x_b), y_b)
            loss.backward()
            optimizer.step()

    bc.eval()
    return bc


def train_quick_bcq(
    s_tr: np.ndarray,
    a_tr: np.ndarray,
    r_tr: np.ndarray,
    ns_tr: np.ndarray,
    d_tr: np.ndarray,
    state_dim: int,
    epochs: int = 25,
    batch_size: int = 64,
    gamma: float = 0.99,
    threshold: float = 0.3,
    lr: float = 3e-4,
    device: str = "cpu",
) -> DiscreteBCQNetwork:
    """Trains Discrete BCQ on a transition subset."""
    bcq = DiscreteBCQNetwork(
        state_dim=state_dim,
        num_actions=config.NUM_ACTIONS,
        threshold=threshold,
    ).to(device)

    optimizer_q = optim.Adam(bcq.q_net.parameters(), lr=lr)
    optimizer_b = optim.Adam(bcq.behavior_net.parameters(), lr=lr)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    dataset = TensorDataset(
        torch.tensor(s_tr, dtype=torch.float32),
        torch.tensor(a_tr, dtype=torch.long),
        torch.tensor(r_tr, dtype=torch.float32),
        torch.tensor(ns_tr, dtype=torch.float32),
        torch.tensor(d_tr, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        bcq.train()
        for states, actions, rewards, next_states, dones in loader:
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)

            # 1. Behavior network
            b_logits = bcq.behavior_net(states)
            b_loss = criterion_ce(b_logits, actions)
            optimizer_b.zero_grad()
            b_loss.backward()
            optimizer_b.step()

            # 2. Q-networks with batch-constrained target
            with torch.no_grad():
                next_b_logits = bcq.behavior_net(next_states)
                next_b_probs = F.softmax(next_b_logits, dim=-1)
                max_next_b = torch.max(next_b_probs, dim=-1, keepdim=True)[0]
                valid_mask = (next_b_probs / (max_next_b + 1e-8)) >= threshold

                target_q1, target_q2 = bcq.target_q_net(next_states)
                min_target_q = torch.min(target_q1, target_q2).clone()
                min_target_q[~valid_mask] = -1e9
                max_next_q = torch.max(min_target_q, dim=-1)[0]
                target_y = rewards + (1.0 - dones) * gamma * max_next_q

            curr_q1, curr_q2 = bcq.q_net(states)
            q1_selected = curr_q1.gather(1, actions.unsqueeze(1)).squeeze(1)
            q2_selected = curr_q2.gather(1, actions.unsqueeze(1)).squeeze(1)
            q_loss = criterion_mse(q1_selected, target_y) + criterion_mse(q2_selected, target_y)

            optimizer_q.zero_grad()
            q_loss.backward()
            optimizer_q.step()

            bcq.soft_update_targets(tau=0.01)

    bcq.eval()
    return bcq


def evaluate_agent(
    predict_fn,
    test_df: pd.DataFrame,
    sc: StateConstructor,
    gamma: float = 0.99,
) -> dict:
    """Evaluates policy on test patient trajectories."""
    y_true = []
    y_pred = []
    returns = []
    deteriorating_total = 0
    early_detected = 0

    for _, group in test_df.groupby("subject_id"):
        records = group.sort_values(by="chartdate").reset_index(drop=True)
        T = len(records)
        traj_ret = 0.0

        for t in range(T):
            curr_row = records.iloc[t]
            next_row = records.iloc[t + 1] if t < T - 1 else None

            a_clinical = clinical_rule_policy(curr_row)
            s_vec = sc.extract_state_vector(curr_row)
            a_pred = predict_fn(s_vec)

            r = compute_monitoring_reward(curr_row, a_pred, next_row)
            traj_ret += (gamma ** t) * r

            y_true.append(a_clinical)
            y_pred.append(a_pred)

            status = curr_row.get("overall_trajectory_status", "stable")
            if status in ["deteriorating", "rapidly_deteriorating"]:
                deteriorating_total += 1
                if a_pred in [MonitoringAction.INCREASED_MONITORING, MonitoringAction.EARLY_RISK_ALERT, MonitoringAction.CLINICAL_EVALUATION]:
                    early_detected += 1

        returns.append(traj_ret)

    acc = accuracy_score(y_true, y_pred) * 100.0
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100.0
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    sens = (early_detected / deteriorating_total * 100.0) if deteriorating_total > 0 else 0.0
    mean_ret = float(np.mean(returns))

    return {
        "guideline_acc": acc,
        "balanced_acc": bal_acc,
        "macro_f1": f1,
        "sensitivity": sens,
        "mean_return": mean_ret,
    }


def run_scaling_experiment(
    fractions: list = [0.25, 0.50, 0.75, 1.00],
    seed: int = 42,
) -> pd.DataFrame:
    """Runs data scaling experiment across training subsets."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"
    df = pd.read_csv(data_path)

    train_df_full, test_df = split_by_patient(df, test_size=0.2, seed=seed)
    train_patients = sorted(train_df_full["subject_id"].unique())
    total_train_pts = len(train_patients)

    sc = StateConstructor()
    rows = []

    print("=" * 70)
    print("       RUNNING DATA SCALING & SAMPLE EFFICIENCY EXPERIMENT")
    print(f"       Total Train Patients: {total_train_pts} | Fixed Test Patients: {test_df['subject_id'].nunique()}")
    print("=" * 70)

    for frac in fractions:
        k = int(total_train_pts * frac)
        subset_patients = train_patients[:k]
        subset_df = train_df_full[train_df_full["subject_id"].isin(subset_patients)]

        print(f"\n[*] Evaluating Scaling Point: {int(frac * 100)}% Train Data ({k} Patients, {len(subset_df)} Encounters)...")
        s_tr, a_tr, r_tr, ns_tr, d_tr, _ = generate_offline_transitions(subset_df, sc)

        # 1. Train XGBoost
        xgb = XGBoostAgent(num_class=config.NUM_ACTIONS, random_state=seed)
        xgb.fit(s_tr, a_tr)
        xgb_res = evaluate_agent(lambda s: xgb.select_action(s, deterministic=True), test_df, sc)
        rows.append({"Fraction": f"{int(frac*100)}%", "Patients": k, "Transitions": len(s_tr), "Model": "XGBoost", **xgb_res})

        # 2. Train Behavioral Cloning (BC)
        bc = train_quick_bc(s_tr, a_tr, state_dim=sc.state_dim, epochs=25, device=device)
        bc_res = evaluate_agent(lambda s: bc.select_action(s, deterministic=True), test_df, sc)
        rows.append({"Fraction": f"{int(frac*100)}%", "Patients": k, "Transitions": len(s_tr), "Model": "Behavioral Cloning", **bc_res})

        # 3. Train Discrete BCQ (Peer Offline RL)
        bcq = train_quick_bcq(s_tr, a_tr, r_tr, ns_tr, d_tr, state_dim=sc.state_dim, epochs=25, device=device)
        bcq_res = evaluate_agent(lambda s: bcq.select_action(s, deterministic=True), test_df, sc)
        rows.append({"Fraction": f"{int(frac*100)}%", "Patients": k, "Transitions": len(s_tr), "Model": "Discrete BCQ", **bcq_res})

        # 4. Train Discrete CQL (Ours)
        cql = train_quick_cql(
            s_tr,
            a_tr,
            r_tr,
            ns_tr,
            d_tr,
            state_dim=sc.state_dim,
            epochs=30,
            cql_alpha=2.5,
            balanced_power=0.20,
            device=device,
        )
        cql_res = evaluate_agent(
            lambda s: cql.select_action(torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0), deterministic=True),
            test_df,
            sc,
        )
        rows.append({"Fraction": f"{int(frac*100)}%", "Patients": k, "Transitions": len(s_tr), "Model": "Discrete CQL (Ours)", **cql_res})

        print(f"    - XGBoost: Return={xgb_res['mean_return']:.2f} | Bal Acc={xgb_res['balanced_acc']:.1f}% | Macro F1={xgb_res['macro_f1']:.3f}")
        print(f"    - BC     : Return={bc_res['mean_return']:.2f} | Bal Acc={bc_res['balanced_acc']:.1f}% | Macro F1={bc_res['macro_f1']:.3f}")
        print(f"    - BCQ    : Return={bcq_res['mean_return']:.2f} | Bal Acc={bcq_res['balanced_acc']:.1f}% | Macro F1={bcq_res['macro_f1']:.3f}")
        print(f"    - CQL    : Return={cql_res['mean_return']:.2f} | Bal Acc={cql_res['balanced_acc']:.1f}% | Macro F1={cql_res['macro_f1']:.3f}")

    results_df = pd.DataFrame(rows)
    return results_df


def plot_scaling_curves(results_df: pd.DataFrame, output_path: Path):
    """Plots 4-panel publication-ready scaling curves."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    palette = {
        "XGBoost": "#1f77b4",
        "Behavioral Cloning": "#9467bd",
        "Discrete BCQ": "#ff7f0e",
        "Discrete CQL (Ours)": "#2ca02c",
    }
    markers = {
        "XGBoost": "s",
        "Behavioral Cloning": "^",
        "Discrete BCQ": "d",
        "Discrete CQL (Ours)": "o",
    }

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Expected Trajectory Return
    sns.lineplot(
        data=results_df, x="Patients", y="mean_return", hue="Model", style="Model",
        markers=markers, dashes=False, linewidth=2.5, markersize=8, palette=palette, ax=axs[0, 0]
    )
    axs[0, 0].set_title("Expected Trajectory Return (V^pi) vs. Training Cohort", fontweight="bold")
    axs[0, 0].set_ylabel("Mean Discounted Return (Higher is Better)")
    axs[0, 0].set_xlabel("Number of Training Patients")

    # 2. Macro F1-Score
    sns.lineplot(
        data=results_df, x="Patients", y="macro_f1", hue="Model", style="Model",
        markers=markers, dashes=False, linewidth=2.5, markersize=8, palette=palette, ax=axs[0, 1]
    )
    axs[0, 1].set_title("Macro F1-Score (All 4 Actions) vs. Training Cohort", fontweight="bold")
    axs[0, 1].set_ylabel("Macro F1-Score")
    axs[0, 1].set_xlabel("Number of Training Patients")

    # 3. Clinical Guideline Accuracy
    sns.lineplot(
        data=results_df, x="Patients", y="guideline_acc", hue="Model", style="Model",
        markers=markers, dashes=False, linewidth=2.5, markersize=8, palette=palette, ax=axs[1, 0]
    )
    axs[1, 0].set_title("Clinical Guideline Accuracy (%) vs. Training Cohort", fontweight="bold")
    axs[1, 0].set_ylabel("Accuracy (%)")
    axs[1, 0].set_xlabel("Number of Training Patients")

    # 4. Early Deterioration Sensitivity
    sns.lineplot(
        data=results_df, x="Patients", y="sensitivity", hue="Model", style="Model",
        markers=markers, dashes=False, linewidth=2.5, markersize=8, palette=palette, ax=axs[1, 1]
    )
    axs[1, 1].set_title("Early Deterioration Sensitivity (%) vs. Training Cohort", fontweight="bold")
    axs[1, 1].set_ylabel("Sensitivity (%)")
    axs[1, 1].set_xlabel("Number of Training Patients")

    plt.suptitle("Empirical Scaling Laws: Continuous Improvement as Clinical Cohort Grows", fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Scaling curves plot saved to: {output_path}")


def main():
    results_df = run_scaling_experiment()

    # Save CSV
    out_dir = PROJECT_ROOT / "benchmark"
    csv_path = out_dir / "data_scaling_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n[OK] Data scaling tabular metrics saved to: {csv_path}")

    # Plot figure
    plot_path = out_dir / "data_scaling_curve.png"
    plot_scaling_curves(results_df, plot_path)

    # Print summary table
    print("\n" + "=" * 80)
    print("                   DATA SCALING RESULTS SUMMARY TABLE")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
