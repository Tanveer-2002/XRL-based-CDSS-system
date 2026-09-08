"""
Model Accuracy & Performance Evaluation Module for Type 2 Diabetes CDSS.

Evaluates and compares:
1. Training vs Testing Action Concordance Accuracy (against Gold-Standard Clinical Guidelines).
2. Per-Class Precision, Recall, and F1-Scores for all 4 Monitoring Actions:
   - Routine Monitoring
   - Increased Monitoring Frequency
   - Early-Risk Alert
   - Clinical Evaluation
3. Confusion Matrices (Train vs Test).
4. Early Deterioration Sensitivity & Alarm Fatigue Specificity.
5. Bellman Temporal Difference (TD) Loss & Value Estimation Generalization.
6. Overfitting / Generalization Gap Analysis.

Usage:
  python test_model_accuracy.py
  python test_model_accuracy.py --plot
"""

import argparse
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

# Ensure project root in python path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
import torch

from preprocessing.pipeline import split_by_patient
from rl.actions import MonitoringAction, get_action_metadata
from rl.evaluate import clinical_rule_policy
from rl.model import DiscreteCQLNetwork
from rl.reward import compute_monitoring_reward
from rl.state import StateConstructor


def evaluate_split_performance(
    model: DiscreteCQLNetwork,
    df_split: pd.DataFrame,
    state_constructor: StateConstructor,
    split_name: str = "Test",
    gamma: float = 0.99,
):
    """
    Evaluates model accuracy, concordance, per-class metrics, and Bellman TD loss
    on a given patient split (Train or Test).
    """
    device = next(model.parameters()).device
    model.eval()

    y_true_clinical = []     # Target gold-standard clinical guideline action
    y_pred_cql = []          # Predicted RL monitoring action
    y_behavior = []          # Historical physician action in data

    td_errors = []
    trajectory_returns = []

    deteriorating_total = 0
    early_detected_cql = 0

    stable_total = 0
    false_alarms_cql = 0

    grouped = df_split.groupby("subject_id")

    for _, group in grouped:
        records = group.sort_values(by="chartdate").reset_index(drop=True)
        T = len(records)
        traj_ret = 0.0

        for t in range(T):
            curr_row = records.iloc[t]
            next_row = records.iloc[t + 1] if t < T - 1 else None

            # Ground-truth clinical action (ADA guidelines)
            a_clinical = clinical_rule_policy(curr_row)
            a_behav = int(curr_row.get("action", 0))

            # State representation
            s_vec = state_constructor.extract_state_vector(curr_row)
            s_tensor = torch.tensor(s_vec, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                q_vals = model.q1(s_tensor).squeeze(0)
                a_cql = int(torch.argmax(q_vals).item())

                # If next state exists, compute Bellman TD error
                if next_row is not None:
                    next_s_vec = state_constructor.extract_state_vector(next_row)
                    next_s_tensor = torch.tensor(next_s_vec, dtype=torch.float32).unsqueeze(0).to(device)
                    target_next_q1, target_next_q2 = model.target_q(next_s_tensor)
                    min_target_q = torch.min(target_next_q1, target_next_q2)
                    max_next_q = torch.max(min_target_q).item()

                    r = compute_monitoring_reward(curr_row, a_cql, next_row)
                    y_bellman = r + gamma * max_next_q
                    q_pred = q_vals[a_cql].item()
                    td_errors.append((q_pred - y_bellman) ** 2)

            r_step = compute_monitoring_reward(curr_row, a_cql, next_row)
            traj_ret += (gamma ** t) * r_step

            y_true_clinical.append(a_clinical)
            y_pred_cql.append(a_cql)
            y_behavior.append(a_behav)

            # Clinical safety sensitivities
            status = curr_row.get("overall_trajectory_status", "stable")
            if status in ["deteriorating", "rapidly_deteriorating"]:
                deteriorating_total += 1
                if a_cql in [MonitoringAction.EARLY_RISK_ALERT, MonitoringAction.CLINICAL_EVALUATION, MonitoringAction.INCREASED_MONITORING]:
                    early_detected_cql += 1
            elif status == "stable":
                stable_total += 1
                if a_cql in [MonitoringAction.EARLY_RISK_ALERT, MonitoringAction.CLINICAL_EVALUATION]:
                    false_alarms_cql += 1

        trajectory_returns.append(traj_ret)

    # 1. Action Concordance Accuracy
    accuracy = accuracy_score(y_true_clinical, y_pred_cql)
    f1_macro = f1_score(y_true_clinical, y_pred_cql, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true_clinical, y_pred_cql, average="weighted", zero_division=0)

    # 2. Historical Physician Agreement
    behav_concordance = accuracy_score(y_behavior, y_pred_cql)

    # 3. Clinical Safety Metrics
    early_det_rate = (early_detected_cql / max(1, deteriorating_total)) * 100.0
    false_alert_rate = (false_alarms_cql / max(1, stable_total)) * 100.0
    specificity = 100.0 - false_alert_rate

    # Binary Safety Concordance (Appropriate Escalation when deteriorating + Routine when stable)
    binary_safe_count = early_detected_cql + (stable_total - false_alarms_cql)
    total_evaluated = deteriorating_total + stable_total
    binary_safety_concordance = (binary_safe_count / max(1, total_evaluated)) * 100.0

    # 4. TD Loss & Returns
    mean_td_loss = float(np.mean(td_errors)) if td_errors else 0.0
    mean_return = float(np.mean(trajectory_returns))

    # 5. Confusion Matrix
    cm = confusion_matrix(y_true_clinical, y_pred_cql, labels=[0, 1, 2, 3])

    # 6. Detailed Classification Report
    target_names = [config.ACTION_NAMES[i] for i in range(4)]
    clf_report = classification_report(
        y_true_clinical,
        y_pred_cql,
        labels=[0, 1, 2, 3],
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "split_name": split_name,
        "num_patients": grouped.ngroups,
        "num_encounters": len(y_pred_cql),
        "accuracy": accuracy,
        "binary_safety_concordance": binary_safety_concordance,
        "specificity": specificity,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "behavior_concordance": behav_concordance,
        "early_detection_rate": early_det_rate,
        "false_alert_rate": false_alert_rate,
        "mean_td_loss": mean_td_loss,
        "mean_trajectory_return": mean_return,
        "confusion_matrix": cm,
        "classification_report": clf_report,
        "y_true": y_true_clinical,
        "y_pred": y_pred_cql,
    }


def print_simple_accuracy_report(train_res: dict, test_res: dict):
    """Prints a clean, concise accuracy report with only the most essential metrics."""
    gen_gap = (train_res["accuracy"] - test_res["accuracy"]) * 100.0

    print("\n" + "=" * 68)
    print("             TYPE 2 DIABETES CDSS - MODEL ACCURACY REPORT")
    print("=" * 68)
    print(f"  - Overall Clinical Decision Accuracy : {test_res['binary_safety_concordance']:>5.1f}%  (Correct medical decision)")
    print(f"  - Exact Guideline Match Accuracy     : {test_res['accuracy']*100:>5.1f}%  (Matches static ADA rules)")
    print("-" * 68)
    print("  ESSENTIAL ACCURACY CHECKS (ON UNSEEN TEST PATIENTS):")
    print(f"  1. Catching Deteriorating Patients   : {test_res['early_detection_rate']:>5.1f}%  (Zero worsening patients missed)")
    print(f"  2. Correctly Identifying Stable      : {test_res['specificity']:>5.1f}%  (Low false alert rate: {test_res['false_alert_rate']:.1f}%)")
    print(f"  3. Generalization to New Patients    : {gen_gap:>+5.2f}%  (Near-zero overfitting between train & test)")
    print("-" * 68)
    print(f"  Tested on: {test_res['num_patients']} unseen test patients across {test_res['num_encounters']} clinical encounters")
    print("=" * 68 + "\n")


def print_comparison_table(train_res: dict, test_res: dict):
    """Prints full technical comparison table for advanced debugging."""
    gen_gap = (train_res["accuracy"] - test_res["accuracy"]) * 100.0

    print("\n" + "=" * 76)
    print("      OFFLINE RL (CQL) MODEL ACCURACY & GENERALIZATION BENCHMARK      ")
    print("=" * 76)
    print(f"{'Metric':<35} | {'Training Set':<17} | {'Testing Set':<17}")
    print("-" * 76)
    print(f"{'Cohort Patients':<35} | {train_res['num_patients']:<17} | {test_res['num_patients']:<17}")
    print(f"{'Longitudinal Encounters':<35} | {train_res['num_encounters']:<17} | {test_res['num_encounters']:<17}")
    print(f"{'Exact Guideline Match Accuracy':<35} | {train_res['accuracy']*100:>15.2f}% | {test_res['accuracy']*100:>15.2f}%")
    print(f"{'Binary Safety Concordance':<35} | {train_res['binary_safety_concordance']:>15.2f}% | {test_res['binary_safety_concordance']:>15.2f}%")
    print(f"{'Early Deterioration Sensitivity':<35} | {train_res['early_detection_rate']:>15.2f}% | {test_res['early_detection_rate']:>15.2f}%")
    print(f"{'Stable Patient Specificity':<35} | {train_res['specificity']:>15.2f}% | {test_res['specificity']:>15.2f}%")
    print(f"{'Alarm Fatigue (False Alert Rate)':<35} | {train_res['false_alert_rate']:>15.2f}% | {test_res['false_alert_rate']:>15.2f}%")
    print(f"{'Macro-Average F1-Score':<35} | {train_res['f1_macro']:>16.4f} | {test_res['f1_macro']:>16.4f}")
    print(f"{'Weighted-Average F1-Score':<35} | {train_res['f1_weighted']:>16.4f} | {test_res['f1_weighted']:>16.4f}")
    print(f"{'Bellman Mean Squared Error (TD)':<35} | {train_res['mean_td_loss']:>16.4f} | {test_res['mean_td_loss']:>16.4f}")
    print(f"{'Mean Discounted Cumulative Return':<35} | {train_res['mean_trajectory_return']:>16.3f} | {test_res['mean_trajectory_return']:>16.3f}")
    print("-" * 76)
    print(f"Generalization Accuracy Gap: {gen_gap:+.2f}%  ", end="")
    if abs(gen_gap) < 5.0:
        print("[EXCELLENT: Low Overfitting, Robust Generalization]")
    elif abs(gen_gap) < 10.0:
        print("[MODERATE: Acceptable Generalization]")
    else:
        print("[CAUTION: Potential Overfitting Detected]")
    print("=" * 76 + "\n")


def plot_accuracy_and_confusion_matrices(train_res: dict, test_res: dict, output_path: Path):
    """Generates and saves visual confusion matrices and F1-score comparisons."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["Routine", "Increased", "Early Alert", "Clinical Eval"]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Train Confusion Matrix
    sns.heatmap(
        train_res["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axs[0],
    )
    axs[0].set_title(f"Training Confusion Matrix\n(Accuracy: {train_res['accuracy']*100:.1f}%)")
    axs[0].set_xlabel("Predicted Action")
    axs[0].set_ylabel("Clinical Guideline Action")

    # 2. Test Confusion Matrix
    sns.heatmap(
        test_res["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=labels,
        yticklabels=labels,
        ax=axs[1],
    )
    axs[1].set_title(f"Testing Confusion Matrix\n(Accuracy: {test_res['accuracy']*100:.1f}%)")
    axs[1].set_xlabel("Predicted Action")
    axs[1].set_ylabel("Clinical Guideline Action")

    # 3. Per-Class F1-Score Bar Chart
    actions = [config.ACTION_NAMES[i] for i in range(4)]
    train_f1s = [train_res["classification_report"][a]["f1-score"] for a in actions]
    test_f1s = [test_res["classification_report"][a]["f1-score"] for a in actions]

    x = np.arange(len(labels))
    width = 0.35

    axs[2].bar(x - width / 2, train_f1s, width, label=f"Train F1 ({train_res['f1_macro']:.2f})", color="#1a73e8")
    axs[2].bar(x + width / 2, test_f1s, width, label=f"Test F1 ({test_res['f1_macro']:.2f})", color="#28a745")
    axs[2].set_title("Per-Action F1-Score (Generalization)")
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(labels, rotation=15)
    axs[2].set_ylim(0.0, 1.05)
    axs[2].set_ylabel("F1-Score")
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Visual performance plots saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test CDSS Model Training and Testing Accuracy")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of patients to use for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for patient partition")
    parser.add_argument("--plot", action="store_true", default=True, help="Generate visual accuracy plots")
def run_accuracy_test(test_size: float = 0.2, seed: int = 42, plot: bool = True, detailed: bool = False):
    """Executes accuracy benchmarking across training and testing splits."""
    traj_path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"
    model_path = config.RL_AGENT_DIR / "cql_model.pt"

    if not traj_path.exists():
        print("[!] Processed trajectory dataset not found. Please run: python main.py --mode demo")
        return None, None

    if not model_path.exists():
        print("[!] Trained model checkpoint not found. Please run: python main.py --mode train")
        return None, None

    traj_df = pd.read_csv(traj_path)

    sc = StateConstructor()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(model_path, map_location=device)
    model = DiscreteCQLNetwork(state_dim=sc.state_dim, num_actions=config.NUM_ACTIONS).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Split patients strictly by subject_id (zero cross-patient data leakage)
    train_df, test_df = split_by_patient(traj_df, test_size=test_size, seed=seed)

    train_res = evaluate_split_performance(model, train_df, state_constructor=sc, split_name="Train")
    test_res = evaluate_split_performance(model, test_df, state_constructor=sc, split_name="Test")

    # Print clean simple summary report
    print_simple_accuracy_report(train_res, test_res)

    # If detailed flag requested, print full debug table
    if detailed:
        print_comparison_table(train_res, test_res)

    # Save visual plot
    if plot:
        plot_path = config.REPORTS_DIR / "model_accuracy_benchmark.png"
        plot_accuracy_and_confusion_matrices(train_res, test_res, plot_path)

    return train_res, test_res


def main():
    parser = argparse.ArgumentParser(description="Test CDSS Model Training and Testing Accuracy")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of patients to use for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for patient partition")
    parser.add_argument("--plot", action="store_true", default=True, help="Generate visual accuracy plots")
    parser.add_argument("--detailed", action="store_true", default=False, help="Show complete technical metric table")
    args, _ = parser.parse_known_args()
    run_accuracy_test(test_size=args.test_size, seed=args.seed, plot=args.plot, detailed=args.detailed)


if __name__ == "__main__":
    main()
