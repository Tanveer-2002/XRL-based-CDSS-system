"""
Visualization Module for Cross-Model CDSS Benchmarks.

Generates:
1. Multi-panel side-by-side comparison bar charts.
2. Grid of confusion matrices comparing all models on identical test data.
"""

from pathlib import Path
from typing import Any, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def plot_side_by_side_benchmark(
    summary_df: pd.DataFrame,
    all_results: Dict[str, Dict[str, Any]],
    output_path: Path = None,
):
    """Generates comprehensive publication-ready multi-metric comparison plots."""
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "model_benchmark_comparison.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    models = summary_df["Model"].tolist()
    palette = ["#6c757d", "#1f77b4", "#9467bd", "#ff7f0e", "#2ca02c"]

    fig, axs = plt.subplots(2, 4, figsize=(22, 10))
    axs = axs.flatten()

    # 1. Guideline Accuracy
    sns.barplot(data=summary_df, x="Model", y="Guideline Acc (%)", hue="Model", legend=False, ax=axs[0], palette=palette[:len(models)])
    axs[0].set_title("Clinical Guideline Accuracy (%)", fontweight="bold", pad=10)
    axs[0].set_ylabel("Accuracy (%)")
    axs[0].tick_params(axis="x", rotation=25)
    axs[0].set_ylim(0, 100)
    for p in axs[0].patches:
        axs[0].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                        ha="center", va="center", color="white", fontweight="bold")

    # 2. Balanced Accuracy
    sns.barplot(data=summary_df, x="Model", y="Balanced Acc (%)", hue="Model", legend=False, ax=axs[1], palette=palette[:len(models)])
    axs[1].set_title("Balanced Accuracy (%) (Macro Recall)", fontweight="bold", pad=10)
    axs[1].set_ylabel("Balanced Acc (%)")
    axs[1].tick_params(axis="x", rotation=25)
    axs[1].set_ylim(0, 100)
    for p in axs[1].patches:
        axs[1].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                        ha="center", va="center", color="white", fontweight="bold")

    # 3. Macro F1-Score
    sns.barplot(data=summary_df, x="Model", y="Macro F1", hue="Model", legend=False, ax=axs[2], palette=palette[:len(models)])
    axs[2].set_title("Macro F1-Score (All 4 Actions)", fontweight="bold", pad=10)
    axs[2].set_ylabel("Macro F1")
    axs[2].tick_params(axis="x", rotation=25)
    axs[2].set_ylim(0, 1.0)
    for p in axs[2].patches:
        axs[2].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                        ha="center", va="center", color="white", fontweight="bold")

    # 4. Deterioration Sensitivity
    sns.barplot(data=summary_df, x="Model", y="Sensitivity (%)", hue="Model", legend=False, ax=axs[3], palette=palette[:len(models)])
    axs[3].set_title("Early Deterioration Sensitivity (%)", fontweight="bold", pad=10)
    axs[3].set_ylabel("Sensitivity (%)")
    axs[3].tick_params(axis="x", rotation=25)
    axs[3].set_ylim(0, 105)
    for p in axs[3].patches:
        axs[3].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                        ha="center", va="center", color="white", fontweight="bold")

    # 5. False Alerts per Patient-Year (Alarm Fatigue)
    sns.barplot(data=summary_df, x="Model", y="False / Pt-Yr", hue="Model", legend=False, ax=axs[4], palette=palette[:len(models)])
    axs[4].set_title("False Alerts / Patient-Year (Lower is Better)", fontweight="bold", pad=10)
    axs[4].set_ylabel("Alerts / Pt-Year")
    axs[4].tick_params(axis="x", rotation=25)
    max_fa = max(1.0, summary_df["False / Pt-Yr"].max() * 1.25)
    axs[4].set_ylim(0, max_fa)
    for p in axs[4].patches:
        axs[4].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                        ha="center", va="center", color="white" if p.get_height() > 0.2 else "black", fontweight="bold")

    # 6. Action Adjacency Concordance (Near-Miss Safety Rate)
    sns.barplot(data=summary_df, x="Model", y="Adjacent Acc (%)", hue="Model", legend=False, ax=axs[5], palette=palette[:len(models)])
    axs[5].set_title("Adjacent Concordance (+/-1 Tier)", fontweight="bold", pad=10)
    axs[5].set_ylabel("Concordance (%)")
    axs[5].tick_params(axis="x", rotation=25)
    axs[5].set_ylim(0, 105)
    for p in axs[5].patches:
        axs[5].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                        ha="center", va="center", color="white", fontweight="bold")

    # 7. Mean Trajectory Return (V)
    sns.barplot(data=summary_df, x="Model", y="Mean Return (V)", hue="Model", legend=False, ax=axs[6], palette=palette[:len(models)])
    axs[6].set_title("Trajectory Return (V^pi) (Higher is Better)", fontweight="bold", pad=10)
    axs[6].set_ylabel("Discounted Return")
    axs[6].tick_params(axis="x", rotation=25)
    min_ret = min(0.0, summary_df["Mean Return (V)"].min())
    max_ret = max(10.0, summary_df["Mean Return (V)"].max() * 1.25)
    axs[6].set_ylim(min_ret, max_ret)
    for p in axs[6].patches:
        axs[6].annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                        ha="center", va="center", color="white", fontweight="bold")

    # 8. Fitted Q-Evaluation (FQE Value)
    sns.barplot(data=summary_df, x="Model", y="FQE Value", hue="Model", legend=False, ax=axs[7], palette=palette[:len(models)])
    axs[7].set_title("FQE Policy Value Estimate (OPE)", fontweight="bold", pad=10)
    axs[7].set_ylabel("FQE Value")
    axs[7].tick_params(axis="x", rotation=25)
    max_fqe = max(5.0, summary_df["FQE Value"].max() * 1.25)
    axs[7].set_ylim(0, max_fqe)
    for p in axs[7].patches:
        axs[7].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                        ha="center", va="center", color="white", fontweight="bold")

    plt.suptitle("Comprehensive CDSS Multi-Model Benchmark (Test Cohort: 100 Patients)", fontsize=16, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Multi-metric comparison chart saved to: {output_path}")

    # Plot Grid of Confusion Matrices
    cm_path = Path(__file__).resolve().parent / "confusion_matrices_all.png"
    num_models = len(all_results)
    fig_cm, axs_cm = plt.subplots(1, num_models, figsize=(5.5 * num_models, 5))
    if num_models == 1:
        axs_cm = [axs_cm]

    labels = ["ROUTINE", "INCREASED", "ALERT", "CLINICAL"]
    cm_palettes = ["Blues", "Greens", "Purples", "Oranges", "YlGn"]

    for idx, (m_name, res) in enumerate(all_results.items()):
        ax = axs_cm[idx]
        cmap = cm_palettes[idx % len(cm_palettes)]
        sns.heatmap(
            res["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap=cmap,
            xticklabels=labels,
            yticklabels=labels,
            cbar=False,
            ax=ax,
        )
        ax.set_title(f"{m_name}\nAcc: {res['accuracy_guideline']:.1f}% | F1: {res['f1_macro']:.2f}", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Guideline" if idx == 0 else "")

    plt.suptitle("Side-by-Side Confusion Matrices (Guideline vs Model Actions)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Side-by-side confusion matrix grid saved to: {cm_path}")


def plot_general_classification_metrics(
    general_df: pd.DataFrame,
    output_path: Path = None,
):
    """
    Plots 4-panel publication-ready comparison bar charts for standard classification metrics:
    - Panel 1: Overall Accuracy (%)
    - Panel 2: Macro F1-Score
    - Panel 3: Macro Precision (%)
    - Panel 4: Macro Recall / Balanced Accuracy (%)
    """
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "general_metrics_comparison.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.05)

    palette = ["#6c757d", "#1f77b4", "#9467bd", "#ff7f0e", "#2ca02c"]
    models = general_df["Model"].tolist()

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Overall Accuracy
    sns.barplot(data=general_df, x="Model", y="Accuracy (%)", hue="Model", legend=False, ax=axs[0, 0], palette=palette[:len(models)])
    axs[0, 0].set_title("Overall Clinical Accuracy (%)", fontweight="bold", pad=10)
    axs[0, 0].set_ylabel("Accuracy (%)")
    axs[0, 0].tick_params(axis="x", rotation=20)
    axs[0, 0].set_ylim(0, 100)
    for p in axs[0, 0].patches:
        axs[0, 0].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                           ha="center", va="center", color="white", fontweight="bold")

    # 2. Macro F1-Score
    sns.barplot(data=general_df, x="Model", y="Macro F1", hue="Model", legend=False, ax=axs[0, 1], palette=palette[:len(models)])
    axs[0, 1].set_title("Macro F1-Score (Unweighted All Classes)", fontweight="bold", pad=10)
    axs[0, 1].set_ylabel("Macro F1")
    axs[0, 1].tick_params(axis="x", rotation=20)
    axs[0, 1].set_ylim(0, 1.0)
    for p in axs[0, 1].patches:
        axs[0, 1].annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                           ha="center", va="center", color="white", fontweight="bold")

    # 3. Macro Precision
    sns.barplot(data=general_df, x="Model", y="Macro Precision (%)", hue="Model", legend=False, ax=axs[1, 0], palette=palette[:len(models)])
    axs[1, 0].set_title("Macro Precision (%) (Positive Predictive Value)", fontweight="bold", pad=10)
    axs[1, 0].set_ylabel("Precision (%)")
    axs[1, 0].tick_params(axis="x", rotation=20)
    axs[1, 0].set_ylim(0, 100)
    for p in axs[1, 0].patches:
        axs[1, 0].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                           ha="center", va="center", color="white", fontweight="bold")

    # 4. Macro Recall (Balanced Accuracy)
    sns.barplot(data=general_df, x="Model", y="Macro Recall (%)", hue="Model", legend=False, ax=axs[1, 1], palette=palette[:len(models)])
    axs[1, 1].set_title("Macro Recall (%) (Balanced Accuracy / Sensitivity)", fontweight="bold", pad=10)
    axs[1, 1].set_ylabel("Recall (%)")
    axs[1, 1].tick_params(axis="x", rotation=20)
    axs[1, 1].set_ylim(0, 100)
    for p in axs[1, 1].patches:
        axs[1, 1].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                           ha="center", va="center", color="white", fontweight="bold")

    plt.suptitle("General Classification Metrics: Accuracy, F1-Score, Precision, and Recall", fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] General classification metrics chart saved to: {output_path}")
