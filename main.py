"""
Type 2 Diabetes Clinical Decision Support System (CDSS) - Unified CLI & Pipeline Runner.

Longitudinal Health Trajectory Analysis + Offline RL (CQL) + Explainable AI (SHAP) + Cumulative Reporting

Usage:
  python main.py --mode demo                     # Run complete end-to-end pipeline & demo
  python main.py --mode extract                  # Extract cohort from MIMIC-IV / synthetic generator
  python main.py --mode preprocess               # Execute preprocessing and artifact persistence
  python main.py --mode trajectory               # Compute longitudinal trajectories and trends
  python main.py --mode train                    # Train Discrete Conservative Q-Learning (CQL) agent
  python main.py --mode evaluate                 # Benchmark trained CQL agent against baselines
  python main.py --mode report --patient-id P1001# Generate cumulative report for a specific patient
  python main.py --mode serve --port 8000        # Launch FastAPI REST API server
"""

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import sys
# Ensure project root is in python path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
import numpy as np
import pandas as pd
import torch

from preprocessing.load_data import load_raw_data
from preprocessing.normalization import ClinicalScaler
from preprocessing.pipeline import PreprocessingPipeline, split_by_patient
from trajectory.pipeline import TrajectoryPipeline
from rl.actions import MonitoringAction, get_action_metadata
from rl.dataset import create_rl_dataloaders
from rl.evaluate import evaluate_policy
from rl.model import DiscreteCQLNetwork
from rl.state import StateConstructor
from rl.train import train_cql_agent
from rl.transitions import generate_offline_transitions
from xai.explanation_generator import generate_clinical_explanation
from xai.shap_explainer import RLShapExplainer
from report.report_generator import (
    generate_cumulative_report,
    render_html_dashboard,
    render_markdown_report,
)


def run_extract(source: str = "auto", num_patients: int = 500) -> pd.DataFrame:
    """Extracts raw cohort records."""
    print("\n=======================================================")
    print("STEP 1: EXTRACTING LONGITUDINAL HEALTHCARE DATA")
    print("=======================================================")
    raw_df = load_raw_data(source=source, num_patients=num_patients)
    print(f"[OK] Extracted {len(raw_df)} longitudinal records for {raw_df['subject_id'].nunique()} patients.")
    return raw_df


def run_preprocess(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, PreprocessingPipeline]:
    """Runs data cleaning, imputation, and feature engineering."""
    print("\n=======================================================")
    print("STEP 2: CLINICAL DATA PREPROCESSING")
    print("=======================================================")
    train_raw, val_raw = split_by_patient(raw_df, test_size=0.2, seed=42)

    pipeline = PreprocessingPipeline()
    train_prep = pipeline.fit_transform_initial(train_raw)
    val_prep = pipeline.transform_initial(val_raw)

    combined_prep = pd.concat([train_prep, val_prep], ignore_index=True)
    return combined_prep, pipeline


def run_trajectory(prep_df: pd.DataFrame) -> pd.DataFrame:
    """Computes health trajectory metrics on raw clinical values."""
    print("\n=======================================================")
    print("STEP 3: HEALTH TRAJECTORY ANALYSIS (LONGITUDINAL ENGINE)")
    print("=======================================================")
    traj_engine = TrajectoryPipeline()
    traj_df = traj_engine.run(prep_df)
    traj_engine.save(traj_df)

    status_dist = traj_df["overall_trajectory_status"].value_counts(normalize=True).to_dict()
    print("[*] Trajectory Status Distribution:")
    for status, pct in status_dist.items():
        print(f"    - {status}: {pct*100:.1f}%")

    return traj_df


def run_scaling(traj_df: pd.DataFrame, pipeline: PreprocessingPipeline) -> pd.DataFrame:
    """Scales numerical and trajectory features strictly using training stats."""
    print("\n=======================================================")
    print("STEP 4: POST-TRAJECTORY LEAK-FREE NORMALIZATION")
    print("=======================================================")
    train_traj, val_traj = split_by_patient(traj_df, test_size=0.2, seed=42)

    # Detect numerical columns including continuous trajectory metrics
    num_cols = [c for c in config.BIOMARKERS if c in traj_df.columns]
    for marker in config.BIOMARKERS:
        for suffix in ["_change", "_rate", "_moving_avg"]:
            col_name = f"{marker}{suffix}"
            if col_name in traj_df.columns:
                num_cols.append(col_name)
    num_cols.extend(["age", "days_since_last_visit"])

    train_scaled = pipeline.fit_and_apply_scaler(train_traj, numerical_cols=num_cols)
    val_scaled = pipeline.apply_scaler(val_traj)
    scaled_all = pd.concat([train_scaled, val_scaled], ignore_index=True)

    # Save preprocessing objects
    pipeline.save_artifacts()

    # Save processed trajectory dataset with scaled features
    output_path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"
    scaled_all.to_csv(output_path, index=False)
    return scaled_all


def run_train_rl(traj_df: pd.DataFrame, epochs: int = 30) -> Tuple[DiscreteCQLNetwork, StateConstructor]:
    """Prepares transitions and trains Discrete Conservative Q-Learning (CQL) agent."""
    print("\n=======================================================")
    print("STEP 5: OFFLINE REINFORCEMENT LEARNING (DISCRETE CQL)")
    print("=======================================================")
    state_constructor = StateConstructor()

    train_loader, val_loader, _, _ = create_rl_dataloaders(
        trajectory_df=traj_df,
        state_constructor=state_constructor,
        batch_size=config.RL_CONFIG["batch_size"],
    )

    model, history = train_cql_agent(
        train_loader=train_loader,
        val_loader=val_loader,
        state_dim=state_constructor.state_dim,
        epochs=epochs,
        lr=config.RL_CONFIG["lr"],
        gamma=config.RL_CONFIG["gamma"],
        cql_alpha=config.RL_CONFIG["cql_alpha"],
    )
    return model, state_constructor


def run_evaluate(traj_df: pd.DataFrame, model: DiscreteCQLNetwork, state_constructor: StateConstructor):
    """Benchmarks trained CQL policy against baselines."""
    print("\n=======================================================")
    print("STEP 6: OFFLINE POLICY EVALUATION & BENCHMARKING")
    print("=======================================================")
    evaluate_policy(model, traj_df, state_constructor=state_constructor)


def run_demo_patient_report(
    traj_df: pd.DataFrame,
    model: DiscreteCQLNetwork,
    state_constructor: StateConstructor,
    patient_id: Optional[str] = None,
):
    """Generates an end-to-end sample report with SHAP explanation."""
    print("\n=======================================================")
    print("STEP 7: EXPLAINABLE AI (SHAP) & CUMULATIVE REPORT DEMO")
    print("=======================================================")
    if patient_id is None:
        # Pick a deteriorating or interesting patient to demonstrate early risk detection
        det_patients = traj_df[traj_df["overall_trajectory_status"].isin(["deteriorating", "rapidly_deteriorating"])]["subject_id"].unique()
        patient_id = det_patients[0] if len(det_patients) > 0 else traj_df["subject_id"].iloc[0]

    patient_records = traj_df[traj_df["subject_id"] == patient_id].sort_values(by="chartdate").reset_index(drop=True)
    latest_row = patient_records.iloc[-1]

    # Predict Action
    state_vec = state_constructor.extract_state_vector(latest_row)
    state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)

    model.eval()
    q_vals = model.get_q_values(state_tensor).cpu().numpy()
    action_idx = int(np.argmax(q_vals))

    meta = get_action_metadata(action_idx)
    q_dict = {config.ACTION_NAMES[i]: round(float(q_vals[i]), 3) for i in range(len(q_vals))}

    # Explain with SHAP
    bg_states = state_constructor.extract_state_matrix(traj_df.sample(min(40, len(traj_df))))
    explainer = RLShapExplainer(model=model, background_data=bg_states, state_constructor=state_constructor)
    shap_res = explainer.explain_action(state_vec, action=action_idx)

    # Natural language explanation
    explanation = generate_clinical_explanation(
        action=action_idx,
        obs_data=latest_row,
        attributions=shap_res["attributions"],
    )

    rl_rec = {
        "action_code": action_idx,
        "action_name": meta["name"],
        "recommended_interval": meta["interval"],
        "urgency": meta["urgency"],
        "q_values": q_dict,
        "description": meta["description"],
    }

    report = generate_cumulative_report(
        patient_history_df=patient_records,
        rl_recommendation_dict=rl_rec,
        xai_explanation_dict=explanation,
    )

    # Export to reports directory
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = config.REPORTS_DIR / f"{patient_id}_report.json"
    md_path = config.REPORTS_DIR / f"{patient_id}_report.md"
    html_path = config.REPORTS_DIR / f"{patient_id}_report.html"

    import json
    with open(json_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown_report(report))

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(render_html_dashboard(report))

    print(f"\n[OK] Generated Cumulative Health Report for Patient {patient_id}:")
    print(f"    - Trajectory Status:     {report.overall_trajectory_status.upper()}")
    print(f"    - RL Monitoring Action:  {meta['name']} (Interval: {meta['interval']}, Urgency: {meta['urgency']})")
    print(f"    - Summary Statement:     {explanation['summary_statement']}")
    print(f"    - Saved JSON Report:     {json_path}")
    print(f"    - Saved Markdown Report: {md_path}")
    print(f"    - Saved HTML Dashboard:  {html_path}")


def main():
    parser = argparse.ArgumentParser(description="Type 2 Diabetes CDSS Runner")
    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=["extract", "preprocess", "trajectory", "train", "retrain", "evaluate", "accuracy", "report", "serve", "demo"],
        help="Pipeline execution mode",
    )
    parser.add_argument("--source", type=str, default="auto", help="Data source: 'mimic', 'synthetic', or 'auto'")
    parser.add_argument("--num-patients", type=int, default=500, help="Number of cohort patients to process")
    parser.add_argument("--epochs", type=int, default=30, help="Offline RL training epochs")
    parser.add_argument("--patient-id", type=str, default=None, help="Target patient ID for reporting")
    parser.add_argument("--port", type=int, default=8000, help="Port for FastAPI server")

    args = parser.parse_args()

    if args.mode == "accuracy":
        from test_model_accuracy import run_accuracy_test
        run_accuracy_test()

    elif args.mode == "extract":
        run_extract(source=args.source, num_patients=args.num_patients)

    elif args.mode == "preprocess":
        raw_df = load_raw_data(source=args.source, num_patients=args.num_patients)
        prep_df, pipeline = run_preprocess(raw_df)
        traj_df = run_trajectory(prep_df)
        run_scaling(traj_df, pipeline)

    elif args.mode == "trajectory":
        prep_path = config.INTERIM_DATA_DIR / "t2d_cohort_raw.csv"
        df = pd.read_csv(prep_path)
        run_trajectory(df)

    elif args.mode == "train":
        traj_path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"
        df = pd.read_csv(traj_path)
        run_train_rl(df, epochs=args.epochs)

    elif args.mode == "retrain":
        from rl.cql import fine_tune_cql_with_feedback
        print("[*] Initiating Continual RL Retraining with Clinician Feedback & Longitudinal Outcomes...")
        epochs_to_run = args.epochs if args.epochs != 30 else 5
        res = fine_tune_cql_with_feedback(epochs=epochs_to_run)
        print(f"[OK] Retraining finished. Status: {res['status']} | Safety Passed: {res['passed_safety']}")

    elif args.mode == "evaluate":
        traj_path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"
        df = pd.read_csv(traj_path)
        model_path = config.RL_AGENT_DIR / "cql_model.pt"
        sc = StateConstructor()
        model = DiscreteCQLNetwork(state_dim=sc.state_dim)
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        run_evaluate(df, model, sc)

    elif args.mode == "report":
        traj_path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"
        df = pd.read_csv(traj_path)
        model_path = config.RL_AGENT_DIR / "cql_model.pt"
        sc = StateConstructor()
        model = DiscreteCQLNetwork(state_dim=sc.state_dim)
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        run_demo_patient_report(df, model, sc, patient_id=args.patient_id)

    elif args.mode == "serve":
        import uvicorn
        print(f"[*] Starting FastAPI Server on port {args.port}...")
        uvicorn.run("api.main:app", host="0.0.0.0", port=args.port, reload=False)

    elif args.mode == "demo":
        print("=========================================================================")
        print("  TYPE 2 DIABETES CLINICAL DECISION SUPPORT SYSTEM (CDSS) - PIPELINE DEMO")
        print("=========================================================================")
        # 1. Extract
        raw_df = run_extract(source=args.source, num_patients=args.num_patients)
        # 2. Preprocess
        prep_df, pipeline = run_preprocess(raw_df)
        # 3. Trajectory analysis
        traj_df = run_trajectory(prep_df)
        # 4. Scaling
        scaled_df = run_scaling(traj_df, pipeline)
        # 5. RL Training
        model, state_constructor = run_train_rl(scaled_df, epochs=args.epochs)
        # 6. Evaluation
        run_evaluate(scaled_df, model, state_constructor)
        # 7. XAI & Report Demo
        run_demo_patient_report(scaled_df, model, state_constructor, patient_id=args.patient_id)
        print("\n=========================================================================")
        print("  [OK] ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
        print("=========================================================================")


if __name__ == "__main__":
    main()
