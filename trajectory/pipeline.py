"""
Consolidated Trajectory Pipeline Module for Type 2 Diabetes CDSS.

Combines:
1. Patient Timeline Construction (build_patient_timeline)
2. Biomarker Changes & Rates of Change (compute_biomarker_changes)
3. Rolling Moving Averages & EMA (compute_moving_averages)
4. Clinical Trend Classification (classify_trend, classify_biomarker_trends)
5. Trajectory Feature Assembly (build_trajectory_features)
6. Overall Longitudinal Trajectory Progression Status Evaluation (evaluate_trajectory_status)
7. Trajectory Pipeline Orchestrator (TrajectoryPipeline)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


# =========================================================================
# 1. PATIENT TIMELINE CONSTRUCTION
# =========================================================================

def build_patient_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs a validated chronological patient timeline:
    - 1-indexed sequential visit counter (visit_number).
    - Time interval between consecutive visits in days (days_since_last_visit).
    - Cumulative days elapsed from patient's first baseline record (cumulative_days).
    """
    timeline_df = df.copy()
    timeline_df["chartdate"] = pd.to_datetime(timeline_df["chartdate"], format="mixed")
    timeline_df = timeline_df.sort_values(by=["subject_id", "chartdate"]).reset_index(drop=True)

    timeline_df["visit_number"] = timeline_df.groupby("subject_id").cumcount() + 1

    prev_dates = timeline_df.groupby("subject_id")["chartdate"].shift(1)
    time_diff_days = (timeline_df["chartdate"] - prev_dates).dt.total_seconds() / (24.0 * 3600.0)
    timeline_df["days_since_last_visit"] = time_diff_days.fillna(0.0).round(1)
    timeline_df.loc[timeline_df["visit_number"] == 1, "days_since_last_visit"] = 0.0

    baseline_dates = timeline_df.groupby("subject_id")["chartdate"].transform("min")
    timeline_df["cumulative_days"] = (
        (timeline_df["chartdate"] - baseline_dates).dt.total_seconds() / (24.0 * 3600.0)
    ).round(1)

    return timeline_df


# =========================================================================
# 2. BIOMARKER CHANGES AND RATES OF CHANGE
# =========================================================================

def compute_biomarker_changes(
    df: pd.DataFrame,
    biomarkers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Computes absolute delta (x_t - x_{t-1}) and monthly / daily rates of change.
    """
    if biomarkers is None:
        biomarkers = config.BIOMARKERS

    result_df = df.copy()
    grouped = result_df.groupby("subject_id")

    for marker in biomarkers:
        if marker not in result_df.columns:
            continue

        prev_val = grouped[marker].shift(1)
        delta_col = f"{marker}_change"
        result_df[delta_col] = (result_df[marker] - prev_val).fillna(0.0).round(3)
        result_df.loc[result_df["visit_number"] == 1, delta_col] = 0.0

        rate_monthly_col = f"{marker}_rate"
        rate_daily_col = f"{marker}_rate_daily"

        dt = result_df["days_since_last_visit"].copy()
        is_positive_dt = (dt > 0) & (result_df["visit_number"] > 1)

        result_df[rate_daily_col] = 0.0
        result_df[rate_monthly_col] = 0.0

        if is_positive_dt.any():
            daily = result_df.loc[is_positive_dt, delta_col] / dt.loc[is_positive_dt]
            result_df.loc[is_positive_dt, rate_daily_col] = daily.round(5)
            result_df.loc[is_positive_dt, rate_monthly_col] = (daily * 30.0).round(4)

    return result_df


# =========================================================================
# 3. ROLLING MOVING AVERAGES & EMA
# =========================================================================

def compute_moving_averages(
    df: pd.DataFrame,
    biomarkers: Optional[List[str]] = None,
    window: int = 3,
) -> pd.DataFrame:
    """
    Computes 3-visit rolling average and exponential moving average (EMA)
    to isolate true trajectory drifts from transient laboratory noise.
    """
    if biomarkers is None:
        biomarkers = config.BIOMARKERS

    result_df = df.copy()

    for marker in biomarkers:
        if marker not in result_df.columns:
            continue

        ma_col = f"{marker}_moving_avg"
        ema_col = f"{marker}_ema"

        result_df[ma_col] = (
            result_df.groupby("subject_id")[marker]
            .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
            .round(2)
        )

        result_df[ema_col] = (
            result_df.groupby("subject_id")[marker]
            .transform(lambda s: s.ewm(span=window, min_periods=1).mean())
            .round(2)
        )

    return result_df


# =========================================================================
# 4. CLINICAL TREND CLASSIFICATION (ADA GUIDELINES)
# =========================================================================

def classify_trend(change_val: float, threshold: float) -> str:
    """Classifies delta into 'increasing', 'decreasing', or 'stable'."""
    if np.isnan(change_val):
        return "stable"
    if change_val >= threshold:
        return "increasing"
    elif change_val <= -threshold:
        return "decreasing"
    return "stable"


def classify_biomarker_trends(
    df: pd.DataFrame,
    biomarkers: Optional[List[str]] = None,
    thresholds: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Classifies trend direction for each biomarker based on clinical thresholds.
    """
    if biomarkers is None:
        biomarkers = config.BIOMARKERS
    if thresholds is None:
        thresholds = config.CLINICAL_THRESHOLDS

    result_df = df.copy()
    trend_mapping = {"increasing": 1, "stable": 0, "decreasing": -1}

    for marker in biomarkers:
        change_col = f"{marker}_change"
        trend_col = f"{marker}_trend"
        trend_num_col = f"{marker}_trend_num"

        if change_col not in result_df.columns:
            continue

        thresh_info = thresholds.get(marker, {})
        sig_threshold = thresh_info.get("sig_change", 0.0)

        result_df[trend_col] = result_df[change_col].apply(
            lambda val: classify_trend(val, sig_threshold)
        )
        result_df[trend_num_col] = result_df[trend_col].map(trend_mapping).fillna(0).astype(int)

        result_df.loc[result_df["visit_number"] == 1, trend_col] = "stable"
        result_df.loc[result_df["visit_number"] == 1, trend_num_col] = 0

    return result_df


# =========================================================================
# 5. TRAJECTORY FEATURES ASSEMBLY
# =========================================================================

def build_trajectory_features(
    df: pd.DataFrame,
    biomarkers: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Constructs the complete trajectory feature set:
    - Timeline and elapsed days
    - Biomarker deltas and monthly rates
    - 3-visit moving averages
    - Discrete clinical trend flags
    """
    if biomarkers is None:
        biomarkers = config.BIOMARKERS

    traj_df = build_patient_timeline(df)
    traj_df = compute_biomarker_changes(traj_df, biomarkers=biomarkers)
    traj_df = compute_moving_averages(traj_df, biomarkers=biomarkers, window=3)
    traj_df = classify_biomarker_trends(traj_df, biomarkers=biomarkers)

    trajectory_cols = []
    for marker in biomarkers:
        trajectory_cols.extend([
            f"{marker}_change",
            f"{marker}_rate",
            f"{marker}_moving_avg",
            f"{marker}_trend_num",
        ])

    return traj_df, trajectory_cols


# =========================================================================
# 6. OVERALL TRAJECTORY STATUS EVALUATION
# =========================================================================

def evaluate_trajectory_status(row: pd.Series) -> str:
    """
    Evaluates multi-indicator longitudinal status following ADA and KDIGO clinical guidelines:
    - "stable"
    - "improving"
    - "deteriorating"
    - "rapidly_deteriorating"
    """
    if row.get("visit_number", 1) == 1:
        h1 = row.get("hba1c", 7.0)
        return "deteriorating" if h1 >= 9.5 else "stable"

    score = 0.0

    # Glycemic Trajectory
    h_curr = row.get("hba1c", 7.0)
    h_change = row.get("hba1c_change", 0.0)
    h_rate = row.get("hba1c_rate", 0.0)
    glu_change = row.get("glucose_change", 0.0)

    if h_rate >= 0.15 or h_change >= 0.8 or (h_curr >= 9.5 and h_change > 0.1):
        score += 3.0
    elif h_change >= 0.3:
        score += 1.5 if h_curr < 8.0 else 2.0
    elif h_change <= -0.5 and h_curr < 8.5:
        score -= 2.0
    elif h_change <= -0.3:
        score -= 1.0

    if glu_change >= 25.0:
        score += 1.0
    elif glu_change <= -25.0:
        score -= 0.5

    # Cardiovascular Trajectory
    sbp_change = row.get("sbp_change", 0.0)
    sbp_curr = row.get("sbp", 125.0)
    if sbp_change >= 12.0 or sbp_curr >= 150.0:
        score += 1.0
    elif sbp_change <= -12.0:
        score -= 0.5

    # Renal & Metabolic Trajectory
    creat_change = row.get("creatinine_change", 0.0)
    creat_curr = row.get("creatinine", 1.0)
    if creat_change >= 0.25 or creat_curr >= 1.6:
        score += 1.5

    bmi_change = row.get("bmi_change", 0.0)
    if bmi_change >= 1.2:
        score += 0.5
    elif bmi_change <= -1.2:
        score -= 0.5

    ldl_change = row.get("ldl_change", 0.0)
    if ldl_change >= 20.0:
        score += 0.5
    elif ldl_change <= -20.0:
        score -= 0.5

    if score >= 3.5:
        return "rapidly_deteriorating"
    elif score >= 1.5:
        return "deteriorating"
    elif score <= -1.5:
        return "improving"
    else:
        return "stable"


# =========================================================================
# 7. TRAJECTORY PIPELINE ORCHESTRATOR
# =========================================================================

class TrajectoryPipeline:
    """Executes full longitudinal health trajectory extraction."""

    def __init__(self, biomarkers: Optional[List[str]] = None):
        self.biomarkers = biomarkers or config.BIOMARKERS

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        print("[*] Trajectory Pipeline: Computing biomarker timelines, deltas, rates, and moving averages...")
        traj_df, _ = build_trajectory_features(df, biomarkers=self.biomarkers)

        print("[*] Trajectory Pipeline: Evaluating clinical overall trajectory status...")
        traj_df["overall_trajectory_status"] = traj_df.apply(evaluate_trajectory_status, axis=1)

        # Print distribution
        dist = traj_df["overall_trajectory_status"].value_counts(normalize=True) * 100
        print("[*] Trajectory Status Distribution:")
        for status, pct in dist.items():
            print(f"    - {status}: {pct:.1f}%")

        return traj_df

    # Alias for backward compatibility
    run = process

    def save(self, df: pd.DataFrame, file_path: Optional[Path] = None):
        if file_path is None:
            file_path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"[OK] Trajectory dataset saved successfully to {file_path}")
