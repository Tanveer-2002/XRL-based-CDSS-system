"""
Data Loading Module for Type 2 Diabetes CDSS.

Supports:
1. Extraction from MIMIC-IV 3.1 hospital database (diagnoses, labs, OMR, patients).
2. Clinically calibrated synthetic longitudinal cohort generation.
3. Automated cached loading with pipeline integration.
"""

import gzip
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


# Mapping of MIMIC-IV item IDs to clinical biomarker names
MIMIC_LAB_ITEM_MAP = {
    50852: "hba1c",          # % Hemoglobin A1c
    50854: "hba1c",          # Absolute A1c
    50931: "glucose",        # Glucose (Blood Chemistry)
    50809: "glucose",        # Glucose (Blood Gas)
    52027: "glucose",        # Glucose (Whole Blood)
    50912: "creatinine",      # Creatinine (Blood)
    50907: "cholesterol",     # Cholesterol, Total
    50905: "ldl",            # Cholesterol, LDL, Calculated
    50906: "ldl",            # Cholesterol, LDL, Measured
    50904: "hdl",            # Cholesterol, HDL
    51000: "triglycerides",   # Triglycerides
}


def extract_mimic_cohort(
    mimic_dir: Optional[Path] = None,
    max_patients: int = 1500,
    chunk_size: int = 500_000,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Extracts a longitudinal Type 2 Diabetes cohort from MIMIC-IV hosp files.
    """
    if mimic_dir is None:
        mimic_dir = config.MIMIC_IV_ROOT
    else:
        mimic_dir = Path(mimic_dir)

    print(f"[*] Extracting MIMIC-IV cohort from: {mimic_dir}")
    if not mimic_dir.exists():
        raise FileNotFoundError(f"MIMIC-IV directory not found at {mimic_dir}")

    # 1. Find T2D patients via ICD-9 / ICD-10 diagnoses
    diag_file = mimic_dir / "diagnoses_icd.csv.gz"
    print(f"[*] Reading diagnoses from {diag_file.name}...")
    diag_df = pd.read_csv(diag_file)
    t2d_mask = (
        diag_df["icd_code"].astype(str).str.startswith("E11")
        | diag_df["icd_code"].astype(str).str.startswith("250")
    )
    t2d_subject_ids = set(diag_df[t2d_mask]["subject_id"].unique())
    print(f"[+] Found {len(t2d_subject_ids)} T2D patients in diagnoses.")

    if len(t2d_subject_ids) > max_patients:
        t2d_subject_ids = set(sorted(list(t2d_subject_ids))[:max_patients])
        print(f"[*] Limiting cohort to first {len(t2d_subject_ids)} patients for efficient processing.")

    # 2. Extract demographics from patients.csv.gz
    patients_file = mimic_dir / "patients.csv.gz"
    print(f"[*] Reading demographics from {patients_file.name}...")
    patients_df = pd.read_csv(patients_file)
    cohort_patients = patients_df[patients_df["subject_id"].isin(t2d_subject_ids)].copy()
    cohort_patients["gender"] = cohort_patients["gender"].map({"F": "Female", "M": "Male"}).fillna("Unknown")
    cohort_patients["age"] = cohort_patients["anchor_age"]
    demo_dict = cohort_patients.set_index("subject_id")[["gender", "age"]].to_dict(orient="index")

    # 3. Extract Blood Pressure and BMI from OMR
    omr_file = mimic_dir / "omr.csv.gz"
    omr_records = []
    if omr_file.exists():
        print(f"[*] Reading vital signs / BMI from {omr_file.name}...")
        omr_iter = pd.read_csv(omr_file, chunksize=chunk_size)
        for chunk in omr_iter:
            subset = chunk[chunk["subject_id"].isin(t2d_subject_ids)]
            if not subset.empty:
                omr_records.append(subset)

    vitals_df = pd.DataFrame()
    if omr_records:
        omr_df = pd.concat(omr_records, ignore_index=True)
        # Parse BP (e.g., '120/80')
        bp_df = omr_df[omr_df["result_name"].str.contains("Blood Pressure", case=False, na=False)].copy()
        bp_split = bp_df["result_value"].str.extract(r"(?P<sbp>\d{2,3})\s*/\s*(?P<dbp>\d{2,3})")
        bp_df["sbp"] = pd.to_numeric(bp_split["sbp"], errors="coerce")
        bp_df["dbp"] = pd.to_numeric(bp_split["dbp"], errors="coerce")

        bmi_df = omr_df[omr_df["result_name"].str.contains("BMI", case=False, na=False)].copy()
        bmi_df["bmi"] = pd.to_numeric(bmi_df["result_value"], errors="coerce")

        vitals_merged = pd.concat([
            bp_df[["subject_id", "chartdate", "sbp", "dbp"]].dropna(subset=["sbp", "dbp"]),
            bmi_df[["subject_id", "chartdate", "bmi"]].dropna(subset=["bmi"])
        ], ignore_index=True)
        vitals_df = vitals_merged.groupby(["subject_id", "chartdate"]).mean().reset_index()

    # 4. Extract target lab measurements from labevents.csv.gz
    lab_file = mimic_dir / "labevents.csv.gz"
    target_item_ids = set(MIMIC_LAB_ITEM_MAP.keys())
    print(f"[*] Scanning {lab_file.name} for target diabetes biomarkers...")

    lab_chunks = []
    chunk_count = 0
    with pd.read_csv(lab_file, chunksize=chunk_size) as reader:
        for chunk in reader:
            chunk_count += 1
            chunk_subset = chunk[
                chunk["subject_id"].isin(t2d_subject_ids)
                & chunk["itemid"].isin(target_item_ids)
            ]
            if not chunk_subset.empty:
                lab_chunks.append(
                    chunk_subset[["subject_id", "charttime", "itemid", "valuenum"]].dropna(subset=["valuenum"])
                )
            if chunk_count % 10 == 0:
                print(f"  Processed {chunk_count * chunk_size:,} lab rows... found {sum(len(c) for c in lab_chunks)} matching records")

    if not lab_chunks:
        raise ValueError("No matching lab records found in MIMIC-IV for selected cohort.")

    labs_df = pd.concat(lab_chunks, ignore_index=True)
    labs_df["chartdate"] = pd.to_datetime(labs_df["charttime"]).dt.strftime("%Y-%m-%d")
    labs_df["biomarker"] = labs_df["itemid"].map(MIMIC_LAB_ITEM_MAP)

    # Pivot labs by (subject_id, chartdate)
    pivot_labs = labs_df.pivot_table(
        index=["subject_id", "chartdate"],
        columns="biomarker",
        values="valuenum",
        aggfunc="mean"
    ).reset_index()

    # Combine vitals and labs
    if not vitals_df.empty:
        merged = pd.merge(pivot_labs, vitals_df, on=["subject_id", "chartdate"], how="outer")
    else:
        merged = pivot_labs

    # Add demographics and lifestyle proxies
    merged["gender"] = merged["subject_id"].map(lambda sid: demo_dict.get(sid, {}).get("gender", "Female"))
    merged["age"] = merged["subject_id"].map(lambda sid: demo_dict.get(sid, {}).get("age", 60.0))
    # Synthetic default for smoking and physical activity if not recorded in hosp tables
    np.random.seed(42)
    smoking_choices = ["never", "former", "current"]
    smoking_probs = [0.55, 0.30, 0.15]
    activity_choices = ["sedentary", "moderate", "active"]
    activity_probs = [0.45, 0.40, 0.15]

    unique_sids = merged["subject_id"].unique()
    sid_smoking = {s: np.random.choice(smoking_choices, p=smoking_probs) for s in unique_sids}
    sid_activity = {s: np.random.choice(activity_choices, p=activity_probs) for s in unique_sids}

    merged["smoking_status"] = merged["subject_id"].map(sid_smoking)
    merged["physical_activity"] = merged["subject_id"].map(sid_activity)

    # Reorder columns
    expected_cols = [
        "subject_id", "chartdate", "age", "gender", "smoking_status", "physical_activity",
        "hba1c", "glucose", "bmi", "sbp", "dbp", "cholesterol", "ldl", "hdl", "triglycerides", "creatinine"
    ]
    for c in expected_cols:
        if c not in merged.columns:
            merged[c] = np.nan

    merged = merged[expected_cols].sort_values(["subject_id", "chartdate"]).reset_index(drop=True)

    if output_path is None:
        output_path = config.INTERIM_DATA_DIR / "t2d_cohort_raw.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"[OK] Extracted {len(merged)} records across {merged['subject_id'].nunique()} patients to {output_path}")
    return merged


def generate_synthetic_cohort(
    num_patients: int = 500,
    min_visits: int = 4,
    max_visits: int = 12,
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generates a realistic, clinically calibrated longitudinal Type 2 Diabetes cohort.
    
    Includes 4 realistic clinical trajectories:
    1. Stable / Well-controlled (35%): HbA1c ~6.2-6.8%, normal rate of change
    2. Improving (20%): HbA1c gradually decreases from ~8.5% -> 6.8%
    3. Deteriorating (30%): HbA1c increases from ~6.8% -> 8.8%, rising BP and BMI
    4. Rapidly deteriorating (15%): HbA1c surges from 7.5% -> 11.0%, severe glycemic spikes
    """
    np.random.seed(seed)
    records = []

    # Trajectory types and probabilities
    trajectories = ["stable", "improving", "deteriorating", "rapid_deteriorating"]
    traj_probs = [0.35, 0.20, 0.30, 0.15]

    for pid in range(1001, 1001 + num_patients):
        subject_id = f"P{pid}"
        gender = np.random.choice(["Male", "Female"], p=[0.52, 0.48])
        age_start = int(np.random.normal(61, 10))
        age_start = max(30, min(88, age_start))
        
        smoking_status = np.random.choice(["never", "former", "current"], p=[0.50, 0.35, 0.15])
        physical_activity = np.random.choice(["sedentary", "moderate", "active"], p=[0.45, 0.40, 0.15])
        
        traj_type = np.random.choice(trajectories, p=traj_probs)
        num_visits = np.random.randint(min_visits, max_visits + 1)
        
        # Start date
        start_year = np.random.randint(2018, 2021)
        start_month = np.random.randint(1, 12)
        start_day = np.random.randint(1, 28)
        current_date = pd.Timestamp(year=start_year, month=start_month, day=start_day)

        # Baseline clinical parameters
        if traj_type == "stable":
            base_hba1c = np.random.uniform(6.0, 6.9)
            base_bmi = np.random.uniform(24.0, 31.0)
            base_sbp = np.random.uniform(118.0, 134.0)
            base_dbp = np.random.uniform(74.0, 84.0)
            hba1c_drift = np.random.normal(0.0, 0.05)
        elif traj_type == "improving":
            base_hba1c = np.random.uniform(8.0, 9.8)
            base_bmi = np.random.uniform(29.0, 37.0)
            base_sbp = np.random.uniform(135.0, 155.0)
            base_dbp = np.random.uniform(85.0, 95.0)
            hba1c_drift = -np.random.uniform(0.18, 0.35)
        elif traj_type == "deteriorating":
            base_hba1c = np.random.uniform(6.8, 7.6)
            base_bmi = np.random.uniform(27.0, 34.0)
            base_sbp = np.random.uniform(128.0, 142.0)
            base_dbp = np.random.uniform(80.0, 90.0)
            hba1c_drift = np.random.uniform(0.15, 0.32)
        else: # rapid_deteriorating
            base_hba1c = np.random.uniform(7.4, 8.8)
            base_bmi = np.random.uniform(29.0, 38.0)
            base_sbp = np.random.uniform(138.0, 160.0)
            base_dbp = np.random.uniform(88.0, 100.0)
            hba1c_drift = np.random.uniform(0.40, 0.85)

        base_creatinine = np.random.uniform(0.7, 1.2)
        base_chol = np.random.uniform(160.0, 230.0)
        base_ldl = base_chol * np.random.uniform(0.55, 0.65)
        base_hdl = np.random.uniform(38.0, 58.0)
        base_trig = np.random.uniform(130.0, 240.0)

        curr_hba1c = base_hba1c
        curr_bmi = base_bmi
        curr_sbp = base_sbp
        curr_dbp = base_dbp
        curr_creat = base_creatinine
        curr_chol = base_chol
        curr_ldl = base_ldl
        curr_hdl = base_hdl
        curr_trig = base_trig

        for v in range(num_visits):
            # Days elapsed: typically 60 to 180 days (2 to 6 months) between observations
            if v > 0:
                days_gap = int(np.random.normal(120, 30))
                days_gap = max(30, min(365, days_gap))
                current_date += pd.Timedelta(days=days_gap)
                
                # Apply trajectory drift
                curr_hba1c = max(4.5, min(16.0, curr_hba1c + hba1c_drift + np.random.normal(0, 0.12)))
                curr_bmi = max(18.0, min(55.0, curr_bmi + (hba1c_drift * 0.4) + np.random.normal(0, 0.2)))
                curr_sbp = max(80.0, min(220.0, curr_sbp + (hba1c_drift * 4.0) + np.random.normal(0, 2.5)))
                curr_dbp = max(50.0, min(130.0, curr_dbp + (hba1c_drift * 2.0) + np.random.normal(0, 1.8)))
                curr_creat = max(0.4, min(6.0, curr_creat + (0.04 if traj_type.startswith("deteriorating") else 0.0) + np.random.normal(0, 0.03)))
                curr_ldl = max(40.0, min(300.0, curr_ldl + (hba1c_drift * 6.0) + np.random.normal(0, 5.0)))
                curr_trig = max(50.0, min(600.0, curr_trig + (hba1c_drift * 12.0) + np.random.normal(0, 8.0)))
                curr_chol = curr_ldl + curr_hdl + (curr_trig / 5.0)

            # Glucose strongly correlates with HbA1c (eAG = 28.7 * HbA1c - 46.7)
            glucose_mean = 28.7 * curr_hba1c - 46.7
            curr_glucose = max(40.0, min(500.0, np.random.normal(glucose_mean, 18.0)))

            age_at_visit = age_start + int((current_date.year - start_year))

            record = {
                "subject_id": subject_id,
                "chartdate": current_date.strftime("%Y-%m-%d"),
                "age": age_at_visit,
                "gender": gender,
                "smoking_status": smoking_status,
                "physical_activity": physical_activity,
                "hba1c": round(curr_hba1c, 2),
                "glucose": round(curr_glucose, 1),
                "bmi": round(curr_bmi, 1),
                "sbp": round(curr_sbp, 1),
                "dbp": round(curr_dbp, 1),
                "cholesterol": round(curr_chol, 1),
                "ldl": round(curr_ldl, 1),
                "hdl": round(curr_hdl, 1),
                "triglycerides": round(curr_trig, 1),
                "creatinine": round(curr_creat, 2),
            }

            # Clinically simulate occasional missing lab (e.g. lipids or creatinine not ordered at every visit)
            if np.random.rand() < 0.12 and v > 0:
                record["cholesterol"] = np.nan
                record["ldl"] = np.nan
                record["hdl"] = np.nan
                record["triglycerides"] = np.nan
            if np.random.rand() < 0.08 and v > 0:
                record["creatinine"] = np.nan

            records.append(record)

    df = pd.DataFrame(records)
    if output_path is None:
        output_path = config.INTERIM_DATA_DIR / "t2d_cohort_raw.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Generated synthetic cohort with {len(df)} visits across {num_patients} patients at {output_path}")
    return df


def load_raw_data(
    source: str = "auto",
    num_patients: int = 500,
    force_new: bool = False,
) -> pd.DataFrame:
    """
    Main loader function:
    - If cached interim file exists and not force_new, loads it.
    - If source == 'mimic' or 'auto' (and MIMIC hosp files exist), attempts extraction.
    - Otherwise generates calibrated synthetic cohort.
    """
    cache_path = config.INTERIM_DATA_DIR / "t2d_cohort_raw.csv"
    if cache_path.exists() and not force_new:
        print(f"[*] Loading existing cohort from cache: {cache_path}")
        return pd.read_csv(cache_path)

    if source == "mimic":
        if config.MIMIC_IV_ROOT.exists() and (config.MIMIC_IV_ROOT / "diagnoses_icd.csv.gz").exists():
            try:
                return extract_mimic_cohort(mimic_dir=config.MIMIC_IV_ROOT, max_patients=num_patients)
            except Exception as e:
                print(f"[!] MIMIC-IV extraction encountered: {e}. Falling back to synthetic cohort generation.")

    print("[*] Generating high-fidelity calibrated longitudinal cohort...")
    return generate_synthetic_cohort(num_patients=num_patients, output_path=cache_path)
