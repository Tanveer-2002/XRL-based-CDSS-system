"""
Consolidated Preprocessing Pipeline Module for Type 2 Diabetes CDSS.

Combines:
1. Data Cleaning & Bounds Verification (clean_records)
2. Temporal Feature Engineering (engineer_features)
3. Patient-Aware Longitudinal Missing Value Imputation (LongitudinalImputer)
4. Categorical Encoding (CategoricalEncoder)
5. Leak-Free Normalization (ClinicalScaler)
6. Patient-Level Splitting (split_by_patient)
7. Unified Preprocessing Pipeline (PreprocessingPipeline)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


# =========================================================================
# 1. DATA CLEANING & PHYSIOLOGICAL BOUNDS VERIFICATION
# =========================================================================

def clean_records(
    df: pd.DataFrame,
    min_visits: int = 2,
    apply_bounds: bool = True,
) -> pd.DataFrame:
    """
    Cleans raw longitudinal patient records:
    - Parses and validates datetimes.
    - Chronologically orders encounters per patient.
    - Deduplicates same-day multi-records via clinical averaging.
    - Filters out-of-bounds physiological values to NaN for imputation.
    - Enforces minimum visit count requirement per patient.
    """
    cleaned = df.copy()

    # Datetime parsing & dropping missing primary keys
    cleaned["chartdate"] = pd.to_datetime(cleaned["chartdate"], errors="coerce")
    cleaned = cleaned.dropna(subset=["subject_id", "chartdate"])

    # Sort chronologically
    cleaned = cleaned.sort_values(by=["subject_id", "chartdate"]).reset_index(drop=True)

    # Handle same-day duplicates by averaging measurements
    group_cols = ["subject_id", "chartdate"]
    agg_dict = {}

    for col in config.BIOMARKERS:
        if col in cleaned.columns:
            agg_dict[col] = "mean"

    for col in config.CATEGORICAL_FEATURES:
        if col in cleaned.columns:
            agg_dict[col] = "first"

    for col in ["age"]:
        if col in cleaned.columns:
            agg_dict[col] = "max"

    if agg_dict:
        cleaned = cleaned.groupby(group_cols, as_index=False).agg(agg_dict)

    # Filter physiological bounds
    if apply_bounds:
        for col, (low, high) in config.PHYSIOLOGICAL_BOUNDS.items():
            if col in cleaned.columns:
                mask_out_of_bounds = (cleaned[col] < low) | (cleaned[col] > high)
                if mask_out_of_bounds.any():
                    cleaned.loc[mask_out_of_bounds, col] = np.nan

    # Filter out patients with fewer than min_visits
    visit_counts = cleaned["subject_id"].value_counts()
    valid_subjects = visit_counts[visit_counts >= min_visits].index
    cleaned = cleaned[cleaned["subject_id"].isin(valid_subjects)].copy()

    return cleaned.sort_values(by=["subject_id", "chartdate"]).reset_index(drop=True)


# =========================================================================
# 2. TEMPORAL & VISIT FEATURE ENGINEERING
# =========================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives temporal and visit metadata:
    - visit_number: 1-indexed sequential encounter count per patient
    - days_since_last_visit: delta time in days from previous observation (dt)
    - cumulative_days: total days elapsed since patient's first baseline record
    """
    engineered = df.copy()
    engineered["chartdate"] = pd.to_datetime(engineered["chartdate"])
    engineered = engineered.sort_values(by=["subject_id", "chartdate"]).reset_index(drop=True)

    engineered["visit_number"] = engineered.groupby("subject_id").cumcount() + 1

    engineered["prev_date"] = engineered.groupby("subject_id")["chartdate"].shift(1)
    engineered["days_since_last_visit"] = (
        (engineered["chartdate"] - engineered["prev_date"]).dt.total_seconds() / (24 * 3600)
    ).fillna(0.0).round(1)
    engineered.loc[engineered["visit_number"] == 1, "days_since_last_visit"] = 0.0

    first_dates = engineered.groupby("subject_id")["chartdate"].transform("min")
    engineered["cumulative_days"] = (
        (engineered["chartdate"] - first_dates).dt.total_seconds() / (24 * 3600)
    ).round(1)

    engineered = engineered.drop(columns=["prev_date"])
    return engineered


# =========================================================================
# 3. LONGITUDINAL MISSING VALUE IMPUTATION
# =========================================================================

class LongitudinalImputer:
    """
    Longitudinal missing value imputer designed for patient trajectories:
    1. Patient forward-fill (carry observations forward).
    2. Patient backward-fill (carry first observation backward for initial visits).
    3. Population median fallback (fitted strictly on training set).
    """

    def __init__(self, biomarker_cols: Optional[List[str]] = None):
        self.biomarker_cols = biomarker_cols or config.BIOMARKERS
        self.population_medians: Dict[str, float] = {}
        self.categorical_modes: Dict[str, str] = {}
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> "LongitudinalImputer":
        for col in self.biomarker_cols:
            if col in train_df.columns:
                val = train_df[col].dropna()
                self.population_medians[col] = float(val.median()) if not val.empty else 0.0

        for col in config.CATEGORICAL_FEATURES:
            if col in train_df.columns:
                mode_series = train_df[col].dropna().mode()
                self.categorical_modes[col] = mode_series.iloc[0] if not mode_series.empty else "Unknown"

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("LongitudinalImputer must be fitted before transform.")

        imputed = df.copy()
        imputed["chartdate"] = pd.to_datetime(imputed["chartdate"])
        imputed = imputed.sort_values(by=["subject_id", "chartdate"]).reset_index(drop=True)

        present_biomarkers = [c for c in self.biomarker_cols if c in imputed.columns]
        if present_biomarkers:
            imputed[present_biomarkers] = imputed.groupby("subject_id")[present_biomarkers].ffill()
            imputed[present_biomarkers] = imputed.groupby("subject_id")[present_biomarkers].bfill()

        for col, median_val in self.population_medians.items():
            if col in imputed.columns:
                imputed[col] = imputed[col].fillna(median_val)

        for col, mode_val in self.categorical_modes.items():
            if col in imputed.columns:
                imputed[col] = imputed[col].fillna(mode_val)

        return imputed

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)

    def save(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"medians": self.population_medians, "modes": self.categorical_modes}, file_path)

    def load(self, file_path: Path) -> "LongitudinalImputer":
        state = joblib.load(file_path)
        self.population_medians = state["medians"]
        self.categorical_modes = state["modes"]
        self.is_fitted = True
        return self


# =========================================================================
# 4. CATEGORICAL ENCODING
# =========================================================================

DEFAULT_MAPPINGS = {
    "gender": {"female": 0, "f": 0, "male": 1, "m": 1, "unknown": 0},
    "smoking_status": {"never": 0, "former": 1, "current": 2, "unknown": 0},
    "physical_activity": {"sedentary": 0, "moderate": 1, "active": 2, "unknown": 0},
}


class CategoricalEncoder:
    """Encodes categorical clinical variables with artifact persistence."""

    def __init__(self, cat_cols: Optional[List[str]] = None, mappings: Optional[Dict] = None):
        self.cat_cols = cat_cols or config.CATEGORICAL_FEATURES
        self.mappings = mappings or DEFAULT_MAPPINGS
        self.encoded_cols: List[str] = [f"{col}_encoded" for col in self.cat_cols]
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "CategoricalEncoder":
        self.encoded_cols = [f"{col}_encoded" for col in self.cat_cols]
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed = df.copy()
        for col in self.cat_cols:
            if col in transformed.columns:
                mapping = self.mappings.get(col, {})
                col_lower = transformed[col].astype(str).str.lower()
                transformed[f"{col}_encoded"] = col_lower.map(mapping).fillna(0).astype(int)
            else:
                transformed[f"{col}_encoded"] = 0
        return transformed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"mappings": self.mappings, "cat_cols": self.cat_cols}, file_path)

    def load(self, file_path: Path) -> "CategoricalEncoder":
        state = joblib.load(file_path)
        self.mappings = state["mappings"]
        self.cat_cols = state["cat_cols"]
        self.encoded_cols = [f"{col}_encoded" for col in self.cat_cols]
        self.is_fitted = True
        return self


# =========================================================================
# 5. LEAK-FREE CLINICAL NORMALIZATION
# =========================================================================

class ClinicalScaler:
    """
    StandardScaler wrapper fitted strictly on training patients to prevent leakage.
    Scaled columns are saved as `{col}_scaled`, keeping raw values intact for XAI and reports.
    """

    def __init__(self, numerical_cols: Optional[List[str]] = None):
        self.numerical_cols = numerical_cols
        self.scaler = StandardScaler()
        self.is_fitted: bool = False
        self.feature_names_in_: List[str] = []

    def fit(self, train_df: pd.DataFrame, numerical_cols: Optional[List[str]] = None) -> "ClinicalScaler":
        if numerical_cols is not None:
            self.numerical_cols = numerical_cols
        elif self.numerical_cols is None:
            detected = [c for c in config.BIOMARKERS if c in train_df.columns]
            detected += [c for c in config.CONTEXT_FEATURES if c in train_df.columns]
            self.numerical_cols = detected

        self.feature_names_in_ = [col for col in self.numerical_cols if col in train_df.columns]
        train_vals = train_df[self.feature_names_in_].fillna(0.0).values
        self.scaler.fit(train_vals)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("ClinicalScaler must be fitted before transforming data.")

        transformed = df.copy()
        vals = transformed[self.feature_names_in_].fillna(0.0).values
        scaled_vals = self.scaler.transform(vals)

        for idx, col in enumerate(self.feature_names_in_):
            transformed[f"{col}_scaled"] = scaled_vals[:, idx]

        return transformed

    def fit_transform(self, train_df: pd.DataFrame, numerical_cols: Optional[List[str]] = None) -> pd.DataFrame:
        return self.fit(train_df, numerical_cols).transform(train_df)

    def inverse_transform(self, scaled_array, feature_names: Optional[List[str]] = None):
        return self.scaler.inverse_transform(scaled_array)

    def save(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": self.scaler,
                "feature_names_in_": self.feature_names_in_,
                "numerical_cols": self.numerical_cols,
            },
            file_path,
        )

    def load(self, file_path: Path) -> "ClinicalScaler":
        data = joblib.load(file_path)
        self.scaler = data["scaler"]
        self.feature_names_in_ = data["feature_names_in_"]
        self.numerical_cols = data["numerical_cols"]
        self.is_fitted = True
        return self


# =========================================================================
# 6. PATIENT-LEVEL PARTITIONING (ZERO LEAKAGE)
# =========================================================================

def split_by_patient(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits DataFrame strictly by subject_id to ensure zero leakage across patient trajectories.
    """
    unique_patients = df["subject_id"].unique()
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=test_size, random_state=seed
    )
    train_df = df[df["subject_id"].isin(train_patients)].copy().reset_index(drop=True)
    test_df = df[df["subject_id"].isin(test_patients)].copy().reset_index(drop=True)
    return train_df, test_df


# =========================================================================
# 7. UNIFIED PREPROCESSING PIPELINE ORCHESTRATOR
# =========================================================================

class PreprocessingPipeline:
    """
    Unified Preprocessing Pipeline orchestrating cleaning, feature engineering,
    imputation, encoding, and leak-free scaling.
    """

    def __init__(self):
        self.imputer = LongitudinalImputer()
        self.encoder = CategoricalEncoder()
        self.scaler = ClinicalScaler()
        self.is_fitted: bool = False

    def fit_transform_initial(self, train_raw_df: pd.DataFrame) -> pd.DataFrame:
        print("[*] Preprocessing Pipeline: Cleaning records...")
        cleaned = clean_records(train_raw_df)

        print("[*] Preprocessing Pipeline: Engineering visit features...")
        engineered = engineer_features(cleaned)

        print("[*] Preprocessing Pipeline: Fitting & applying longitudinal imputer...")
        imputed = self.imputer.fit_transform(engineered)

        print("[*] Preprocessing Pipeline: Fitting & applying categorical encoders...")
        encoded = self.encoder.fit_transform(imputed)

        self.is_fitted = True
        return encoded

    def transform_initial(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transforming new data.")

        cleaned = clean_records(raw_df, min_visits=1)
        engineered = engineer_features(cleaned)
        imputed = self.imputer.transform(engineered)
        encoded = self.encoder.transform(imputed)
        return encoded

    def fit_and_apply_scaler(
        self,
        train_trajectory_df: pd.DataFrame,
        numerical_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        print("[*] Preprocessing Pipeline: Fitting ClinicalScaler on training trajectories...")
        return self.scaler.fit_transform(train_trajectory_df, numerical_cols=numerical_cols)

    def apply_scaler(self, trajectory_df: pd.DataFrame) -> pd.DataFrame:
        return self.scaler.transform(trajectory_df)

    def save_artifacts(self, objects_dir: Optional[Path] = None):
        if objects_dir is None:
            objects_dir = config.PREPROCESSING_OBJECTS_DIR
        objects_dir.mkdir(parents=True, exist_ok=True)
        self.imputer.save(objects_dir / "imputer.pkl")
        self.encoder.save(objects_dir / "encoders.pkl")
        self.scaler.save(objects_dir / "scaler.pkl")
        print(f"[OK] Preprocessing artifacts saved successfully to {objects_dir}")

    def load_artifacts(self, objects_dir: Optional[Path] = None):
        if objects_dir is None:
            objects_dir = config.PREPROCESSING_OBJECTS_DIR
        self.imputer.load(objects_dir / "imputer.pkl")
        self.encoder.load(objects_dir / "encoders.pkl")
        self.scaler.load(objects_dir / "scaler.pkl")
        self.is_fitted = True
        return self
