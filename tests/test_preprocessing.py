"""
Unit Tests for Preprocessing Module.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from preprocessing.cleaning import clean_records
from preprocessing.encoding import CategoricalEncoder
from preprocessing.feature_engineering import engineer_features
from preprocessing.missing_values import LongitudinalImputer
from preprocessing.normalization import ClinicalScaler
from preprocessing.pipeline import split_by_patient


@pytest.fixture
def sample_raw_data():
    """Generates small dummy DataFrame with typical raw data anomalies."""
    data = [
        {"subject_id": "P1", "chartdate": "2020-01-10", "hba1c": 6.5, "glucose": 115.0, "sbp": 120.0, "gender": "Female", "smoking_status": "never"},
        # Same day duplicate visit
        {"subject_id": "P1", "chartdate": "2020-01-10", "hba1c": 6.7, "glucose": 125.0, "sbp": 124.0, "gender": "Female", "smoking_status": "never"},
        # Subsequent visit
        {"subject_id": "P1", "chartdate": "2020-05-15", "hba1c": 7.2, "glucose": 140.0, "sbp": 132.0, "gender": "Female", "smoking_status": "never"},
        # Third visit with missing glucose
        {"subject_id": "P1", "chartdate": "2020-09-20", "hba1c": 7.8, "glucose": np.nan, "sbp": 140.0, "gender": "Female", "smoking_status": "never"},
        
        # Second patient P2
        {"subject_id": "P2", "chartdate": "2019-03-01", "hba1c": 8.5, "glucose": 180.0, "sbp": 145.0, "gender": "Male", "smoking_status": "former"},
        {"subject_id": "P2", "chartdate": "2019-07-01", "hba1c": 8.0, "glucose": 160.0, "sbp": 138.0, "gender": "Male", "smoking_status": "former"},
        
        # Third patient P3 with only 1 visit (should be filtered out by min_visits=2)
        {"subject_id": "P3", "chartdate": "2021-01-01", "hba1c": 6.0, "glucose": 100.0, "sbp": 115.0, "gender": "Female", "smoking_status": "never"},
    ]
    return pd.DataFrame(data)


def test_clean_records(sample_raw_data):
    cleaned = clean_records(sample_raw_data, min_visits=2)

    # Verify P3 dropped (fewer than 2 visits)
    assert "P3" not in cleaned["subject_id"].values
    # Verify same day duplicate for P1 collapsed into 1 row
    p1_rows = cleaned[cleaned["subject_id"] == "P1"]
    assert len(p1_rows) == 3
    # Check that hba1c for first visit is averaged: (6.5 + 6.7) / 2 = 6.6
    assert round(p1_rows.iloc[0]["hba1c"], 2) == 6.6
    # Check chronological ordering
    assert p1_rows.iloc[0]["chartdate"] < p1_rows.iloc[1]["chartdate"]


def test_feature_engineering(sample_raw_data):
    cleaned = clean_records(sample_raw_data, min_visits=2)
    engineered = engineer_features(cleaned)

    # Verify visit number is 1, 2, 3
    p1_records = engineered[engineered["subject_id"] == "P1"].sort_values("visit_number")
    assert list(p1_records["visit_number"]) == [1, 2, 3]

    # Verify days_since_last_visit
    assert p1_records.iloc[0]["days_since_last_visit"] == 0.0
    assert p1_records.iloc[1]["days_since_last_visit"] > 0.0
    assert p1_records.iloc[2]["cumulative_days"] > p1_records.iloc[1]["cumulative_days"]


def test_longitudinal_imputer(sample_raw_data):
    cleaned = clean_records(sample_raw_data, min_visits=2)
    engineered = engineer_features(cleaned)

    imputer = LongitudinalImputer()
    imputed = imputer.fit_transform(engineered)

    # Ensure no NaN remains in biomarker columns
    assert not imputed["glucose"].isna().any()
    # Check that P1 visit 3 (which was NaN) was forward-filled with visit 2's glucose (140.0)
    p1_v3 = imputed[(imputed["subject_id"] == "P1") & (imputed["visit_number"] == 3)]
    assert float(p1_v3["glucose"].iloc[0]) == 140.0


def test_categorical_encoding(sample_raw_data):
    cleaned = clean_records(sample_raw_data, min_visits=2)
    encoder = CategoricalEncoder()
    encoded = encoder.fit_transform(cleaned)

    assert "gender_encoded" in encoded.columns
    assert "smoking_status_encoded" in encoded.columns
    # Female: 0, Male: 1
    assert encoded[encoded["subject_id"] == "P1"]["gender_encoded"].iloc[0] == 0
    assert encoded[encoded["subject_id"] == "P2"]["gender_encoded"].iloc[0] == 1


def test_leak_free_normalization(sample_raw_data):
    cleaned = clean_records(sample_raw_data, min_visits=2)
    train_df, val_df = split_by_patient(cleaned, test_size=0.5, seed=42)

    # Verify patient separation
    train_sids = set(train_df["subject_id"].unique())
    val_sids = set(val_df["subject_id"].unique())
    assert len(train_sids.intersection(val_sids)) == 0

    scaler = ClinicalScaler(numerical_cols=["hba1c", "glucose"])
    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)

    # Scaled columns exist
    assert "hba1c_scaled" in train_scaled.columns
    assert "hba1c_scaled" in val_scaled.columns

    # Original unscaled columns are intact for clinical interpretability
    assert "hba1c" in train_scaled.columns
    assert train_scaled["hba1c"].iloc[0] == train_df["hba1c"].iloc[0]
