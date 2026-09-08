"""
Central Configuration for Type 2 Diabetes Clinical Decision Support System (CDSS)
"""

import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREPROCESSING_OBJECTS_DIR = PROCESSED_DATA_DIR / "preprocessing_objects"

# Path to MIMIC-IV hosp folder (auto-detects standalone 'hosp' or 'mimic-iv-3.1/hosp')
_MIMIC_CANDIDATES = [
    Path(os.environ.get("MIMIC_IV_PATH", "")),
    PROJECT_ROOT / "mimic-iv-3.1" / "hosp",
    PROJECT_ROOT / "hosp",
    RAW_DATA_DIR / "mimic-iv-3.1" / "hosp",
    RAW_DATA_DIR / "hosp",
    PROJECT_ROOT / "mimic-iv" / "hosp",
    RAW_DATA_DIR / "mimic-iv" / "hosp",
]
MIMIC_IV_ROOT = next((p for p in _MIMIC_CANDIDATES if p.exists() and (p / "diagnoses_icd.csv.gz").exists()), PROJECT_ROOT / "mimic-iv-3.1" / "hosp")

MODELS_DIR = BASE_DIR / "models"
RL_AGENT_DIR = MODELS_DIR / "rl_agent"
SCALERS_DIR = MODELS_DIR / "scalers"
ENCODERS_DIR = MODELS_DIR / "encoders"

REPORTS_DIR = BASE_DIR / "reports"
FEEDBACK_DIR = DATA_DIR / "feedback"

# Ensure essential directories exist
for directory in [
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    PREPROCESSING_OBJECTS_DIR,
    RL_AGENT_DIR,
    SCALERS_DIR,
    ENCODERS_DIR,
    REPORTS_DIR,
    FEEDBACK_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Clinical Biomarkers
BIOMARKERS = [
    "hba1c",
    "glucose",
    "bmi",
    "sbp",
    "dbp",
    "cholesterol",
    "ldl",
    "hdl",
    "triglycerides",
    "creatinine",
]

# Demographic and Categorical Features
CATEGORICAL_FEATURES = [
    "gender",
    "smoking_status",
    "physical_activity",
]

# Context / Metadata Features
CONTEXT_FEATURES = [
    "age",
    "visit_number",
    "days_since_last_visit",
]

# Derived Trajectory Metrics calculated for each biomarker
TRAJECTORY_METRIC_TYPES = [
    "change",       # delta = x_t - x_{t-1}
    "rate",         # rate = delta / dt (per 30 days)
    "moving_avg",   # rolling 3-observation average
    "trend_num",    # numerical trend: 1 (increasing), 0 (stable), -1 (decreasing)
]

# Clinical Significance Thresholds for Trajectory Changes (ADA / Clinical Guidelines)
# Absolute change >= threshold is considered clinically significant
CLINICAL_THRESHOLDS = {
    "hba1c": {
        "sig_change": 0.3,       # % change
        "target_max": 7.0,       # %
        "poor_control": 8.0,     # %
        "critical": 9.5,         # %
        "unit": "%",
    },
    "glucose": {
        "sig_change": 15.0,      # mg/dL
        "target_max": 130.0,     # fasting mg/dL
        "poor_control": 180.0,   # mg/dL
        "critical": 250.0,       # mg/dL
        "unit": "mg/dL",
    },
    "bmi": {
        "sig_change": 1.0,       # kg/m^2
        "target_max": 25.0,      # kg/m^2
        "obese": 30.0,           # kg/m^2
        "severe_obese": 35.0,    # kg/m^2
        "unit": "kg/m²",
    },
    "sbp": {
        "sig_change": 10.0,      # mmHg
        "target_max": 130.0,     # mmHg
        "stage2": 140.0,         # mmHg
        "crisis": 180.0,         # mmHg
        "unit": "mmHg",
    },
    "dbp": {
        "sig_change": 5.0,       # mmHg
        "target_max": 80.0,      # mmHg
        "stage2": 90.0,          # mmHg
        "crisis": 120.0,         # mmHg
        "unit": "mmHg",
    },
    "cholesterol": {
        "sig_change": 20.0,      # mg/dL
        "target_max": 200.0,     # mg/dL
        "high": 240.0,           # mg/dL
        "unit": "mg/dL",
    },
    "ldl": {
        "sig_change": 15.0,      # mg/dL
        "target_max": 100.0,     # mg/dL
        "high": 160.0,           # mg/dL
        "unit": "mg/dL",
    },
    "hdl": {
        "sig_change": 5.0,       # mg/dL (lower is worse)
        "target_min": 40.0,      # mg/dL (men: 40, women: 50)
        "optimal": 60.0,         # mg/dL
        "unit": "mg/dL",
    },
    "triglycerides": {
        "sig_change": 30.0,      # mg/dL
        "target_max": 150.0,     # mg/dL
        "high": 200.0,           # mg/dL
        "critical": 500.0,       # mg/dL
        "unit": "mg/dL",
    },
    "creatinine": {
        "sig_change": 0.2,       # mg/dL
        "target_max": 1.2,       # mg/dL
        "elevated": 1.5,         # mg/dL
        "critical": 2.5,         # mg/dL
        "unit": "mg/dL",
    },
}

# Physiological Sanity Bounds (Filtering invalid sensor/measurement entries)
PHYSIOLOGICAL_BOUNDS = {
    "hba1c": (3.5, 20.0),
    "glucose": (30.0, 800.0),
    "bmi": (12.0, 75.0),
    "sbp": (60.0, 260.0),
    "dbp": (40.0, 160.0),
    "cholesterol": (50.0, 600.0),
    "ldl": (20.0, 400.0),
    "hdl": (10.0, 150.0),
    "triglycerides": (30.0, 1500.0),
    "creatinine": (0.2, 18.0),
    "age": (18.0, 105.0),
}

# Offline RL Action Definitions
ACTION_NAMES = {
    0: "ROUTINE_MONITORING",
    1: "INCREASED_MONITORING",
    2: "EARLY_RISK_ALERT",
    3: "CLINICAL_EVALUATION",
}

ACTION_DESCRIPTIONS = {
    0: "Standard routine monitoring (e.g., standard 6-month checkup interval)",
    1: "Increased monitoring frequency (e.g., close 1-3 month checkup with repeat HbA1c/BP)",
    2: "Early-risk alert (clinician flag for deteriorating trajectory and escalating metabolic risk)",
    3: "Clinical evaluation (urgent comprehensive diagnostic assessment and specialist referral)",
}

NUM_ACTIONS = len(ACTION_NAMES)

# Offline Reinforcement Learning Hyperparameters (Discrete CQL)
RL_CONFIG = {
    "gamma": 0.99,
    "cql_alpha": 1.0,               # Temperature/scale for conservative penalty logsumexp
    "lr": 3e-4,
    "batch_size": 64,
    "hidden_dim": 128,
    "num_hidden_layers": 2,
    "tau": 0.005,                   # Soft target network update rate
    "target_update_interval": 1,
    "num_epochs": 40,
    "device": "cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu",
    "val_split": 0.2,
    "seed": 42,
}

# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "title": "Type 2 Diabetes CDSS API",
    "version": "1.0.0",
}
