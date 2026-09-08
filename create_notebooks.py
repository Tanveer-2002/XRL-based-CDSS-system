"""
Utility script to generate the 5 comprehensive Jupyter Notebooks for diabetes-cdss.
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.11",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "file_extension": ".py",
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
                "language": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 2,
    }


def md_cell(source):
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code_cell(source):
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": lines,
    }


# ==========================================
# 01_data_exploration.ipynb
# ==========================================
nb1_cells = [
    md_cell("""# Notebook 01: Exploratory Data Analysis (EDA) of Type 2 Diabetes Longitudinal Cohort
This notebook explores the healthcare records, biomarker distributions, encounter intervals, and patient demographics for the longitudinal Type 2 Diabetes cohort."""),
    code_cell("""import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config
from preprocessing.load_data import load_raw_data

# Set styling
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)"""),
    md_cell("## 1. Load Longitudinal Cohort Records"),
    code_cell("""# Load cached interim cohort or generate calibrated cohort
df = load_raw_data(source="auto", num_patients=500)
print(f"Total observations: {len(df)}")
print(f"Unique patients: {df['subject_id'].nunique()}")
df.head(10)"""),
    md_cell("## 2. Demographic and Baseline Patient Profiles"),
    code_cell("""patient_first = df.groupby("subject_id").first().reset_index()

fig, axs = plt.subplots(1, 3, figsize=(16, 4))
sns.histplot(patient_first["age"], bins=15, kde=True, ax=axs[0], color="#1a73e8")
axs[0].set_title("Age Distribution at Baseline")

sns.countplot(x="gender", data=patient_first, ax=axs[1], palette="Blues_r")
axs[1].set_title("Gender Distribution")

sns.countplot(x="smoking_status", data=patient_first, ax=axs[2], palette="Greens_r")
axs[2].set_title("Smoking Status Distribution")
plt.tight_layout()
plt.show()"""),
    md_cell("## 3. Clinical Biomarker Distributions & Guideline Targets"),
    code_cell("""fig, axs = plt.subplots(2, 3, figsize=(16, 8))
biomarkers_to_plot = ["hba1c", "glucose", "bmi", "sbp", "ldl", "creatinine"]

for idx, marker in enumerate(biomarkers_to_plot):
    ax = axs[idx // 3, idx % 3]
    sns.histplot(df[marker].dropna(), kde=True, ax=ax, color="#0d6efd")
    thresh = config.CLINICAL_THRESHOLDS.get(marker, {})
    target = thresh.get("target_max")
    if target:
        ax.axvline(target, color="red", linestyle="--", label=f"Target ({target})")
        ax.legend()
    ax.set_title(f"{marker.upper()} Distribution")

plt.tight_layout()
plt.show()"""),
    md_cell("## 4. Longitudinal Visit Intervals and Follow-Up Patterns"),
    code_cell("""visit_counts = df.groupby("subject_id")["chartdate"].count()
print("Visits per patient summary:")
print(visit_counts.describe())

plt.figure(figsize=(8, 4))
sns.countplot(x=visit_counts, palette="viridis")
plt.title("Distribution of Total Visits per Patient")
plt.xlabel("Number of Encounters")
plt.ylabel("Number of Patients")
plt.show()"""),
]

# ==========================================
# 02_preprocessing.ipynb
# ==========================================
nb2_cells = [
    md_cell("""# Notebook 02: Clinical Data Preprocessing & Leakage Prevention
This notebook walks through cleaning, temporal feature engineering, longitudinal imputation, categorical encoding, and leak-free train/test scaling."""),
    code_cell("""import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import pandas as pd
import numpy as np
import config
from preprocessing.load_data import load_raw_data
from preprocessing.cleaning import clean_records
from preprocessing.feature_engineering import engineer_features
from preprocessing.missing_values import LongitudinalImputer
from preprocessing.encoding import CategoricalEncoder
from preprocessing.normalization import ClinicalScaler
from preprocessing.pipeline import PreprocessingPipeline, split_by_patient"""),
    md_cell("## 1. Load Raw Healthcare Records"),
    code_cell("""raw_df = load_raw_data(source="auto", num_patients=200)
print(f"Raw shape: {raw_df.shape}")
print(f"Missing values before preprocessing:\n{raw_df.isna().sum()[raw_df.isna().sum() > 0]}")"""),
    md_cell("## 2. Data Cleaning & Physiological Bounds"),
    code_cell("""cleaned_df = clean_records(raw_df, min_visits=2)
print(f"Cleaned shape: {cleaned_df.shape}")
cleaned_df.head(5)"""),
    md_cell("## 3. Temporal Feature Engineering (Visit Index, dt)"),
    code_cell("""engineered_df = engineer_features(cleaned_df)
engineered_df[["subject_id", "chartdate", "visit_number", "days_since_last_visit", "cumulative_days"]].head(8)"""),
    md_cell("## 4. Patient-level Train / Test Split (Zero Cross-Patient Leakage)"),
    code_cell("""train_df, test_df = split_by_patient(engineered_df, test_size=0.2, seed=42)
print(f"Train patients: {train_df['subject_id'].nunique()}, Records: {len(train_df)}")
print(f"Test patients:  {test_df['subject_id'].nunique()}, Records: {len(test_df)}")

# Assert zero patient intersection
assert len(set(train_df['subject_id']).intersection(set(test_df['subject_id']))) == 0
print("OK Verified zero patient overlap between train and test splits!")"""),
    md_cell("## 5. Fit & Save Imputer, Encoder, and Normalization Scaler"),
    code_cell("""imputer = LongitudinalImputer().fit(train_df)
train_imputed = imputer.transform(train_df)
test_imputed = imputer.transform(test_df)

encoder = CategoricalEncoder().fit(train_imputed)
train_encoded = encoder.transform(train_imputed)
test_encoded = encoder.transform(test_imputed)

print("Imputed and encoded sample:")
train_encoded[["subject_id", "visit_number", "gender_encoded", "smoking_status_encoded", "hba1c"]].head()"""),
]

# ==========================================
# 03_trajectory_analysis.ipynb
# ==========================================
nb3_cells = [
    md_cell("""# Notebook 03: Health Trajectory Analysis Engine
Demonstrates longitudinal trajectory analysis: timeline building, delta calculation, monthly rate of change, moving averages, ADA clinical trend categorization, and overall trajectory classification."""),
    code_cell("""import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config
from preprocessing.load_data import load_raw_data
from preprocessing.pipeline import PreprocessingPipeline
from trajectory.timeline import build_patient_timeline
from trajectory.changes import compute_biomarker_changes
from trajectory.moving_average import compute_moving_averages
from trajectory.trends import classify_biomarker_trends
from trajectory.pipeline import TrajectoryPipeline"""),
    md_cell("## 1. Execute Trajectory Engine on Patient Encounters"),
    code_cell("""raw_df = load_raw_data(source="auto", num_patients=150)
prep = PreprocessingPipeline()
clean_df = prep.fit_transform_initial(raw_df)

traj_engine = TrajectoryPipeline()
traj_df = traj_engine.run(clean_df)
traj_df[["subject_id", "chartdate", "hba1c", "hba1c_change", "hba1c_rate", "hba1c_moving_avg", "hba1c_trend", "overall_trajectory_status"]].head(10)"""),
    md_cell("## 2. Trajectory Distribution Across the Cohort"),
    code_cell("""plt.figure(figsize=(8, 4))
sns.countplot(x="overall_trajectory_status", data=traj_df, palette="Set2")
plt.title("Overall Trajectory Status Distribution")
plt.xlabel("Longitudinal Progression Status")
plt.ylabel("Observation Count")
plt.show()"""),
    md_cell("## 3. Visualizing Trajectories for Different Clinical Phenotypes"),
    code_cell("""# Pick a deteriorating patient and an improving/stable patient
sample_det = traj_df[traj_df["overall_trajectory_status"] == "rapidly_deteriorating"]["subject_id"].iloc[0]
sample_sta = traj_df[traj_df["overall_trajectory_status"] == "stable"]["subject_id"].iloc[0]

fig, axs = plt.subplots(1, 2, figsize=(15, 5))

for idx, (sid, title) in enumerate([(sample_det, "Rapidly Deteriorating Patient"), (sample_sta, "Stable Patient")]):
    sub = traj_df[traj_df["subject_id"] == sid].sort_values("visit_number")
    axs[idx].plot(sub["visit_number"], sub["hba1c"], marker="o", label="HbA1c (%)", color="#dc3545" if "Det" in title else "#28a745", linewidth=2)
    axs[idx].plot(sub["visit_number"], sub["hba1c_moving_avg"], linestyle="--", label="3-Visit Moving Avg", color="#6c757d")
    axs[idx].axhline(7.0, color="orange", linestyle=":", label="ADA Target (7.0%)")
    axs[idx].set_title(f"{title} ({sid})")
    axs[idx].set_xlabel("Visit Number")
    axs[idx].set_ylabel("HbA1c (%)")
    axs[idx].legend()

plt.tight_layout()
plt.show()"""),
]

# ==========================================
# 04_rl_training.ipynb
# ==========================================
nb4_cells = [
    md_cell("""# Notebook 04: Offline Reinforcement Learning (Discrete CQL) Training
Prepares Offline RL transitions (state, action, reward, next_state, done), constructs PyTorch datasets with zero patient leakage, and trains the Conservative Q-Learning agent."""),
    code_cell("""import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config
from rl.state import StateConstructor
from rl.transitions import generate_offline_transitions
from rl.dataset import create_rl_dataloaders
from rl.model import DiscreteCQLNetwork
from rl.train import train_cql_agent"""),
    md_cell("## 1. Load Processed Trajectory Dataset"),
    code_cell("""traj_path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"
if not traj_path.exists():
    from main import run_demo_pipeline
    print("Running initial pipeline to build trajectory dataset...")
traj_df = pd.read_csv(traj_path)
print(f"Loaded {len(traj_df)} trajectory records.")
traj_df.head(5)"""),
    md_cell("## 2. Inspect Offline RL State Representation and Actions"),
    code_cell("""sc = StateConstructor()
print(f"State dimensionality: {sc.state_dim}")
print("State features:", sc.feature_names[:10], "... [total", len(sc.feature_names), "]")
print("Action space:", config.ACTION_NAMES)"""),
    md_cell("## 3. Create Offline RL DataLoaders & Train Discrete CQL Agent"),
    code_cell("""train_loader, val_loader, train_ds, val_ds = create_rl_dataloaders(
    trajectory_df=traj_df,
    state_constructor=sc,
    batch_size=64,
    val_split=0.2,
)

model, history = train_cql_agent(
    train_loader=train_loader,
    val_loader=val_loader,
    state_dim=sc.state_dim,
    epochs=25,
    lr=3e-4,
    cql_alpha=1.0,
)"""),
    md_cell("## 4. Plot Training Convergence & Loss Curves"),
    code_cell("""fig, axs = plt.subplots(1, 3, figsize=(16, 4))

axs[0].plot(history["epoch"], history["td_loss"], label="Train TD Loss", color="#0d6efd")
axs[0].plot(history["epoch"], history["val_td_loss"], label="Val TD Loss", color="#dc3545", linestyle="--")
axs[0].set_title("Bellman Temporal Difference (TD) Loss")
axs[0].set_xlabel("Epoch")
axs[0].legend()

axs[1].plot(history["epoch"], history["cql_loss"], label="CQL Penalty", color="#ffc107")
axs[1].set_title("Conservative Q-Learning (CQL) Penalty")
axs[1].set_xlabel("Epoch")
axs[1].legend()

axs[2].plot(history["epoch"], history["avg_q"], label="Avg Q-value", color="#198754")
axs[2].set_title("Average Policy Q-Value")
axs[2].set_xlabel("Epoch")
axs[2].legend()

plt.tight_layout()
plt.show()"""),
]

# ==========================================
# 05_evaluation.ipynb
# ==========================================
nb5_cells = [
    md_cell("""# Notebook 05: Policy Evaluation, Explainable AI (SHAP), and Cumulative Reports
Evaluates the learned CQL policy against clinical baselines, computes SHAP feature attributions, generates natural language explanations, and builds interactive cumulative health reports."""),
    code_cell("""import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import torch
import pandas as pd
import numpy as np
import config
from rl.state import StateConstructor
from rl.model import DiscreteCQLNetwork
from rl.evaluate import evaluate_policy
from xai.shap_explainer import RLShapExplainer
from xai.explanation_generator import generate_clinical_explanation
from report.report_generator import generate_cumulative_report, render_markdown_report, render_html_dashboard"""),
    md_cell("## 1. Load Trained CQL Model and Trajectory Data"),
    code_cell("""traj_df = pd.read_csv(config.PROCESSED_DATA_DIR / "trajectory_data.csv")
sc = StateConstructor()

model_path = config.RL_AGENT_DIR / "cql_model.pt"
checkpoint = torch.load(model_path, map_location="cpu")
model = DiscreteCQLNetwork(state_dim=sc.state_dim)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("OK CQL Model loaded successfully.")"""),
    md_cell("## 2. Policy Benchmarking: CQL vs Clinical Rule vs Behavior vs Random"),
    code_cell("""results = evaluate_policy(model, traj_df, state_constructor=sc)"""),
    md_cell("## 3. Explainable AI: SHAP Attributions for a Specific Patient"),
    code_cell("""# Select a patient with a deteriorating trajectory
det_sids = traj_df[traj_df["overall_trajectory_status"] == "deteriorating"]["subject_id"].unique()
target_sid = det_sids[0]
patient_records = traj_df[traj_df["subject_id"] == target_sid].sort_values("visit_number").reset_index(drop=True)
latest_obs = patient_records.iloc[-1]

state_vec = sc.extract_state_vector(latest_obs)
action_idx = model.select_action(torch.tensor(state_vec, dtype=torch.float32))

bg_matrix = sc.extract_state_matrix(traj_df.sample(40))
explainer = RLShapExplainer(model=model, background_data=bg_matrix, state_constructor=sc)
shap_res = explainer.explain_action(state_vec, action=action_idx)

explanation = generate_clinical_explanation(
    action=action_idx,
    obs_data=latest_obs,
    attributions=shap_res["attributions"]
)

print("RECOMMENDED ACTION:", explanation["action_name"])
print("SUMMARY RATIONALE:\n", explanation["summary_statement"])
print("\nTOP KEY DRIVING FACTORS:")
for kf in explanation["key_factors"]:
    print(f" - {kf['clinical_description']} (SHAP: {kf['shap_attribution']:+.4f})")"""),
    md_cell("## 4. Generate Cumulative Health Report"),
    code_cell("""from rl.actions import get_action_metadata
meta = get_action_metadata(action_idx)

rl_rec = {
    "action_code": action_idx,
    "action_name": meta["name"],
    "recommended_interval": meta["interval"],
    "urgency": meta["urgency"],
    "q_values": {config.ACTION_NAMES[i]: round(float(v), 3) for i, v in enumerate(model.get_q_values(torch.tensor(state_vec)).numpy())},
    "description": meta["description"],
}

report = generate_cumulative_report(
    patient_history_df=patient_records,
    rl_recommendation_dict=rl_rec,
    xai_explanation_dict=explanation,
)

md_content = render_markdown_report(report)
print(md_content[:1200] + "\\n... [truncated] ...")"""),
]

# Write out all notebooks
notebooks = {
    "01_data_exploration.ipynb": nb1_cells,
    "02_preprocessing.ipynb": nb2_cells,
    "03_trajectory_analysis.ipynb": nb3_cells,
    "04_rl_training.ipynb": nb4_cells,
    "05_evaluation.ipynb": nb5_cells,
}

for nb_name, cells in notebooks.items():
    nb_path = NOTEBOOKS_DIR / nb_name
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(make_notebook(cells), f, indent=2)
    print(f"[OK] Created notebook: {nb_path}")
