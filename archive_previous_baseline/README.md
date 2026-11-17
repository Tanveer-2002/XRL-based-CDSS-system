# Archived Previous Baseline (Historical Record)

This directory preserves the exact model checkpoint, evaluation tables, and benchmark figures from the **initial baseline Discrete CQL implementation** prior to the class-balanced sampling and conservatism hyperparameter optimization.

---

## 1. Directory Structure

```
archive_previous_baseline/
├── README.md                           # This documentation file
├── benchmark/
│   ├── BENCHMARK_REPORT.md             # Benchmark markdown report generated from previous baseline
│   ├── benchmark_results.csv           # Multi-model evaluation metrics table (previous baseline)
│   ├── confusion_matrices_all.png      # 4-model confusion matrices (previous baseline)
│   ├── data_scaling_curve.png          # 4-panel data scaling curves (previous baseline)
│   ├── data_scaling_results.csv        # Tabular data scaling metrics (previous baseline)
│   └── model_benchmark_comparison.png  # 8-panel side-by-side comparison grid (previous baseline)
└── models/
    └── rl_agent/
        └── cql_model.pt                # Historical CQL PyTorch model checkpoint (Epoch 5 checkpoint)
```

---

## 2. Configuration Differences: Previous Baseline vs. Active Champion

| Component | Archived Baseline (`archive_previous_baseline/`) | Active Champion (`models/rl_agent/cql_model.pt`) |
| :--- | :--- | :--- |
| **Sampling Strategy** | Uniform random batch sampling | **Class-balanced sampling** (`WeightedRandomSampler`, power = 0.20) |
| **Conservative Penalty ($\alpha$)** | $\alpha = 1.0$ (Data Scaling) / $\alpha = 0.5$ (Benchmark) | **$\alpha = 2.5$** (Robust OOD lower-bounding) |
| **Checkpoint Convergence** | Frozen at early validation TD-loss minimum (Epoch 5) | **Full convergence checkpoint** (Epoch 35) |
| **Target Q-Network Update** | Polyak $\tau = 0.005$ | Polyak $\tau = 0.01$ |
| **Target Double Q Calculation** | Single-step target min | Argmax policy Double-Q target estimation |

---

## 3. Benchmark Metric Comparison (Test Cohort)

| Metric | Archived Baseline | Active Champion | Impact / Clinical Interpretation |
| :--- | :---: | :---: | :--- |
| **Clinical Guideline Accuracy** | 36.76% | **47.43%** | +10.67% increase in alignment with clinical practice |
| **Balanced Accuracy** | 30.57% | **53.83%** | **+23.26% jump**; eliminates majority-class collapse |
| **Macro F1-Score** | 0.2222 | **0.4335** | Near-doubling of multi-class F1 across all 4 actions |
| **Deterioration Sensitivity** | 18.64% | **81.36%** | **> 4x improvement** in flagging worsening patients |
| **Action Adjacency Concordance** | 88.54% | **91.70%** | 91.7% of predictions are within +/- 1 intensity level |
| **False Alerts per Patient-Year** | 0.35 | **0.30** | Lowest false alert rate among all predictive models |
| **Mean Discounted Return ($V^\pi$)** | 13.78 | **14.22** | Beats Historical Clinician baseline (13.76) |

---

## 4. How to Restore or Compare

- **To inspect archived metrics**: Open `archive_previous_baseline/benchmark/benchmark_results.csv`
- **To inspect active metrics**: Open `benchmark/benchmark_results.csv`
- **To load the archived model in Python**:
  ```python
  import torch
  from rl.cql import DiscreteCQLNetwork
  checkpoint = torch.load("archive_previous_baseline/models/rl_agent/cql_model.pt", map_location="cpu")
  model = DiscreteCQLNetwork(state_dim=checkpoint["state_dim"])
  model.load_state_dict(checkpoint["model_state_dict"])
  model.eval()
  ```
