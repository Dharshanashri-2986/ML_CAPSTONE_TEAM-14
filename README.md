# AeroShield — Turbofan Engine Predictive Maintenance (ML Capstone, Team 14)

**Course:** 23CSE301 Machine Learning — B.Tech CSE, III Year, Amrita Vishwa Vidyapeetham (2026-27)
**Scope:** Review 1 — EDA & Preprocessing · Regression · Classification Part A

## Team

| Member | Roll No. | Primary track owned |
|---|---|---|
| _Name_ | _Roll no._ | Regression |
| _Name_ | _Roll no._ | Classification |
| _Name_ | _Roll no._ | EDA & Preprocessing |

## Problem Statement

Commercial turbofan engines degrade gradually until failure. Using multivariate sensor telemetry, we:

1. **Regression** — predict the **Remaining Useful Life (RUL)** of an engine, in operating cycles, at every cycle.
2. **Classification** — flag engines in the **critical maintenance window** (`RUL <= 30` cycles → class 1, else class 0).

## Dataset

**NASA C-MAPSS Turbofan Engine Degradation Simulation — subset FD004** (Saxena et al., 2008, NASA Prognostics Center of Excellence). Files are in `data/raw/NASA_FD004/`.

| File | Contents |
|---|---|
| `train_FD004.txt` | 249 engines run to failure — 61,249 rows |
| `test_FD004.txt` | 248 engines, trajectories stopped before failure — 41,214 rows |
| `RUL_FD004.txt` | True RUL at the last observed cycle of each test engine |

Each row = one engine cycle, 26 space-separated columns: `unit_id`, `cycle`, 3 operational settings (altitude, Mach number, throttle resolver angle) and 21 sensor channels (temperatures, pressures, spool speeds, bleed flows). FD004 is the hardest C-MAPSS subset: **6 operating conditions and 2 fault modes** (HPC and fan degradation). No missing values and no duplicate rows.

**Target construction:** train `RUL = last_cycle − cycle`; test `RUL = true_RUL_at_end + (last_cycle − cycle)`.

## Pipeline

| Step | What we do |
|---|---|
| Audit & EDA | Shapes, dtypes, missing values, duplicates, distributions of every feature, correlation heatmap, feature–target scatter plots |
| Outliers | IQR audit on all 24 channels; retained (they correspond to real flight regimes / degradation) |
| Feature engineering | Per-engine 5-cycle rolling mean and cycle-to-cycle change for `sensor_2`, `sensor_11`, `sensor_14` → 30 features |
| Split | Engine-level (grouped by `unit_id`) 80:20 train/validation split, `random_state=42` — an engine's trajectory never spans both sets. NASA test set kept fully held out |
| Scaling | `StandardScaler` fitted on the training engines only, then applied to validation and test |
| Cross-validation | `GroupKFold` by engine for CV and GridSearch |

## Results Summary

All models are evaluated on the same held-out NASA test set (248 engines, 41,214 cycles).

### Regression — RUL (Review 1)

| Rank | Model | R² | RMSE (cycles) | MAE (cycles) |
|---|---|---|---|---|
| 1 | Gradient Boosting | 0.3658 | 73.40 | 53.94 |
| 2 | Random Forest | 0.3561 | 73.96 | 54.37 |
| 3 | Linear Regression | 0.3063 | 76.76 | 56.37 |
| 4 | Ridge | 0.2892 | 77.70 | 57.54 |
| 5 | Decision Tree | 0.2878 | 77.78 | 57.54 |
| 6 | Lasso | 0.2777 | 78.33 | 58.02 |
| 7 | KNN Regressor | 0.2432 | 80.18 | 59.99 |
| 8 | Polynomial (deg-2) | -0.0106 | 92.66 | 69.27 |
| 9 | ElasticNet | -0.0718 | 95.42 | 70.95 |
| 10 | SVR (RBF) | -0.1261 | 97.81 | 71.96 |

5-fold GroupKFold CV R² on the 200 training engines: **Gradient Boosting 0.5842 ± 0.0433**, **Random Forest 0.5807 ± 0.0458** (`notebooks/02_Regression.ipynb`, section C3).

### Classification Part A — Critical-risk flag (Review 1)

| Rank | Model | Accuracy | Precision (w) | Recall (w) | F1 (w) | ROC-AUC |
|---|---|---|---|---|---|---|
| 1 | Logistic Regression | 0.9835 | 0.9813 | 0.9835 | 0.9820 | 0.9857 |
| 2 | Decision Tree | 0.9827 | 0.9806 | 0.9827 | 0.9814 | 0.8710 |
| 3 | K-Nearest Neighbors | 0.9795 | 0.9743 | 0.9795 | 0.9758 | 0.9303 |
| 4 | SVC (RBF) | 0.9782 | 0.9717 | 0.9782 | 0.9738 | 0.9565 |
| 5 | Gaussian NB | 0.9404 | 0.9608 | 0.9404 | 0.9503 | 0.6009 |

(w) = weighted average. The test set is heavily imbalanced (2.1% critical-risk cycles), so also see the per-class recall in `notebooks/03_Classification.ipynb`.

## Review 1 Coverage Map

| Rubric | Where it is |
|---|---|
| A1 Dataset loading & audit | `01_EDA_Preprocessing.ipynb` → §2 |
| A2 EDA visualisations (every feature's distribution, heatmap, target, scatter plots) | `01_EDA_Preprocessing.ipynb` → §3–7 |
| A3 Insight commentary | Observation cells under each plot |
| B1 Data cleaning (missing, duplicates, outliers) | `01_EDA_Preprocessing.ipynb` → §2 and §9 |
| B2 Encoding, scaling & splitting | `01_EDA_Preprocessing.ipynb` → §11 (no categorical columns, so no encoding needed) |
| B3 Feature engineering | `01_EDA_Preprocessing.ipynb` → §10 |
| C1 All 10 regression algorithms | `02_Regression.ipynb` → C1 (+ C1b per-algorithm checks) |
| C2 Comparative table ranked by R² | `02_Regression.ipynb` → C1 summary |
| C3 Hyperparameter tuning (≥ 2 models) | `02_Regression.ipynb` → C2 |
| Mandatory 5-fold CV of top-2 models | `02_Regression.ipynb` → C3 |
| C4 Residual, predicted-vs-actual, feature importance | `02_Regression.ipynb` → C4 |
| D1 All 5 Part-A classifiers | `03_Classification.ipynb` |
| D2 Accuracy, weighted F1, confusion matrices, comparison table | `03_Classification.ipynb` → D1 summary & D2 |

## Repository Structure

Follows guideline §8.

```
├── README.md                 # this file
├── requirements.txt          # Python dependencies with versions
├── data/
│   └── raw/NASA_FD004/       # raw dataset files (train / test / RUL)
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb  # audit, EDA, cleaning, feature engineering, leakage-free split & scaling
│   ├── 02_Regression.ipynb         # 10 regression algorithms, comparison, tuning, 5-fold CV, diagnostics
│   └── 03_Classification.ipynb     # 5 Part-A classifiers, comparison, confusion matrices, tuning, CV, ROC
├── models/                   # best-model summaries (.txt); .joblib files are written when the notebooks run
├── app/                      # reserved for GUI code (not used in Review 1)
├── src/                      # shared, commented Python code used by every notebook
│   ├── preprocessing/        #   loading, RUL target, audit, features, leakage-free split/scale, get_dataset()
│   └── plotting.py           #   shared colourblind-friendly plot style
└── reports/                  # result tables exported as CSV by the notebooks
```

Every notebook loads its data through `src.preprocessing.get_dataset()`, so all algorithms see exactly the same split. `01_EDA_Preprocessing.ipynb` builds the same arrays step by step and explains each preprocessing decision.

## Environment Setup

Python 3.10+ (tested on 3.12).

```bash
git clone https://github.com/Dharshanashri-2986/ML_CAPSTONE_TEAM-14.git
cd ML_CAPSTONE_TEAM-14
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run

```bash
jupyter notebook
```

Open the notebooks in `notebooks/` in order (01 → 02 → 03) and choose **Kernel → Restart & Run All** for each. Each notebook is self-contained and locate the repository root automatically, so they work whichever folder Jupyter was started from. Seeds are fixed (`random_state=42`), so results are reproducible.

- `01_EDA_Preprocessing.ipynb` takes under a minute.
- `02_Regression.ipynb` takes several minutes (tree ensembles, GridSearch, 5-fold CV).
- `03_Classification.ipynb` takes a couple of minutes.

Outputs: result tables are written to `reports/`, best models and summaries to `models/`.

## Use of AI Tools (Guideline 7.5)

Generative AI (Anthropic's Claude) was used for **code scaffolding only**:
- reconstructing the shared `src/preprocessing` module;
- restructuring the repository to the guideline layout;
- robust path handling, engine-grouped cross-validation, and warning clean-up;
- adding the per-algorithm parameter-sweep and data-preparation code cells, and code comments;
- drafting this README's structure.

All analysis, interpretation, feature-engineering decisions and written observations in the notebooks are the team's own.

## References

- A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", *PHM 2008*.
- NASA Prognostics Center of Excellence Data Repository — C-MAPSS dataset.
- scikit-learn documentation — https://scikit-learn.org
