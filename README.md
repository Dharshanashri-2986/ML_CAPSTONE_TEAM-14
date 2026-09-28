# AeroShield — Turbofan Engine Predictive Maintenance (ML Capstone, Team 14)

**Course:** 23CSE301 Machine Learning — B.Tech CSE, III Year, Amrita Vishwa Vidyapeetham (2026-27)
**Tracks:** Regression · Classification · Clustering

## Team

| Member | Roll No. | Primary track owned |
|---|---|---|
| _Name_ | _Roll no._ | Regression |
| _Name_ | _Roll no._ | Classification |
| _Name_ | _Roll no._ | Clustering |

## Problem Statement

Commercial turbofan engines degrade gradually until failure. Using multivariate sensor telemetry, we:

1. **Regression** — predict the **Remaining Useful Life (RUL)** of an engine, in operating cycles, at every cycle.
2. **Classification** — flag engines in the **critical maintenance window** (`RUL <= 30` cycles → class 1, else class 0).
3. **Clustering** — discover operating / health-state groupings in the telemetry without labels (Review 2).

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

### Classification Part B & Clustering (Review 2)

_To be added._

## Repository Structure

```
├── README.md
├── requirements.txt
├── data/raw/NASA_FD004/        # raw dataset files
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb   # dataset audit, EDA, cleaning, feature engineering, split & scaling
│   ├── 02_Regression.ipynb          # 10 regression algorithms, tuning, CV, diagnostics
│   └── 03_Classification.ipynb      # classification Part A (5 algorithms)
├── src/
│   ├── preprocessing/          # loading, RUL targets, audit, features, leakage-free split/scale, get_dataset()
│   └── plotting.py             # shared plot style (colourblind palette)
├── reports/                    # result tables exported as CSV by the notebooks
├── models/                     # best-model summaries (.txt); .joblib files are generated locally
└── app/                        # GUI / deployment (bonus, Review 2)
```

All notebooks load data through `src.preprocessing.get_dataset()`, so every algorithm in a track sees exactly the same preprocessed split.

## Environment Setup

Python 3.10+ recommended.

```bash
git clone https://github.com/Dharshanashri-2986/ML_CAPSTONE_TEAM-14.git
cd ML_CAPSTONE_TEAM-14
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run

```bash
cd notebooks
jupyter notebook
```

Run the notebooks in order — **01 → 02 → 03** — with *Kernel → Restart & Run All*. The notebooks must be started from inside `notebooks/` (they add the repo root to `sys.path` to import `src`). Notebook 02 takes several minutes because of the tree ensembles, GridSearch and 5-fold CV.

Outputs: result tables are written to `reports/`, best models to `models/`.

## Use of AI Tools (Guideline 7.5)

Generative AI (Anthropic's Claude) was used for **code scaffolding only**: reconstructing the shared `src/preprocessing` module, fixing notebook import/save paths, switching cross-validation to engine-grouped folds, adding parameter-sweep code cells, and drafting this README's structure. All analysis, interpretation, feature-engineering decisions and written observations in the notebooks are the team's own.

## References

- A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", *PHM 2008*.
- NASA Prognostics Center of Excellence Data Repository — C-MAPSS dataset.
- scikit-learn documentation — https://scikit-learn.org
