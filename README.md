# AeroShield — Turbofan Engine Predictive Maintenance

**Machine Learning Capstone · Team 14 · 23CSE301**  
B.Tech Computer Science and Engineering · Amrita Vishwa Vidyapeetham · Academic Year 2026–27


> **Try the application:** [mlcapstoneteam-14.streamlit.app](https://mlcapstoneteam-14.streamlit.app/)  
> **Source code:** [ML_CAPSTONE_TEAM-14 on GitHub](https://github.com/Dharshanashri-2986/ML_CAPSTONE_TEAM-14)

## Table of Contents

- [Project Overview](#project-overview)
- [Project Goals](#project-goals)
- [Dataset](#dataset)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Project Tracks](#project-tracks)
- [Team](#team)
- [Repository Structure](#repository-structure)
- [Technology Stack](#technology-stack)
- [Run Locally](#run-locally)
- [Results and Evaluation](#results-and-evaluation)
- [Limitations](#limitations)
- [References](#references)

## Project Overview

AeroShield is a machine-learning project focused on **turbofan engine health and predictive maintenance**. Aircraft engines produce multivariate sensor readings as they operate and gradually degrade. Analyzing these readings can help estimate remaining useful life and identify operation close to a maintenance threshold.

The project explores three machine-learning tracks:

1. **Regression** — estimate Remaining Useful Life (RUL), measured in operating cycles.
2. **Classification** — classify engine-cycle observations into maintenance-risk categories.
3. **Clustering** — explore groups or patterns in the prepared data using unsupervised learning.

The repository includes data exploration and preprocessing notebooks, model experiments, evaluation outputs, and a Streamlit-based demonstration interface.

## Project Goals

- Explore sensor telemetry and understand relationships between features and engine degradation.
- Prepare data for machine learning with consistent preprocessing and feature engineering.
- Implement and compare multiple regression and classification algorithms.
- Investigate clustering as an unsupervised analysis approach.
- Evaluate models using suitable metrics rather than relying on a single score.
- Present project outputs through an accessible web application.

## Dataset

The regression and predictive-maintenance workflow uses the **NASA C-MAPSS Turbofan Engine Degradation Simulation dataset, subset FD004**.

The FD004 files are located under `data/raw/NASA_FD004/`.

| File | Description |
|---|---|
| `train_FD004.txt` | Engine run-to-failure trajectories for training |
| `test_FD004.txt` | Engine trajectories that stop before failure |
| `RUL_FD004.txt` | Reference remaining useful life for each test engine at its final observed cycle |

Each row represents an engine at a particular operating cycle and contains engine/cycle identifiers, operational settings, and sensor measurements.

### Target construction

- **Training trajectories:** RUL is derived from the final observed cycle for each engine minus the current cycle.
- **Test trajectories:** the reference RUL at the final observed cycle is combined with the number of cycles remaining to that observation.

The test set is kept separate from training when evaluating the RUL prediction workflow.

> **Note:** Check the individual notebook and application descriptions for the exact data and target used by each classification and clustering experiment. Different tracks may use different target definitions or prepared data.

## Machine Learning Workflow

The project follows a reproducible workflow:

1. **Load and audit data** — inspect dimensions, data types, missing values, duplicates, and feature ranges.
2. **Exploratory data analysis (EDA)** — examine distributions, correlations, and relationships between sensor values and RUL.
3. **Preprocess and engineer features** — prepare inputs for modeling and derive additional sensor features where appropriate.
4. **Split data carefully** — keep related observations from the same engine together when splitting or validating engine-cycle data.
5. **Train candidate models** — compare several algorithms within each project track.
6. **Evaluate and compare** — use regression, classification, and clustering metrics appropriate to each task.
7. **Present findings** — use tables and visualizations in the notebooks and the Streamlit application.

## Project Tracks

### 1. Regression

Regression models estimate the remaining useful life of an engine in operating cycles. Experiments include linear, regularized, instance-based, support-vector, and tree-based approaches.

**Evaluation metrics:** R², root mean squared error (RMSE), and mean absolute error (MAE).

The existing regression notebook reports Gradient Boosting as the strongest model in its reported held-out test comparison, with **R² = 0.3658, RMSE = 73.40 cycles, and MAE = 53.94 cycles**. Refer to the notebook and generated reports for the full comparison and evaluation context.

### 2. Classification

Classification experiments investigate how engine-cycle observations can be assigned to maintenance-risk classes. The repository covers the classification work completed across the team's assigned portions.

**Evaluation metrics may include:** accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC, depending on the experiment.

Because risk datasets can be imbalanced, per-class performance and confusion matrices should be considered alongside overall accuracy.

### 3. Clustering

Clustering explores structure in data without using a target label during model fitting. It can help identify groups of observations with similar characteristics.

Review the clustering notebook and its outputs for the algorithms, feature preparation, selection criteria, visualizations, and metrics used in the current implementation. Clusters should be interpreted as patterns in the analyzed data, not as confirmed real-world diagnoses or failure causes.

## Team

| Member | Roll No. | Primary track owned |
|---|---|---|
| **Vuribindi Rohan Reddy** | `CB.SC.U4CSE24359` | EDA, preprocessing, first 5 regression models, and last 5 classification models |
| **S B Priyadharshika** | `CB.SC.U4CSE24344` | Last 5 regression models and clustering |
| **Dharshanashri S** | `CB.SC.U4CSE24314` | Classification and user interface (UI) |

Track ownership indicates each member's primary area of responsibility; the project is a collaborative team effort.

## Live Demonstration

Explore the deployed application here:

### [🚀 Launch AeroShield — Streamlit Demo](https://mlcapstoneteam-14.streamlit.app/)

The application provides a browser-based way to explore the project. The available screens and interactions depend on the currently deployed version.

## Repository Structure

```text
ML_CAPSTONE_TEAM-14/
├── app/                      # Streamlit application
├── data/
│   └── raw/
│       └── NASA_FD004/       # C-MAPSS FD004 source files
├── models/                   # Model artifacts and/or model summaries
├── notebooks/                # EDA, preprocessing, regression,
│                             # classification, and clustering experiments
├── reports/                  # Exported tables and experiment outputs
├── src/                      # Shared preprocessing and plotting code
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

Some generated model files or reports may appear only after running the relevant notebooks. Consult the current repository contents for exact filenames.

## Technology Stack

- **Language:** Python
- **Data processing:** NumPy, pandas
- **Machine learning:** scikit-learn, SciPy, joblib
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Interactive application:** Streamlit
- **Development and experiments:** Jupyter Notebook
- **Version control:** Git and GitHub

The dependency list is maintained in [`requirements.txt`](requirements.txt).

## Run Locally

### Prerequisites

- Python 3.10 or newer is recommended.
- Git
- A virtual environment is recommended.

### 1. Clone the repository

```bash
git clone https://github.com/Dharshanashri-2986/ML_CAPSTONE_TEAM-14.git
cd ML_CAPSTONE_TEAM-14
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit application

First, inspect `app/` and locate the application's Python entry-point file. Then run Streamlit with that path. For example, if the entry point is `app.py` inside `app/`:

```bash
streamlit run app/app.py
```

If the entry-point filename differs, substitute the actual filename. The app's model artifacts and data files must be present at the paths expected by the application.

### 5. Run the notebooks

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Open the notebooks in `notebooks/` and run them in the order indicated by their filenames or internal instructions. Run each notebook from top to bottom. Some experiments, model tuning, or cross-validation runs may take several minutes.

## Results and Evaluation

Use the notebook outputs and files in `reports/` as the source of truth for the current experimental results. When comparing models:

- Use the same evaluation split when a fair comparison is intended.
- Keep training and test data separate.
- For engine-cycle data, avoid placing cycles from the same engine in both training and validation when that could lead to leakage.
- Consider regression errors as well as R².
- For classification, examine precision, recall, F1-score, and the confusion matrix, especially when classes are imbalanced.
- For clustering, consider the chosen internal metrics and whether the resulting groups are interpretable.

Reported metrics are experimental results on the specified data and split; they do not guarantee equivalent performance in operational aircraft maintenance.

## Limitations

- This is an academic machine-learning project and is not a certified aircraft maintenance or safety system.
- Results depend on the dataset, feature engineering, train/test strategy, and selected evaluation metrics.
- A predicted RUL or risk class should not be treated as a real-world maintenance decision without domain validation and extensive testing.
- Clustering reveals statistical structure and does not by itself establish causal failure modes.

## References

1. A. Saxena, K. Goebel, D. Simon, and N. Eklund, “Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation,” *International Conference on Prognostics and Health Management (PHM)*, 2008.
2. NASA Prognostics Center of Excellence, C-MAPSS Turbofan Engine Degradation Simulation dataset.
3. [scikit-learn documentation](https://scikit-learn.org/stable/)
4. [Streamlit documentation](https://docs.streamlit.io/)

---

**Course:** 23CSE301 — Machine Learning  
**Project:** AeroShield · ML Capstone Team 14  
**Demo:** [mlcapstoneteam-14.streamlit.app](https://mlcapstoneteam-14.streamlit.app/)
