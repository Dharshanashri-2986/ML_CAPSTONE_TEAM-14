from pathlib import Path
import sys
import re
import warnings
import base64
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import (
    get_dataset,
    RISK_THRESHOLD,
    load_raw,
    compute_rul,
    engineer_features,
)

REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
SEED = 42


@st.cache_resource
def load_test_history():
    train_raw, test_raw, rul_raw = load_raw()
    return compute_rul(train_raw, test_raw, rul_raw)[1]


# =========================================================
# PAGE CONFIGURATION AND UI
# =========================================================

st.set_page_config(
    page_title="AeroShield | Engine Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, #dcecf7 0%, transparent 34%),
            linear-gradient(180deg, #f5f8fc, #edf3f8);
        color: #172b42;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101d32, #183958);
    }

    [data-testid="stSidebar"] * {
        color: #edf6ff;
    }

    /* Improve navigation contrast against the dark sidebar. */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] [data-testid="stRadio"] p,
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: #EAF4FF !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: #466078 !important;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #C4D6E8 !important;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        color: #143858;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dce7f0;
        padding: 16px;
        border-radius: 15px;
        box-shadow: 0 5px 18px #1839580c;
    }

    [data-testid="stMetricLabel"] {
        color: #526980 !important;
    }

    [data-testid="stMetricValue"] {
        color: #15567a !important;
    }

    .hero {
        padding: 27px 30px;
        border-radius: 20px;
        color: white;
        background: linear-gradient(115deg, #10233d, #175579 65%, #208b96);
        margin-bottom: 20px;
        box-shadow: 0 12px 28px #10233d22;
    }

    .hero h1 {
        color: white !important;
        margin: 0 0 7px;
    }

    .hero p {
        color: #F0F8FF !important;
        opacity: 1 !important;
        margin: 0;
        font-size: 1.03rem;
        line-height: 1.55;
    }

    .eyebrow {
        font-size: .74rem;
        letter-spacing: .14em;
        text-transform: uppercase;
        color: #9bd7e1;
        font-weight: 700;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #17628a, #208c98);
        border: 0;
        border-radius: 10px;
        font-weight: 700;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #dce7f0;
        border-radius: 10px;
    }

    /* Improve form labels and select-box contrast on the light background. */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"] {
        color: #263f58;
    }

    [data-testid="stSelectbox"] label,
    [data-testid="stNumberInput"] label,
    [data-testid="stRadio"] label {
        color: #173954;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# High-contrast chart theme
CHART_BG = "#0B1220"
CHART_TEXT = "#EDF4FC"
CHART_GRID = "#2A3A50"
BLUE = "#38BDF8"
TEAL = "#2DD4BF"
ORANGE = "#FBBF24"
RED = "#FB7185"


# =========================================================
# LOAD REPORTS, DATA AND EXISTING MODELS
# =========================================================

@st.cache_data
def read_csv(name):
    path = REPORTS / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_resource
def load_datasets():
    return {
        task: get_dataset(
            task=task,
            dataset_id="FD004",
            random_state=SEED,
        )
        for task in ("regression", "classification", "clustering")
    }


@st.cache_resource
def load_models():
    paths = {
        "regression": [
            "regression_best.joblib",
            "best_regression_model.joblib",
        ],
        "classification": [
            "classification_best.joblib",
            "best_classification_model.joblib",
        ],
        "clustering": ["clustering_best.joblib"],
    }

    result = {}

    for task, candidates in paths.items():
        path = next(
            (
                MODELS / name
                for name in candidates
                if (MODELS / name).exists()
            ),
            None,
        )

        if path is None:
            raise FileNotFoundError(
                f"No saved {task} model found in {MODELS}"
            )

        result[task] = joblib.load(path)

    return result


try:
    DATA = load_datasets()
    MODELS_MAP = load_models()
except Exception as exc:
    st.error(f"Unable to load project data or models: {exc}")
    st.info(
        "Run the app from the repository root and check "
        "the data/, src/, and models/ directories."
    )
    st.stop()


# =========================================================
# GENERAL HELPERS
# =========================================================

def hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">AEROSHIELD · NASA C-MAPSS FD004</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def to_array(value):
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return np.asarray(value)


def numeric_column(df, column):
    return pd.to_numeric(df[column], errors="coerce")


def dark_layout(fig, title=None, height=440):
    fig.update_layout(
        title=dict(
            text=title or "",
            font=dict(size=17, color=CHART_TEXT),
            x=0.02,
        ),
        height=height,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(
            family="DM Sans, sans-serif",
            size=12,
            color=CHART_TEXT,
        ),
        margin=dict(l=28, r=42, t=68, b=50),
        hoverlabel=dict(
            bgcolor="#18263A",
            font_color="#FFFFFF",
            bordercolor=CHART_GRID,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=CHART_TEXT, size=11),
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=CHART_GRID,
        zerolinecolor="#64748B",
        linecolor=CHART_GRID,
        tickfont=dict(color=CHART_TEXT, size=11),
        title_font=dict(color=CHART_TEXT),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=CHART_GRID,
        zerolinecolor="#64748B",
        linecolor=CHART_GRID,
        tickfont=dict(color=CHART_TEXT, size=11),
        title_font=dict(color=CHART_TEXT),
    )
    return fig


def show_chart(fig, key):
    # Explicit unique keys prevent StreamlitDuplicateElementId errors.
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
        config={"displaylogo": False, "responsive": True},
    )



def _fmt(x):
    if pd.isna(x):
        return ""
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.4f}"


def scorecard(df, label, metrics, key, title):
    """Heatmap: colour is relative rank within each metric; text is the real value."""
    if df.empty or label not in df.columns:
        st.warning("Cannot display scorecard: model label or report data is missing.")
        return

    cols = [metric for metric, _ in metrics if metric in df.columns]
    if not cols:
        st.warning("Cannot display scorecard: none of the requested metric columns exist.")
        return

    lower = dict(metrics)
    vals = df[cols].apply(pd.to_numeric, errors="coerce")
    z = pd.DataFrame(index=df.index)
    for col in cols:
        span = vals[col].max() - vals[col].min()
        if pd.isna(span):
            z[col] = np.nan
        else:
            scaled = (vals[col] - vals[col].min()) / span if span else vals[col] * 0 + 0.5
            z[col] = 1 - scaled if lower[col] else scaled

    fig = go.Figure(go.Heatmap(
        z=z.values,
        x=cols,
        y=df[label].astype(str).tolist(),
        text=vals.apply(lambda col: col.map(_fmt)).values,
        texttemplate="%{text}",
        colorscale=[[0, "#7F1D1D"], [0.5, "#1E3A5F"], [1, "#0F766E"]],
        zmin=0,
        zmax=1,
        showscale=False,
        xgap=2,
        ygap=2,
        hovertemplate="Model: %{y}<br>Metric: %{x}<br>Value: %{text}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    dark_layout(fig, title, max(380, 44 * len(df) + 120))
    show_chart(fig, key=key)
    st.caption("Colour is relative inside each column, so small gaps can look big. Read the numbers.")

def report_table(df):
    if df.empty:
        st.warning("The corresponding report CSV was not found.")
    else:
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )


def normalize_name(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def notebook_output_image(notebook_name, source_marker):
    """
    Read a saved PNG from a matching notebook cell.

    The notebook is never executed here. The function only reads the image
    already embedded in the .ipynb file's saved output.
    """
    candidates = [
        ROOT / notebook_name,
        ROOT / "notebooks" / notebook_name,
    ]

    notebook_path = next(
        (path for path in candidates if path.exists()),
        None,
    )

    if notebook_path is None:
        return None

    try:
        with notebook_path.open("r", encoding="utf-8") as handle:
            notebook = json.load(handle)

        for cell in notebook.get("cells", []):
            source = "".join(cell.get("source", []))

            if source_marker.lower() not in source.lower():
                continue

            for output in cell.get("outputs", []):
                data = output.get("data", {})
                png = data.get("image/png")

                if png:
                    encoded = "".join(png) if isinstance(png, list) else png
                    return base64.b64decode(encoded)

    except Exception as exc:
        warnings.warn(
            f"Could not read saved output from {notebook_name}: {exc}"
        )

    return None


def artifact_candidates(task, model_name):
    """
    Discover an existing model artifact matching a report model name.
    No model training is performed.
    """
    aliases = {
        "regression": {
            "gradientboosting": [
                "regression_best.joblib",
                "best_regression_model.joblib",
            ],
        },
        "classification": {
            "gradientboosting": [
                "classification_best.joblib",
                "best_classification_model.joblib",
            ],
        },
        "clustering": {
            "kmeans": ["clustering_best.joblib"],
        },
    }

    norm = normalize_name(model_name)
    names = aliases.get(task, {}).get(norm, [])

    # Also accept artifacts named after their model, if present.
    for path in MODELS.glob("*.joblib"):
        stem = normalize_name(path.stem)
        if norm and (norm in stem or stem in norm):
            names.append(path.name)

    result = []
    for name in names:
        path = MODELS / name
        if path.exists() and path not in result:
            result.append(path)

    return result


def load_selected_artifact(task, model_name):
    candidates = artifact_candidates(task, model_name)
    if not candidates:
        return None
    try:
        return joblib.load(candidates[0])
    except Exception:
        return None


def model_names(df, column):
    if df.empty or column not in df.columns:
        return []
    return df[column].dropna().astype(str).tolist()



def draw_confusion_matrix(cm, key):
    fig = px.imshow(
        cm,
        x=["Predicted normal", "Predicted critical"],
        y=["Actual normal", "Actual critical"],
        text_auto=True,
        labels={
            "x": "Predicted class",
            "y": "Actual class",
            "color": "Cycles",
        },
        color_continuous_scale=["#132238", "#17628A", "#2DD4BF"],
        aspect="auto",
    )
    fig.update_traces(
        textfont=dict(size=17, color="white"),
        hovertemplate=(
            "Actual: %{y}<br>Predicted: %{x}<br>"
            "Count: %{z}<extra></extra>"
        ),
    )
    dark_layout(fig, "Confusion matrix", 450)
    show_chart(fig, key=key)


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.markdown("# ✈️ AeroShield")
    st.caption("ENGINE HEALTH INTELLIGENCE")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Regression",
            "Classification",
            "Clustering",
            "Interactive ML Lab",
        ],
        key="workspace_navigation",
    )

    st.divider()
    st.caption("NASA C-MAPSS · FD004")
    st.caption("Existing saved models · No retraining")
    st.caption("Team 14 · ML Capstone")


# =========================================================
# 1. OVERVIEW
# =========================================================

if page == "Overview":
    hero(
        "Predict before failure.",
        "One workspace for remaining useful life, failure-risk "
        "classification, and engine operating regimes.",
    )

    reg = DATA["regression"]
    a, b, c, d = st.columns(4)
    a.metric("Input features", len(reg["feature_names"]))
    b.metric("Training rows", f"{len(reg['X_train']):,}")
    c.metric("Validation rows", f"{len(reg['X_val']):,}")
    d.metric("Test rows", f"{len(reg['X_test']):,}")

    st.subheader("Three complementary ML tracks")
    a, b, c = st.columns(3)

    with a:
        st.markdown("### 📈 Regression")
        st.write("Estimate remaining useful life (RUL).")
        st.caption("R² · RMSE · MAE")

    with b:
        st.markdown("### 🎯 Classification")
        st.write(f"Flag critical cycles (RUL ≤ {RISK_THRESHOLD}).")
        st.caption("F1 · ROC-AUC · Missed failures")

    with c:
        st.markdown("### 🧩 Clustering")
        st.write("Discover groups of similar engine-cycle patterns.")
        st.caption("Silhouette · Davies–Bouldin · Calinski–Harabasz")

    reg_report = read_csv("regression_results.csv")
    clf_report = read_csv("classification_results.csv")
    clu_report = read_csv("clustering_results.csv")

    st.subheader("Best recorded results by task")
    a, b, c = st.columns(3)

    if not reg_report.empty and "R2 test" in reg_report.columns:
        temp = reg_report.copy()
        temp["R2 test"] = numeric_column(temp, "R2 test")
        best = temp.loc[temp["R2 test"].idxmax()]
        a.metric(
            str(best["Model"]),
            f"R² {best['R2 test']:.4f}",
            f"RMSE {float(best['RMSE test']):.2f} cycles",
        )

    if not clf_report.empty and "F1w test" in clf_report.columns:
        temp = clf_report.copy()
        temp["F1w test"] = numeric_column(temp, "F1w test")
        best = temp.loc[temp["F1w test"].idxmax()]
        b.metric(
            str(best["Model"]),
            f"F1 {best['F1w test']:.4f}",
            f"Critical recall {float(best['Class 1 Recall']):.3f}",
        )

    if not clu_report.empty and "Algorithm" in clu_report.columns:
        rows = clu_report[
            clu_report["Algorithm"].astype(str).str.contains(
                "K-Means", case=False, na=False
            )
        ]
        if not rows.empty:
            best = rows.iloc[0]
            c.metric(
                "K-Means",
                f"Silhouette {float(best['Silhouette Score']):.4f}",
                f"DB {best['Davies-Bouldin']:.3f}",
            )


# =========================================================
# 2. REGRESSION
# =========================================================

elif page == "Regression":
    hero(
        "Regression analytics",
        "Compare all evaluated regressors and inspect the saved "
        "best model's test-set predictions and errors.",
    )

    df = read_csv("regression_results.csv")
    if df.empty:
        st.error("Missing reports/regression_results.csv")
        st.stop()

    with st.expander("Full results table (from reports/)"):
        report_table(df)

    st.subheader("Final comparison graphs")
    scorecard(
        df, "Model",
        [("R2 test", False), ("R2 val", False), ("RMSE test", True), ("MAE test", True)],
        key="regression_scorecard",
        title="Regression scorecard (greener = better in each column)",
    )

    gap_cols = {"R2 val", "R2 test", "Model"}
    if gap_cols.issubset(df.columns):
        gap = df.copy()
        gap["R2 val"] = pd.to_numeric(gap["R2 val"], errors="coerce")
        gap["R2 test"] = pd.to_numeric(gap["R2 test"], errors="coerce")
        gap = gap.dropna(subset=["R2 val", "R2 test"]).sort_values("R2 test")
        fig = go.Figure()
        for _, row in gap.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["R2 val"], row["R2 test"]], y=[row["Model"]] * 2,
                mode="lines", line=dict(color="#64748B", width=3),
                showlegend=False, hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=gap["R2 val"], y=gap["Model"], mode="markers",
            name="Validation", marker=dict(color=BLUE, size=11),
        ))
        fig.add_trace(go.Scatter(
            x=gap["R2 test"], y=gap["Model"], mode="markers",
            name="Test", marker=dict(color=ORANGE, size=11),
        ))
        fig.update_xaxes(title="R²")
        dark_layout(fig, "Validation vs test R² (long line = generalisation gap)", 500)
        show_chart(fig, key="regression_gap")
    else:
        st.info("Validation/test R² columns are unavailable for the dumbbell chart.")

    ranked = df.copy()
    ranked["R2 test"] = pd.to_numeric(ranked["R2 test"], errors="coerce")
    ranked = ranked.dropna(subset=["R2 test"]).sort_values("R2 test", ascending=False)
    fig = px.bar(
        ranked, x="Model", y="R2 test", color="R2 test",
        color_continuous_scale=["#7F1D1D", "#1E3A5F", "#0F766E"],
        text="R2 test",
        hover_data=[c for c in ["RMSE test", "MAE test"] if c in ranked.columns],
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-35)
    dark_layout(fig, "All regressors · Test R² (higher is better)", 520)
    show_chart(fig, key="regression_all_models_final")

    # CHANGE: distinguish the saved tuned artifact from the baseline report row.
    st.subheader("Saved champion · Gradient Boosting (tuned)")

    model = load_selected_artifact("regression", "Gradient Boosting")

    if model is None:
        st.error(
            "The saved Gradient Boosting regression artifact was not found. "
            "The recorded comparison results remain available above."
        )
    else:
        ds = DATA["regression"]
        X = to_array(ds["X_test"])
        y = to_array(ds["y_test"]).reshape(-1)

        try:
            pred = np.asarray(model.predict(X)).reshape(-1)

            live = pd.DataFrame({
                "Metric": ["R²", "RMSE", "MAE"],
                "Recomputed from saved tuned Gradient Boosting": [
                    r2_score(y, pred),
                    np.sqrt(mean_squared_error(y, pred)),
                    mean_absolute_error(y, pred),
                ],
            })

            st.dataframe(live, use_container_width=True, hide_index=True)

            plot_df = pd.DataFrame({
                "Actual RUL": y,
                "Predicted RUL": pred,
                "Residual": y - pred,
            })

            # Deterministic downsampling keeps scatter plots responsive.
            plot_df = plot_df.sample(
                min(8000, len(plot_df)),
                random_state=SEED,
            )

            c1, c2 = st.columns(2)

            with c1:
                fig = px.scatter(
                    plot_df,
                    x="Actual RUL",
                    y="Predicted RUL",
                    opacity=0.55,
                    labels={
                        "Actual RUL": "Actual RUL (cycles)",
                        "Predicted RUL": "Predicted RUL (cycles)",
                    },
                )
                low = float(
                    min(
                        plot_df["Actual RUL"].min(),
                        plot_df["Predicted RUL"].min(),
                    )
                )
                high = float(
                    max(
                        plot_df["Actual RUL"].max(),
                        plot_df["Predicted RUL"].max(),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[low, high],
                        y=[low, high],
                        mode="lines",
                        name="Ideal prediction",
                        line=dict(color=ORANGE, dash="dash", width=2),
                    )
                )
                dark_layout(
                    fig,
                    "Gradient Boosting · Actual vs predicted RUL",
                    460,
                )
                show_chart(fig, key="regression_saved_actual_predicted")

            with c2:
                fig = px.scatter(
                    plot_df,
                    x="Predicted RUL",
                    y="Residual",
                    opacity=0.55,
                )
                fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color=ORANGE,
                )
                fig.update_xaxes(title="Predicted RUL (cycles)")
                fig.update_yaxes(title="Residual (actual − predicted)")
                dark_layout(
                    fig,
                    "Gradient Boosting · Residual plot",
                    460,
                )
                show_chart(fig, key="regression_saved_residuals")

            c1, c2 = st.columns(2)

            with c1:
                fig = px.histogram(
                    plot_df,
                    x="Residual",
                    nbins=45,
                )
                fig.update_traces(marker_color=TEAL)
                dark_layout(fig, "Residual distribution", 400)
                show_chart(fig, key="regression_saved_error_distribution")

            with c2:
                importance = getattr(model, "feature_importances_", None)
                feature_names = list(ds["feature_names"])

                if importance is not None and len(importance) == len(feature_names):
                    imp = pd.DataFrame({
                        "Feature": feature_names,
                        "Importance": importance,
                    }).nlargest(20, "Importance").sort_values("Importance")

                    fig = px.bar(
                        imp,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        text="Importance",
                    )
                    fig.update_traces(
                        texttemplate="%{text:.4f}",
                        textposition="outside",
                        cliponaxis=False,
                        marker_color=BLUE,
                    )
                    fig.update_yaxes(title="")
                    dark_layout(
                        fig,
                        "Gradient Boosting · Feature importance",
                        500,
                    )
                    show_chart(fig, key="regression_saved_importance")
                else:
                    st.info(
                        "The saved model does not expose feature_importances_. "
                        "No feature-importance graph is shown."
                    )

        except Exception as exc:
            st.error(
                f"Unable to generate saved Gradient Boosting diagnostics: {exc}"
            )


# =========================================================
# 3. CLASSIFICATION
# =========================================================

elif page == "Classification":
    hero(
        "Failure-risk classification",
        "Compare all evaluated classifiers, inspect ROC-AUC, and "
        "investigate false alarms and missed critical cycles.",
    )

    df = read_csv("classification_results.csv")
    if df.empty:
        st.error("Missing reports/classification_results.csv")
        st.stop()

    with st.expander("Full results table (from reports/)"):
        report_table(df)

    st.subheader("Final comparison graphs")
    scorecard(
        df, "Model",
        [("Accuracy test", False), ("Precision test", False), ("Recall test", False),
         ("F1w test", False), ("ROC AUC", False), ("Class 1 Recall", False),
         ("FN (Missed Failures)", True), ("FP (False Alarms)", True)],
        key="classification_scorecard",
        title="Classification scorecard (greener = better in each column)",
    )

    ranked = df.copy()
    ranked["Class 1 Recall"] = pd.to_numeric(ranked["Class 1 Recall"], errors="coerce")
    ranked = ranked.dropna(subset=["Class 1 Recall"]).sort_values("Class 1 Recall", ascending=False)
    hover_cols = [c for c in ["FN (Missed Failures)", "FP (False Alarms)"] if c in ranked.columns]
    fig = px.bar(
        ranked, x="Model", y="Class 1 Recall", color="Class 1 Recall",
        color_continuous_scale=["#7F1D1D", "#1E3A5F", "#0F766E"],
        text="Class 1 Recall", hover_data=hover_cols,
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-35, yaxis_range=[0, 1.12])
    dark_layout(fig, "All classifiers · Critical-class recall (higher is better)", 520)
    show_chart(fig, key="classification_all_models_final")
    st.caption("Recall alone ignores false alarms. Hover over a bar to see FN and FP counts.")

    # Use the saved ROC figure from the classification notebook.
    # The notebook is not executed and no classifier is retrained.
    st.subheader("Combined ROC curves · Saved notebook output")

    roc_image = notebook_output_image(
        "03_Classification.ipynb",
        "Complete ROC Curve Comparison",
    )

    if roc_image is not None:
        st.image(
            roc_image,
            use_container_width=True,
        )
    else:
        st.warning(
            "The saved ROC figure could not be found. Keep "
            "03_Classification.ipynb in the repository root or notebooks/ "
            "folder. The recorded AUC table remains available."
        )

    # Per-model confusion matrix dropdown.
    # FN and FP come from the project's recorded report. TN and TP are
    # reconstructed using the current test-set class totals.
    st.subheader("Model-specific confusion matrix")

    classifier_names = model_names(df, "Model")
    selected = st.selectbox(
        "Select classifier",
        classifier_names,
        key="classification_model_selector",
    )

    selected_rows = df[df["Model"].astype(str) == str(selected)]

    if selected_rows.empty:
        st.warning("No recorded result row was found for this classifier.")
    else:
        result_row = selected_rows.iloc[0]
        required = {
            "FN (Missed Failures)",
            "FP (False Alarms)",
        }

        if not required.issubset(df.columns):
            st.warning(
                "The classification report does not contain both FN and FP "
                "counts, so a model-specific confusion matrix cannot be reconstructed."
            )
        else:
            try:
                y_test = to_array(DATA["classification"]["y_test"]).reshape(-1)

                negative_total = int(np.sum(y_test == 0))
                positive_total = int(np.sum(y_test == 1))

                fn = int(result_row["FN (Missed Failures)"])
                fp = int(result_row["FP (False Alarms)"])

                # Matrix rows: actual [normal, critical].
                # Matrix columns: predicted [normal, critical].
                tn = negative_total - fp
                tp = positive_total - fn

                if min(tn, fp, fn, tp) < 0:
                    st.error(
                        "The recorded FN/FP counts do not agree with the "
                        "current test-set class totals. Check that the report "
                        "CSV and preprocessing split come from the same notebook run."
                    )
                else:
                    cm = np.array(
                        [[tn, fp], [fn, tp]],
                        dtype=int,
                    )

                    draw_confusion_matrix(
                        cm,
                        key=f"classification_cm_{normalize_name(selected)}",
                    )

                    a, b, c, d = st.columns(4)
                    a.metric("True negatives", f"{tn:,}")
                    b.metric("False alarms", f"{fp:,}")
                    c.metric("Missed failures", f"{fn:,}")
                    d.metric("True positives", f"{tp:,}")

                    critical_precision = (
                        tp / (tp + fp) if (tp + fp) else 0.0
                    )
                    critical_recall = (
                        tp / (tp + fn) if (tp + fn) else 0.0
                    )
                    critical_f1 = (
                        2 * critical_precision * critical_recall
                        / (critical_precision + critical_recall)
                        if (critical_precision + critical_recall)
                        else 0.0
                    )

                    a, b, c = st.columns(3)
                    a.metric(
                        "Critical precision",
                        f"{critical_precision:.4f}",
                    )
                    b.metric(
                        "Critical recall",
                        f"{critical_recall:.4f}",
                    )
                    c.metric(
                        "Critical F1",
                        f"{critical_f1:.4f}",
                    )

            except Exception as exc:
                st.error(
                    f"Unable to reconstruct the confusion matrix for {selected}: {exc}"
                )


# =========================================================
# 4. CLUSTERING
# =========================================================

elif page == "Clustering":
    hero(
        "Clustering & operating regimes",
        "Compare unsupervised algorithms and inspect cluster structure "
        "using PCA and hierarchical linkage.",
    )

    df = read_csv("clustering_results.csv")
    if df.empty:
        st.error("Missing reports/clustering_results.csv")
        st.stop()

    with st.expander("Full results table (from reports/)"):
        report_table(df)

    st.subheader("Final comparison graphs")
    scorecard(
        df, "Algorithm",
        [("Silhouette Score", False), ("Davies-Bouldin", True),
         ("Calinski-Harabasz", False), ("Noise %", True), ("Fit Time (s)", True)],
        key="clustering_scorecard",
        title="Clustering scorecard (greener = better in each column)",
    )

    ranked = df.copy()
    ranked["Silhouette Score"] = pd.to_numeric(ranked["Silhouette Score"], errors="coerce")
    ranked = ranked.dropna(subset=["Silhouette Score"]).sort_values("Silhouette Score", ascending=False)
    fig = px.bar(
        ranked, x="Algorithm", y="Silhouette Score", color="Silhouette Score",
        color_continuous_scale=["#7F1D1D", "#1E3A5F", "#0F766E"],
        text="Silhouette Score",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-25)
    dark_layout(fig, "All clustering algorithms · Silhouette score (higher is better)", 520)
    show_chart(fig, key="clustering_all_models_final")
    st.caption(
        "Ranked by Silhouette only. K-Means is the model used in this app: it assigns "
        "every point to a cluster and supports native prediction for new data, "
        "while standard DBSCAN labels noise points and has no native predict() method."
    )

    # NEW: optional model-selection charts from existing report CSVs.
    # These charts are shown only when the expected CSV and columns exist.
    st.subheader("K-Means elbow and GMM model selection")
    elbow = read_csv("clustering_kmeans_elbow.csv")
    gmm_sel = read_csv("clustering_gmm_selection.csv")
    c1, c2 = st.columns(2)

    if not elbow.empty and {"k", "Inertia"}.issubset(elbow.columns):
        fig = px.line(elbow, x="k", y="Inertia", markers=True)
        fig.update_traces(line_color=TEAL)
        fig.add_vline(x=6, line_dash="dash", line_color=ORANGE)
        dark_layout(fig, "K-Means elbow curve (inertia vs k)", 400)
        with c1:
            show_chart(fig, key="clustering_elbow")
    else:
        with c1:
            st.info(
                "K-Means elbow chart unavailable: expected CSV/columns not found."
            )

    if not gmm_sel.empty and {"k", "BIC", "AIC"}.issubset(gmm_sel.columns):
        fig = go.Figure([
            go.Scatter(
                x=gmm_sel["k"],
                y=gmm_sel[metric],
                name=metric,
                mode="lines+markers",
            )
            for metric in ("BIC", "AIC")
        ])
        fig.update_xaxes(title="k")
        fig.update_yaxes(title="Score")
        dark_layout(fig, "GMM model selection (BIC / AIC)", 400)
        with c2:
            show_chart(fig, key="clustering_gmm_selection")
    else:
        with c2:
            st.info(
                "GMM selection chart unavailable: expected CSV/columns not found."
            )

    st.subheader("Saved best model · K-Means")

    X_train = to_array(DATA["clustering"]["X_train"])
    rng = np.random.RandomState(SEED)
    sample = X_train[
        rng.choice(
            len(X_train),
            size=min(5000, len(X_train)),
            replace=False,
        )
    ]

    model = MODELS_MAP["clustering"]

    try:
        labels = np.asarray(model.predict(sample))

        # Match the notebook: fit PCA on full training data, then project sample.
        pca = PCA(n_components=10, random_state=SEED).fit(X_train)
        projected = pca.transform(sample)[:, :2]

        plot_df = pd.DataFrame({
            "PC1": projected[:, 0],
            "PC2": projected[:, 1],
            "Cluster": labels.astype(str),
        })

        plot_df = plot_df.sample(
            min(3000, len(plot_df)),
            random_state=SEED,
        )

        fig = px.scatter(
            plot_df,
            x="PC1",
            y="PC2",
            color="Cluster",
            opacity=0.7,
            render_mode="webgl",
            color_discrete_sequence=[
                "#38BDF8",
                "#2DD4BF",
                "#FBBF24",
                "#A78BFA",
                "#FB7185",
                "#F97316",
                "#34D399",
                "#60A5FA",
            ],
        )
        fig.update_traces(marker=dict(size=5))
        dark_layout(
            fig,
            "Saved K-Means · PCA operating-regime map",
            520,
        )
        show_chart(fig, key="clustering_saved_kmeans_pca")

        variance = pca.explained_variance_ratio_[:2].sum() * 100
        st.caption(
            f"The first two principal components explain {variance:.2f}% "
            "of the variance in the training data. This is a visualization, "
            "not a cluster-quality metric."
        )

        counts = pd.Series(labels).value_counts().sort_index()
        cluster_counts = pd.DataFrame({
            "Cluster": counts.index.astype(str),
            "Rows": counts.values,
        })

        st.subheader("Cluster sizes in plotted sample")
        st.dataframe(
            cluster_counts,
            use_container_width=True,
            hide_index=True,
        )

    except Exception as exc:
        st.error(f"Unable to display saved K-Means visualization: {exc}")

    st.subheader("Agglomerative dendrogram")

    c1, c2 = st.columns(2)
    method = c1.selectbox(
        "Linkage method",
        ["ward", "average"],
        key="dendrogram_linkage",
    )
    n = c2.selectbox(
        "Rows in dendrogram",
        [100, 200, 300, 500],
        index=2,
        key="dendrogram_rows",
    )

    try:
        dendro_sample = sample[:min(int(n), len(sample))]
        Z = linkage(dendro_sample, method=method)
        dendro = dendrogram(
            Z,
            truncate_mode="lastp",
            p=30,
            no_plot=True,
        )

        fig = go.Figure()

        for xs, ys in zip(dendro["icoord"], dendro["dcoord"]):
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=TEAL, width=1.8),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        fig.update_xaxes(title="Merge position")
        fig.update_yaxes(title="Linkage distance")

        if method == "ward" and int(n) == 300:
            dendrogram_title = (
                "Ward linkage · Last 30 merge groups "
                "(notebook-matched, n=300)"
            )
        else:
            dendrogram_title = (
                f"{method.title()} linkage · Last 30 merge groups "
                f"(sample n={len(dendro_sample)})"
            )

        dark_layout(
            fig,
            dendrogram_title,
            470,
        )
        show_chart(fig, key=f"dendrogram_{method}_{n}")

    except Exception as exc:
        st.error(f"Unable to generate dendrogram: {exc}")


# =========================================================
# 5. INTERACTIVE ML LAB
# =========================================================

elif page == "Interactive ML Lab":
    hero(
        "Interactive ML Lab",
        "Pick a track, choose a real engine and cycle, tweak sensors, "
        "and see the saved model's output.",
    )

    track = st.radio(
        "1 · Choose your ML track",
        ["Regression", "Classification", "Clustering"],
        horizontal=True,
        key="lab_track",
    ).lower()

    ds = DATA["regression"]  # shared engineered features, split and scaler
    model = MODELS_MAP[track]
    test_raw = load_test_history()

    required_columns = {
        "unit_id", "cycle", "RUL",
        "sensor_2", "sensor_11", "sensor_14",
    }
    missing = required_columns - set(test_raw.columns)

    if missing:
        st.error(
            "The raw test-history data is missing expected columns: "
            + ", ".join(sorted(missing))
        )
        st.stop()

    c1, c2 = st.columns(2)
    engine = c1.selectbox(
        "2 · Test engine",
        sorted(test_raw["unit_id"].unique()),
        key="lab_engine",
    )

    eng_df = test_raw[test_raw["unit_id"] == engine].sort_values("cycle")
    last = int(eng_df["cycle"].max())

    cycle = c2.slider(
        "Cycle",
        int(eng_df["cycle"].min()),
        last,
        last,
        key=f"lab_cycle_{engine}",
    )

    # Only history up to the selected cycle is used.
    hist = eng_df[eng_df["cycle"] <= cycle].copy()
    cur = hist.iloc[-1]

    st.markdown("#### 3 · Optional: change sensor readings at this cycle")

    sensor_columns = st.columns(3)
    vals = {}

    for col, sensor in zip(
        sensor_columns,
        ["sensor_2", "sensor_11", "sensor_14"],
    ):
        vals[sensor] = col.number_input(
            sensor,
            value=float(cur[sensor]),
            format="%.4f",
            key=f"lab_{sensor}_{engine}_{cycle}",
        )

    if st.button("Run prediction", type="primary", key="lab_run"):
        changed = {
            sensor: value
            for sensor, value in vals.items()
            if not np.isclose(value, float(cur[sensor]))
        }

        # Only edit the selected cycle; earlier history remains unchanged.
        for sensor, value in changed.items():
            hist.loc[hist.index[-1], sensor] = value

        # Recompute rolling means / cycle-to-cycle features using history only.
        engineered = engineer_features(hist)
        feature_names = list(ds["feature_names"])
        missing_features = set(feature_names) - set(engineered.columns)

        if missing_features:
            st.error(
                "Feature engineering did not produce the model's expected features: "
                + ", ".join(sorted(missing_features))
            )
            st.stop()

        row = engineered.iloc[[-1]][feature_names]

        # Use the scaler fitted on training data; never refit it here.
        X = ds["scaler"].transform(row.values)
        actual = int(cur["RUL"])

        st.caption(
            f"Engine {engine}, cycle {cycle} of {last} · "
            f"sensor overrides: {changed or 'none'}"
        )

        if track == "regression":
            pred = float(model.predict(X)[0])

            a, b = st.columns(2)
            a.metric(
                "Predicted RUL",
                f"{pred:.1f} cycles",
                f"{pred - actual:+.1f} vs actual",
                delta_color="off",
            )
            b.metric("Actual RUL", f"{actual} cycles")

            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pred,
                delta={"reference": actual},
                title={"text": "Predicted RUL (cycles); amber line = actual"},
                gauge={
                    "axis": {
                        "range": [0, max(300, pred * 1.3, actual * 1.3)]
                    },
                    "bar": {"color": TEAL},
                    "steps": [
                        {"range": [0, RISK_THRESHOLD], "color": "#7F1D1D"}
                    ],
                    "threshold": {
                        "line": {"color": ORANGE, "width": 4},
                        "value": actual,
                    },
                },
            ))
            dark_layout(fig, None, 340)
            show_chart(fig, key="lab_regression_gauge")

        elif track == "classification":
            label = int(model.predict(X)[0])

            if not hasattr(model, "predict_proba"):
                st.error(
                    "The saved classifier does not support probability estimates."
                )
                st.stop()

            # Select probability by the actual class label, not a hard-coded column.
            classes = list(model.classes_)

            if 1 not in classes:
                st.error("The saved classifier does not contain critical class 1.")
                st.stop()

            critical_index = classes.index(1)
            prob = float(model.predict_proba(X)[0, critical_index])

            a, b, c = st.columns(3)
            a.metric(
                "Predicted class",
                "Critical" if label == 1 else "Normal",
            )
            b.metric("P(critical)", f"{prob:.3f}")
            c.metric(
                "Actual class",
                "Critical" if actual <= RISK_THRESHOLD else "Normal",
                f"actual RUL {actual}",
                delta_color="off",
            )

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                number={"valueformat": ".3f"},
                title={"text": "Probability of the critical window"},
                gauge={
                    "axis": {"range": [0, 1]},
                    "bar": {"color": RED if label == 1 else TEAL},
                    "steps": [
                        {"range": [0, 0.5], "color": "#134E4A"},
                        {"range": [0.5, 1], "color": "#7F1D1D"},
                    ],
                },
            ))
            dark_layout(fig, None, 340)
            show_chart(fig, key="lab_classification_gauge")

        else:
            st.metric("K-Means cluster", int(model.predict(X)[0]))
            st.caption("Cluster IDs are arbitrary labels, not diagnoses.")

        if changed:
            st.warning(
                "Edited sensor readings are hypothetical. The actual RUL shown "
                "belongs to the original unedited data point."
            )

        with st.expander("Model input vector (before scaling)"):
            st.dataframe(row.T, use_container_width=True)


# =========================================================
# FOOTER
# =========================================================

st.divider()
st.caption(
    "AeroShield · Team 14 · NASA C-MAPSS FD004 · "
    "Educational prototype."
)