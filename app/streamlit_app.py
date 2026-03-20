import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
METRICS_PATH = MODELS_DIR / "best_model_metrics.json"
RESULTS_PATH = MODELS_DIR / "model_results.csv"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.csv"
TUNING_RESULTS_PATH = MODELS_DIR / "random_forest_tuning_results.json"

st.set_page_config(page_title="House Price Prediction", layout="wide")
st.title("House Price Prediction")

col1, col2 = st.columns(2)
with col1:
    status_df = pd.DataFrame(
        [
            {"artifact": "Best model metrics", "available": METRICS_PATH.exists()},
            {"artifact": "Model comparison", "available": RESULTS_PATH.exists()},
            {"artifact": "Feature importance", "available": FEATURE_IMPORTANCE_PATH.exists()},
            {"artifact": "RF tuning results", "available": TUNING_RESULTS_PATH.exists()},
        ]
    )
    st.subheader("Artifact status")
    st.dataframe(status_df, hide_index=True, use_container_width=True)

    st.subheader("Best model metrics")
    if METRICS_PATH.exists():
        st.json(json.loads(METRICS_PATH.read_text(encoding="utf-8")))
    else:
        st.info("Train the project first to generate model metrics.")
with col2:
    st.subheader("Model comparison")
    if RESULTS_PATH.exists():
        results_df = pd.read_csv(RESULTS_PATH).sort_values("RMSE").reset_index(drop=True)
        st.dataframe(results_df, use_container_width=True)
    else:
        st.info("Run training first to generate model comparison results.")

if FEATURE_IMPORTANCE_PATH.exists():
    st.subheader("Top feature importances")
    importance_df = pd.read_csv(FEATURE_IMPORTANCE_PATH).head(20)
    st.bar_chart(importance_df.set_index("feature"))

if TUNING_RESULTS_PATH.exists():
    st.subheader("Random Forest tuning")
    st.json(json.loads(TUNING_RESULTS_PATH.read_text(encoding="utf-8")))
