import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"


MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
    "XGBoost (Ensemble)": "xgboost_ensemble.joblib",
}


def load_metrics():
    with (MODEL_DIR / "metrics.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_feature_info():
    with (MODEL_DIR / "feature_names.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_test_data():
    return pd.read_csv(MODEL_DIR / "test_data.csv")


def load_model(model_name):
    model_path = MODEL_DIR / MODEL_FILES[model_name]
    return joblib.load(model_path)


def render_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig, clear_figure=True)


def render_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).T
    st.dataframe(df_report.style.format(precision=4))


def main():
    st.set_page_config(page_title="ML Assignment 2 - Classifier Demo", layout="wide")
    st.title("ML Assignment 2 - Classification Models")

    metrics = load_metrics()
    feature_info = load_feature_info()
    feature_names = feature_info["features"]
    target_column = feature_info["target"]

    st.markdown(
        """
This app demonstrates six classification models trained on the **Breast Cancer**
dataset. Use the sidebar to select a model and optionally upload a CSV file for
prediction. If the uploaded CSV includes the target column (`target`), the app
will compute evaluation metrics and a confusion matrix on your uploaded data.
"""
    )

    st.sidebar.header("Model Selection")
    model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))
    model = load_model(model_name)

    col_metrics, col_compare = st.columns([1, 2])
    with col_metrics:
        st.subheader("Selected Model Metrics (Test Set)")
        selected_metrics = metrics[model_name]
        metric_df = pd.DataFrame(
            {
                "Metric": list(selected_metrics.keys()),
                "Value": [round(v, 4) for v in selected_metrics.values()],
            }
        )
        st.table(metric_df)

    with col_compare:
        st.subheader("Model Comparison (Test Set)")
        comparison_rows = []
        for name, values in metrics.items():
            comparison_rows.append(
                {
                    "Model": name,
                    "Accuracy": values["Accuracy"],
                    "AUC": values["AUC"],
                    "Precision": values["Precision"],
                    "Recall": values["Recall"],
                    "F1 Score": values["F1 Score"],
                    "MCC": values["MCC"],
                }
            )
        comparison_df = pd.DataFrame(comparison_rows).set_index("Model")
        st.dataframe(comparison_df.style.format(precision=4))

    st.subheader("Confusion Matrix / Classification Report (Test Set)")
    test_data = load_test_data()
    x_test = test_data[feature_names]
    y_test = test_data[target_column]
    y_pred = model.predict(x_test)
    render_confusion_matrix(y_test, y_pred)
    render_classification_report(y_test, y_pred)

    st.divider()
    st.subheader("Upload Dataset for Evaluation (CSV)")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        missing = [col for col in feature_names if col not in uploaded_df.columns]
        if missing:
            st.error(
                "Uploaded CSV is missing required feature columns: "
                + ", ".join(missing)
            )
            return

        x_upload = uploaded_df[feature_names]
        y_upload = (
            uploaded_df[target_column]
            if target_column in uploaded_df.columns
            else None
        )

        preds = model.predict(x_upload)
        uploaded_df["prediction"] = preds
        st.write("Preview with predictions:")
        st.dataframe(uploaded_df.head())

        if y_upload is not None:
            st.markdown("### Uploaded Data Metrics")
            render_confusion_matrix(y_upload, preds)
            render_classification_report(y_upload, preds)


if __name__ == "__main__":
    main()
