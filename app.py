import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
    "XGBoost (Ensemble)": "xgboost_ensemble.pkl",
}

MAX_UPLOAD_MB = 5


def load_metrics():
    with (MODEL_DIR / "metrics.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_feature_info():
    with (MODEL_DIR / "feature_names.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_label_encoder():
    with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
        return pickle.load(f)


def load_test_data():
    return pd.read_csv(MODEL_DIR / "test_data.csv")


def load_model(model_name):
    with open(MODEL_DIR / MODEL_FILES[model_name], "rb") as f:
        return pickle.load(f)


def render_confusion_matrix(y_true, y_pred, class_names=None, show_percent=True):
    """Enhanced confusion matrix with class labels, counts, and optional percentages."""
    cm = confusion_matrix(y_true, y_pred)
    if class_names is not None:
        # Ensure order matches confusion_matrix output
        labels = class_names
    else:
        labels = [str(i) for i in range(len(cm))]

    fig, ax = plt.subplots(figsize=(8, 6))
    if show_percent:
        row_sums = cm.sum(axis=1)[:, None]
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_pct = 100 * cm.astype("float") / row_sums
        annot = [
            [f"{v}\n({p:.1f}%)" for v, p in zip(row_v, row_p)]
            for row_v, row_p in zip(cm, cm_pct)
        ]
        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            ax=ax,
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Count"},
        )
    else:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Count"},
        )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def render_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).T
    st.dataframe(df_report.style.format(precision=4))


def main():
    st.set_page_config(page_title="ML Assignment 2 - Bank Marketing", layout="wide")
    st.title("ML Assignment 2 - Classification Models")
    st.caption("Bank Marketing Dataset (UCI) | Predict term deposit subscription (yes/no)")

    metrics = load_metrics()
    feature_info = load_feature_info()
    feature_names = feature_info["features"]
    target_column = feature_info["target"]
    class_names = feature_info.get("classes")
    le = load_label_encoder()

    st.markdown(
        """
This app demonstrates six classification models trained on the **Bank Marketing** (UCI) dataset.
Predict whether a client will subscribe to a term deposit. Use the sidebar to select a model.
Upload a CSV for predictions (max **5 MB**). If the CSV includes the target column (`y`), metrics will be computed.
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

    st.subheader("Confusion Matrix (Test Set)")
    show_pct = st.checkbox("Show percentages in cells", value=True, key="show_pct")
    test_data = load_test_data()
    x_test = test_data[feature_names]
    y_test = test_data[target_column]
    pred_enc = model.predict(x_test)
    y_pred = le.inverse_transform(pred_enc)
    render_confusion_matrix(y_test, y_pred, class_names=class_names, show_percent=show_pct)

    st.subheader("Classification Report (Test Set)")
    render_classification_report(y_test, y_pred)

    st.divider()
    st.subheader("Upload Dataset for Evaluation (CSV)")
    st.info(
        f"Upload a CSV with the same {len(feature_names)} feature columns ({', '.join(feature_names[:5])}...). Keep file under 5 MB."
    )
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_UPLOAD_MB:
            st.error(
                f"File too large ({file_size_mb:.1f} MB). Streamlit Cloud has limits. "
                f"Please use a smaller file (max {MAX_UPLOAD_MB} MB) or use the test_data.csv sample."
            )
            return

        try:
            uploaded_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            return

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

        try:
            preds_enc = model.predict(x_upload)
            preds = le.inverse_transform(preds_enc)
        except Exception as e:
            st.error(f"Prediction failed: {e}. Check that columns match the training data.")
            return

        result_df = uploaded_df.copy()
        result_df["prediction"] = preds
        st.write("Preview with predictions:")
        st.dataframe(result_df.head())

        if y_upload is not None:
            st.markdown("### Uploaded Data Metrics")
            render_confusion_matrix(y_upload, preds, class_names=class_names)
            render_classification_report(y_upload, preds)


if __name__ == "__main__":
    main()
