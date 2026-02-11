"""
Train 6 classification models on Bank Marketing dataset (UCI).
Dataset: Predict if client subscribes to term deposit (yes/no).
- 45,211 instances, 16+ features - meets assignment requirements.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "y"


def load_data() -> tuple[pd.DataFrame, list[str], str]:
    """Load Bank Marketing dataset from OpenML."""
    print("Fetching Bank Marketing dataset from OpenML...")
    data = fetch_openml(name="bank-marketing", version=1, as_frame=True, parser="auto")
    df = pd.concat([data.data, data.target], axis=1)
    df.columns = df.columns.astype(str)
    # Target is the last column (from data.target)
    target_col = df.columns[-1]
    df = df.rename(columns={target_col: TARGET_COLUMN})
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip().str.lower()
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    return df, feature_cols, TARGET_COLUMN


def get_preprocessor(feature_cols: list[str], df: pd.DataFrame):
    """Build ColumnTransformer for numeric + categorical features."""
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in numeric_cols]
    if not cat_cols:
        cat_cols = [
            c
            for c in feature_cols
            if df[c].dtype == "object" or str(df[c].dtype) == "category"
        ]

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if cat_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            )
        )
    if not transformers:
        transformers.append(("num", StandardScaler(), feature_cols))
    return ColumnTransformer(transformers, remainder="passthrough")


def make_models(preprocessor):
    """Create model pipelines. Binary classification (yes/no)."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "KNN": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "Naive Bayes": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", GaussianNB()),
            ]
        ),
        "Random Forest (Ensemble)": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=12,
                        min_samples_leaf=5,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "XGBoost (Ensemble)": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model, x_test, y_test):
    """Compute metrics for binary classification."""
    preds = model.predict(x_test)
    try:
        probs = model.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
    except Exception:
        auc = 0.0

    return {
        "Accuracy": accuracy_score(y_test, preds),
        "AUC": auc,
        "Precision": precision_score(y_test, preds, average="binary", zero_division=0),
        "Recall": recall_score(y_test, preds, average="binary", zero_division=0),
        "F1 Score": f1_score(y_test, preds, average="binary", zero_division=0),
        "MCC": matthews_corrcoef(y_test, preds),
    }


def main():
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df, feature_cols, target_col = load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Target classes: {df[target_col].unique().tolist()}")
    print(f"Features: {len(feature_cols)}")

    x = df[feature_cols]
    y = df[target_col]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
    )

    preprocessor = get_preprocessor(feature_cols, df)
    preprocessor.fit(x_train)

    models = make_models(preprocessor)
    metrics = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(x_train, y_train)
        metrics[name] = evaluate_model(model, x_test, y_test)
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        with open(output_dir / f"{safe_name}.pkl", "wb") as f:
            pickle.dump(model, f)

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    test_df = x_test.copy()
    test_df[target_col] = le.inverse_transform(y_test)
    test_df.to_csv(output_dir / "test_data.csv", index=False)

    with (output_dir / "feature_names.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "features": feature_cols,
                "target": target_col,
                "classes": le.classes_.tolist(),
            },
            f,
            indent=2,
        )

    with open(output_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("Saved models (.pkl) and metrics to:", output_dir)


if __name__ == "__main__":
    main()
