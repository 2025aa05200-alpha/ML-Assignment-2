# ML Assignment 2 - Classification Models

## Problem Statement
Build and evaluate six classification models on a single dataset and deploy an
interactive Streamlit app that compares model performance and supports CSV
uploads for predictions and evaluation.

## Dataset Description
**Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset (UCI)

- **Type:** Binary classification
- **Instances:** 569
- **Features:** 30 numeric features
- **Target:** `target` (0 = malignant, 1 = benign)

## Models Used
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes (Gaussian)
- Random Forest (Ensemble)
- XGBoost (Ensemble)

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) | 0.9474 | 0.9937 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |
| XGBoost (Ensemble) | 0.9561 | 0.9950 | 0.9467 | 0.9861 | 0.9660 | 0.9058 |

### Model Performance Observations

| ML Model Name | Observation about Model Performance |
| --- | --- |
| Logistic Regression | Best overall accuracy and strong balanced metrics on this dataset. |
| Decision Tree | Lower recall and MCC, indicating weaker generalization than other models. |
| KNN | Strong recall and F1, performs well after scaling. |
| Naive Bayes | Solid AUC but lower overall accuracy than top models. |
| Random Forest (Ensemble) | High AUC and stable metrics with good balance. |
| XGBoost (Ensemble) | High recall and competitive accuracy, strong ensemble performance. |

## Streamlit App Features
- Dataset upload option (CSV)
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix and classification report

## How to Run Locally
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Train and save models:
   ```
   python model/train_models.py
   ```
3. Start the Streamlit app:
   ```
   streamlit run app.py
   ```
