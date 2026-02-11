# ML Assignment 2 - Classification Models

## Problem Statement
Build and evaluate six classification models on a single dataset and deploy an
interactive Streamlit app that compares model performance and supports CSV
uploads for predictions and evaluation.

## Dataset Description
**Dataset:** Bank Marketing (UCI Machine Learning Repository)

- **Type:** Binary classification
- **Instances:** 45,211
- **Features:** 16 (V1–V16: age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome)
- **Target:** `y` – whether client subscribed to a term deposit (yes/no)


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
| Logistic Regression | 0.9012 | 0.9056 | 0.6445 | 0.3478 | 0.4518 | 0.4261 |
| Decision Tree | 0.8746 | 0.7015 | 0.4649 | 0.4754 | 0.4701 | 0.3990 |
| KNN | 0.8962 | 0.8277 | 0.5990 | 0.3403 | 0.4340 | 0.4001 |
| Naive Bayes | 0.8548 | 0.8101 | 0.4059 | 0.5198 | 0.4559 | 0.3774 |
| Random Forest (Ensemble) | 0.9073 | 0.9291 | 0.6698 | 0.4102 | 0.5088 | 0.4778 |
| XGBoost (Ensemble) | 0.9092 | 0.9340 | 0.6671 | 0.4471 | 0.5354 | 0.4992 |

### Model Performance Observations

| ML Model Name | Observation about Model Performance |
| --- | --- |
| Logistic Regression | Good accuracy and AUC; moderate precision/recall balance. |
| Decision Tree | Lower AUC; prone to overfitting on imbalanced data. |
| KNN | Decent accuracy; recall limited on minority class. |
| Naive Bayes | Lower accuracy; Gaussian assumption may not fit all features well. |
| Random Forest (Ensemble) | Strong AUC; handles imbalance better; good overall metrics. |
| XGBoost (Ensemble) | Best F1 and MCC; best balance for imbalanced classification. |

## Streamlit App Features
- Dataset upload option 
- Model selection dropdown
- Display of evaluation metrics
- Enhanced confusion matrix (class labels, counts, percentages)
- Classification report

