# ML Assignment 2 - Classification Models

## Problem Statement

This project develops and evaluates six machine learning classification models on the Bank Marketing dataset to predict whether a client will subscribe to a term deposit (yes/no) based on demographic, financial, and campaign-related features. The models are deployed in an interactive web application that allows users to compare performance metrics, visualize results, and test predictions on custom data uploads.

## Dataset Description

**Dataset:** Bank Marketing (UCI Machine Learning Repository)

- **Type:** Binary classification
- **Instances:** 45,211
- **Features:** 16 (V1–V16: age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome)
- **Target:** `y` – whether the client subscribed to a term deposit (yes/no)

**Splitting strategy:** The data is split into training and test sets using an 80–20 stratified split. Stratification ensures that the proportion of positive and negative classes in both splits mirrors the original dataset, which is important given the class imbalance (minority class ~11%).

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
| Random Forest (Ensemble) | 0.8986 | 0.9247 | 0.7143 | 0.2221 | 0.3389 | 0.3611 |
| XGBoost (Ensemble) | 0.9092 | 0.9340 | 0.6671 | 0.4471 | 0.5354 | 0.4992 |

### Model Performance Observations

| ML Model Name | Observation about Model Performance |
| --- | --- |
| Logistic Regression | Good accuracy (90.1%) and AUC (0.906); moderate precision/recall balance. Interpretable and suitable for baseline comparisons. |
| Decision Tree | Lower AUC (0.70); prone to overfitting on imbalanced data. Trade-off between interpretability and generalization. |
| KNN | Decent accuracy; recall limited on minority class. Sensitive to feature scaling and local patterns. |
| Naive Bayes | Lower accuracy (85.5%); Gaussian assumption may not fit all features well. Fast but less suited to this domain. |
| Random Forest (Ensemble) | Smaller forest trades recall for model size but maintains solid AUC. Handles mixed feature types well. |
| XGBoost (Ensemble) | Best F1 and MCC; best balance for imbalanced classification. Strong AUC and practical performance. |

### Conclusion

Ensemble methods (Random Forest and XGBoost) achieve the highest AUC and strongest overall metrics on this imbalanced dataset. XGBoost offers the best trade-off between accuracy, F1, and MCC, making it the preferred choice for production use. Logistic Regression provides a solid baseline with interpretability. For this binary prediction task on bank marketing data, gradient boosting (XGBoost) and tree-based ensembles are the most suitable models.

## Streamlit App Features

- Dataset upload option
- Sample dataset download
- Model selection dropdown
- Display of evaluation metrics
- Enhanced confusion matrix (class labels, counts, percentages)
- Classification report
