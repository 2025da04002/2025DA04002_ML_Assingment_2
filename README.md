# Student Performance ML Classification & Web Application

## a. Problem Statement
Predicting student academic success early in a course is critical for educational institutions to provide timely interventions and support. The objective of this project is to build and evaluate multiple machine learning classification models to predict whether a student will achieve **Above Average Academic Performance** (defined as a final Portuguese course grade `G3 >= 12` on a 0–20 grading scale) based on demographic, social, and study habit features.

## b. Dataset Description
- **Dataset Name:** Student Performance Data Set (Portuguese Language Course - `student-por.csv`) from the UCI Machine Learning Repository / Kaggle.
- **Total Instances:** 649 students (Meets minimum instance size of 500).
- **Total Features:** 30 independent attributes (Meets minimum feature size of 12).
- **Target Variable:** Binary classification target (`target`), where `1` indicates `G3 >= 12` (Above Average / Good Performance) and `0` indicates `G3 < 12`.
- **Class Distribution:** 348 Positive instances (`1`) and 301 Negative instances (`0`), making it a well-balanced classification dataset.
- **Feature Types:** Attributes include student demographics (age, sex, address), family background (parental education/jobs, family support), study habits (study time, past failures, travel time), and social habits (free time, alcohol consumption, absences).

## c. GitHub Repository Link
- **GitHub Repo:** [https://github.com/2025da04002/2025DA04002_ML_Assingment_2]

## d. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.7538** | 0.8081 | **0.7317** | **0.8571** | **0.7895** | **0.5067** |
| **Decision Tree** | 0.6692 | 0.6919 | 0.6552 | 0.8143 | 0.7261 | 0.3330 |
| **kNN** | 0.6538 | 0.7138 | 0.6344 | 0.8429 | 0.7239 | 0.3051 |
| **Naive Bayes** | 0.7385 | 0.7676 | 0.7308 | 0.8143 | 0.7703 | 0.4725 |
| **Random Forest (Ensemble)** | 0.7385 | **0.8119** | 0.7250 | 0.8286 | 0.7733 | 0.4733 |

### Observations on Model Performance
| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the highest overall **Accuracy (75.38%)**, **Precision (73.17%)**, **Recall (85.71%)**, **F1 Score (0.7895)**, and **MCC (0.5067)**. Because the tabular features have strong linear relationships with academic discipline, regularized linear modeling generalized exceptionally well without overfitting. |
| **Decision Tree** | Achieved **66.92% accuracy** and an **MCC of 0.3330**. Single decision trees suffered from high variance and overfitting on categorical attributes compared to linear and ensemble approaches. |
| **kNN** | Recorded the lowest accuracy (**65.38%**) and MCC (**0.3051**). Distance-based classification struggled due to the high dimensionality (30 features) after one-hot encoding categorical variables. |
| **Naive Bayes** | Performed strongly with **73.85% accuracy** and an **MCC of 0.4725**. Despite the conditional independence assumption, Gaussian Naive Bayes separated high and low performing students effectively. |
| **Random Forest (Ensemble)** | Achieved the highest **AUC Score (0.8119)** alongside **73.85% accuracy** and **0.4733 MCC**. By averaging multiple randomized decision trees, it successfully reduced variance and captured non-linear feature interactions. |
| **Overall Winner for your dataset?** | **Logistic Regression** is the overall winner for decision-threshold metrics (highest Accuracy, F1, and MCC of 0.5067), while **Random Forest** is the best model for overall ranking discrimination (highest AUC of 0.8119). |
