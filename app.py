import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="Student Performance ML Classifier", layout="wide")

st.title("🎓 Student Performance Classification Web App")
st.markdown("Predict whether a student achieves **Above Average Academic Performance** (`G3 >= 12`) based on demographic, social, and study habits.")

# Sidebar - Model Selection & File Upload
st.sidebar.header("1. Upload Test Dataset")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

st.sidebar.header("2. Select Machine Learning Model")
model_options = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "kNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Random Forest (Ensemble)": "model/random_forest_ensemble.joblib"
}

selected_model_name = st.sidebar.selectbox("Choose Model", list(model_options.keys()))

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Preview of Uploaded Dataset")
    st.dataframe(data.head(5), use_container_width=True)
    
    if "target" not in data.columns:
        st.error("Error: The uploaded CSV must contain the 'target' column.")
    else:
        X_test = data.drop(columns=["target"])
        y_test = data["target"]
        
        # Load selected model
        model_path = model_options[selected_model_name]
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Compute Metrics
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            st.divider()
            st.subheader(f"📈 Evaluation Metrics: {selected_model_name}")
            
            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{acc:.4f}")
            col1.metric("Recall", f"{rec:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}")
            col2.metric("F1 Score", f"{f1:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col3.metric("MCC Score", f"{mcc:.4f}")
            
            st.divider()
            col_cm, col_cr = st.columns(2)
            
            with col_cm:
                st.subheader("🧩 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                            xticklabels=["Below Avg (0)", "Above Avg (1)"],
                            yticklabels=["Below Avg (0)", "Above Avg (1)"], ax=ax)
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")
                st.pyplot(fig)
                
            with col_cr:
                st.subheader("📋 Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
        else:
            st.error(f"Model file not found at `{model_path}`. Please run `train_models.py` first.")
else:
    st.info("👆 Please upload `test_data.csv` from the sidebar to evaluate the models.")