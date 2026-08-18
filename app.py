import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Student Performance ML Classifier", layout="wide")
st.title("🎓 Student Performance Classification Web App")
st.markdown("Predict whether a student achieves **Above Average Academic Performance** (`G3 >= 12`) based on demographic, social, and study habits.")

# --- 1. LIVE MODEL TRAINING LOGIC ---
@st.cache_resource # This ensures the models only train once when the app loads, keeping it fast!
def train_models_live():
    df = pd.read_csv('student-por.csv', sep=None, engine='python')
    df['target'] = (df['G3'] >= 12).astype(int)
    
    X = df.drop(columns=['G1', 'G2', 'G3', 'target'])
    y = df['target']
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ]
    )
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'kNN': KNeighborsClassifier(n_neighbors=7),
        'Naive Bayes': GaussianNB(),
        'Random Forest (Ensemble)': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    }
    
    trained_pipelines = {}
    for name, model in models.items():
        clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        clf.fit(X_train, y_train)
        trained_pipelines[name] = clf
        
    return trained_pipelines

# Train models in the background
try:
    trained_models = train_models_live()
except Exception as e:
    st.error(f"Error training models. Ensure 'student-por.csv' is in the repository. Error: {e}")
    st.stop()

# --- 2. STREAMLIT UI ---
st.sidebar.header("1. Dataset Selection")
uploaded_file = st.sidebar.file_uploader("Upload custom CSV (Optional)", type=["csv"])

st.sidebar.header("2. Select Machine Learning Model")
selected_model_name = st.sidebar.selectbox("Choose Model", list(trained_models.keys()))

data = None
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom CSV uploaded successfully!")
elif os.path.exists("test_data.csv"):
    data = pd.read_csv("test_data.csv")
    st.sidebar.info("Using default `test_data.csv` from repository.")
else:
    st.error("No dataset available. Please upload `test_data.csv`.")

if data is not None:
    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head(5), use_container_width=True)
    
    if "target" not in data.columns:
        st.error("Error: The dataset must contain the 'target' column.")
    else:
        X_test = data.drop(columns=["target"])
        y_test = data["target"]
        
        # Get the live-trained model
        model = trained_models[selected_model_name]
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        st.divider()
        st.subheader(f"📈 Evaluation Metrics: {selected_model_name}")
        
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
