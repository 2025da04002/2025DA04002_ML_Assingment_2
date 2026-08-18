import os
import joblib
import numpy as np
import pandas as pd

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
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# 1. Create model directory
os.makedirs('model', exist_ok=True)

# 2. Load dataset
df = pd.read_csv('student-por.csv', sep=None, engine='python')

# 3. Create binary target: 1 if G3 >= 12 (Above Average/Good), else 0
df['target'] = (df['G3'] >= 12).astype(int)

# Drop grade columns so models predict using demographic, social & study features
X = df.drop(columns=['G1', 'G2', 'G3', 'target'])
y = df['target']

# 4. Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# 5. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Save test set for Streamlit app upload
test_df = X_test.copy()
test_df['target'] = y_test.values
test_df.to_csv('test_data.csv', index=False)
print("Saved test_data.csv successfully!")

# 6. Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ]
)

# 7. Define the 5 Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'kNN': KNeighborsClassifier(n_neighbors=7),
    'Naive Bayes': GaussianNB(),
    'Random Forest (Ensemble)': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
}

# 8. Train, evaluate, and save each model
results = []
for name, model in models.items():
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    clf.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    # Calculate evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    results.append({
        'ML Model Name': name,
        'Accuracy': round(acc, 4),
        'AUC': round(auc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1': round(f1, 4),
        'MCC': round(mcc, 4)
    })
    
    # Save trained pipeline
    filename = name.lower().replace(' ', '_').replace('(', '').replace(')', '') + '.joblib'
    joblib.dump(clf, os.path.join('model', filename))
    print(f"Trained and saved: {name} -> model/{filename}")

# Display Comparison Table
results_df = pd.DataFrame(results)
print("\n=== MODEL COMPARISON TABLE ===")
print(results_df.to_string(index=False))