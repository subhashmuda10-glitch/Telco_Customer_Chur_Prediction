import pandas as pd
import numpy as np
import pickle

# ------------------------------
# 1. Load Dataset
# ------------------------------
df = pd.read_csv("data/Telco-Customer-Churn-dataset.csv")

# Fix TotalCharges (some spaces exist)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

# ------------------------------
# 2. Split X and y
# ------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

# ------------------------------
# 3. Label Encode the target
# ------------------------------
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# ------------------------------
# 4. Column Transformer for categorical + numeric
# ------------------------------
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

column_transformer = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ]
)

X_processed = column_transformer.fit_transform(X)

# ------------------------------
# 5. Train-test split
# ------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# ------------------------------
# 6. Train several models
# ------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "CatBoost": CatBoostClassifier(verbose=0)
}

results = {}

from sklearn.metrics import accuracy_score

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results[name] = acc
    print(name, "Accuracy:", acc)

# ------------------------------
# 7. Find best model
# ------------------------------
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)
print("Accuracy:", results[best_model_name])

# ------------------------------
# 8. Save model + preprocessors
# ------------------------------
pickle.dump(column_transformer, open("../models/column_transformer.pkl", "wb"))
pickle.dump(label_encoder, open("../models/label_encoder.pkl", "wb"))
pickle.dump(best_model, open("../models/best_model.pkl", "wb"))

print("\nSaved all artifacts successfully!")
