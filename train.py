from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from src.data_utils import load_data, clean_data

RANDOM_STATE = 42
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

df = clean_data(load_data())
y = df.pop("Churn").map({"No": 0, "Yes": 1})
if y.isna().any():
    raise ValueError("Unexpected values found in Churn. Expected only 'Yes' and 'No'.")
X = df.copy()

# First reserve a final test set. Split the remaining data into train and validation.
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.20,
    random_state=RANDOM_STATE, stratify=y_train_val
)

numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()

def make_pipeline(classifier):
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # Dense output allows all three example estimators to consume the features.
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", classifier)
    ])

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE)
}

# Compare candidates on validation data; do not use the final test set to choose a model.
validation_results = []
fitted_models = {}

for name, classifier in models.items():
    pipeline = make_pipeline(classifier)
    print(f"\nTraining {name}...")
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_val)
    proba = pipeline.predict_proba(X_val)[:, 1]
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_val, pred)),
        "precision": float(precision_score(y_val, pred, zero_division=0)),
        "recall": float(recall_score(y_val, pred, zero_division=0)),
        "f1": float(f1_score(y_val, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_val, proba)),
        "pr_auc": float(average_precision_score(y_val, proba))
    }
    validation_results.append(metrics)
    fitted_models[name] = pipeline
    print("Validation metrics:", metrics)

validation_df = pd.DataFrame(validation_results).sort_values("roc_auc", ascending=False)
validation_df.to_csv(REPORT_DIR / "validation_model_comparison.csv", index=False)
print("\n--- Validation model comparison (sorted by ROC-AUC) ---")
print(validation_df.to_string(index=False))

best_name = validation_df.iloc[0]["model"]
selected_pipeline = fitted_models[best_name]

# Refit the selected model on all training + validation rows, then evaluate once on test.
selected_pipeline.fit(X_train_val, y_train_val)
test_pred = selected_pipeline.predict(X_test)
test_proba = selected_pipeline.predict_proba(X_test)[:, 1]

test_metrics = {
    "selected_model": best_name,
    "accuracy": float(accuracy_score(y_test, test_pred)),
    "precision": float(precision_score(y_test, test_pred, zero_division=0)),
    "recall": float(recall_score(y_test, test_pred, zero_division=0)),
    "f1": float(f1_score(y_test, test_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, test_proba)),
    "pr_auc": float(average_precision_score(y_test, test_proba))
}
print(f"\nSelected model based on validation ROC-AUC: {best_name}")
print("\n--- Final held-out test metrics ---")
print(json.dumps(test_metrics, indent=2))
print("\nClassification report:")
print(classification_report(y_test, test_pred, target_names=["No Churn", "Churn"], zero_division=0))
print("Confusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_test, test_pred))

joblib.dump(selected_pipeline, MODEL_DIR / "churn_pipeline.joblib")
with open(MODEL_DIR / "model_info.json", "w", encoding="utf-8") as f:
    json.dump({
        "model_name": best_name,
        "features": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "random_state": RANDOM_STATE,
        "train_validation_test_split": "64% / 16% / 20%",
        "selection_metric": "validation ROC-AUC",
        "test_metrics": test_metrics
    }, f, indent=2)

predictions = X_test.copy()
predictions["actual_churn"] = y_test.map({0: "No", 1: "Yes"}).values
predictions["predicted_churn"] = pd.Series(test_pred, index=X_test.index).map({0: "No", 1: "Yes"})
predictions["churn_probability"] = test_proba
predictions.to_csv(REPORT_DIR / "test_predictions.csv", index=False)

print(f"\nSaved model to {MODEL_DIR / 'churn_pipeline.joblib'}")
print(f"Saved reports to {REPORT_DIR}/")
print("Metrics above are from this run; performance is not guaranteed.")
