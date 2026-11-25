"""
AquaGuard AI – Water Quality Classification System
Model Training, Preprocessing, Evaluation, and Optimization Pipeline

Description:
  1. Loads the Water Potability dataset and splits it 80/20 (stratified).
  2. Trains the ORIGINAL models (no class weights, tuned for accuracy) so we have a "before" result.
  3. Trains 7 IMPROVED models (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM,
     Random Forest, AdaBoost): class_weight='balanced' (where supported) and
     GridSearchCV that optimises F1-score with stratified 5-fold cross-validation.
  4. Picks the best model by cross-validation F1 (training data only, never the test set).
  5. Saves the chosen pipeline to water_model.joblib and all metrics to model_results.json.

Run:  python train_model.py
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET_COL = "Potability"

# Models compared in the "bake-off" (all syllabus classifiers, same split and same pipeline).
# Each entry: display name, estimator, hyperparameter grid, supports class_weight?
# KNN, Naive Bayes and AdaBoost have no class_weight option in scikit-learn.
MODEL_CONFIGS = {
    "logistic_regression": (
        "Logistic Regression",
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        {"model__C": [0.01, 0.1, 1, 10]},
        True,
    ),
    "decision_tree": (
        "Decision Tree",
        DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        {"model__max_depth": [3, 5, 8, None], "model__min_samples_leaf": [1, 5, 10]},
        True,
    ),
    "knn": (
        "K-Nearest Neighbors",
        KNeighborsClassifier(),
        {"model__n_neighbors": [3, 5, 9, 15, 25]},
        False,
    ),
    "naive_bayes": (
        "Naive Bayes (Gaussian)",
        GaussianNB(),
        {"model__var_smoothing": [1e-9, 1e-7, 1e-5]},
        False,
    ),
    "svm": (
        "Support Vector Machine (RBF)",
        SVC(class_weight="balanced", probability=True, random_state=RANDOM_STATE),
        {"model__C": [0.5, 1, 5, 10], "model__gamma": ["scale", 0.05, 0.1]},
        True,
    ),
    "random_forest": (
        "Random Forest",
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5, 10],
        },
        True,
    ),
    "adaboost": (
        "AdaBoost",
        AdaBoostClassifier(random_state=RANDOM_STATE),
        {"model__n_estimators": [50, 100, 200], "model__learning_rate": [0.1, 0.5, 1.0]},
        False,
    ),
}


def make_pipeline(model):
    """Preprocessing + model in one Pipeline, so the imputer/scaler are fitted on training data only."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def evaluate(pipeline, X_test, y_test):
    """Return accuracy, precision, recall, F1, ROC-AUC and confusion matrix on the test set."""
    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def clean_params(params):
    """Make GridSearch params JSON friendly (drop 'model__' prefix, None -> 'None')."""
    return {k.replace("model__", ""): ("None" if v is None else v) for k, v in params.items()}


def print_metrics(label, m):
    print(
        f"      {label:<34} Acc={m['accuracy']:.4f}  Prec={m['precision']:.4f}  "
        f"Rec={m['recall']:.4f}  F1={m['f1_score']:.4f}  AUC={m['roc_auc']:.4f}"
    )


def main():
    print("=" * 70)
    print(" AquaGuard AI – Model Training and Optimization Pipeline")
    print("=" * 70)

    # ---------------------------------------------------------------- 1. Data
    dataset_path = "water_potability.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Please place water_potability.csv in the project folder."
        )

    print(f"\n[1/6] Loading dataset: {dataset_path}...")
    df = pd.read_csv(dataset_path)
    total_records, total_columns = df.shape
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    missing_dict = df[feature_cols].isnull().sum().to_dict()
    duplicates_count = int(df.duplicated().sum())
    class_counts = df[TARGET_COL].value_counts().to_dict()
    print(f"      Records: {total_records}, features: {len(feature_cols)}, duplicates: {duplicates_count}")
    print(f"      Missing values: {missing_dict}")
    print(f"      Target: Non-potable (0)={class_counts.get(0, 0)}, Potable (1)={class_counts.get(1, 0)}")

    # Feature summary stats (used by the UI for default input values)
    feature_stats = {
        col: {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "median": float(df[col].median()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        }
        for col in feature_cols
    }

    # Clean example rows for the live-demo preset buttons
    sample_potable = df[df[TARGET_COL] == 1].dropna().head(1)[feature_cols].iloc[0].to_dict()
    sample_non_potable = df[df[TARGET_COL] == 0].dropna().head(1)[feature_cols].iloc[0].to_dict()

    # ------------------------------------------------------------- 2. Split
    print("\n[2/6] Stratified split (80% train, 20% test)...")
    X = df[feature_cols]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"      Train: {len(X_train)}  Test: {len(X_test)}")

    # ------------------------------------------- 3. BEFORE: original approach
    print("\n[3/6] BEFORE: original models (no class weights, tuned for accuracy)...")
    lr_before = make_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    lr_before.fit(X_train, y_train)
    lr_before_m = evaluate(lr_before, X_test, y_test)
    print_metrics("Logistic Regression", lr_before_m)

    rf_base = make_pipeline(RandomForestClassifier(random_state=RANDOM_STATE))
    rf_base.fit(X_train, y_train)
    rf_base_m = evaluate(rf_base, X_test, y_test)
    print_metrics("Random Forest (default)", rf_base_m)

    old_grid = GridSearchCV(
        make_pipeline(RandomForestClassifier(random_state=RANDOM_STATE)),
        param_grid=MODEL_CONFIGS["random_forest"][2],
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    old_grid.fit(X_train, y_train)
    rf_before_m = evaluate(old_grid.best_estimator_, X_test, y_test)
    rf_before_m["best_parameters"] = clean_params(old_grid.best_params_)
    rf_before_m["best_cv_accuracy"] = float(old_grid.best_score_)
    print_metrics("Random Forest (tuned for accuracy)", rf_before_m)

    # --------------------------------------- 4. AFTER: imbalance-aware tuning
    print("\n[4/6] AFTER / BAKE-OFF: class_weight='balanced' + GridSearchCV(scoring=F1, StratifiedKFold)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    improved = {}
    fitted = {}
    for key, (name, estimator, grid, has_class_weight) in MODEL_CONFIGS.items():
        search = GridSearchCV(
            make_pipeline(estimator), param_grid=grid, cv=cv, scoring="f1", n_jobs=-1
        )
        search.fit(X_train, y_train)
        metrics = evaluate(search.best_estimator_, X_test, y_test)
        metrics.update({
            "name": name,
            "class_weight": "balanced" if has_class_weight else "not supported",
            "best_parameters": clean_params(search.best_params_),
            "cv_f1": float(search.best_score_),
        })
        improved[key] = metrics
        fitted[key] = search.best_estimator_
        print_metrics(f"{name} (cv F1={search.best_score_:.3f})", metrics)

    # ----------------------------------------------- 5. Select the best model
    print("\n[5/6] Selecting the best model by cross-validation F1 (training data only)...")
    best_key = max(improved, key=lambda k: improved[k]["cv_f1"])
    best_name = improved[best_key]["name"]
    best_pipeline = fitted[best_key]
    print(f"      Selected: {best_name} (CV F1 = {improved[best_key]['cv_f1']:.4f})")

    # Permutation importance works for any model: how much F1 drops when a feature is shuffled
    perm = permutation_importance(
        best_pipeline, X_test, y_test, scoring="f1", n_repeats=10, random_state=RANDOM_STATE
    )
    importances = dict(sorted(
        zip(feature_cols, perm.importances_mean.astype(float).tolist()),
        key=lambda item: item[1], reverse=True,
    ))

    # ------------------------------------------------------------ 6. Save
    print("\n[6/6] Saving artifacts...")
    joblib.dump(best_pipeline, "water_model.joblib")

    results_data = {
        "dataset_info": {
            "filename": "water_potability.csv",
            "source": "https://www.kaggle.com/datasets/adityakadiwal/water-potability",
            "total_records": total_records,
            "total_columns": total_columns,
            "feature_count": len(feature_cols),
            "features": feature_cols,
            "target": TARGET_COL,
            "missing_values": missing_dict,
            "duplicate_records": duplicates_count,
            "class_distribution": {
                "Non-Potable (0)": int(class_counts.get(0, 0)),
                "Potable (1)": int(class_counts.get(1, 0)),
            },
            "train_size": len(X_train),
            "test_size": len(X_test),
            "split_ratio": "80% Training / 20% Testing (Stratified)",
        },
        "feature_statistics": feature_stats,
        "presentation_presets": {
            "potable_sample": sample_potable,
            "non_potable_sample": sample_non_potable,
        },
        # BEFORE: the original approach (kept so the app can show before vs after)
        "models": {
            "logistic_regression": {
                "name": "Logistic Regression (Original)",
                "description": "Linear classifier, no class weights.",
                **lr_before_m,
            },
            "random_forest_baseline": {
                "name": "Random Forest (Default)",
                "description": "100 trees with default parameters, no class weights.",
                **rf_base_m,
            },
            "random_forest_optimized": {
                "name": "Random Forest (Tuned for Accuracy)",
                "description": "Original GridSearchCV tuned for accuracy, no class weights.",
                **rf_before_m,
            },
        },
        # AFTER: imbalance-aware models, tuned for F1 with stratified CV
        "tuning": {
            "scoring": "f1",
            "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
            "class_weight": "balanced (where the algorithm supports it)",
        },
        "improved_models": improved,
        "selected_model": {
            "key": best_key,
            "name": f"{best_name} (Balanced, F1-tuned)",
            "selection_criterion": (
                "Highest 5-fold stratified cross-validation F1-score on the training data "
                f"(CV F1 = {improved[best_key]['cv_f1']:.4f}). The test set was not used to choose the model."
            ),
            "importance_method": "Permutation importance on the test set (mean drop in F1 when a feature is shuffled)",
            "feature_importances": importances,
        },
    }

    with open("model_results.json", "w") as f:
        json.dump(results_data, f, indent=4)
    print("      Saved: water_model.joblib, model_results.json")

    print("\n" + "=" * 70)
    print(" Training, evaluation, and artifact generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
