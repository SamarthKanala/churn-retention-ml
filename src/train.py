import os
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
TARGET = "Exited"

NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
           "IsActiveMember", "EstimatedSalary", "BalanceSalaryRatio", "ZeroBalance",
           "TenureByAge", "InactiveWithBalance"]
CATEGORICAL = ["Geography", "Gender", "AgeGroup", "ProductGroup"]


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def main():
    df = pd.read_parquet("data/features.parquet")
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # handles ~4:1 imbalance

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=pos_weight,
            eval_metric="logloss", random_state=RANDOM_STATE),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision",
               "recall": "recall", "f1": "f1"}

    mlflow.set_experiment("churn-prediction")
    os.makedirs("models", exist_ok=True)
    results = {}

    for name, clf in models.items():
        pipe = Pipeline([("prep", make_preprocessor()), ("clf", clf)])

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_params({f"clf__{k}": v for k, v in clf.get_params().items()
                               if isinstance(v, (int, float, str, bool, type(None)))})

            # 1) Stratified 5-fold CV on the training set only
            cvres = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
            cv_metrics = {f"cv_{m}": float(np.mean(cvres[f"test_{m}"])) for m in scoring}
            mlflow.log_metrics(cv_metrics)

            # 2) Fit on full training set, evaluate once on the untouched test set
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)
            test_metrics = {
                "test_roc_auc": roc_auc_score(y_test, proba),
                "test_pr_auc": average_precision_score(y_test, proba),
                "test_recall": recall_score(y_test, pred),
                "test_precision": precision_score(y_test, pred),
                "test_f1": f1_score(y_test, pred),
            }
            mlflow.log_metrics(test_metrics)

            path = f"models/{name}.joblib"
            joblib.dump(pipe, path)
            mlflow.log_artifact(path)

            results[name] = {**cv_metrics, **test_metrics}
            print(f"\n{name}")
            for k, v in results[name].items():
                print(f"  {k}: {v:.3f}")

    # Pick the best model by CV PR-AUC (not test, to avoid peeking at the test set)
    best = max(results, key=lambda n: results[n]["cv_pr_auc"])
    joblib.dump(joblib.load(f"models/{best}.joblib"), "models/best_model.joblib")
    print(f"\nBest model by CV PR-AUC: {best}")


if __name__ == "__main__":
    main()