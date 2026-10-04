import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

from api import add_features
from train import CATEGORICAL, NUMERIC, RANDOM_STATE, TARGET

RAW = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
       "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
EPS = 1e-4


def _psi_from_props(p_ref, p_cur):
    p_ref, p_cur = np.clip(p_ref, EPS, None), np.clip(p_cur, EPS, None)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def psi_categorical(ref, cur):
    cats = sorted(set(ref.astype(str)) | set(cur.astype(str)))
    p_ref = ref.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).values
    p_cur = cur.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).values
    return _psi_from_props(p_ref, p_cur)


def psi_numeric(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    p_ref = np.histogram(ref, edges)[0] / len(ref)
    p_cur = np.histogram(cur, edges)[0] / len(cur)
    return _psi_from_props(p_ref, p_cur)


def status(psi):
    return "stable" if psi < 0.10 else "moderate" if psi < 0.25 else "SIGNIFICANT"


def drift_report(ref, cur, ref_scores, cur_scores):
    rows = []
    for col in NUMERIC + CATEGORICAL:
        is_cat = col in CATEGORICAL or ref[col].nunique() <= 10
        psi = psi_categorical(ref[col], cur[col]) if is_cat else psi_numeric(ref[col], cur[col])
        ks_p = None if is_cat else float(ks_2samp(ref[col], cur[col]).pvalue)
        rows.append({"feature": col, "psi": round(psi, 4), "ks_pvalue": ks_p, "status": status(psi)})
    s = psi_numeric(ref_scores, cur_scores)
    rows.append({"feature": "PREDICTED_SCORE", "psi": round(s, 4), "ks_pvalue":
                 float(ks_2samp(ref_scores, cur_scores).pvalue), "status": status(s)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def simulate_drift(raw, seed=0):
    rng = np.random.default_rng(seed)
    d = raw.copy()
    d["Age"] = (d["Age"] + 8).clip(upper=100)            # older customer base
    d["Balance"] = d["Balance"] * 1.3                     # higher balances
    active = d["IsActiveMember"] == 1
    flip = active & (rng.random(len(d)) < 0.30)           # 30% of active members go inactive
    d.loc[flip, "IsActiveMember"] = 0
    return d


df = pd.read_parquet("data/features.parquet")
X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
X_train, X_test, _, _ = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)  # same split as train.py

model = joblib.load("models/best_model.joblib")
ref_scores = model.predict_proba(X_train)[:, 1]

batches = {
    "real_test_batch": X_test,
    "simulated_drift_batch": add_features(simulate_drift(X_test[RAW])),
}

for name, batch in batches.items():
    cur_scores = model.predict_proba(batch[NUMERIC + CATEGORICAL])[:, 1]
    rep = drift_report(X_train, batch, ref_scores, cur_scores)
    rep.to_csv(f"reports/drift_{name}.csv", index=False)
    print(f"\n=== {name} ===")
    print(rep.head(8).to_string(index=False))
    alerts = rep[rep["status"] == "SIGNIFICANT"]
    if len(alerts):
        print(f"ALERT: {len(alerts)} feature(s) with PSI >= 0.25 -> investigate, consider retraining.")
    else:
        print("No significant drift.")