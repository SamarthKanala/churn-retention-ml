import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

from train import CATEGORICAL, NUMERIC, RANDOM_STATE, TARGET

os.makedirs("reports", exist_ok=True)

df = pd.read_parquet("data/features.parquet")
X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)  # same split as train.py

pipe = joblib.load("models/xgboost.joblib")
prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]

Xt = prep.transform(X_test)
if hasattr(Xt, "toarray"):
    Xt = Xt.toarray()
names = [n.split("__", 1)[1].replace("<", "lt_").replace("[", "(").replace("]", ")")
         for n in prep.get_feature_names_out()]
Xt = pd.DataFrame(Xt, columns=names)

# ---- 1) SHAP ----
sv = shap.TreeExplainer(clf).shap_values(Xt)

plt.figure()
shap.summary_plot(sv, Xt, show=False)
plt.savefig("reports/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure()
shap.summary_plot(sv, Xt, plot_type="bar", show=False)
plt.savefig("reports/shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()

imp = pd.Series(np.abs(sv).mean(axis=0), index=names).sort_values(ascending=False)
print("Top 10 features by mean |SHAP|:\n", imp.head(10).round(3))

# ---- 2) Fairness check (Gender, Geography) ----
proba = pipe.predict_proba(X_test)[:, 1]
res = X_test[["Gender", "Geography"]].copy()
res["y"], res["pred"], res["proba"] = y_test.values, (proba >= 0.5).astype(int), proba

for col in ["Gender", "Geography"]:
    rows = []
    for g, d in res.groupby(col):
        rows.append({col: g, "n": len(d), "actual_churn": d.y.mean(),
                     "flagged": d.pred.mean(),
                     "recall": recall_score(d.y, d.pred),
                     "precision": precision_score(d.y, d.pred)})
    print(f"\nFairness by {col}:\n", pd.DataFrame(rows).round(3).to_string(index=False))

# ---- 3) Business lift: how many churners sit in the riskiest customers ----
res = res.sort_values("proba", ascending=False).reset_index(drop=True)
total = res.y.sum()
print("\nChurners captured by contacting the top X% riskiest customers:")
for pct in (10, 20, 30):
    k = int(len(res) * pct / 100)
    cap = res.y.iloc[:k].sum() / total
    print(f"  Top {pct}%: {cap:.1%} of churners (lift {cap / (pct / 100):.1f}x)")