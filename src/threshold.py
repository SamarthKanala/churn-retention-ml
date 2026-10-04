import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from train import CATEGORICAL, NUMERIC, RANDOM_STATE, TARGET

# ---- Illustrative business assumptions (replace with your own) ----
VALUE_RETAINED = 10000   # value of keeping one customer
OFFER_COST = 500         # cost of contacting one customer with an offer
SAVE_RATE = 0.30         # share of contacted would-be churners the offer actually keeps


def profit(y_true, flagged, cost=OFFER_COST):
    tp = int(((flagged == 1) & (y_true == 1)).sum())
    return tp * SAVE_RATE * VALUE_RETAINED - int(flagged.sum()) * cost


df = pd.read_parquet("data/features.parquet")
X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)  # same split as train.py

pipe = joblib.load("models/xgboost.joblib")

# Out-of-fold probabilities on the training set (no test data used to pick the threshold)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof = cross_val_predict(clone(pipe), X_train, y_train, cv=cv, method="predict_proba")[:, 1]

thresholds = np.arange(0.05, 0.96, 0.01)
ytr = y_train.values


def best_threshold(cost):
    p = [profit(ytr, (oof >= t).astype(int), cost) for t in thresholds]
    return thresholds[int(np.argmax(p))], np.array(p)


best_t, profits = best_threshold(OFFER_COST)
print(f"Best threshold under assumptions: {best_t:.2f}")

# Sensitivity: how does the best threshold move if the offer cost changes?
for c in (250, 500, 1000, 2000):
    print(f"  offer cost {c}: best threshold {best_threshold(c)[0]:.2f}")

# One-time evaluation on the held-out test set
proba = pipe.predict_proba(X_test)[:, 1]
yte = y_test.values
print("\nTest set (2,000 customers):")
for label, t in [("default 0.50", 0.50), (f"tuned {best_t:.2f}", best_t)]:
    flagged = (proba >= t).astype(int)
    per_1000 = profit(yte, flagged) / len(yte) * 1000
    print(f"  {label}: flagged {flagged.mean():.1%}, recall {recall_score(yte, flagged):.3f}, "
          f"precision {precision_score(yte, flagged):.3f}, net benefit per 1,000 customers {per_1000:,.0f}")

# Save the threshold and a plot
with open("models/threshold.json", "w") as f:
    json.dump({"threshold": round(float(best_t), 2),
               "assumptions": {"value_retained": VALUE_RETAINED,
                               "offer_cost": OFFER_COST, "save_rate": SAVE_RATE}}, f, indent=2)

plt.figure(figsize=(7, 4))
plt.plot(thresholds, profits / len(ytr) * 1000)
plt.axvline(best_t, color="red", linestyle="--", label=f"best = {best_t:.2f}")
plt.axvline(0.5, color="gray", linestyle=":", label="default = 0.50")
plt.xlabel("Decision threshold")
plt.ylabel("Net benefit per 1,000 customers (illustrative)")
plt.legend()
plt.tight_layout()
plt.savefig("reports/threshold_profit.png", dpi=150)