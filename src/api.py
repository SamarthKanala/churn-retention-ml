from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"
model = joblib.load(MODEL_PATH)

NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
           "IsActiveMember", "EstimatedSalary", "BalanceSalaryRatio", "ZeroBalance",
           "TenureByAge", "InactiveWithBalance"]
CATEGORICAL = ["Geography", "Gender", "AgeGroup", "ProductGroup"]

app = FastAPI(title="Churn Prediction API")


class Customer(BaseModel):
    CreditScore: int = Field(ge=300, le=900)
    Geography: Literal["France", "Germany", "Spain"]
    Gender: Literal["Female", "Male"]
    Age: int = Field(ge=18, le=100)
    Tenure: int = Field(ge=0, le=50)
    Balance: float = Field(ge=0)
    NumOfProducts: int = Field(ge=1, le=4)
    HasCrCard: int = Field(ge=0, le=1)
    IsActiveMember: int = Field(ge=0, le=1)
    EstimatedSalary: float = Field(ge=0)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same feature logic as src/features.py (PySpark), reimplemented for single-row serving."""
    df = df.copy()
    df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["ZeroBalance"] = (df["Balance"] == 0).astype(int)
    df["TenureByAge"] = df["Tenure"] / df["Age"]
    df["AgeGroup"] = pd.cut(
        df["Age"], [-1, 29, 39, 49, 59, 200],
        labels=["<30", "30-39", "40-49", "50-59", "60+"],
    ).astype(str)
    df["ProductGroup"] = df["NumOfProducts"].map(
        lambda n: "1" if n == 1 else "2" if n == 2 else "3+")
    df["InactiveWithBalance"] = (
        (df["IsActiveMember"] == 0) & (df["Balance"] > 0)).astype(int)
    return df


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    df = add_features(pd.DataFrame([customer.model_dump()]))
    prob = float(model.predict_proba(df[NUMERIC + CATEGORICAL])[0, 1])
    band = "High" if prob >= 0.5 else "Medium" if prob >= 0.25 else "Low"
    return {"churn_probability": round(prob, 4),
            "predicted_churn": prob >= 0.5,
            "risk_band": band}