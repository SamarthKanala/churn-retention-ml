from fastapi.testclient import TestClient

from api import app

client = TestClient(app)

BASE = {"CreditScore": 650, "Geography": "France", "Gender": "Male", "Age": 40,
        "Tenure": 5, "Balance": 80000.0, "NumOfProducts": 1, "HasCrCard": 1,
        "IsActiveMember": 1, "EstimatedSalary": 90000.0}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_valid():
    r = client.post("/predict", json=BASE)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_band"] in {"Low", "Medium", "High"}


def test_predict_rejects_invalid_age():
    r = client.post("/predict", json={**BASE, "Age": 5})
    assert r.status_code == 422


def test_risk_ordering_is_sensible():
    risky = {**BASE, "Age": 55, "Geography": "Germany", "Gender": "Female",
             "IsActiveMember": 0, "NumOfProducts": 1}
    safe = {**BASE, "Age": 28, "IsActiveMember": 1, "NumOfProducts": 2}
    p_risky = client.post("/predict", json=risky).json()["churn_probability"]
    p_safe = client.post("/predict", json=safe).json()["churn_probability"]
    assert p_risky > p_safe