import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import pytest
from fastapi.testclient import TestClient
from app.rag import Index, evaluate
from scripts.eval_retrieval import LABELLED

GOOD = dict(credit_score=620, utilization=0.9, on_time_ratio=0.7, account_age_months=24, recent_inquiries=5, debt_to_income=0.5, late_30d_count=3, income_k=40)
SAFE = dict(credit_score=790, utilization=0.1, on_time_ratio=0.99, account_age_months=200, recent_inquiries=0, debt_to_income=0.1, late_30d_count=0, income_k=150)


@pytest.fixture(scope="module")
def client():
    if not pathlib.Path("artifacts/risk_model.joblib").exists():
        from app.train import main; main(n=20000, trials=2)
    from app.main import app
    return TestClient(app)


def test_risky_scores_higher_than_safe(client):
    r, s = client.post("/risk-score", json=GOOD).json(), client.post("/risk-score", json=SAFE).json()
    assert r["risk_probability"] > s["risk_probability"]


def test_explanations_have_reasons(client):
    assert len(client.post("/explanations", json=GOOD).json()["reasons"]) == 3


def test_validation_rejects_bad_payload(client):
    assert client.post("/risk-score", json={**GOOD, "credit_score": 1000}).status_code == 422


def test_knowledge_search_grounded(client):
    r = client.post("/knowledge/search", json={"query": "adverse action notice timing"}).json()
    assert r["sources"][0]["id"] == "pol-003" and "pol-003" in r["answer"]


def test_hybrid_not_worse_than_keyword():
    idx = Index()
    assert evaluate(idx, LABELLED, "hybrid", 1)["mrr"] >= evaluate(idx, LABELLED, "keyword", 1)["mrr"] - 1e-9
