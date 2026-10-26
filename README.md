# Lending Risk Scoring + Policy Knowledge (RAG) Platform

An end-to-end reference implementation of an **ML risk-scoring service with explainability** and a **retrieval-augmented policy Q&A service**, behind one FastAPI app with audit logging.

> Independent portfolio project. Built on **synthetic data** (150K generated applicants and 6 invented policy snippets) to demonstrate the architecture and engineering practices I use in production fintech ML work. It contains no employer code or data.

## What it demonstrates
| Area | Implementation |
|---|---|
| Feature engineering | credit, utilization, repayment, account-history features from 150K records (`app/data.py`) |
| Model selection | XGBoost vs LightGBM tuned with **Optuna**, class weighting, **threshold calibration** under a review-volume cap (`app/train.py`) |
| Explainability | **SHAP** top-3 reason indicators per decision (`/explanations`) |
| ML lifecycle | **MLflow** experiment tracking per model, metrics/params logged, best model persisted as a versioned artifact |
| RAG | ingestion → keyword (BM25) + semantic (TF-IDF cosine) + hybrid search, metadata filter, title rerank, grounded answers with citations (`app/rag.py`); optional **Azure OpenAI** generation via env vars |
| Retrieval evaluation | hit-rate@k and MRR on a labelled set, logged to MLflow (`scripts/eval_retrieval.py`) |
| API | FastAPI + Pydantic contracts: `/risk-score`, `/explanations`, `/knowledge/search`, `/health` |
| Persistence & caching | SQLAlchemy audit table (SQLite here, PostgreSQL by URL swap), in-process cache (Redis in prod) |
| Observability | request-id tracing middleware, structured JSON logs, full audit trail of inputs/outputs |
| Delivery | PyTest suite, Dockerfile, GitHub Actions CI (tests + image build) |

## Measured results (this repo, synthetic data)
| Model | PR-AUC | Recall @ calibrated threshold | Precision | Review rate |
|---|---|---|---|---|
| XGBoost (selected) | 0.567 | 0.397 | 0.599 | 18.8% |
| LightGBM | 0.567 | 0.415 | 0.591 | 19.9% |

Retrieval on the tiny demo corpus: keyword, semantic and hybrid all score hit-rate@1 = MRR = 0.875 (corpus is too small to separate them; swap in a real corpus and the eval script will).

## Run it
```bash
pip install -r requirements.txt
python -m app.train            # trains, tunes, logs to ./mlruns, saves artifacts/
pytest -q                      # 5 tests
uvicorn app.main:app --reload  # http://localhost:8000/docs
mlflow ui                      # compare runs
```
Example:
```bash
curl -s localhost:8000/risk-score -H 'content-type: application/json' -d '{"credit_score":620,"utilization":0.9,"on_time_ratio":0.7,"account_age_months":24,"recent_inquiries":5,"debt_to_income":0.5,"late_30d_count":3,"income_k":40}'
curl -s localhost:8000/knowledge/search -H 'content-type: application/json' -d '{"query":"adverse action notice timing"}'
```

## Production notes
Azure mapping: Azure ML registry for models, Azure AI Search for the index, Azure OpenAI for generation, AKS + Key Vault for deployment/secrets, Azure Monitor for telemetry. Not included here to keep the repo runnable offline.
