import json, logging, time, uuid
import joblib, numpy as np, shap
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, func
from sqlalchemy.orm import declarative_base, sessionmaker
from .data import FEATURES
from .rag import Index, answer

log = logging.getLogger("risk"); logging.basicConfig(level=logging.INFO)
Base = declarative_base()
engine = create_engine("sqlite:///audit.db", connect_args={"check_same_thread": False})  # swap URL for PostgreSQL
Session = sessionmaker(engine)


class Audit(Base):
    __tablename__ = "audit"
    id = Column(Integer, primary_key=True); request_id = Column(String(36), index=True)
    endpoint = Column(String(40)); payload = Column(Text); result = Column(Text)
    created = Column(DateTime, server_default=func.now())


Base.metadata.create_all(engine)
app = FastAPI(title="Lending Risk & Policy Knowledge Platform")
INDEX = Index(); _bundle = {}; _cache: dict = {}


def bundle():
    if not _bundle:
        b = joblib.load("artifacts/risk_model.joblib"); _bundle.update(b, explainer=shap.TreeExplainer(b["model"]))
    return _bundle


class Applicant(BaseModel):
    credit_score: float = Field(ge=300, le=850); utilization: float = Field(ge=0, le=1)
    on_time_ratio: float = Field(ge=0, le=1); account_age_months: float = Field(ge=0)
    recent_inquiries: int = Field(ge=0); debt_to_income: float = Field(ge=0, le=1)
    late_30d_count: int = Field(ge=0); income_k: float = Field(gt=0)


class Query(BaseModel):
    query: str = Field(min_length=3); dept: str | None = None; mode: str = "hybrid"


@app.middleware("http")
async def trace(request: Request, call_next):
    rid = request.headers.get("x-request-id", str(uuid.uuid4())); t = time.perf_counter()
    resp = await call_next(request); resp.headers["x-request-id"] = rid
    log.info(json.dumps({"rid": rid, "path": request.url.path, "ms": round((time.perf_counter() - t) * 1000, 1), "status": resp.status_code}))
    request.state.rid = rid
    return resp


def audit(endpoint, payload, result, rid=""):
    with Session() as s:
        s.add(Audit(request_id=rid, endpoint=endpoint, payload=json.dumps(payload), result=json.dumps(result))); s.commit()


def _score(a: Applicant):
    b = bundle(); x = np.array([[getattr(a, f) for f in FEATURES]])
    p = float(b["model"].predict_proba(x)[0, 1])
    sv = b["explainer"].shap_values(x); sv = sv[1] if isinstance(sv, list) else sv
    sv = np.asarray(sv).reshape(-1)
    reasons = [{"feature": FEATURES[i], "impact": round(float(sv[i]), 4)} for i in np.argsort(-np.abs(sv))[:3]]
    return {"risk_probability": round(p, 4), "needs_review": p >= b["threshold"], "threshold": b["threshold"], "reasons": reasons}


@app.get("/health")
def health(): return {"status": "ok"}


@app.post("/risk-score")
def risk_score(a: Applicant):
    key = a.model_dump_json()
    if key not in _cache: _cache[key] = _score(a)  # Redis in production; in-process dict here
    audit("/risk-score", a.model_dump(), _cache[key]); return _cache[key]


@app.post("/explanations")
def explanations(a: Applicant):
    r = _score(a); audit("/explanations", a.model_dump(), r)
    return {"reasons": r["reasons"], "narrative": "Top drivers: " + ", ".join(f"{x['feature']} ({x['impact']:+.2f})" for x in r["reasons"])}


@app.post("/knowledge/search")
def knowledge(q: Query):
    if q.mode not in ("hybrid", "keyword", "semantic"): raise HTTPException(422, "bad mode")
    hits = INDEX.search(q.query, dept=q.dept, mode=q.mode)
    out = {"answer": answer(q.query, hits), "sources": [h.__dict__ for h in hits]}
    audit("/knowledge/search", q.model_dump(), out); return out
