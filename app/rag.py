"""Hybrid retrieval (BM25-style keyword + TF-IDF cosine + metadata filter + rerank) over policy docs.
The generation step is pluggable: set AZURE_OPENAI_* env vars to use Azure OpenAI, else an extractive answer is returned."""
import math, os, re
from collections import Counter
from dataclasses import dataclass

DOCS = [
    {"id": "pol-001", "dept": "lending", "title": "Debt-to-income limits",
     "text": "Applications with a debt-to-income ratio above 43 percent require manual underwriting review and documented compensating factors."},
    {"id": "pol-002", "dept": "lending", "title": "Utilization and revolving credit",
     "text": "Revolving credit utilization above 75 percent is treated as a risk indicator and must be listed as a reason code on the decision."},
    {"id": "pol-003", "dept": "compliance", "title": "Adverse action notices",
     "text": "When an application is declined or priced adversely, the applicant must receive an adverse action notice listing the principal reasons within 30 days."},
    {"id": "pol-004", "dept": "compliance", "title": "Model explainability",
     "text": "Every automated risk score must be accompanied by feature-level reason indicators so analysts can interpret and challenge the decision."},
    {"id": "pol-005", "dept": "ops", "title": "Audit logging retention",
     "text": "Risk decisions and retrieved policy context are stored in the audit log and retained for seven years for regulatory examination."},
    {"id": "pol-006", "dept": "lending", "title": "Late payment handling",
     "text": "Two or more payments thirty days past due in the last twelve months escalate the application to a senior analyst."},
]
_tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())


@dataclass
class Hit:
    id: str; title: str; dept: str; score: float; text: str


class Index:
    def __init__(self, docs=DOCS):
        self.docs = docs
        self.toks = [_tok(d["title"] + " " + d["text"]) for d in docs]
        self.df = Counter(t for ts in self.toks for t in set(ts))
        self.avg = sum(map(len, self.toks)) / len(self.toks)

    def _bm25(self, q, i, k1=1.5, b=0.75):
        tf, n = Counter(self.toks[i]), len(self.docs)
        s = 0.0
        for t in q:
            if t in tf:
                idf = math.log(1 + (n - self.df[t] + .5) / (self.df[t] + .5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(self.toks[i]) / self.avg))
        return s

    def _tfidf(self, q, i):
        n = len(self.docs); tf = Counter(self.toks[i])
        w = lambda t: tf[t] * math.log((n + 1) / (self.df.get(t, 0) + 1))
        num = sum(w(t) for t in set(q) if t in tf)
        den = math.sqrt(sum(w(t) ** 2 for t in tf)) or 1
        return num / (den * math.sqrt(len(set(q)) or 1))

    def search(self, query, k=3, dept=None, mode="hybrid"):
        q = _tok(query); hits = []
        for i, d in enumerate(self.docs):
            if dept and d["dept"] != dept:
                continue
            kw, sem = self._bm25(q, i), self._tfidf(q, i)
            s = {"keyword": kw, "semantic": sem, "hybrid": 0.6 * kw + 0.4 * sem * 5}[mode]
            title_boost = 0.5 * len(set(q) & set(_tok(d["title"])))  # rerank: title overlap
            hits.append(Hit(d["id"], d["title"], d["dept"], round(s + title_boost, 4), d["text"]))
        return sorted([h for h in hits if h.score > 0], key=lambda h: -h.score)[:k]


def answer(query, hits):
    if not hits:
        return "No grounded answer found in the policy corpus."
    if os.getenv("AZURE_OPENAI_ENDPOINT"):  # optional generation path
        from openai import AzureOpenAI  # pip install openai
        c = AzureOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], api_key=os.environ["AZURE_OPENAI_API_KEY"], api_version="2024-06-01")
        ctx = "\n".join(f"[{h.id}] {h.text}" for h in hits)
        r = c.chat.completions.create(model=os.environ["AZURE_OPENAI_DEPLOYMENT"], messages=[
            {"role": "system", "content": "Answer ONLY from the context and cite ids."}, {"role": "user", "content": f"{ctx}\n\nQ: {query}"}])
        return r.choices[0].message.content
    return f"{hits[0].text} [{hits[0].id}]"


def evaluate(index, labelled, mode="hybrid", k=3):
    """labelled: list of (query, relevant_doc_id). Returns hit-rate@k and MRR."""
    hr = mrr = 0
    for q, rel in labelled:
        ids = [h.id for h in index.search(q, k=k, mode=mode)]
        hr += rel in ids; mrr += 1 / (ids.index(rel) + 1) if rel in ids else 0
    return {"hit_rate_at_k": hr / len(labelled), "mrr": mrr / len(labelled)}
