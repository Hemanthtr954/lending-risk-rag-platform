"""Compare keyword / semantic / hybrid retrieval on a labelled set, log to MLflow."""
import mlflow
from app.rag import Index, evaluate
LABELLED = [
    ("when does debt to income need manual review", "pol-001"), ("what is the utilization threshold for reason codes", "pol-002"),
    ("how long to send an adverse action notice", "pol-003"), ("do scores need explanations for analysts", "pol-004"),
    ("how long are decisions kept in audit logs", "pol-005"), ("escalation for late payments", "pol-006"),
    ("declined applicant must be told the reasons", "pol-003"), ("high revolving balance risk indicator", "pol-002"),
]
if __name__ == "__main__":
    idx = Index(); mlflow.set_experiment("retrieval-eval")
    for mode in ("keyword", "semantic", "hybrid"):
        m = evaluate(idx, LABELLED, mode=mode, k=1)
        with mlflow.start_run(run_name=mode): mlflow.log_metrics(m)
        print(mode, m)
