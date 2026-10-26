"""Train XGBoost vs LightGBM, tune with Optuna, calibrate a threshold, log everything to MLflow."""
import json, pathlib, sys
import joblib, mlflow, numpy as np, optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from .data import FEATURES, make_dataset

ART = pathlib.Path("artifacts")


def pick_threshold(y, p, max_review_rate=0.20):
    """Lowest-cost threshold: maximise recall while keeping review volume under a cap."""
    best_t, best_r = 0.5, -1
    for t in np.linspace(0.05, 0.9, 86):
        flagged = (p >= t).mean()
        if flagged <= max_review_rate:
            r = recall_score(y, p >= t)
            if r > best_r:
                best_t, best_r = float(t), r
    return best_t


def main(n=150_000, trials=10):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    df = make_dataset(n)
    X, y = df[FEATURES], df.high_risk
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=1)
    spw = (ytr == 0).sum() / max(1, (ytr == 1).sum())  # class weighting

    def build(kind, trial):
        if kind == "xgb":
            return XGBClassifier(n_estimators=trial.suggest_int("n", 100, 300), max_depth=trial.suggest_int("d", 3, 7),
                                 learning_rate=trial.suggest_float("lr", 0.03, 0.2, log=True),
                                 scale_pos_weight=min(spw, 5), eval_metric="aucpr", n_jobs=-1)
        return LGBMClassifier(n_estimators=trial.suggest_int("n", 100, 300), num_leaves=trial.suggest_int("nl", 15, 63),
                              learning_rate=trial.suggest_float("lr", 0.03, 0.2, log=True),
                              class_weight="balanced", verbose=-1, n_jobs=-1)

    mlflow.set_experiment("lending-risk")
    results = {}
    for kind in ("xgb", "lgbm"):
        def objective(trial):
            m = build(kind, trial).fit(Xtr, ytr)
            return average_precision_score(yva, m.predict_proba(Xva)[:, 1])
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials)
        with mlflow.start_run(run_name=kind):
            class Fixed:  # replay best params
                def __init__(s, p): s.p = p
                def suggest_int(s, k, a, b): return s.p[k]
                def suggest_float(s, k, a, b, log=False): return s.p[k]
            model = build(kind, Fixed(study.best_params)).fit(Xtr, ytr)
            pv = model.predict_proba(Xva)[:, 1]
            thr = pick_threshold(yva, pv)
            pt = model.predict_proba(Xte)[:, 1]
            m = {"pr_auc": average_precision_score(yte, pt), "recall": recall_score(yte, pt >= thr),
                 "precision": precision_score(yte, pt >= thr), "review_rate": float((pt >= thr).mean()), "threshold": thr}
            mlflow.log_params(study.best_params); mlflow.log_metrics(m)
            results[kind] = (m, model, thr)
    best = max(results, key=lambda k: results[k][0]["pr_auc"])
    m, model, thr = results[best]
    ART.mkdir(exist_ok=True)
    joblib.dump({"model": model, "threshold": thr, "features": FEATURES, "kind": best}, ART / "risk_model.joblib")
    (ART / "metrics.json").write_text(json.dumps({k: v[0] for k, v in results.items()} | {"selected": best}, indent=2))
    print(json.dumps({k: v[0] for k, v in results.items()} | {"selected": best}, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150_000, int(sys.argv[2]) if len(sys.argv) > 2 else 10)
