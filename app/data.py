"""Synthetic lending data generator (no real customer data is used anywhere in this repo)."""
import numpy as np
import pandas as pd

FEATURES = [
    "credit_score", "utilization", "on_time_ratio", "account_age_months",
    "recent_inquiries", "debt_to_income", "late_30d_count", "income_k",
]


def make_dataset(n: int = 150_000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "credit_score": rng.normal(690, 70, n).clip(300, 850),
        "utilization": rng.beta(2, 4, n),
        "on_time_ratio": rng.beta(8, 1.2, n),
        "account_age_months": rng.gamma(4, 18, n).clip(1, 480),
        "recent_inquiries": rng.poisson(1.4, n),
        "debt_to_income": rng.beta(2.5, 5, n),
        "late_30d_count": rng.poisson(0.5, n),
        "income_k": rng.lognormal(4.2, 0.45, n),
    })
    z = (-3.0 - 0.012 * (df.credit_score - 690) + 2.4 * df.utilization
         - 2.8 * (df.on_time_ratio - 0.88) + 0.25 * df.recent_inquiries
         + 2.0 * df.debt_to_income + 0.45 * df.late_30d_count
         - 0.004 * (df.income_k - 70) - 0.003 * df.account_age_months)
    p = 1 / (1 + np.exp(-z))
    df["high_risk"] = (rng.random(n) < p).astype(int)
    return df
