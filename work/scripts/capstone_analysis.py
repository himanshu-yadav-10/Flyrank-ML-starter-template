"""
Capstone analysis (Lane 2 - Refresh / Content Opportunity Scoring).

Rebuilds the validated Gradient-Boosting model from the week-5 notebook on the
same client-held-out split, then produces the ranked opportunity queue with
reason codes and every figure the deployed paper embeds.

Everything here is decision-support. The label (is_declining_label) is observed
from trend_direction == 'down'; the model scores P(declining | signals) and the
reason codes describe the observed feature profile that correlates with that
probability, not a causal mechanism.

Run:  python work/scripts/capstone_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from ml_utils import MODEL_NUMERIC_FEATURES, MODEL_CATEGORICAL_FEATURES, precision_at_k

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = REPO / "data" / "processed" / "refresh_feature_vector.csv"
FIG = REPO / "work" / "figures"
OUT = REPO / "work" / "outputs"
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

BRAND = "#6F4E7C"
SECOND = "#2E86AB"


def main() -> None:
    df = pd.read_csv(DATA)
    print(f"Loaded {len(df):,} rows x {len(df.columns)} cols")

    # ---------------- grouped split by client (identical to week 5) -------------
    rng = np.random.default_rng(RANDOM_STATE)
    clients = df["client_id"].drop_duplicates().to_numpy()
    shuffled = rng.permutation(clients)
    n_test = max(1, int(round(len(shuffled) * 0.2)))
    test_clients = set(shuffled[:n_test])
    test_mask = df["client_id"].isin(test_clients).to_numpy()
    train_mask = ~test_mask

    num = [c for c in MODEL_NUMERIC_FEATURES if c in df.columns]
    Xnum = df[num].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    cat = [c for c in MODEL_CATEGORICAL_FEATURES if c in df.columns]
    Xcat = pd.get_dummies(df[cat].fillna("unknown").astype(str), prefix=cat, dtype=float)
    X = pd.concat([Xnum.reset_index(drop=True), Xcat.reset_index(drop=True)], axis=1)
    y = df["is_declining_label"].astype(int)
    yte = y[test_mask].reset_index(drop=True)
    Xtr, Xte = X[train_mask].reset_index(drop=True), X[test_mask].reset_index(drop=True)
    ytr = y[train_mask].reset_index(drop=True)

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve

    model = GradientBoostingClassifier(max_depth=3, n_estimators=150, learning_rate=0.1, random_state=RANDOM_STATE)
    model.fit(Xtr, ytr)
    s = model.predict_proba(Xte)[:, 1]

    base_rate = float(yte.mean())
    metrics = {
        "base_rate_test_fold": base_rate,
        "precision_at_20": precision_at_k(yte, s, 20),
        "precision_at_50": precision_at_k(yte, s, 50),
        "precision_at_100": precision_at_k(yte, s, 100),
        "roc_auc": float(roc_auc_score(yte, s)),
        "average_precision": float(average_precision_score(yte, s)),
    }
    w4 = _week4_baseline(df, train_mask, test_mask)
    metrics["w4_precision_at_50"] = precision_at_k(yte, w4, 50)
    metrics["seed"] = RANDOM_STATE
    metrics["n_train_rows"] = int(train_mask.sum())
    metrics["n_test_rows"] = int(test_mask.sum())
    metrics["n_train_clients"] = int(df.loc[train_mask, "client_id"].nunique())
    metrics["n_test_clients"] = int(len(test_clients))
    print("Key metrics (gradient boosting, client-holdout test):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    (OUT / "capstone_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))

    # ---------------- figures ----------------
    _fig_topk(metrics, base_rate, w4, s, yte)
    _fig_feature_importance(model, X.columns)
    _fig_pr_curve(yte, s, base_rate)
    _fig_roc_curve(yte, s)

    # ---------------- ranked opportunity queue with reason codes ----------------
    queue = _build_queue(df, s, test_mask)
    queue_path = OUT / "capstone_opportunity_queue.csv"
    queue.head(2000).to_csv(queue_path, index=False)
    print(f"\nWrote ranked queue (top 2000 rows) -> {queue_path.relative_to(REPO)}")

    # top-10 reason code tally for the paper
    tally = queue["reason_code"].value_counts()
    print("Top-10 reason codes:")
    print(tally.head(10).to_string())
    (OUT / "capstone_reason_codes.json").write_text(
        json.dumps({"reason_codes": tally.head(12).to_dict()}, indent=2, sort_keys=True))


def _week4_baseline(df: pd.DataFrame, train_mask: np.ndarray, test_mask: np.ndarray) -> np.ndarray:
    train_logsorted = np.sort(np.log1p(
        pd.to_numeric(df.loc[train_mask, "impressions_90d"], errors="coerce").fillna(0).to_numpy()))
    u = np.linspace(0.0, 1.0, len(train_logsorted))
    log_imp = np.log1p(pd.to_numeric(df["impressions_90d"], errors="coerce").fillna(0).to_numpy())
    vis = np.interp(log_imp, train_logsorted, u)
    stale = (pd.to_numeric(df["days_since_last_update"], errors="coerce").fillna(0) >= 180).astype(float)
    visible = (pd.to_numeric(df["impressions_90d"], errors="coerce").fillna(0) >= 500).astype(float)
    w4 = stale * visible * vis
    return w4[test_mask]


def _fig_topk(metrics: dict, base_rate: float, w4: np.ndarray, s: np.ndarray, yte: pd.Series) -> None:
    ks = [10, 20, 30, 50, 75, 100, 150, 200]
    model_p = [precision_at_k(yte, s, k) for k in ks]
    w4_p = [precision_at_k(yte, w4, k) for k in ks]
    rand_p = [base_rate] * len(ks)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, model_p, marker="o", color=BRAND, label="Gradient boosting (model)")
    ax.plot(ks, w4_p, marker="s", color=SECOND, label="Week-4 rule (baseline)")
    ax.plot(ks, rand_p, "k--", label="Random ordering (base rate %.2f)" % base_rate)
    ax.set_xlabel("Top-K pages reviewed, by score rank")
    ax.set_ylabel("Precision@K (fraction that are declining)")
    ax.set_title("Model vs baseline on the same held-out clients")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "precision_at_k.png", dpi=150)
    plt.close(fig)


def _fig_feature_importance(model, columns) -> None:
    imp = pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False)
    top = imp.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index, top.values, color=BRAND)
    ax.set_xlabel("Gini feature importance")
    ax.set_title("What the model leans on (top 10)")
    fig.tight_layout()
    fig.savefig(FIG / "feature_importance.png", dpi=150)
    plt.close(fig)


def _fig_pr_curve(yte, s, base_rate) -> None:
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(yte, s)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color=BRAND, lw=2)
    ax.axhline(base_rate, color="k", ls="--", label="Base rate %.2f" % base_rate)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (AP=%.3f)" % average_precision_score(yte, s))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "pr_curve.png", dpi=150)
    plt.close(fig)


def _fig_roc_curve(yte, s) -> None:
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(yte, s)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=BRAND, lw=2, label="AUC=%.3f" % auc(fpr, tpr))
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (held-out clients)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "roc_curve.png", dpi=150)
    plt.close(fig)


def _build_queue(df: pd.DataFrame, s: np.ndarray, test_mask: np.ndarray) -> pd.DataFrame:
    """
    Rank pages on P(declining) and attach a decision-support reason code from the
    observed feature profile. Codes are heuristic descriptions of the profile that
    correlates with predicted decline - not a claim that the feature caused it.
    """
    test_df = df[test_mask].copy().reset_index(drop=True)
    test_df["opportunity_score"] = s

    def code(row) -> str:
        p = row["opportunity_score"]
        stale = row["days_since_last_update"] >= 180
        visible = row["impressions_90d"] >= 500
        deep = row["avg_position"] > 20 or row["avg_position"] == 0
        low_ctr = row["ctr"] < 1.0 if row["impressions_90d"] >= 100 else False
        low_reach = row["impressions_90d"] < 100
        mature = row["content_age_days"] >= 300
        # diminishing reach: page still logs impressions but on few distinct days
        reach_loss = (row["impressions_90d"] >= 100) and (row["days_with_impressions"] <= 30)
        if p >= 0.6:
            if stale and visible:
                return "stale_visible_decline"
            if reach_loss:
                return "reach_loss"
            if low_ctr and visible:
                return "visible_low_ctr"
            if deep:
                return "position_loss"
            if mature:
                return "mature_content_decline"
            if low_reach:
                return "low_reach_review"
            return "review_decline_risk"
        if p >= 0.45:
            return "monitor_borderline"
        return "monitor_stable"

    test_df["reason_code"] = test_df.apply(code, axis=1)

    def action(c: str) -> str:
        mapping = {
            "stale_visible_decline": "refresh",
            "reach_loss": "review",
            "visible_low_ctr": "metadata_review",
            "position_loss": "review",
            "mature_content_decline": "refresh_or_rewrite",
            "low_reach_review": "review_low_priority",
            "review_decline_risk": "review",
            "monitor_borderline": "monitor",
            "monitor_stable": "monitor",
        }
        return mapping[c]

    test_df["action"] = test_df["reason_code"].map(action)
    test_df = test_df.sort_values("opportunity_score", ascending=False).reset_index(drop=True)
    test_df.insert(0, "opportunity_rank", np.arange(1, len(test_df) + 1))
    return test_df


if __name__ == "__main__":
    main()
