"""Shared train/eval helpers: covers both regression (pIC50) and classification (Active/Inactive).

Classification output rule (project convention): softmax probabilities
(sklearn `predict_proba`, torch `torch.softmax(logits, dim=1)`).
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np

RANDOM_SEED = 57


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"RMSE": rmse, "MAE": float(mean_absolute_error(y_true, y_pred)), "R2": float(r2_score(y_true, y_pred))}


def classification_metrics(y_true: np.ndarray, y_pred_logits: np.ndarray) -> dict:
    """y_pred_logits: (n,2) array; softmax applied here once for the project-wide prob convention."""
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score, accuracy_score
    ex = np.exp(y_pred_logits - y_pred_logits.max(axis=1, keepdims=True))
    probs = (ex / ex.sum(axis=1, keepdims=True))[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "AUC": float(roc_auc_score(y_true, probs)),
        "AAP": float(average_precision_score(y_true, probs)),
        "balanced_acc": float(balanced_accuracy_score(y_true, preds)),
        "accuracy": float(accuracy_score(y_true, preds)),
    }


def process_in_chunks(model, X, chunk=20000):
    """Predict a large library in memory-safe slices (returns np.array)."""
    n = X.shape[0]
    out = []
    for i in range(0, n, chunk):
        out.append(np.asarray(model.predict_proba(X[i:i + chunk])))
    return np.vstack(out)


def bootstrap_ci(metric_fn, y_true, y_pred, n_boot=1000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            continue
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def smoke_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny subset, single pass")
    ap.add_argument("--train-csv", default=None)
    return ap


def run_smoke_report(model_name, metrics, extras=None, runtime=0.0):
    print(json.dumps({"model": model_name, "metrics": metrics, "extras": extras or {}, "runtime_s": round(runtime, 2)}))
