"""CatBoost — [Paper: Dorogush, Ershov & Gulin, "CatBoost: gradient boosting with categorical features support", arXiv:1810.11363 (2018)]
[Code: github.com/catboost/catboost]
[Reference use: part21_fresh_multitask]

Run:  python models/catboost_model.py --smoke
"""
from __future__ import annotations
import time
import numpy as np

from ml_common import RANDOM_SEED, regression_metrics, classification_metrics, smoke_argparser, run_smoke_report
import featurize

def build(kind="reg", seed=RANDOM_SEED):
    from catboost import CatBoostRegressor, CatBoostClassifier
    base = dict(iterations=400, learning_rate=0.05, depth=6, loss_function=("RMSE" if kind == "reg" else "Logloss"),
                random_seed=seed, verbose=0, thread_count=-1)
    return (CatBoostRegressor(**base) if kind == "reg" else CatBoostClassifier(**base))


def main(argv=None):
    args = smoke_argparser().parse_known_args(argv)[0]
    df = featurize.load_training_table(args.train_csv)
    smiles = featurize.smiles_column(df).tolist()
    y_cls = featurize.label_encode(df)
    y_reg = df["pIC50"].astype(float).to_numpy()
    tr, va, te = featurize.scaffold_split(smiles, y_reg, y_cls)
    if args.smoke:
        tr, va, te = tr[:80], va[:60], te[:40]
    X = featurize.smiles_to_fingerprints(smiles, family="morgan")
    t0 = time.time()
    reg = build(kind="reg").fit(X[tr], y_reg[tr], eval_set=(X[va], y_reg[va]), verbose=0)
    reg_m = regression_metrics(y_reg[va], reg.predict(X[va]))
    cls = build(kind="cls").fit(X[tr], y_cls[tr], eval_set=(X[va], y_cls[va]), verbose=0)
    cls_m = classification_metrics(y_cls[va], np.log(np.clip(cls.predict_proba(X[va]), 1e-12, 1.0)))
    run_smoke_report("catboost", {"reg_val": reg_m, "cls_val": cls_m}, runtime=time.time() - t0)


if __name__ == "__main__":
    main()

