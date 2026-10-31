"""XGBoost — [Paper: Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System", KDD 2016, DOI 10.1145/2939672.2939785]
[Code: github.com/dmlc/xgboost]
[Reference use: part21_fresh_multitask]

Run:  python models/xgboost_model.py --smoke
"""
from __future__ import annotations
import time
import numpy as np

from ml_common import RANDOM_SEED, regression_metrics, classification_metrics, smoke_argparser, run_smoke_report
import featurize

def build(kind="reg", seed=RANDOM_SEED):
    from xgboost import XGBRegressor, XGBClassifier
    base = dict(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.8,
                reg_lambda=1.0, tree_method="hist", random_state=seed, n_jobs=-1)
    return (XGBRegressor(**base) if kind == "reg" else XGBClassifier(**base))


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
    reg = build(kind="reg").fit(X[tr], y_reg[tr], eval_set=[(X[va], y_reg[va])], verbose=False)
    reg_m = regression_metrics(y_reg[va], reg.predict(X[va]))
    cls = build(kind="cls").fit(X[tr], y_cls[tr], eval_set=[(X[va], y_cls[va])], verbose=False)
    cls_m = classification_metrics(y_cls[va], np.log(np.clip(cls.predict_proba(X[va]), 1e-12, 1.0)))
    run_smoke_report("xgboost", {"reg_val": reg_m, "cls_val": cls_m}, runtime=time.time() - t0)


if __name__ == "__main__":
    main()

