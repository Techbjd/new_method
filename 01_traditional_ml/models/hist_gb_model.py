"""Hist Gradient Boosting — [Paper: sklearn HistGradientBoosting (inspired by LightGBM; see Ke et al., LightGBM NeurIPS 2017)]
[Reference use: part21_fresh_multitask]

Run:  python models/hist_gb_model.py --smoke
"""
from __future__ import annotations
import time
import numpy as np

from ml_common import RANDOM_SEED, regression_metrics, classification_metrics, smoke_argparser, run_smoke_report
import featurize

def build(kind="reg", seed=RANDOM_SEED):
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    base = dict(max_iter=400, early_stopping=True, validation_fraction=0.15, random_state=seed)
    return (HistGradientBoostingRegressor(**base) if kind == "reg" else HistGradientBoostingClassifier(**base))


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
    reg = build(kind="reg").fit(X[tr], y_reg[tr])
    reg_m = regression_metrics(y_reg[va], reg.predict(X[va]))
    cls = build(kind="cls").fit(X[tr], y_cls[tr])
    cls_m = classification_metrics(y_cls[va], np.log(np.clip(cls.predict_proba(X[va]), 1e-12, 1.0)))
    run_smoke_report("hist_gb", {"reg_val": reg_m, "cls_val": cls_m}, runtime=time.time() - t0)


if __name__ == "__main__":
    main()

