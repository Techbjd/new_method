"""Extra Trees — [Paper: Geurts et al., "Extremely Randomized Trees", Machine Learning 63:3–42, 2006, DOI 10.1007/s10994-006-6226-1]
[Reference use: part21_fresh_multitask model comparison]

Run:  python models/extra_trees_model.py --smoke
"""
from __future__ import annotations
import time
import numpy as np

from ml_common import RANDOM_SEED, regression_metrics, classification_metrics, smoke_argparser, run_smoke_report
import featurize

def build(kind="reg", seed=RANDOM_SEED, n_estimators=500):
    from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
    base = dict(n_estimators=n_estimators, max_features="sqrt", n_jobs=-1, random_state=seed)
    return (ExtraTreesRegressor(**base) if kind == "reg" else ExtraTreesClassifier(**base))


def main(argv=None):
    args = smoke_argparser().parse_known_args(argv)[0]
    df = featurize.load_training_table(args.train_csv)
    smiles = featurize.smiles_column(df).tolist()
    y_cls = featurize.label_encode(df)
    y_reg = df["pIC50"].astype(float).to_numpy()
    tr, va, te = featurize.scaffold_split(smiles, y_reg, y_cls)
    if args.smoke:
        tr, va, te = tr[:80], va[:60], te[:40]
    X = featurize.smiles_to_fingerprints(smiles, family="maccs")
    t0 = time.time()
    reg = build(kind="reg").fit(X[tr], y_reg[tr])
    reg_m = regression_metrics(y_reg[va], reg.predict(X[va]))
    cls = build(kind="cls").fit(X[tr], y_cls[tr])
    cls_m = classification_metrics(y_cls[va], np.log(np.clip(cls.predict_proba(X[va]), 1e-12, 1.0)))
    run_smoke_report("extra_trees", {"reg_val": reg_m, "cls_val": cls_m}, runtime=time.time() - t0)


if __name__ == "__main__":
    main()

