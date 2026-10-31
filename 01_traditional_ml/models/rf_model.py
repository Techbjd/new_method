"""Random Forest — [Paper: Breiman, "Random Forests", Machine Learning 45:5–32, 2001, DOI 10.1023/A:1010933404324]
[Reference use in this project: Budha et al., Adv. Theory Simul. 2025, DOI 10.1002/adts.202501502 (Morgan r3/512)]
[Code origin: sklearn RandomForest; Arjun sir Parts 6/8 (20-classifier RF catalog + COCONUT screen)]

Run:  python models/rf_model.py --smoke
"""
from __future__ import annotations

import numpy as np
import time

from ml_common import RANDOM_SEED, regression_metrics, classification_metrics, bootstrap_ci, smoke_argparser, run_smoke_report
import featurize

def build(n_estimators=500, seed=RANDOM_SEED, kind="reg"):
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    base = dict(n_estimators=n_estimators, max_features="sqrt", n_jobs=-1, random_state=seed)
    return (RandomForestRegressor(**{**base, "criterion": "squared_error"}) if kind == "reg"
            else RandomForestClassifier(**base))


def main(argv=None):
    args = smoke_argparser().parse_known_args(argv)[0]
    from sklearn.metrics import mean_squared_error, average_precision_score
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
    lo, hi = bootstrap_ci(mean_squared_error, y_reg[te], reg.predict(X[te]), n_boot=300)
    run_smoke_report("random_forest", {"reg_val": reg_m, "cls_val": cls_m}, {"rmse_boot95": [round(lo, 4), round(np.sqrt(hi), 4)]}, time.time() - t0)


if __name__ == "__main__":
    main()

