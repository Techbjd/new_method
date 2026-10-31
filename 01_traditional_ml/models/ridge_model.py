"""Ridge (6 RDKit descriptors) — [Paper: Hoerl & Kennard, "Ridge Regression: Biased Estimation...", Technometrics 12(1):55–67, 1970, DOI 10.1080/00401706.1970.10488634]
[Reference use: Part 20 leak-safe linear baseline on selected descriptors]

Run:  python models/ridge_model.py --smoke
"""
from __future__ import annotations
import time
import numpy as np

from ml_common import RANDOM_SEED, regression_metrics, smoke_argparser, run_smoke_report
import featurize

def build(seed=RANDOM_SEED):
    from sklearn.linear_model import Ridge
    return Ridge(alpha=1.0, random_state=seed)


def main(argv=None):
    args = smoke_argparser().parse_known_args(argv)[0]
    df = featurize.load_training_table(args.train_csv)
    smiles = featurize.smiles_column(df).tolist()
    y_cls = featurize.label_encode(df)
    y_reg = df["pIC50"].astype(float).to_numpy()
    tr, va, te = featurize.scaffold_split(smiles, y_reg, y_cls)
    if args.smoke:
        tr, va, te = tr[:80], va[:60], te[:40]
    X = featurize.smiles_to_descriptors(smiles)[:, :6]  # MW, logP, TPSA, rotatable, HBA, HBD
    t0 = time.time()
    reg = build().fit(X[tr], y_reg[tr])
    reg_m = regression_metrics(y_reg[va], reg.predict(X[va]))
    run_smoke_report("ridge_descriptors6", {"reg_val": reg_m}, runtime=time.time() - t0)


if __name__ == "__main__":
    main()

