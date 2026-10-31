"""DeepChem GCN — [Method: Ramsundar et al., deepchem/deepchem — github.com/deepchem/deepchem]
[Origin in this repo: Arjun sir Part_9_GNN_DeepChem.ipynb ([ARJUN COPY]) — GCN from scratch on RDKit-converted graphs]

GUARDED: skipped with a clear message if `deepchem` is missing (same behaviour
as every other optional heavy dependency; see docs/ERRORS_AND_FIXES.md).

Run:  python models/deepchem_gcn_model.py --smoke
"""
from __future__ import annotations
import argparse

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--task", default="cls", choices=["cls", "reg"])
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_known_args(argv)[0]
    try:
        import deepchem as dc
        from deepchem.models import GCNModel
    except Exception:
        print("[skipped] deepchem_gcn_model: deepchem not installed (pip install deepchem; "
              "code github.com/deepchem/deepchem — Arjun sir Part 9 origin)")
        return

    import numpy as np, sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "..", "01_traditional_ml", "models"))
    import featurize as fz
    df = fz.load_training_table()
    smiles = fz.smiles_column(df).tolist()
    y_cls = fz.label_encode(df)
    y_reg = df["pIC50"].astype(float).to_numpy()
    tr, va, te = fz.scaffold_split(smiles, y_reg, y_cls)
    if args.smoke:
        tr, va, te = tr[:80], va[:60], te[:40]

    featurizer = dc.feat.ConvMolFeaturizer()
    datasets = []
    for idx in (tr, va, te):
        yy = (y_cls[idx] if args.task == "cls" else y_reg[idx].reshape(-1, 1))
        datasets.append(dc.data.NumpyDataset(X=np.array(smiles)[idx], y=yy, ids=np.array(smiles)[idx]))
        datasets[-1].X = featurizer.featurize(datasets[-1].X)
    mode = ("classification" if args.task == "cls" else "regression")
    model = GCNModel(mode=mode, n_tasks=1, learning_rate=1e-3, batch_size=32, epochs=args.epochs)
    model.fit(datasets[0], nb_epoch=args.epochs)
    from sklearn.metrics import roc_auc_score, mean_squared_error
    pred = model.predict(datasets[1])
    if args.task == "cls":
        prob = pred[:, 1] if pred.ndim == 2 else pred.ravel()
        print(f"[DeepChem-GCN] val AUC={roc_auc_score(datasets[1].y.reshape(-1).astype(int), prob):.4f}")
    else:
        print(f"[DeepChem-GCN] val RMSE={float(np.sqrt(mean_squared_error(datasets[1].y.reshape(-1), pred.reshape(-1)))):.4f}")


if __name__ == "__main__":
    main()

