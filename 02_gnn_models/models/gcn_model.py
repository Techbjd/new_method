"""GCN — [Paper: Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017, arXiv:1609.02907]
[Faithful replication source: github.com/HFooladi/GNNs-For-Chemists (notebook 06 style; 9-dim feats, 80/10/10 split seed 42, batch 32, BCEWithLogitsLoss)]
[Method origin in this repo: Arjun sir Part 9 DeepChem GCN from scratch (deepchem/deepchem)]
[Replication in this repo: Part_9 train_gcn_hfooladi.py; here re-cut as a single standalone file]

Run:  python models/gcn_model.py --smoke
"""
from __future__ import annotations
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

import graph_featurizer as gf
import dl_utils


class GCN(nn.Module):
    def __init__(self, num_node_features: int = 78, hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(num_node_features if i == 0 else hidden_dim, hidden_dim)
                                    for i in range(3)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(3)])
        self.dropout = float(dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Dropout(self.dropout),
                                  nn.Linear(64, num_classes))

    def forward(self, data) -> Tensor:
        x, ei, bat = data.x.float(), data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = torch.relu(bn(conv(x, ei)))
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat([global_mean_pool(x, bat), global_max_pool(x, bat)], dim=1)
        return self.head(x)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--task", default="cls", choices=["cls", "reg"])
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_known_args(argv)[0]

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "..", "01_traditional_ml", "models"))
    import featurize as fz
    df = fz.load_training_table()
    smiles = fz.smiles_column(df).tolist()
    y_cls = fz.label_encode(df)
    y_reg = df["pIC50"].astype(float).to_numpy()
    tr, va, te = fz.scaffold_split(smiles, y_reg, y_cls)
    if args.smoke:
        tr, va, te = tr[:60], va[:40], te[:30]
    graphs, dropped = gf.dataset_from_smiles(smiles, labels=(y_cls if args.task == "cls" else y_reg))
    tl, vl, el = dl_utils.make_loaders(graphs, tr, va, te)
    dl_utils.set_seed()
    model = GCN(num_node_features=graphs[0].x.shape[1]).to(dl_utils.DEVICE)
    t0 = time.time()
    model, hist = dl_utils.train_loop(model, tl, vl, task=args.task, epochs=args.epochs)
    metrics = dl_utils.evaluate(model, vl, task=args.task)
    print(f"[GCN] dropped graphs: {dropped} | val: {metrics} | val epochs: {len(hist)} | {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

