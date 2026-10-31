"""GAT + Hybrid GCN-GAT — [Paper: Veličković et al., "Graph Attention Networks", ICLR 2018, arXiv:1710.10903]
[Hybrid reference: HDTI-IC50, J. Comput.-Aided Mol. Des. 2025]
[Origin in this repo: Part_13_HybridGCNGAT (user; ~60k params)]

Run:  python models/gat_model.py --smoke
"""
from __future__ import annotations
import argparse
import time
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool

import graph_featurizer as gf
import dl_utils


class GAT(nn.Module):
    def __init__(self, num_node_features: int = 78, hidden_dim: int = 128, heads: int = 4,
                 num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.l1 = GATConv(num_node_features, hidden_dim, heads=heads, dropout=dropout)
        self.l2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim * heads) for _ in range(2)])
        self.dropout = float(dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim * heads * 2, 64), nn.ReLU(),
                                  nn.Dropout(self.dropout), nn.Linear(64, num_classes))

    def forward(self, data) -> Tensor:
        x, ei, bat = data.x.float(), data.edge_index, data.batch
        x = torch.relu(self.bns[0](self.l1(x, ei)))
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = torch.relu(self.bns[1](self.l2(x, ei)))
        x = torch.cat([global_mean_pool(x, bat), global_max_pool(x, bat)], dim=1)
        return self.head(x)


class HybridGCNGAT(nn.Module):
    """GCN featurizer/propagation path + GAT attention path, fused at readout (Part 13 design)."""
    def __init__(self, num_node_features: int = 78, hidden_dim: int = 64, heads: int = 2,
                 num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gcn1 = GCNConv(num_node_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gat1 = GATConv(num_node_features, hidden_dim, heads=heads, concat=False)
        self.dropout = float(dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim * 4, 32), nn.ReLU(),
                                  nn.Dropout(self.dropout), nn.Linear(32, num_classes))

    def forward(self, data) -> Tensor:
        x, ei, bat = data.x.float(), data.edge_index, data.batch
        g1 = torch.relu(self.gcn1(x, ei))
        g2 = torch.relu(self.gcn2(nn.functional.dropout(g1, p=self.dropout, training=self.training), ei))
        a1 = torch.relu(self.gat1(x, ei))
        pooled = torch.cat([global_mean_pool(g2, bat), global_max_pool(g2, bat),
                            global_mean_pool(a1, bat), global_max_pool(a1, bat)], dim=1)
        return self.head(pooled)


def pick(name="gat"):
    return GAT if name == "gat" else HybridGCNGAT


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--task", default="cls", choices=["cls", "reg"])
    ap.add_argument("--arch", default="gat", choices=["gat", "hybrid"])
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
    cls = pick(args.arch)
    nf = graphs[0].x.shape[1]
    model = cls(num_node_features=nf).to(dl_utils.DEVICE)
    t0 = time.time()
    model, hist = dl_utils.train_loop(model, tl, vl, task=args.task, epochs=args.epochs)
    metrics = dl_utils.evaluate(model, vl, task=args.task)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.arch.upper()}] dropped: {dropped} | params: {n_params} | val: {metrics} | {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

