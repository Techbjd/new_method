"""GIN — [Paper: Xu et al., "How Powerful are Graph Neural Networks?", ICLR 2019, arXiv:1810.00826]
[Pretraining flavor: MolCLR — Wang et al., NeurIPS 2020 — github.com/yuyangw/MolCLR (28/63 layers matched; Part_10 retrained from scratch)]
[Faithful replication source: github.com/HFooladi/GNNs-For-Chemists (notebook 06 GIN)]

Run:  python models/gin_model.py --smoke
"""
from __future__ import annotations
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import scatter
from torch_geometric.nn import global_mean_pool, global_max_pool

import graph_featurizer as gf
import dl_utils


class GINEncoder(nn.Module):
    """MolCLR-style GIN: 5 linear layers, hidden 300, learnable eps, custom neighbor aggregation."""
    def __init__(self, num_node_features: int = 78, hidden: int = 300, num_layers: int = 5):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([nn.Linear(num_node_features if i == 0 else hidden, hidden)
                                     for i in range(num_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(num_layers)])
        self.eps = nn.Parameter(torch.zeros(num_layers))

    def forward(self, data) -> Tensor:
        x, ei, bat = data.x.float(), data.edge_index, data.batch
        h = x
        for li in range(self.num_layers):
            agg = scatter(h[ei[0]], ei[1], dim=0, dim_size=h.size(0), reduce="sum")  # neighbor sum
            out = self.layers[li]((1 + self.eps[li]) * h + agg)
            h = F.relu(self.bns[li](out))
        return torch.cat([global_mean_pool(h, bat), global_max_pool(h, bat)], dim=1)  # 600


class GINNet(nn.Module):
    """Full net: GINEncoder + task head returning num_classes logits (cls) or 1 (reg)."""
    def __init__(self, num_node_features: int = 78, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.encoder = GINEncoder(num_node_features)
        out_dim = num_classes
        self.head = nn.Sequential(nn.Linear(600, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, out_dim))

    def forward(self, data) -> Tensor:
        return self.head(self.encoder(data))


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
    nf = graphs[0].x.shape[1]
    model = GINNet(num_node_features=nf).to(dl_utils.DEVICE)
    t0 = time.time()
    model, hist = dl_utils.train_loop(model, tl, vl, task=args.task, epochs=args.epochs)
    metrics = dl_utils.evaluate(model, vl, task=args.task)
    print(f"[GIN] dropped: {dropped} | val: {metrics} | {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

