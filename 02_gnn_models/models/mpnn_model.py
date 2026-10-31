"""D-MPNN (Chemprop-style directed message passing) — [Paper: Yang et al., "Analyzing Learned Molecular Representations for Property Prediction", JCIM 2019, 59(8):3370–3388, DOI 10.1021/acs.jcim.9b00237]
[Official code: github.com/chemprop/chemprop (this file is a compact PyG re-implementation; the full chemprop trainer remains Colab-only)]
[Origin in this repo: part21 mpnn model on RDKit graphs]

Run:  python models/mpnn_model.py --smoke
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


class DMPNN(nn.Module):
    """Chemprop-style directed-edge message passing.

    My graph_featurizer emits bonds as adjacent pairs [i->j, j->i], so under PyG
    batch concatenation the reverse of directed edge k is always k ^ 1. The
        h_e^{t+1} = ReLU(h_e^0 + W_h (Σ_f≠rev f→src(e) h_f^t)) .
    """
    def __init__(self, num_node_features: int = 78, hidden: int = 128, depth: int = 3,
                 num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.depth = depth
        self.W_i = nn.Linear(num_node_features, hidden)
        self.W_h = nn.Linear(hidden, hidden)
        self.W_o = nn.Linear(hidden, hidden)
        self.dropout = float(dropout)
        out_dim = num_classes
        self.head = nn.Sequential(nn.Linear(hidden * 2, 64), nn.ReLU(),
                                  nn.Dropout(self.dropout), nn.Linear(64, out_dim))

    def forward(self, data) -> Tensor:
        x, ei, bat = data.x.float(), data.edge_index, data.batch
        src, dst = ei[0], ei[1]
        n_nodes = data.num_nodes
        rev = torch.arange(src.size(0), device=src.device) ^ 1  # adjacent-pair layout
        e_state = self.W_i(x)[src]                              # h_e^0 per directed edge
        m = e_state.clone()
        for _ in range(self.depth):
            in_sum = scatter(m, dst, dim=0, dim_size=n_nodes, reduce="sum")  # inbox per node
            m_sum = in_sum[src] - m[rev]                        # exclude the reverse edge
            m = nn.functional.dropout(F.relu(e_state + self.W_h(m_sum)),
                                      p=self.dropout, training=self.training)
        node_msg = scatter(m, dst, dim=0, dim_size=n_nodes, reduce="sum")
        node_h = F.relu(self.W_o(node_msg))
        x = torch.cat([global_mean_pool(node_h, bat), global_max_pool(node_h, bat)], dim=1)
        return self.head(F.relu(x))


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
    model = DMPNN(num_node_features=nf).to(dl_utils.DEVICE)
    t0 = time.time()
    model, hist = dl_utils.train_loop(model, tl, vl, task=args.task, epochs=args.epochs)
    metrics = dl_utils.evaluate(model, vl, task=args.task)
    print(f"[D-MPNN] dropped: {dropped} | val: {metrics} | {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

