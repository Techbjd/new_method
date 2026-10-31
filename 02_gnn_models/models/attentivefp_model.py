"""AttentiveFP — [Paper: Xiong et al., "Pushing the Boundaries of Molecular Representations for Drug Discovery with the Graph Attention Mechanism", J. Med. Chem. 2020, 63(16):8749–8760, DOI 10.1021/acs.jmedchem.9b00959]
[Code: github.com/OpenDrugAI/AttentiveFP; PyG port: torch_geometric.nn.AttentiveFP]
[Origin in this repo: Part_12_AttentiveFP (user; hidden 128, K=2 timesteps)]

Run:  python models/attentivefp_model.py --smoke
"""
from __future__ import annotations
import argparse
import time
import torch
from torch import Tensor
from torch_geometric.nn import AttentiveFP

import graph_featurizer as gf
import dl_utils


class AttentiveFPNet(torch.nn.Module):
    """PyG AttentiveFP (num_timesteps=K iterations of graph-level attention); 6-dim bond feats."""
    def __init__(self, num_node_features: int = 78, edge_dim: int = 6, hidden: int = 128,
                 num_timesteps: int = 2, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.net = AttentiveFP(in_channels=num_node_features, hidden_channels=hidden,
                               out_channels=num_classes, edge_dim=edge_dim, num_layers=3,
                               num_timesteps=num_timesteps, dropout=dropout)

    def forward(self, data) -> Tensor:
        return self.net(data.x.float(), data.edge_index, data.edge_attr.float(), data.batch)


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
    model = AttentiveFPNet(num_node_features=nf).to(dl_utils.DEVICE)
    t0 = time.time()
    model, hist = dl_utils.train_loop(model, tl, vl, task=args.task, epochs=args.epochs)
    metrics = dl_utils.evaluate(model, vl, task=args.task)
    print(f"[AttentiveFP] dropped: {dropped} | val: {metrics} | {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

