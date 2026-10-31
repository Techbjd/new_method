"""SMILES Character CNN — [Paper lineage: Gómez-Bombarelli et al., "Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules", ACS Cent. Sci. 2018, 4(2):268–276, DOI 10.1021/acscentsci.7b00572]
[Origin in this repo: part21_fresh_multitask char-CNN on tokenized SMILES]

Run:  python models/smiles_cnn_model.py --smoke
"""
from __future__ import annotations
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import dl_utils

CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
             "[]()#+-=@%/\\.") + ["PAD", "UNK", "BOS", "EOS"]

def encode_smiles(smiles_list, max_len=96):
    """One-hot char codes (L,) ints in [0, len(CHARS)); PAD=0 via lookup offset."""
    vocab = {c: i + 1 for i, c in enumerate(CHARS)}
    out = np.zeros((len(smiles_list), max_len), dtype=np.int64)
    for i, s in enumerate(smiles_list):
        for j, c in enumerate(str(s)[:max_len]):
            out[i, j] = vocab.get(c, vocab["UNK"])
    return torch.tensor(out)

class SMILESCNN(nn.Module):
    def __init__(self, vocab_size=None, emb=96, hidden=128, num_classes=2, dropout=0.2):
        super().__init__()
        vocab_size = vocab_size or (len(CHARS) + 1)
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb, hidden, k, padding=k // 2) for k in (3, 5, 7)])
        self.dropout = float(dropout)
        out_dim = num_classes
        self.head = nn.Sequential(nn.Linear(hidden * 3, 64), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(64, out_dim))

    def forward(self, codes) -> torch.Tensor:
        z = self.emb(codes).transpose(1, 2)
        feats = [F.max_pool1d(F.relu(c(z)), z.size(2)).squeeze(2) for c in self.convs]
        return self.head(torch.cat(feats, dim=1))

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--task", default="cls", choices=["cls", "reg"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
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
        tr, va, te = tr[:80], va[:60], te[:40]
    X = encode_smiles(smiles)
    y = torch.tensor(y_cls if args.task == "cls" else y_reg, dtype=torch.float)
    ds_tr = TensorDataset(X[tr], y[tr]); ds_va = TensorDataset(X[va], y[va])
    dl = lambda d, sh: DataLoader(d, batch_size=args.batch, shuffle=sh)
    train_dl, val_dl = dl(ds_tr, False), dl(ds_va, False)
    dl_utils.set_seed()
    model = SMILESCNN().to(dl_utils.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = (lambda o, t: F.cross_entropy(o, t.long()) if args.task == "cls"
               else F.mse_loss(o.squeeze(-1), t))
    best_val, best_state, bad = (-np.inf if args.task == "cls" else np.inf), None, 0
    key = "AUC" if args.task == "cls" else "RMSE"
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(dl_utils.DEVICE), yb.to(dl_utils.DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            outs, ys = [], []
            for xb, yb in val_dl:
                outs.append(model(xb.to(dl_utils.DEVICE)).detach().cpu())
                ys.append(yb)
        outs = torch.cat(outs); ys = torch.cat(ys)
        if args.task == "cls":
            probs = torch.softmax(outs, dim=1)[:, 1].numpy()
            from sklearn.metrics import roc_auc_score
            val = roc_auc_score(ys.int(), probs)
        else:
            from sklearn.metrics import mean_squared_error
            val = float(np.sqrt(mean_squared_error(ys.numpy(), outs.squeeze(-1).numpy())))
        improved = (val > best_val) if args.task == "cls" else (val < best_val)
        if improved:
            best_val, bad = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    print(f"[SMILES-CNN] val_{key}={best_val:.4f} | {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

