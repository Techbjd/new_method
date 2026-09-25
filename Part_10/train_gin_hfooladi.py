"""Part 10 (HFooladi-exact): GIN for MDM2 inhibitor classification.

Faithful replication of HFooladi/GNNs-For-Chemists notebook 06
(`06_GNN_GIN.ipynb`) applied to the local MDM2 dataset:

* Featurization — byte-identical to notebook 04 / ``notebooks/utils/featurization.py``
  (also imported, not redefined, from notebook 05 onward in HFooladi):
  9-dim integer atom features + 3-dim bond features, ``mol_to_graph(smiles)``.
* Model — ``MLP`` + ``GIN`` identical to notebook 06: ``node_encoder`` Linear,
  ``num_layers`` × ``GINConv(MLP, train_eps=True)`` + BatchNorm + ReLU + Dropout,
  ``global_mean_pool``, final prediction ``MLP``; plus ``get_embeddings``.
* Training — ``train_and_evaluate`` identical to notebook 06 (BBBP setup):
  ``BCEWithLogitsLoss``, 80/10/10 random split (seed 42), ``DataLoader``
  batch 32, Adam (lr 1e-3, weight_decay 1e-4), 100 epochs, best-val-AUC
  checkpointing, ROC-AUC reporting.

Dataset: ``Part_10/ro5_properties_filtered.csv`` (645 MDM2 compounds,
``bioactivity_class`` -> y in {0, 1}).

Usage:
    python train_gin_hfooladi.py
    python train_gin_hfooladi.py --epochs 100 --hidden 64 --layers 3
    python train_gin_hfooladi.py --screen input.csv --screen-out preds.csv
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE, "ro5_properties_filtered.csv")
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Featurization — identical to HFooladi notebooks 04/06
# ---------------------------------------------------------------------------
def atom_features(atom):
    """Extract a feature vector for an RDKit atom (shape (9,), dtype long)."""
    return torch.tensor([
        atom.GetAtomicNum(),
        int(atom.GetChiralTag()),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        atom.GetNumRadicalElectrons(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
    ], dtype=torch.long)


def bond_features(bond):
    """Extract a feature vector for an RDKit bond (shape (3,), dtype long)."""
    return torch.tensor([
        int(bond.GetBondTypeAsDouble()),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ], dtype=torch.long)


def mol_to_graph(smiles):
    """Convert a SMILES into a PyTorch Geometric graph (undirected, 2 directed
    edges per bond). Returns Data(x [N,9], edge_index [2,E], edge_attr [E,3])."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()], dim=0)
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append((i, j))
        edge_index.append((j, i))
        edge_attr.append(bond_features(bond))
        edge_attr.append(bond_features(bond))
    if edge_index:  # guard for single-atom molecules (no bonds); HFooladi's
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # tutorial set has none
        edge_attr = torch.stack(edge_attr, dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = None
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)


def mol_to_graph_labeled(smiles, y):
    """mol_to_graph + float label ``y`` of shape [1] for BCEWithLogitsLoss."""
    try:
        d = mol_to_graph(smiles)
    except ValueError:
        return None
    d.y = torch.tensor([[float(y)]], dtype=torch.float)
    return d


# ---------------------------------------------------------------------------
# Model — identical to HFooladi notebook 06 (MLP + GIN)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """MLP used both in GINConv layers and for final predictions."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        if num_layers == 1:
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.batch_norms[i](x)
            x = F.relu(x)
        x = self.lins[-1](x)
        return x


class GIN(nn.Module):
    """Graph Isomorphism Network: node_encoder + GINConv layers + mean pool + MLP."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3,
                 dropout=0.5, epsilon=0):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.node_encoder = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP(hidden_channels, hidden_channels, hidden_channels)
            self.convs.append(GINConv(mlp, train_eps=True, eps=epsilon))
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)])
        self.mlp = MLP(hidden_channels, hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.node_encoder(x.float())
        xs = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        x = global_mean_pool(x, batch)
        x = self.mlp(x)
        return x

    def get_embeddings(self, x, edge_index, batch):
        """Node embeddings from all layers (no dropout; for visualisation)."""
        x = self.node_encoder(x.float())
        xs = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            xs.append(x)
        return xs


# ---------------------------------------------------------------------------
# Training — identical to HFooladi notebook 06 train_and_evaluate
# ---------------------------------------------------------------------------
def train_and_evaluate(model, optimizer, train_loader, val_loader, test_loader,
                       dev, epochs=100):
    """Train and evaluate a model for molecular property prediction."""
    train_losses, val_losses, val_aucs = [], [], []
    best_val_auc, best_model = 0, None
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_samples = 0, 0
        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(dev)
            out = model(data.x.float(), data.edge_index, data.batch)
            if out.size(0) != data.y.size(0):
                m = min(out.size(0), data.y.size(0))
                out, data.y = out[:m], data.y[:m]
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            total_samples += data.num_graphs
        avg_loss = total_loss / total_samples
        train_losses.append(avg_loss)

        model.eval()
        val_loss, val_samples, y_true, y_pred = 0, 0, [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(dev)
                out = model(data.x.float(), data.edge_index, data.batch)
                if out.size(0) != data.y.size(0):
                    m = min(out.size(0), data.y.size(0))
                    out, data.y = out[:m], data.y[:m]
                loss = criterion(out, data.y)
                val_loss += loss.item() * data.num_graphs
                val_samples += data.num_graphs
                y_true.append(data.y.cpu().numpy())
                y_pred.append(torch.sigmoid(out).cpu().numpy())
        avg_val_loss = val_loss / val_samples
        val_losses.append(avg_val_loss)
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        val_auc = roc_auc_score(y_true, y_pred)
        val_aucs.append(val_auc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}: Train Loss: {avg_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val AUC: {val_auc:.4f}", flush=True)

    model.load_state_dict(best_model)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(dev)
            out = model(data.x.float(), data.edge_index, data.batch)
            if out.size(0) != data.y.size(0):
                m = min(out.size(0), data.y.size(0))
                out, data.y = out[:m], data.y[:m]
            y_true.append(data.y.cpu().numpy())
            y_pred.append(torch.sigmoid(out).cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    test_auc = roc_auc_score(y_true, y_pred)
    return {"train_losses": train_losses, "val_losses": val_losses,
            "val_aucs": val_aucs, "best_val_auc": best_val_auc,
            "test_auc": test_auc, "y_true": y_true, "y_pred": y_pred}


def load_mdm2_graphs(csv_path=DATA_CSV):
    df = pd.read_csv(csv_path)
    df["y"] = (df["bioactivity_class"] == "Active").astype(int)
    graphs = [g for g in (mol_to_graph_labeled(s, y)
                          for s, y in zip(df["canonical_smiles"], df["y"]))
              if g is not None]
    print(f"Converted {len(graphs)}/{len(df)} molecules to graphs "
          f"(9-dim atom feats: {graphs[0].x.shape})")
    return graphs, df


def random_split(graphs, seed=SEED):
    torch.manual_seed(seed)
    idx = torch.randperm(len(graphs))
    n = len(graphs)
    tr, va = idx[:int(0.8 * n)], idx[int(0.8 * n):int(0.9 * n)]
    te = idx[int(0.9 * n):]
    return [graphs[i] for i in tr], [graphs[i] for i in va], [graphs[i] for i in te]


# ---------------------------------------------------------------------------
# Screening — same mol_to_graph -> sigmoid(model) pattern as training
# ---------------------------------------------------------------------------
@torch.no_grad()
def screen_smiles(model, smiles_list, dev, batch_size=256):
    """Return P(active) for each valid SMILES (NaN for invalid)."""
    model.eval()
    probs = np.full(len(smiles_list), np.nan)
    graphs, keep = [], []
    for i, s in enumerate(smiles_list):
        try:
            graphs.append(mol_to_graph(s))
            keep.append(i)
        except ValueError:
            continue
    for s in range(0, len(graphs), batch_size):
        chunk = graphs[s:s + batch_size]
        loader = DataLoader(chunk, batch_size=len(chunk), shuffle=False)
        b = next(iter(loader)).to(dev)
        p = torch.sigmoid(model(b.x.float(), b.edge_index, b.batch))[:, 0]
        probs[np.array(keep[s:s + batch_size])] = p.cpu().numpy()
    return probs


PROB_COL = "gin_hfooladi_prob"

SMILES_CANDIDATES = ["canonical_smiles", "smiles", "SMILES", "isomeric_smiles",
                     "structure", "clean_smiles"]
ID_CANDIDATES = ["identifier", "coconut_id", "coconutID", "id", "name",
                 "title", "compound_id"]


def resolve_columns(columns, smiles_col="canonical_smiles"):
    """Pick the SMILES + identifier columns, tolerating COCONUT-style names."""
    scol = smiles_col if smiles_col in columns else next(
        (c for c in SMILES_CANDIDATES if c in columns), columns[0])
    idcol = next((c for c in ID_CANDIDATES if c in columns and c != scol), None)
    return scol, idcol


def screen_file(model, input_csv, output_csv, dev, smiles_col="canonical_smiles",
                chunksize=20000, batch_size=256):
    """Memory-safe screening of large libraries (e.g. COCONUT, ~10^5-10^6 rows).

    Reads the input CSV in chunks, featurizes+predicts per chunk, keeps only
    identifier/SMILES/probability, then writes one ranked CSV (NaN last).
    """
    cols = pd.read_csv(input_csv, nrows=0).columns.tolist()
    scol, idcol = resolve_columns(cols, smiles_col)
    keep = ([idcol] if idcol else []) + [scol]
    print(f"SMILES column: {scol} | ID column: {idcol}")
    parts, total, valid = [], 0, 0
    reader = pd.read_csv(input_csv, chunksize=chunksize, usecols=keep)
    for chunk in reader:
        smiles = chunk[scol].astype(str).tolist()
        probs = screen_smiles(model, smiles, dev, batch_size=batch_size)
        out = pd.DataFrame({scol: smiles, PROB_COL: probs})
        if idcol:
            out.insert(0, idcol, chunk[idcol].values)
        parts.append(out)
        total += len(chunk)
        valid += int(np.isfinite(probs).sum())
        print(f"  screened {total} ({valid} valid) ...", flush=True)
    full = pd.concat(parts, ignore_index=True)
    full.sort_values(PROB_COL, ascending=False).to_csv(output_csv, index=False)
    print(f"Screened {total} compounds ({valid} valid) -> {output_csv}")


def main():
    ap = argparse.ArgumentParser(description="HFooladi-exact GIN on MDM2")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--screen", default=None, help="CSV with SMILES to screen")
    ap.add_argument("--screen-out", default=os.path.join(BASE, "gin_hfooladi_screen.csv"))
    ap.add_argument("--weights", default=None, help="Load .pth instead of training")
    args = ap.parse_args()

    in_channels = 9  # HFooladi 9-dim atom features
    model = GIN(in_channels, args.hidden, out_channels=1, num_layers=args.layers,
                dropout=args.dropout, epsilon=0).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,} | device: {device}")

    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        graphs, _ = load_mdm2_graphs()
        tr, va, te = random_split(graphs)
        print(f"Train: {len(tr)}, Validation: {len(va)}, Test: {len(te)}")
        tr_loader = DataLoader(tr, batch_size=32, shuffle=True)
        va_loader = DataLoader(va, batch_size=32, shuffle=False)
        te_loader = DataLoader(te, batch_size=32, shuffle=False)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
        res = train_and_evaluate(model, opt, tr_loader, va_loader, te_loader,
                                 device, args.epochs)
        print(f"Best Val AUC: {res['best_val_auc']:.4f} | Test AUC: {res['test_auc']:.4f}")
        yt = res["y_true"].ravel()
        yp = res["y_pred"].ravel()
        yb = (yp > 0.5).astype(int)
        print(f"Acc {accuracy_score(yt, yb):.4f} Prec "
              f"{precision_score(yt, yb, zero_division=0):.4f} Rec "
              f"{recall_score(yt, yb, zero_division=0):.4f} F1 "
              f"{f1_score(yt, yb, zero_division=0):.4f}")
        torch.save(model.state_dict(), os.path.join(BASE, "gin_hfooladi_mdm2.pth"))
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        ax[0].plot(res["train_losses"], label="GIN"); ax[0].set_title("Training Loss")
        ax[1].plot(res["val_losses"], label="GIN"); ax[1].set_title("Validation Loss")
        ax[2].plot(res["val_aucs"], label="GIN"); ax[2].set_title("Validation AUC")
        for a in ax:
            a.set_xlabel("Epoch"); a.legend(); a.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE, "gin_hfooladi_curves.png"), dpi=150)
        print("Saved gin_hfooladi_mdm2.pth + gin_hfooladi_curves.png")

    if args.screen:
        screen_file(model, args.screen, args.screen_out, device)


if __name__ == "__main__":
    main()
