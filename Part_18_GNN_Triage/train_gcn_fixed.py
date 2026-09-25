"""Train Part-9 GCN on the FIXED 78-dim builder, identical protocol otherwise:
80/20 stratified split seed 57, 100 epochs, Adam lr 1e-3, batch 64.
Then: test metrics + temperature scaling on the held-out 129.

Outputs: models/gcn_fixed.pth, models/gcn_fixed_meta.json
Run this first on a fresh machine (e.g. Colab) before screen_triage.py.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef,
                             confusion_matrix)
from torch_geometric.loader import DataLoader

from common import (RANDOM_SEED, mol_to_graph, GCN, predict_probs, predict_logits)

BASE = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.dirname(BASE)  # repo root (works locally and in Colab)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

df = pd.read_csv(f"{NEW}/Part_9/ro5_properties_filtered.csv")
df['y'] = (df['bioactivity_class'] == 'Active').astype(int)
graphs = [g for g in (mol_to_graph(s, y) for s, y in
                      zip(df['canonical_smiles'], df['y'])) if g is not None]
labels = np.array([g.y.item() for g in graphs])
print(f"graphs {len(graphs)}/{len(df)}")

train_idx, test_idx, y_tr, y_te = train_test_split(
    np.arange(len(graphs)), labels, test_size=0.2,
    random_state=RANDOM_SEED, stratify=labels)
tr_loader = DataLoader([graphs[i] for i in train_idx], batch_size=64, shuffle=True)
te_graphs = [graphs[i] for i in test_idx]
print(f"train {len(train_idx)} test {len(test_idx)}")


def train_epoch(model, loader, opt, crit):
    model.train()
    tot = 0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        loss = crit(model(batch.x, batch.edge_index, batch.batch), batch.y)
        loss.backward()
        opt.step()
        tot += loss.item() * batch.num_graphs
    return tot / len(loader.dataset)


model = GCN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

for epoch in range(1, 101):
    loss = train_epoch(model, tr_loader, opt, crit)
    if epoch % 10 == 0 or epoch == 1:
        p = predict_probs(model, te_graphs, device)
        print(f"epoch {epoch:3d} loss {loss:.4f} test AUC {roc_auc_score(y_te, p):.4f}", flush=True)

os.makedirs(f"{BASE}/models", exist_ok=True)
torch.save(model.state_dict(), f"{BASE}/models/gcn_fixed.pth")

probs = predict_probs(model, te_graphs, device)
pred = (probs > 0.5).astype(int)
metrics = {
    'acc': float(accuracy_score(y_te, pred)),
    'prec': float(precision_score(y_te, pred, zero_division=0)),
    'rec': float(recall_score(y_te, pred, zero_division=0)),
    'f1': float(f1_score(y_te, pred, zero_division=0)),
    'auc': float(roc_auc_score(y_te, probs)),
    'mcc': float(matthews_corrcoef(y_te, pred)),
    'cm': confusion_matrix(y_te, pred).tolist(),
}
print("TEST:", json.dumps({k: v for k, v in metrics.items() if k != 'cm'}, indent=1))

logits = torch.tensor(predict_logits(model, te_graphs, device))
labels_t = torch.tensor(y_te)
T = torch.ones(1, requires_grad=True)
lbfgs = torch.optim.LBFGS([T], lr=0.1, max_iter=100)


def _closure():
    lbfgs.zero_grad()
    loss = nn.CrossEntropyLoss()(logits / T.clamp(min=1e-3), labels_t)
    loss.backward()
    return loss


lbfgs.step(_closure)
T_val = float(T.clamp(min=1e-3).item())
cal = F.softmax(logits / T_val, dim=1)[:, 1].numpy()
metrics['temperature'] = T_val
metrics['auc_cal'] = float(roc_auc_score(y_te, cal))
print(f"T={T_val:.3f} AUC_cal={metrics['auc_cal']:.4f}")
json.dump(metrics, open(f"{BASE}/models/gcn_fixed_meta.json", 'w'), indent=1)
print("saved models/gcn_fixed.pth + gcn_fixed_meta.json")
