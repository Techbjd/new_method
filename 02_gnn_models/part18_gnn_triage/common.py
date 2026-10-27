"""Shared fixed 78-dim graph builder + GCN/GIN model definitions (Parts 9/10).

Feature layout (offsets): atomic_num 0:53 | degree 53:59 | formal_charge 59:65 |
num_hs 65:70 | hybridization 70:75 | aromatic 75 | ring 76 | chiral 77 (=78).
This is the FIXED builder (Part 9 post-fix). All weights in Part_18 are trained
with this builder — never mix with Part_9/10 .pth files (old buggy builder).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from rdkit import Chem

RANDOM_SEED = 57

COMMON_ELEMENTS = [1, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21,
                   22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37,
                   38, 39, 40, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55,
                   56, 78, 79, 80, 82, 83]
ATOM_CHOICES = {
    'atomic_num': COMMON_ELEMENTS + ['OTHER'],  # 53 dims
    'degree': [0, 1, 2, 3, 4, 5],                # 6
    'formal_charge': [-2, -1, 0, 1, 2, 3],       # 6
    'num_hs': [0, 1, 2, 3, 4],                   # 5
    'hybridization': [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2],    # 5
}


def one_hot(val, choices):
    enc = [0] * len(choices)
    if val in choices:
        enc[choices.index(val)] = 1
    return enc


def atom_features(atom):
    f = []
    z = atom.GetAtomicNum()
    f += one_hot(z if z in COMMON_ELEMENTS else 'OTHER', ATOM_CHOICES['atomic_num'])
    f += one_hot(atom.GetTotalDegree(), ATOM_CHOICES['degree'])
    f += one_hot(atom.GetFormalCharge(), ATOM_CHOICES['formal_charge'])
    f += one_hot(atom.GetTotalNumHs(), ATOM_CHOICES['num_hs'])
    f += one_hot(atom.GetHybridization(), ATOM_CHOICES['hybridization'])
    f.append(int(atom.GetIsAromatic()))
    f.append(int(atom.IsInRing()))
    f.append(int(atom.HasProp('_ChiralityPossible')
                 and atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED))
    assert len(f) == 78, f"atom feature dim {len(f)} != 78"
    return f


def mol_to_graph(smiles, label=0):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    node_features = [atom_features(a) for a in mol.GetAtoms()]
    edge_index = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])
    if not edge_index:
        edge_index = [[0, 0]]
    return Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        y=torch.tensor([label], dtype=torch.long),
    )


class GCN(nn.Module):
    """Part 9 architecture, USE_MACCS=False (identical to original paper model)."""

    def __init__(self, num_node_features=78, hidden_dim=128, num_classes=2, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, num_classes))

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.classifier(g)


class GINEncoder(nn.Module):
    """Part 10: 5-layer GIN encoder, MolCLR-style (hidden 300)."""

    def __init__(self, num_node_features=78, hidden_dim=300, num_layers=5, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.gin_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.gin_layers.append(nn.Linear(num_node_features, hidden_dim))
        self.bn_layers.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.gin_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))
        self.eps = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_layers)])

    def forward(self, x, edge_index, batch):
        for i in range(self.num_layers):
            neighbor_sum = self._aggregate(x, edge_index)
            x = self.gin_layers[i]((1 + self.eps[i]) * x + neighbor_sum)
            x = self.bn_layers[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

    @staticmethod
    def _aggregate(x, edge_index):
        src, dst = edge_index
        agg = torch.zeros_like(x)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(x[src]), x[src])
        return agg


class MDM2Classifier(nn.Module):
    """Part 10 head: GIN encoder + classifier (USE_MACCS=False)."""

    def __init__(self, encoder, hidden_dim=600, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x, edge_index, batch):
        return self.classifier(self.encoder(x, edge_index, batch))


def predict_probs(model, graphs, device, batch_size=256):
    model.eval()
    probs = []
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy().tolist())
    return np.array(probs)


def predict_logits(model, graphs, device, batch_size=256):
    model.eval()
    outs = []
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outs.append(model(batch.x, batch.edge_index, batch.batch).cpu().numpy())
    return np.concatenate(outs, axis=0)
