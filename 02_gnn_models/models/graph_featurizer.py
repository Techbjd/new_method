"""Shared graph featurizer — 78-dim atom features (adapted from Part_18_GNN_Triage/common.py, RANDOM_SEED=57 layout kept).

Layout offsets (assert = 78):
atomic_num 0:53 | degree 53:59 | formal_charge 59:65 | num_hs 65:70 |
hybridization 70:75 | aromatic 75 | in_ring 76 | chiral 77
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

RANDOM_SEED = 57

COMMON_ELEMENTS = ["H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Si",
                   "Na", "Mg", "K", "Ca", "Fe", "Zn", "Cu", "Se", "As", "Al",
                   "Li", "He", "Ne", "Ar", "Cr", "Mn", "Co", "Ni", "Sr", "Ti",
                   "V", "Mo", "Sn", "Sb", "Te", "Ba", "La", "W", "Au", "Hg",
                   "Pb", "Bi", "Gd", "Y", "Zr", "Pd", "Ag", "Cd", "Ru", "Pt",
                   ""]

ELEMENT_BLOCK = 53  # 52 element symbols + OTHER fallback (+ 1-2 spare slots) → offsets kept Part_18-canon

def one_hot(val, choices):
    vec = [1.0 if c == val else 0.0 for c in choices]
    if all(v == 0.0 for v in vec):
        vec = [0.0] * (len(choices) - 1) + [1.0]  # send unmatched to the OTHER slot
    return vec

def atom_features(atom):
    """78-dim one-hot block vector."""
    from rdkit import Chem
    hybridizations = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
                      Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
                      Chem.HybridizationType.SP3D2]
    feats = []
    try:
        feats += one_hot(atom.GetSymbol(), COMMON_ELEMENTS)
    except Exception:
        feats += one_hot("OTHER", COMMON_ELEMENTS)
    feats += one_hot(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5])          # 53:59
    feats += one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2, 3])        # 59:65
    feats += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])               # 65:70
    feats += one_hot(atom.GetHybridization(), hybridizations)             # 70:75
    feats += [1.0 if atom.GetIsAromatic() else 0.0]                       # 75
    feats += [1.0 if atom.IsInRing() else 0.0]                            # 76
    feats += [1.0 if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED else 0.0]  # 77
    assert len(feats) == 78, f"feature dim {len(feats)} != 78"
    return feats

def bond_features(bond):
    """6-dim bond block: [single, double, triple, aromatic, conjugated, in_ring]."""
    from rdkit import Chem
    bt = bond.GetBondType()
    t = bt if not isinstance(bt, int) else bt
    return [
        1.0 if bt == Chem.BondType.SINGLE else 0.0,
        1.0 if bt == Chem.BondType.DOUBLE else 0.0,
        1.0 if bt == Chem.BondType.TRIPLE else 0.0,
        1.0 if bt == Chem.BondType.AROMATIC else 0.0,
        1.0 if bond.GetIsConjugated() else 0.0,
        1.0 if bond.IsInRing() else 0.0,
    ]

def mol_to_graph(smiles: str, label=None):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    feats = [atom_features(a) for a in mol.GetAtoms()]
    if len(feats) == 0:
        return None
    x = torch.tensor(np.array(feats, dtype=np.float32))
    edges, bond_feats = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
        bf = bond_features(bond=b)
        bond_feats += [bf, bf]
    if len(edges) == 0:
        edge_index = torch.tensor([[0, 0]], dtype=torch.long)
        edge_attr = torch.zeros((1, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(bond_feats, dtype=np.float32))
    y = None
    if label is not None:
        y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

def dataset_from_smiles(smiles_list, labels=None):
    """Return (graphs, dropped_count); masks out invalid/None molecules."""
    graphs, keep = [], []
    dropped = 0
    for i, s in enumerate(smiles_list):
        g = mol_to_graph(s, label=None if labels is None else int(labels[i]))
        if g is None:
            dropped += 1
            continue
        keep.append(i)
        graphs.append(g)
    return graphs, dropped

def graphs_to_edge_attrs(graphs):
    """Chemprop-style D-MPNN needs directed edges; expand undirected -> directed index."""
    dir_to_src = []
    for g in graphs:
        n_dir = g.edge_index.shape[1]
        src = g.edge_index.clone()
        dir_to_src.append(src)
    return dir_to_src
