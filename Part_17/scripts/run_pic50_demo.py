"""COCONUT pIC50 demo inference: load Part_17 HybridRegression checkpoint,
run on first 1000 COCONUT SMILES, save sample_pIC50_demo.csv.
Featurization + arch EXACTLY as Part_17_Pretrain_Finetune_v1_regression.ipynb Cell 10/12.
"""
import os, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

REPO = Path(__file__).resolve().parents[2]  # Part_17/scripts/ -> repo root
ap = argparse.ArgumentParser(description="COCONUT pIC50 inference with HybridRegression checkpoint")
ap.add_argument("--nrows", type=int, default=1000, help="rows of COCONUT pool to score (default 1000 demo; use 1000000 for full)")
ap.add_argument("--out", type=str, default=str(REPO / "Part_17" / "results" / "sample_pIC50_demo.csv"))
ap.add_argument("--coconut", type=str, default=str(REPO / "Part_8" / "screening_results.csv"))
ap.add_argument("--ckpt", type=str, default=str(REPO / "Part_17" / "models" / "hybrid_pretrain_regression.pth"))
ap.add_argument("--batch", type=int, default=64)
a = ap.parse_args()
CKPT, COCONUT, OUT = a.ckpt, a.coconut, a.out
N, BATCH = a.nrows, a.batch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_MACCS, MORGAN_NBITS, MACCS_DIM = True, 2048, 167
FP_DIM = 2215
_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=MORGAN_NBITS)

def morgan_fp(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None: return None
    bv = _morgan_gen.GetFingerprint(m)
    arr = np.zeros((MORGAN_NBITS,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

def maccs_fp(smi):
    from rdkit.Chem import MACCSkeys
    m = Chem.MolFromSmiles(str(smi))
    if m is None: return None
    bv = MACCSkeys.GenMACCSKeys(m)
    arr = np.zeros((MACCS_DIM,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

def combined_fp(smi):
    a = morgan_fp(smi)
    if a is None: return None
    b = maccs_fp(smi)
    if b is None: return None
    return np.concatenate([a, b]).astype(np.float32)

ATOM_CHOICES={'atomic_num':list(range(1,101)),'degree':[0,1,2,3,4,5],
 'formal_charge':[-2,-1,0,1,2,3],'num_hs':[0,1,2,3,4],
 'hybridization':[Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
  Chem.rdchem.HybridizationType.SP3D2]}
def one_hot(v, choices):
    enc=[0]*len(choices)
    if v in choices: enc[choices.index(v)]=1
    return enc
def atom_features(atom):
    f=[]
    f+=one_hot(atom.GetAtomicNum(), ATOM_CHOICES['atomic_num'])
    f+=one_hot(atom.GetTotalDegree(), ATOM_CHOICES['degree'])
    f+=one_hot(atom.GetFormalCharge(), ATOM_CHOICES['formal_charge'])
    f+=one_hot(atom.GetTotalNumHs(), ATOM_CHOICES['num_hs'])
    f+=one_hot(atom.GetHybridization(), ATOM_CHOICES['hybridization'])
    f.append(int(atom.GetIsAromatic())); f.append(int(atom.IsInRing()))
    return (f+[0]*78)[:78]
def mol_to_graph(smiles):
    m=Chem.MolFromSmiles(str(smiles))
    if m is None: return None
    x=torch.tensor([atom_features(a) for a in m.GetAtoms()], dtype=torch.float)
    ei=[]
    for b in m.GetBonds():
        i,j=b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ei.extend([[i,j],[j,i]])
    if not ei: ei=[[0,0]]
    ei=torch.tensor(ei, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=ei)

class FPMLP(nn.Module):
    def __init__(self, in_dim=2215, hidden=[512,256,128], dropout=0.3):
        super().__init__()
        layers=[]; d=in_dim
        for h in hidden:
            layers+=[nn.Linear(d,h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d=h
        self.enc=nn.Sequential(*layers)
        self.out_dim=hidden[-1]
    def forward(self, fp): return self.enc(fp)

class GCNEncoder(nn.Module):
    def __init__(self, num_node_features=78, hidden_dim=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.dropout=dropout
        self.convs=nn.ModuleList([GCNConv(num_node_features if i==0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        self.bns=nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.out_dim=hidden_dim*2
    def forward(self, x, edge_index, batch):
        for conv,bn in zip(self.convs, self.bns):
            x=F.relu(bn(conv(x, edge_index)))
            x=F.dropout(x, p=self.dropout, training=self.training)
        return torch.cat([global_mean_pool(x,batch), global_max_pool(x,batch)], dim=1)

class HybridRegression(nn.Module):
    def __init__(self, fp_dim=2215, gnn_hidden=128, gnn_layers=3, fp_hidden=[512,256,128], r=0.5, dropout=0.3):
        super().__init__()
        self.fp_enc=FPMLP(fp_dim, fp_hidden, dropout)
        self.gnn_enc=GCNEncoder(78, gnn_hidden, gnn_layers, dropout)
        total=256; g_dim=int(total*r); f_dim=total-g_dim
        self.g_proj=nn.Linear(self.gnn_enc.out_dim, g_dim)
        self.f_proj=nn.Linear(self.fp_enc.out_dim, f_dim)
        self.reg_head=nn.Sequential(nn.Linear(total,128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128,1))
    def forward(self, x, edge_index, batch, fp):
        g=self.g_proj(self.gnn_enc(x, edge_index, batch))
        f=self.f_proj(self.fp_enc(fp))
        return self.reg_head(torch.cat([g,f], dim=1)).squeeze(-1)

print(f"device={device}")
print(f"ckpt={CKPT} exists={os.path.exists(CKPT)}")
coco = pd.read_csv(COCONUT, usecols=["identifier","canonical_smiles"], nrows=N)
print(f"COCONUT sample: {len(coco)} rows from {COCONUT}")
print(coco.head(2).to_string())

model = HybridRegression().to(device)
sd = torch.load(CKPT, map_location=device)
model.load_state_dict(sd)
model.eval()
print(f"loaded {CKPT} ({len(sd)} tensors), fp_dim=2215, heads=reg_only")

fps, graphs, keep = [], [], []
for _, r in coco.iterrows():
    fp = combined_fp(r["canonical_smiles"]); g = mol_to_graph(r["canonical_smiles"])
    if fp is not None and g is not None:
        fps.append(fp); graphs.append(g); keep.append(True)
    else:
        keep.append(False)
fps = np.stack(fps)
print(f"featurized {len(graphs)}/{len(coco)} valid")

preds = []
with torch.no_grad():
    for s in range(0, len(graphs), BATCH):
        bidx = slice(s, s+BATCH)
        b_fp = torch.tensor(fps[s:s+BATCH], dtype=torch.float, device=device)
        loader = DataLoader(graphs[s:s+BATCH], batch_size=len(graphs[s:s+BATCH]), shuffle=False)
        batch = next(iter(loader)).to(device)
        out = model(batch.x, batch.edge_index, batch.batch, b_fp)
        preds.extend(out.cpu().numpy().tolist())
preds = np.array(preds)
print(f"pred_pIC50: mean={preds.mean():.3f} std={preds.std():.3f} min={preds.min():.3f} max={preds.max():.3f}")
print(f"train-range check: pretrain pIC50 spans ~4-11 (ChEMBL 4441); curated 645 similar")

coco_valid = coco[np.array(keep)].copy().reset_index(drop=True)
coco_valid["pred_pIC50"] = preds
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
coco_valid.to_csv(OUT, index=False)
print(f"saved {OUT} rows={len(coco_valid)} cols={list(coco_valid.columns)}")
print(coco_valid[["identifier","pred_pIC50"]].head(10).to_string(index=False))
