import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from pathlib import Path
import tempfile
BASE = str(Path(__file__).resolve().parents[2])  # repo root (local or Colab clone)
TMPD = os.path.join(tempfile.gettempdir(), "mdm2_tmp")
os.makedirs(TMPD, exist_ok=True)
FP_DIM=2215
_morgan_gen=rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
def combined_fp(smi):
    m=Chem.MolFromSmiles(str(smi))
    if m is None: return None
    bv=_morgan_gen.GetFingerprint(m); a=np.zeros((2048,),dtype=np.float32); DataStructs.ConvertToNumpyArray(bv,a)
    bv2=MACCSkeys.GenMACCSKeys(m); b=np.zeros((167,),dtype=np.float32); DataStructs.ConvertToNumpyArray(bv2,b)
    return np.concatenate([a,b]).astype(np.float32)
ATOM_CHOICES={'atomic_num':list(range(1,101)),'degree':[0,1,2,3,4,5],'formal_charge':[-2,-1,0,1,2,3],'num_hs':[0,1,2,3,4],'hybridization':[Chem.rdchem.HybridizationType.SP,Chem.rdchem.HybridizationType.SP2,Chem.rdchem.HybridizationType.SP3,Chem.rdchem.HybridizationType.SP3D,Chem.rdchem.HybridizationType.SP3D2]}
def one_hot(v,choices):
    enc=[0]*len(choices)
    if v in choices: enc[choices.index(v)]=1
    return enc
def atom_features(atom):
    f=[]
    f+=one_hot(atom.GetAtomicNum(),ATOM_CHOICES['atomic_num'])
    f+=one_hot(atom.GetTotalDegree(),ATOM_CHOICES['degree'])
    f+=one_hot(atom.GetFormalCharge(),ATOM_CHOICES['formal_charge'])
    f+=one_hot(atom.GetTotalNumHs(),ATOM_CHOICES['num_hs'])
    f+=one_hot(atom.GetHybridization(),ATOM_CHOICES['hybridization'])
    f.append(int(atom.GetIsAromatic())); f.append(int(atom.IsInRing()))
    return (f+[0]*78)[:78]
def mol_to_graph(smi,label=0):
    m=Chem.MolFromSmiles(str(smi))
    if m is None: return None
    x=torch.tensor([atom_features(a) for a in m.GetAtoms()],dtype=torch.float)
    ei=[]
    for b in m.GetBonds():
        i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx(); ei.extend([[i,j],[j,i]])
    if not ei: ei=[[0,0]]
    ei=torch.tensor(ei,dtype=torch.long).t().contiguous()
    return Data(x=x,edge_index=ei,y=torch.tensor([label],dtype=torch.long))
class GCNEncoder(nn.Module):
    def __init__(self,in_dim=78,hidden=128,layers=3,dropout=0.3):
        super().__init__()
        self.convs=nn.ModuleList([GCNConv(in_dim if i==0 else hidden,hidden) for i in range(layers)])
        self.bns=nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
        self.dropout=dropout; self.out_dim=hidden*2
    def forward(self,x,edge_index,batch):
        for conv,bn in zip(self.convs,self.bns):
            x=F.relu(bn(conv(x,edge_index))); x=F.dropout(x,p=self.dropout,training=self.training)
        return torch.cat([global_mean_pool(x,batch),global_max_pool(x,batch)],dim=1)
class FPMlp(nn.Module):
    def __init__(self,in_dim=FP_DIM,hidden=[512,256,128],dropout=0.3):
        super().__init__()
        layers=[]; d=in_dim
        for h in hidden:
            layers+=[nn.Linear(d,h),nn.BatchNorm1d(h),nn.ReLU(),nn.Dropout(dropout)]; d=h
        self.enc=nn.Sequential(*layers); self.out_dim=hidden[-1]
    def forward(self,fp): return self.enc(fp)
class HybridModel(nn.Module):
    def __init__(self,fp_dim=FP_DIM,gnn_hidden=128,gnn_layers=3,fp_hidden=[512,256,128],r=0.5,dropout=0.3,num_classes=2):
        super().__init__()
        self.r=r; self.fp_enc=FPMlp(fp_dim,fp_hidden,dropout); self.gnn_enc=GCNEncoder(78,gnn_hidden,gnn_layers,dropout)
        total=256; g_dim=int(total*r); f_dim=total-g_dim
        self.g_proj=nn.Linear(self.gnn_enc.out_dim,g_dim); self.f_proj=nn.Linear(self.fp_enc.out_dim,f_dim)
        self.cls=nn.Sequential(nn.Linear(total,128),nn.ReLU(),nn.Dropout(dropout),nn.Linear(128,num_classes))
    def forward(self,x,edge_index,batch,fp):
        g=self.gnn_enc(x,edge_index,batch); g=self.g_proj(g)
        f=self.fp_enc(fp); f=self.f_proj(f)
        return self.cls(torch.cat([g,f],dim=1))

device=torch.device('cpu')
df645=pd.read_csv(os.path.join(BASE,'ro5_properties_filtered.csv'))
fz=np.load(os.path.join(BASE,'Part_17/data/featurized_curated.npz'))
y=fz['cur_labels'].astype(int)
# verify order: compare our fp vs stored fp for first few indices
Xstored=fz['cur_fps']
for i in [0,1,2,3,14]:
    smi=df645.iloc[i]['canonical_smiles']
    fp=combined_fp(smi)
    print(f"idx {i} match={np.allclose(fp,Xstored[i],atol=1e-6)} maxdiff={np.abs(fp-Xstored[i]).max():.2e} label stored={y[i]} bio={df645.iloc[i]['bioactivity_class']}")
print("df len",len(df645),"fps shape",Xstored.shape)

def predict_test(model_path, idx):
    model=HybridModel(r=0.5).to(device)
    sd=torch.load(model_path,map_location=device)
    print(f"{os.path.basename(model_path)} keys sample:", list(sd.keys())[:5], "total", len(sd))
    model.load_state_dict(sd); model.eval()
    fps=[]; graphs=[]
    for i in idx:
        smi=df645.iloc[int(i)]['canonical_smiles']
        fp=combined_fp(smi); g=mol_to_graph(smi)
        fps.append(fp); graphs.append(g)
    fps=np.stack(fps)
    probs=[]
    with torch.no_grad():
        for s in range(0,len(idx),64):
            bfp=torch.tensor(fps[s:s+64],dtype=torch.float)
            bgs=graphs[s:s+64]
            loader=DataLoader(bgs,batch_size=len(bgs),shuffle=False)
            batch=next(iter(loader))
            out=model(batch.x,batch.edge_index,batch.batch,bfp)
            probs.extend(F.softmax(out,dim=1)[:,1].cpu().numpy().tolist())
    return np.array(probs)

for tag,mp in [('scaffold_pretrain','Part_17/models/hybrid_scaffold_pretrain_r0.5.pth'),('scaffold_scratch','Part_17/models/hybrid_scaffold_scratch_r0.5.pth')]:
    try:
        idx=fz['te_cur_new']
        probs=predict_test(os.path.join(BASE,mp),idx)
        print(f"{tag} scaffold test probs mean={probs.mean():.4f} max={probs.max():.4f} min={probs.min():.4f}")
        # save for ECE step
        np.save(os.path.join(TMPD, f"probs_{tag}.npy"), probs)
    except Exception as e:
        import traceback; traceback.print_exc()
