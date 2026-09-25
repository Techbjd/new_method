"""GNN-only virtual-screening triage: classical med-chem filters + calibrated GNN.

Stages (all in one table, no silent drops until the tier column):
  1. valid SMILES -> RDKit properties (MW, logP, HBA, HBD, TPSA, RotB, HAC, QED)
  2. med-chem gates: PAINS / Brenk (FilterCatalog), Lipinski Ro5, Veber
  3. applicability domain vs the 645 training cpds:
       max Morgan(r=3,512-bit) Tanimoto + train MW/logP/HBA/HBD mean+-2sd box
  4. calibrated GCN + GIN (temperature scaling fit on held-out 129) + mean ensemble
  5. shortlist (ensemble_cal > 0.6): MC-dropout uncertainty (dropout-on/BN-frozen,
     medchem-pass only), Butina scaffold clustering, efficiency columns (HAC/MW/logP for LE/LipE
     once a potency/docking score exists)
  6. evidence tier per compound (HIGH / MEDIUM / LOW / OUT-OF-DOMAIN / REJECT)

Inputs : ../Part_9/ro5_properties_filtered.csv (645 train reference)
         ../Natural_MDM2_Inhibitor_Discovery_using_ML/Part_8/screening_results.csv
Outputs: results/triage_full.csv   (one row per valid compound, key columns)
         results/triage_hits.csv   (shortlist, all columns, ranked)
         results/triage_per_cluster.csv (best hit per Butina cluster)
         results/triage_summary.json
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
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, FilterCatalog, QED
from rdkit.Chem import rdFingerprintGenerator as rfg
from rdkit.Chem import DataStructs
from rdkit.ML.Cluster import Butina

from common import mol_to_graph, GCN, GINEncoder, MDM2Classifier

BASE = os.path.dirname(os.path.abspath(__file__))


def _find(path_end, start=BASE):
    """Find a repo-relative file across local + Colab layouts (no slow walks)."""
    cands = [
        os.path.join(os.path.dirname(start), path_end),
        os.path.join(start, path_end),
        os.path.join(os.getcwd(), path_end),
        os.path.join(os.getcwd(), os.path.basename(path_end)),
        os.path.join('/content', path_end),
        os.path.join('/content/new_method', path_end),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    # also try basename match in likely roots (1 level deep only)
    for root in [os.path.dirname(start), start, os.getcwd(), '/content']:
        if os.path.isdir(root):
            try:
                for e in os.listdir(root):
                    full = os.path.join(root, e)
                    if os.path.isdir(full):
                        c = os.path.join(full, path_end)
                        if os.path.exists(c):
                            return c
            except OSError:
                pass
    return cands[0]  # clear error downstream showing expected path


TRAIN_CSV = _find('Part_9/ro5_properties_filtered.csv')
LIB_CSV = _find('Part_8/screening_results.csv')
NEW = os.path.dirname(os.path.dirname(TRAIN_CSV)) if os.path.exists(TRAIN_CSV) else os.path.dirname(BASE)
RES = f"{BASE}/results"
os.makedirs(RES, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device, flush=True)

MORGAN = rfg.GetMorganGenerator(radius=3, fpSize=512)
CHUNK = 20000  # featurize+predict in chunks (memory-safe on 154k)
MC_PASSES = 10  # MC-dropout passes (uncertainty); medchem-pass hits only

# ---------------------------------------------------------------- reference
print("loading training reference...", flush=True)
print("TRAIN_CSV:", TRAIN_CSV, flush=True)
print("LIB_CSV:", LIB_CSV, flush=True)
tr = pd.read_csv(TRAIN_CSV)
tr_mols = [Chem.MolFromSmiles(s) for s in tr['canonical_smiles']]
tr_mols = [m for m in tr_mols if m is not None]
TR_FPS = [MORGAN.GetFingerprint(m) for m in tr_mols]
t_mw = tr['molecular_weight'].values
t_logp = tr['logp'].values
t_hba = tr['n_hba'].values
t_hbd = tr['n_hbd'].values
BOX = {
    'mw': (t_mw.mean() - 2 * t_mw.std(), t_mw.mean() + 2 * t_mw.std()),
    'logp': (t_logp.mean() - 2 * t_logp.std(), t_logp.mean() + 2 * t_logp.std()),
    'hba': (t_hba.mean() - 2 * t_hba.std(), t_hba.mean() + 2 * t_hba.std()),
    'hbd': (t_hbd.mean() - 2 * t_hbd.std(), t_hbd.mean() + 2 * t_hbd.std()),
}
print("AD box:", {k: (round(v[0], 2), round(v[1], 2)) for k, v in BOX.items()}, flush=True)

_p = FilterCatalog.FilterCatalogParams()
_p.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
PAINS = FilterCatalog.FilterCatalog(_p)
_b = FilterCatalog.FilterCatalogParams()
_b.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
BRENK = FilterCatalog.FilterCatalog(_b)


def props(m):
    try:
        q = float(QED.qed(m))
    except Exception:
        q = float('nan')
    return dict(mw=Descriptors.MolWt(m), logp=Descriptors.MolLogP(m),
                hba=Lipinski.NumHAcceptors(m), hbd=Lipinski.NumHDonors(m),
                tpsa=Descriptors.TPSA(m), rotb=Lipinski.NumRotatableBonds(m),
                hac=m.GetNumHeavyAtoms(), qed=q)


# ---------------------------------------------------------------- models
def load_models():
    gcn = GCN().to(device)
    gcn.load_state_dict(torch.load(f"{BASE}/models/gcn_fixed.pth", map_location=device))
    gcn.eval()
    gin = MDM2Classifier(GINEncoder()).to(device)
    gin.load_state_dict(torch.load(f"{BASE}/models/gin_fixed.pth", map_location=device))
    gin.eval()
    t_gcn = json.load(open(f"{BASE}/models/gcn_fixed_meta.json"))["temperature"]
    t_gin = json.load(open(f"{BASE}/models/gin_fixed_meta.json"))["temperature"]
    print(f"loaded GCN (T={t_gcn:.3f}) + GIN (T={t_gin:.3f})", flush=True)
    return gcn, gin, t_gcn, t_gin


def batch_logits(model, graphs, bs=256):
    if len(graphs) == 0:
        return np.zeros((0, 2))
    outs = []
    for s in range(0, len(graphs), bs):
        loader = DataLoader(graphs[s:s + bs], batch_size=len(graphs[s:s + bs]), shuffle=False)
        b = next(iter(loader)).to(device)
        with torch.no_grad():
            outs.append(model(b.x, b.edge_index, b.batch).cpu().numpy())
    return np.concatenate(outs, axis=0)


def _enable_dropout_only(model):
    """Put dropout layers in train mode, keep BatchNorm in eval (stable MC-dropout)."""
    model.eval()
    for mod in model.modules():
        if isinstance(mod, (nn.Dropout,)):
            mod.train()


def mc_dropout_std(model, graphs, passes_=30, bs=256):
    """Predictive std over stochastic (dropout-on, BN-frozen) forward passes."""
    if len(graphs) == 0:
        return np.array([])
    _enable_dropout_only(model)
    allp = []
    for _ in range(passes_):
        outs = []
        for s in range(0, len(graphs), bs):
            loader = DataLoader(graphs[s:s + bs], batch_size=len(graphs[s:s + bs]), shuffle=False)
            b = next(iter(loader)).to(device)
            with torch.no_grad():
                outs.append(F.softmax(model(b.x, b.edge_index, b.batch), dim=1)[:, 1].cpu().numpy())
        allp.append(np.concatenate(outs))
    model.eval()
    return np.std(np.stack(allp), axis=0)


# ---------------------------------------------------------------- main
def main():
    lib = pd.read_csv(LIB_CSV, usecols=["identifier", "canonical_smiles"])
    print(f"library: {len(lib)}", flush=True)
    gcn, gin, T_GCN, T_GIN = load_models()

    out_rows, short = [], []
    short_graphs = []  # (row_idx_in_out) for MC-dropout later
    n_invalid = 0
    for start in range(0, len(lib), CHUNK):
        blk = lib.iloc[start:start + CHUNK]
        mols, fps, gs, keep = [], [], [], []
        for i, r in blk.iterrows():
            m = Chem.MolFromSmiles(str(r['canonical_smiles']))
            if m is None:
                n_invalid += 1
                continue
            g = mol_to_graph(str(r['canonical_smiles']))
            if g is None:
                n_invalid += 1
                continue
            mols.append(m)
            fps.append(MORGAN.GetFingerprint(m))
            gs.append(g)
            keep.append(i)
        lg = batch_logits(gcn, gs)
        li = batch_logits(gin, gs)
        pg = F.softmax(torch.tensor(lg) / T_GCN, dim=1)[:, 1].numpy()
        pi = F.softmax(torch.tensor(li) / T_GIN, dim=1)[:, 1].numpy()
        ens = (pg + pi) / 2
        for j, i in enumerate(keep):
            r = lib.loc[i]
            m = mols[j]
            pr = props(m)
            sims = DataStructs.BulkTanimotoSimilarity(fps[j], TR_FPS)
            maxsim = float(max(sims)) if sims else 0.0
            in_box = (BOX['mw'][0] <= pr['mw'] <= BOX['mw'][1]
                      and BOX['logp'][0] <= pr['logp'] <= BOX['logp'][1]
                      and BOX['hba'][0] <= pr['hba'] <= BOX['hba'][1]
                      and BOX['hbd'][0] <= pr['hbd'] <= BOX['hbd'][1])
            pains_ok = not PAINS.HasMatch(m)
            brenk_ok = not BRENK.HasMatch(m)
            ro5_ok = (pr['mw'] <= 500 and pr['hba'] <= 10 and pr['hbd'] <= 5 and pr['logp'] <= 5)
            veber_ok = (pr['rotb'] <= 10 and pr['tpsa'] <= 140)
            medchem = pains_ok and brenk_ok and ro5_ok and veber_ok
            in_ad = (maxsim >= 0.30) and in_box
            row = dict(identifier=r['identifier'], canonical_smiles=r['canonical_smiles'],
                       gcn_cal=round(float(pg[j]), 4), gin_cal=round(float(pi[j]), 4),
                       ensemble_cal=round(float(ens[j]), 4),
                       max_tanimoto_train=round(maxsim, 3), in_ad_box=bool(in_box),
                       in_ad=bool(in_ad), pains_pass=bool(pains_ok),
                       brenk_pass=bool(brenk_ok), ro5_pass=bool(ro5_ok),
                       veber_pass=bool(veber_ok), medchem_pass=bool(medchem), **pr)
            out_rows.append(row)
            if ens[j] > 0.6:
                short.append(row)
                short_graphs.append(gs[j])
        print(f"chunk {start}-{start + len(blk)}: valid {len(keep)}, short {len(short)}", flush=True)

    full = pd.DataFrame(out_rows)
    full.to_csv(f"{RES}/triage_full.csv", index=False)
    print(f"full: {len(full)} (invalid {n_invalid})", flush=True)

    hits = pd.DataFrame(short)
    if len(hits):
        # MC-dropout uncertainty on medchem-pass shortlist only (speed);
        # the rest get NaN and can never reach HIGH tier.
        sg = F.softmax(torch.tensor(batch_logits(gcn, short_graphs)) / T_GCN, dim=1)[:, 1].numpy()
        si = F.softmax(torch.tensor(batch_logits(gin, short_graphs)) / T_GIN, dim=1)[:, 1].numpy()
        hits['disagreement'] = np.round(np.abs(sg - si), 4)
        g_std = np.full(len(hits), np.nan)
        i_std = np.full(len(hits), np.nan)
        med = hits['medchem_pass'].values
        if med.any():
            sub = [short_graphs[k] for k in np.where(med)[0]]
            g_std[med] = mc_dropout_std(gcn, sub, passes_=MC_PASSES)
            i_std[med] = mc_dropout_std(gin, sub, passes_=MC_PASSES)
        hits['gcn_mc_std'] = np.round(g_std, 4)
        hits['gin_mc_std'] = np.round(i_std, 4)
        # Butina clustering (Tanimoto >= 0.6 neighbours); guard tiny shortlists
        if len(hits) == 1:
            hits['cluster'] = 0
            hits['cluster_size'] = 1
        else:
            sfps = [MORGAN.GetFingerprint(Chem.MolFromSmiles(s)) for s in hits['canonical_smiles']]
            dists = []
            for i in range(1, len(sfps)):
                dists.extend([1 - x for x in DataStructs.BulkTanimotoSimilarity(sfps[i], sfps[:i])])
            clusters = Butina.ClusterData(dists, len(sfps), 0.4, isDistData=True)
            cl = np.zeros(len(sfps), dtype=int)
            for cid, c in enumerate(clusters):
                for idx in c:
                    cl[idx] = cid
            hits['cluster'] = cl
            hits['cluster_size'] = hits.groupby('cluster')['cluster'].transform('size')

        def tier(r):
            if not r['medchem_pass']:
                return 'REJECT'
            if r['ensemble_cal'] < 0.6:
                return 'REJECT'
            try:
                mc_max = max(float(r['gcn_mc_std']), float(r['gin_mc_std']))
            except Exception:
                mc_max = float('nan')
            if r['in_ad'] and r['ensemble_cal'] >= 0.7 and mc_max < 0.10:
                return 'HIGH'
            if r['in_ad'] or r['max_tanimoto_train'] >= 0.30:
                return 'MEDIUM'
            return 'OUT-OF-DOMAIN'

        hits['tier'] = hits.apply(tier, axis=1)
        order = {'HIGH': 0, 'MEDIUM': 1, 'OUT-OF-DOMAIN': 2, 'REJECT': 3}
        hits['_rank'] = hits['tier'].map(order)
        hits = hits.sort_values(['_rank', 'ensemble_cal'],
                                ascending=[True, False]).drop(columns=['_rank']).reset_index(drop=True)
        hits.to_csv(f"{RES}/triage_hits.csv", index=False)
        per = hits.sort_values('ensemble_cal', ascending=False).groupby('cluster').first().reset_index()
        per.to_csv(f"{RES}/triage_per_cluster.csv", index=False)
        print("tiers:", hits['tier'].value_counts().to_dict(), flush=True)
        print(f"clusters: {hits['cluster'].nunique()}", flush=True)

    summary = dict(n_library=int(len(lib)), n_invalid=int(n_invalid), n_valid=int(len(full)),
                   medchem_pass=int(full['medchem_pass'].sum()),
                   in_ad=int(full['in_ad'].sum()),
                   ensemble_gt_06=int((full['ensemble_cal'] > 0.6).sum()),
                   n_hits=int(len(hits)),
                   tiers=hits['tier'].value_counts().to_dict() if len(hits) else {},
                   n_clusters=int(hits['cluster'].nunique()) if len(hits) else 0)
    json.dump(summary, open(f"{RES}/triage_summary.json", 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
