import os
import numpy as np
import pandas as pd

from pathlib import Path
REPO = Path(__file__).resolve().parents[2]  # Part_17/scripts/ -> repo root (local or Colab clone)
base = str(REPO / "Part_17" / "results")
p_cal = os.path.join(base, "coconut_gnn_screening_full_calibrated.csv")
p_dom = os.path.join(base, "coconut_with_domain.csv")
p_paper = str(REPO / "Part_9" / "paper_rf116_filtered_compounds.csv")
p_old116 = os.path.join(base, "coconut_gnn_pretrain_cal_top116.csv")
p_old116_rflike = os.path.join(base, "coconut_gnn_pretrain_top116_RF_like.csv")

out_full = os.path.join(base, "coconut_rescore_rankfusion.csv")
out_top116 = os.path.join(base, "top116_rankfusion.csv")
out_cons = os.path.join(base, "consensus_T1.csv")

EPS = 1e-6

cal = pd.read_csv(p_cal)
dom = pd.read_csv(p_dom)

print(f"cal rows={len(cal)} cols={cal.columns.tolist()}")
print(f"dom rows={len(dom)} cols={dom.columns.tolist()}")

# Verify join key
# cal has identifier, canonical_smiles, prediction, prob_class_0/1, gnn cols
# dom has identifier, canonical_smiles, prediction, prob_class_0/1, mw/logp/..., in_domain
# Join in_domain onto cal via identifier
assert cal['identifier'].duplicated().sum()==0
assert dom['identifier'].duplicated().sum()==0

# Check identifier sets equal
s_cal = set(cal['identifier'])
s_dom = set(dom['identifier'])
print(f"identifier overlap: cal∩dom={len(s_cal & s_dom)}, cal-dom={len(s_cal-s_dom)}, dom-cal={len(s_dom-s_cal)}")

# Check smiles consistency for overlap
m = cal[['identifier','canonical_smiles']].merge(dom[['identifier','canonical_smiles']], on='identifier', suffixes=('_cal','_dom'))
mismatch = (m['canonical_smiles_cal'] != m['canonical_smiles_dom']).sum()
print(f"smiles mismatches: {mismatch}")

# Check RF prob consistency
m2 = cal[['identifier','prob_class_1']].merge(dom[['identifier','prob_class_1']], on='identifier', suffixes=('_cal','_dom'))
maxdiff = (m2['prob_class_1_cal']-m2['prob_class_1_dom']).abs().max()
print(f"max RF prob diff cal vs dom: {maxdiff}")

# Merge in_domain
dom_sub = dom[['identifier','in_domain']].copy()
# normalize in_domain to bool
# csv stores True/False strings or bools
df = cal.merge(dom_sub, on='identifier', how='left')
print(f"merged rows={len(df)}, in_domain nulls={df['in_domain'].isna().sum()}")
print(df['in_domain'].value_counts(dropna=False))

# Core computation
# gnn_cal = gnn_pretrain_prob_cal, rf_prob = prob_class_1
df['gnn_cal'] = df['gnn_pretrain_prob_cal'].astype(float)
df['rf_prob'] = df['prob_class_1'].astype(float)

def logit_clip(p, eps=EPS):
    pc = np.clip(p, eps, 1.0-eps)
    return np.log(pc/(1.0-pc))

df['logit_gnn'] = logit_clip(df['gnn_cal'].values, EPS)
df['logit_rf'] = logit_clip(df['rf_prob'].values, EPS)

mean_gnn = df['logit_gnn'].mean()
std_gnn = df['logit_gnn'].std(ddof=1)
mean_rf = df['logit_rf'].mean()
std_rf = df['logit_rf'].std(ddof=1)
print(f"logit_gnn mean={mean_gnn:.6f} std={std_gnn:.6f} (ddof=1)")
print(f"logit_rf  mean={mean_rf:.6f} std={std_rf:.6f} (ddof=1)")

df['z_gnn'] = (df['logit_gnn']-mean_gnn)/std_gnn
df['z_rf'] = (df['logit_rf']-mean_rf)/std_rf
df['S'] = df['z_gnn'] + df['z_rf']

# Rank by S descending: 1 = best
df = df.sort_values('S', ascending=False).reset_index(drop=True)
df['S_rank'] = np.arange(1, len(df)+1)

# S distribution
print("S describe:")
print(df['S'].describe(percentiles=[0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99]).to_string())
print(f"S mean={df['S'].mean():.6f} std={df['S'].std(ddof=1):.6f} min={df['S'].min():.4f} max={df['S'].max():.4f}")
# thresholds
for q in [0.95, 0.9, 0.5]:
    print(f"S quantile {q}: {df['S'].quantile(q):.4f}")

# Save full rescore with required cols
out_cols = ['identifier','canonical_smiles','gnn_cal','rf_prob','S','S_rank','in_domain']
# ensure in_domain bool preserved
df_out = df[out_cols].copy()
df_out.to_csv(out_full, index=False)
print(f"saved {out_full} rows={len(df_out)}")

# Top116 by S
top116 = df_out.head(116).copy()
top116.to_csv(out_top116, index=False)
print(f"saved {out_top116} rows={len(top116)}")

# Consensus-T1: S_top5% & P_cal>=0.7 & RF>0.5 & in_domain=True
N = len(df)
k5 = int(np.ceil(0.05*N))
S_thresh = df['S'].quantile(0.95)
print(f"N={N}, top5% k={k5}, S>=p95 threshold={S_thresh:.6f}")
# S_top5% defined as S_rank <= k5
cons = df[(df['S_rank']<=k5) & (df['gnn_cal']>=0.7) & (df['rf_prob']>0.5) & (df['in_domain']==True)].copy()
print(f"consensus-T1 n={len(cons)}")
cons_out = cons[out_cols].copy()
cons_out.to_csv(out_cons, index=False)
print(f"saved {out_cons} rows={len(cons_out)}")

# Overlaps
paper = pd.read_csv(p_paper)
old = pd.read_csv(p_old116)
old_rflike = pd.read_csv(p_old116_rflike)
print(f"paper_rf116 n={len(paper)}, old_cal_top116 n={len(old)}, old_rflike n={len(old_rflike)}")

set_new116 = set(top116['identifier'])
set_cons = set(cons_out['identifier'])
set_paper = set(paper['identifier'])
set_old = set(old['identifier'])
set_oldrf = set(old_rflike['identifier'])

print(f"overlap new116 ∩ paper: {len(set_new116 & set_paper)}")
print(f"overlap new116 ∩ old_cal116: {len(set_new116 & set_old)}")
print(f"overlap new116 ∩ old_rflike116: {len(set_new116 & set_oldrf)}")
print(f"overlap cons ∩ paper: {len(set_cons & set_paper)}")
print(f"overlap cons ∩ old_cal116: {len(set_cons & set_old)}")
print(f"overlap cons ∩ new116: {len(set_cons & set_new116)}")

# Also: how many of paper_rf116 are in full set? And their S ranks?
paper_in_full = df[df['identifier'].isin(set_paper)][['identifier','gnn_cal','rf_prob','S','S_rank','in_domain']].sort_values('S_rank')
print(f"paper ids found in full: {len(paper_in_full)}/{len(set_paper)}")
print(paper_in_full.head(20).to_string(index=False))

# old in full ranks
old_in_full = df[df['identifier'].isin(set_old)][['identifier','gnn_cal','rf_prob','S','S_rank','in_domain']].sort_values('S_rank')
print(f"old ids found in full: {len(old_in_full)}/{len(set_old)}")

# Top10 rows
print("TOP10 rankfusion:")
print(df_out.head(10).to_string(index=False))

# Extra: distribution of in_domain in top slices
for k in [10,116,k5]:
    sub = df.head(k)
    print(f"top{k}: in_domain True={int((sub['in_domain']==True).sum())}/{k}, gnn_cal>=0.7={(sub['gnn_cal']>=0.7).sum()}, rf>0.5={(sub['rf_prob']>0.5).sum()}")

# Verify files exist
import pathlib
for p in [out_full, out_top116, out_cons]:
    print(p, pathlib.Path(p).exists(), pathlib.Path(p).stat().st_size)
