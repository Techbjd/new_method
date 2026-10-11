import pandas as pd, numpy as np, os, pickle, json
from pathlib import Path
BASE = str(Path(__file__).resolve().parents[2])  # repo root (local or Colab clone)
RAW = os.path.join(BASE, "Part_17/results/coconut_gnn_screening_full.csv")
CAL = os.path.join(BASE, "Part_17/results/coconut_gnn_screening_full_calibrated.csv")
OUT_AUDIT = os.path.join(BASE, "Part_17/results/calibration_audit.csv")
OUT_REL = os.path.join(BASE, "Part_17/results/reliability_table.csv")
OUT_FLAGS = os.path.join(BASE, "Part_17/results/coconut_disagreement_flags.csv")

def dist_stats(s):
    s = np.asarray(s, dtype=float)
    return dict(n=len(s), mean=float(s.mean()), std=float(s.std(ddof=1) if len(s)>1 else 0.0),
                min=float(s.min()), max=float(s.max()),
                median=float(np.median(s)),
                n_gt_05=int((s>0.5).sum()), n_gt_085=int((s>0.85).sum()), n_gt_09=int((s>0.9).sum()),
                frac_gt_05=float((s>0.5).mean()), frac_gt_085=float((s>0.85).mean()), frac_gt_09=float((s>0.9).mean()))

def ece_nll_table(y_true, y_prob, n_bins=10):
    y_true=np.asarray(y_true,dtype=float); y_prob=np.asarray(y_prob,dtype=float)
    eps=1e-15
    p=np.clip(y_prob,eps,1-eps)
    nll=float(-(y_true*np.log(p)+(1-y_true)*np.log(1-p)).mean())
    brier=float(((y_prob-y_true)**2).mean())
    edges=np.linspace(0,1,n_bins+1)
    rows=[]; ece=0.0; mce=0.0
    for i in range(n_bins):
        lo,hi=edges[i],edges[i+1]
        if i==n_bins-1: m=(y_prob>=lo)&(y_prob<=hi)
        else: m=(y_prob>=lo)&(y_prob<hi)
        # handle lo==0 edge: include 0
        if i==0: m=(y_prob>=lo)&(y_prob<hi) if n_bins>1 else np.ones_like(y_prob,dtype=bool)
        n=int(m.sum())
        if n==0:
            rows.append(dict(bin=i,lo=lo,hi=hi,n=0,frac_pos=np.nan,mean_conf=np.nan,gap=np.nan,weight=0.0,contrib=0.0))
        else:
            acc=float(y_true[m].mean()); conf=float(y_prob[m].mean())
            gap=abs(acc-conf); w=n/len(y_true); contrib=w*gap
            ece+=contrib; mce=max(mce,gap)
            rows.append(dict(bin=i,lo=lo,hi=hi,n=n,frac_pos=acc,mean_conf=conf,gap=gap,weight=w,contrib=contrib))
    return dict(ece=float(ece),mce=float(mce),nll=float(nll),brier=float(brier),table=pd.DataFrame(rows))

print("Loading...")
raw=pd.read_csv(RAW, usecols=['identifier','gnn_pretrain_prob','gnn_scratch_prob','rf_prob'])
cal=pd.read_csv(CAL, usecols=['identifier','prediction','prob_class_0','prob_class_1','gnn_pretrain_prob_raw','gnn_pretrain_prob_cal'])
print(f"raw {len(raw)} cal {len(cal)}")

# 1) distribution stats
cols_raw={'pretrain_raw':raw['gnn_pretrain_prob'],'scratch_raw':raw['gnn_scratch_prob'],'rf_raw':raw['rf_prob']}
cols_cal={'pretrain_raw_in_calfile':cal['gnn_pretrain_prob_raw'],'pretrain_cal_Tscaled':cal['gnn_pretrain_prob_cal'],'rf_in_calfile(prob_class_1)':cal['prob_class_1']}
audit_rows=[]
for name,s in list(cols_raw.items())+list(cols_cal.items()):
    d=dist_stats(s); d['column']=name; audit_rows.append(d)
audit=pd.DataFrame(audit_rows)[['column','n','mean','std','min','max','median','n_gt_05','n_gt_085','n_gt_09','frac_gt_05','frac_gt_085','frac_gt_09']]
print(audit.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
audit.to_csv(OUT_AUDIT,index=False)
print(f"saved {OUT_AUDIT}")

# 2) disagreement
dis=np.abs(raw['gnn_pretrain_prob'].values-raw['gnn_scratch_prob'].values)
distrust=dis>0.3
print(f"\ndisagreement |pretrain-scratch|: mean={dis.mean():.4f} std={dis.std():.4f} median={np.median(dis):.4f} max={dis.max():.4f}")
print(f"distrust (>0.3): n={distrust.sum()} / {len(dis)} = {distrust.mean()*100:.2f}%")
print(f" >0.5: {(dis>0.5).sum()} ({(dis>0.5).mean()*100:.2f}%)")
# quantiles
print("quantiles:", dict(zip([0.5,0.75,0.9,0.95,0.99],[float(np.quantile(dis,q)) for q in [0.5,0.75,0.9,0.95,0.99]])))
# save flags subset (distrust only) + summary row appended? Keep flags file small: distrust subset with probs
m=pd.merge(raw,cal[['identifier','gnn_pretrain_prob_raw','gnn_pretrain_prob_cal']],on='identifier',how='left')
m['disagreement']=np.abs(m['gnn_pretrain_prob']-m['gnn_scratch_prob'])
m['distrust_gt03']=(m['disagreement']>0.3).astype(int)
flags_distrust=m[m['distrust_gt03']==1][['identifier','gnn_pretrain_prob','gnn_scratch_prob','rf_prob','gnn_pretrain_prob_cal','disagreement','distrust_gt03']].sort_values('disagreement',ascending=False)
flags_distrust.to_csv(OUT_FLAGS,index=False)
print(f"saved distrust subset {len(flags_distrust)} rows -> {OUT_FLAGS}")
# also print calibration shrinkage stats
print(f"\ncal shrinkage: raw_in_cal mean={cal['gnn_pretrain_prob_raw'].mean():.4f} -> cal mean={cal['gnn_pretrain_prob_cal'].mean():.4f}; n>0.9 {int((cal['gnn_pretrain_prob_raw']>0.9).sum())} -> {int((cal['gnn_pretrain_prob_cal']>0.9).sum())}; n>0.85 {int((cal['gnn_pretrain_prob_raw']>0.85).sum())} -> {int((cal['gnn_pretrain_prob_cal']>0.85).sum())}")

# 3) ECE/NLL where labels exist
print("\n=== LABEL AVAILABILITY ===")
print("COCONUT screening (154647): NO labels -> ECE/NLL NOT computable on screening set.")
# curated
fz=np.load(os.path.join(BASE,'Part_17/data/featurized_curated.npz'))
print("featurized_curated keys:", list(fz.files), "cur_labels dist:", np.bincount(fz['cur_labels'].astype(int)))
print("te_cur (scaffold) n=",len(fz['te_cur_new']), " te_rand n=",len(fz['te_rand_new']))
# Part_6 legacy
try:
    y6=pd.read_csv(os.path.join(BASE,'Natural_MDM2_Inhibitor_Discovery_using_ML/Part_6/y_test.csv'))
    print("Part_6 y_test:", y6.shape, y6.columns.tolist(), y6.iloc[:,0].value_counts().to_dict() if y6.shape[1]>=1 else '')
except Exception as e:
    print("Part_6 y_test err", e)

# 3a) RF scaffold ECE via reload
rf_path=os.path.join(BASE,'Part_17/models/rf_scaffold.pkl')
rf=pickle.load(open(rf_path,'rb'))
X=fz['cur_fps']; y=fz['cur_labels'].astype(int)
for tag,idx in [('scaffold',fz['te_cur_new']),('random',fz['te_rand_new'])]:
    idx=np.asarray(idx,dtype=int)
    probs=rf.predict_proba(X[idx])[:,1]
    r=ece_nll_table(y[idx],probs,10)
    print(f"\n[RF-scaffold.pkl on curated-{tag} test n={len(idx)} pos={(y[idx]==1).sum()}] ECE={r['ece']:.4f} MCE={r['mce']:.4f} NLL={r['nll']:.4f} Brier={r['brier']:.4f} mean_p={probs.mean():.4f}")
    print(r['table'].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# 3b) labeled GNN test file (hybrid_mask_aug_focal scaffold)
proxy_path=os.path.join(BASE,'Part_17/results_rf_like/test_scaffold_hybrid_mask_aug_focal.csv')
if os.path.exists(proxy_path):
    df=pd.read_csv(proxy_path)
    r=ece_nll_table(df['label'].values,df['prob'].values,10)
    print(f"\n[hybrid_mask_aug_focal scaffold test n={len(df)}] ECE={r['ece']:.4f} MCE={r['mce']:.4f} NLL={r['nll']:.4f} Brier={r['brier']:.4f}")
    print(r['table'].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    r['table'].to_csv(OUT_REL,index=False)
    print(f"saved reliability (focal scaffold) -> {OUT_REL}")

