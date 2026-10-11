import pandas as pd, numpy as np, os, pickle, tempfile
from pathlib import Path
BASE = str(Path(__file__).resolve().parents[2])  # repo root (local or Colab clone)
TMPD = os.path.join(tempfile.gettempdir(), "mdm2_tmp")
os.makedirs(TMPD, exist_ok=True)
OUT_AUDIT=os.path.join(BASE,"Part_17/results/calibration_audit.csv")
OUT_REL=os.path.join(BASE,"Part_17/results/reliability_table.csv")

# --- rebuild calibration_audit with disagreement summary row ---
audit=pd.read_csv(OUT_AUDIT)
print("current audit rows:", len(audit))
# load disagreement stats from previous run logic
raw=pd.read_csv(os.path.join(BASE,"Part_17/results/coconut_gnn_screening_full.csv"), usecols=['gnn_pretrain_prob','gnn_scratch_prob'])
dis=np.abs(raw['gnn_pretrain_prob'].values-raw['gnn_scratch_prob'].values)
dis_row=pd.DataFrame([{'column':'disagreement_abs_pre_minus_scr','n':len(dis),'mean':float(dis.mean()),'std':float(dis.std(ddof=1)),
 'min':float(dis.min()),'max':float(dis.max()),'median':float(np.median(dis)),
 'n_gt_05':int((dis>0.5).sum()),'n_gt_085':int((dis>0.85).sum()),'n_gt_09':int((dis>0.9).sum()),
 'frac_gt_05':float((dis>0.5).mean()),'frac_gt_085':float((dis>0.85).mean()),'frac_gt_09':float((dis>0.9).mean())}])
# add distrust>0.3 as extra columns? append as new cols for all rows
audit['n_gt_03']=np.nan; audit['frac_gt_03']=np.nan
# compute n>0.3 for prob columns
import pandas as pd
cal=pd.read_csv(os.path.join(BASE,"Part_17/results/coconut_gnn_screening_full_calibrated.csv"), usecols=['gnn_pretrain_prob_raw','gnn_pretrain_prob_cal','prob_class_1'])
colmap={'pretrain_raw':raw['gnn_pretrain_prob'],'scratch_raw':raw['gnn_scratch_prob'],'rf_raw':pd.read_csv(os.path.join(BASE,'Part_17/results/coconut_gnn_screening_full.csv'),usecols=['rf_prob'])['rf_prob'],
 'pretrain_raw_in_calfile':cal['gnn_pretrain_prob_raw'],'pretrain_cal_Tscaled':cal['gnn_pretrain_prob_cal'],'rf_in_calfile(prob_class_1)':cal['prob_class_1']}
for i,r in audit.iterrows():
    s=colmap.get(r['column'])
    if s is not None:
        audit.at[i,'n_gt_03']=int((s>0.3).sum()); audit.at[i,'frac_gt_03']=float((s>0.3).mean())
# disagreement row n_gt_03
dis_row['n_gt_03']=int((dis>0.3).sum()); dis_row['frac_gt_03']=float((dis>0.3).mean())
audit_full=pd.concat([audit,dis_row],ignore_index=True)
audit_full.to_csv(OUT_AUDIT,index=False)
print(audit_full.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print(f"rewrote {OUT_AUDIT} rows={len(audit_full)}")

# --- rebuild reliability_table combined ---
def ece_nll_table(y_true,y_prob,n_bins=10):
    y_true=np.asarray(y_true,float); y_prob=np.asarray(y_prob,float)
    eps=1e-15; p=np.clip(y_prob,eps,1-eps)
    nll=float(-(y_true*np.log(p)+(1-y_true)*np.log(1-p)).mean()); brier=float(((y_prob-y_true)**2).mean())
    edges=np.linspace(0,1,n_bins+1); rows=[]; ece=0; mce=0
    for i in range(n_bins):
        lo,hi=edges[i],edges[i+1]
        m=(y_prob>=lo)&(y_prob<=hi) if i==n_bins-1 else (y_prob>=lo)&(y_prob<hi)
        n=int(m.sum())
        if n==0: rows.append(dict(bin=i,lo=lo,hi=hi,n_bin=0,frac_pos=np.nan,mean_conf=np.nan,gap=np.nan,weight=0,contrib=0))
        else:
            acc=float(y_true[m].mean()); conf=float(y_prob[m].mean()); gap=abs(acc-conf); w=n/len(y_true)
            ece+=w*gap; mce=max(mce,gap); rows.append(dict(bin=i,lo=lo,hi=hi,n_bin=n,frac_pos=acc,mean_conf=conf,gap=gap,weight=w,contrib=w*gap))
    return ece,mce,nll,brier,pd.DataFrame(rows)

fz=np.load(os.path.join(BASE,'Part_17/data/featurized_curated.npz')); y=fz['cur_labels'].astype(int)
rf=pickle.load(open(os.path.join(BASE,'Part_17/models/rf_scaffold.pkl'),'rb')); X=fz['cur_fps']
eval_sets=[
 ('rf_scaffold','scaffold',y[np.asarray(fz['te_cur_new'],int)],rf.predict_proba(X[np.asarray(fz['te_cur_new'],int)])[:,1]),
 ('rf_scaffold','random',y[np.asarray(fz['te_rand_new'],int)],rf.predict_proba(X[np.asarray(fz['te_rand_new'],int)])[:,1]),
  ('hybrid_scaffold_pretrain_r0.5','scaffold',y[np.asarray(fz['te_cur_new'],int)],np.load(os.path.join(TMPD,'probs_scaffold_pretrain.npy'))),
  ('hybrid_scaffold_scratch_r0.5','scaffold',y[np.asarray(fz['te_cur_new'],int)],np.load(os.path.join(TMPD,'probs_scaffold_scratch.npy'))),
  ('hybrid_random_pretrain_r0.5','random',y[np.asarray(fz['te_rand_new'],int)],np.load(os.path.join(TMPD,'probs_random_pretrain.npy'))),
  ('hybrid_random_scratch_r0.5','random',y[np.asarray(fz['te_rand_new'],int)],np.load(os.path.join(TMPD,'probs_random_scratch.npy'))),
 ('hybrid_mask_aug_focal','scaffold',pd.read_csv(os.path.join(BASE,'Part_17/results_rf_like/test_scaffold_hybrid_mask_aug_focal.csv'))['label'].values,pd.read_csv(os.path.join(BASE,'Part_17/results_rf_like/test_scaffold_hybrid_mask_aug_focal.csv'))['prob'].values),
]
all_rows=[]
for model,split,yt,yp in eval_sets:
    ece,mce,nll,brier,tab=ece_nll_table(yt,yp,10)
    tab['model']=model; tab['split']=split; tab['n']=len(yt); tab['n_pos']=int((np.asarray(yt)==1).sum())
    tab['ece']=ece; tab['mce']=mce; tab['nll']=nll; tab['brier']=brier
    all_rows.append(tab)
    print(f"{model} [{split}] n={len(yt)} ECE={ece:.4f} MCE={mce:.4f} NLL={nll:.4f} Brier={brier:.4f}")
rel=pd.concat(all_rows,ignore_index=True)
rel=rel[['model','split','n','n_pos','ece','mce','nll','brier','bin','lo','hi','n_bin','frac_pos','mean_conf','gap','weight','contrib']]
rel.to_csv(OUT_REL,index=False)
print(f"saved {OUT_REL} rows={len(rel)}")
print(rel.head(12).to_string(index=False))
