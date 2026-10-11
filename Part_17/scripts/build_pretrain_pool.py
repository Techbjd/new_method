"""Build Option C pretrain pool: MDM2-like, 0.4<maxT<0.85 vs 645 train.
Usage (repo root): python3 Part_17/scripts/build_pretrain_pool.py
Works locally and on Colab (auto-detects repo root, downloads ZINC250k if needed).
"""
import os, sys, tempfile, urllib.request, pandas as pd, numpy as np
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

REPO = Path(__file__).resolve().parents[2]  # Part_17/scripts/ -> repo root
SEED = 57
rng = np.random.RandomState(SEED)
OUT = str(REPO / "Part_17" / "data" / "pretrain_mdm2like_20k.csv")
ZINC_CACHE = os.path.join(tempfile.gettempdir(), "zinc250k.csv")
ZINC_URLS = [
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv",
]
TRAIN_CSV = str(REPO / "ro5_properties_filtered.csv")
_CANDIDATES = [
    REPO / "Part_17" / "data" / "mdm2_chembl_smiles.csv",  # tracked, Colab-safe
    REPO / "Natural_MDM2_Inhibitor_Discovery_using_ML" / "Part_12" / "mdm2_chembl_clean.csv",
]
CHEMBL_CSV = str(next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0]))
print(f"CHEMBL source: {CHEMBL_CSV}", flush=True)

def canon(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        if m is None: return None
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None

def fp_of(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)

print("== 1) Load train 645 ==", flush=True)
df_train = pd.read_csv(TRAIN_CSV)
print(f"train shape {df_train.shape}, cols {df_train.columns.tolist()}")
assert "canonical_smiles" in df_train.columns, "canonical_smiles missing in train!"
train_canon_list = [canon(s) for s in tqdm(df_train["canonical_smiles"].astype(str), desc="canon train")]
train_canon_list = [s for s in train_canon_list if s is not None]
print(f"train valid canon: {len(train_canon_list)}/{len(df_train)}")
train_set = set(train_canon_list)
print(f"train unique canon: {len(train_set)}")
train_fps = [fp_of(s) for s in tqdm(train_canon_list, desc="fp train")]
train_fps = [f for f in train_fps if f is not None]
print(f"train fps: {len(train_fps)}")

print("\n== 2) Load ChEMBL ==", flush=True)
df_c = pd.read_csv(CHEMBL_CSV, usecols=["canonical_smiles"])
print(f"chembl raw: {len(df_c)}")
c_canon = []
for s in tqdm(df_c["canonical_smiles"].astype(str), desc="canon chembl"):
    c = canon(s)
    if c is not None:
        c_canon.append(c)
print(f"chembl valid: {len(c_canon)}/{len(df_c)}")

print("\n== 3) ZINC250k ==", flush=True)
zinc_smiles_raw = None
# check local caches
local_hits = []
for root, _, files in os.walk("."):
    for fn in files:
        if "zinc" in fn.lower() and fn.endswith(".csv"):
            local_hits.append(os.path.join(root, fn))
    if len(local_hits) > 5: break
print(f"local zinc cache search: {local_hits[:5] if local_hits else 'none'}")
if os.path.exists(ZINC_CACHE):
    print(f"using cached {ZINC_CACHE}")
    zinc_csv = ZINC_CACHE
else:
    zinc_csv = ZINC_CACHE
    ok = False
    for url in ZINC_URLS:
        try:
            print(f"downloading {url} ...", flush=True)
            urllib.request.urlretrieve(url, zinc_csv)
            print(f"saved to {zinc_csv}, size {os.path.getsize(zinc_csv)}")
            ok = True
            break
        except Exception as e:
            print(f"download failed {url}: {e}")
    if not ok:
        # try HF datasets
        try:
            from datasets import load_dataset
            print("trying HF edmanft/zinc250k ...", flush=True)
            ds = load_dataset("edmanft/zinc250k", split="train")
            dfz = ds.to_pandas()
            print(f"HF cols: {dfz.columns.tolist()}, len {len(dfz)}")
            # find smiles col
            scol = [c for c in dfz.columns if "smil" in c.lower()][0]
            dfz.to_csv(zinc_csv, index=False)
            print(f"saved HF to {zinc_csv}")
            ok = True
        except Exception as e:
            print(f"HF failed: {e}")
            sys.exit(1)

dfz = pd.read_csv(zinc_csv)
print(f"zinc csv cols: {dfz.columns.tolist()}, shape {dfz.shape}")
scol = [c for c in dfz.columns if "smil" in c.lower()][0]
print(f"smiles column: {scol}")
z_canon = []
for s in tqdm(dfz[scol].astype(str), desc="canon zinc"):
    c = canon(s)
    if c is not None:
        z_canon.append(c)
print(f"zinc valid: {len(z_canon)}/{len(dfz)}")

print("\n== 4) Dedup + remove train exact matches ==", flush=True)
# dedup within each source, then across sources, track source priority: chembl first
seen = set()
pool = []  # (canon, source)
n_chembl_dup = 0
chembl_unique = set()
for c in c_canon:
    if c in chembl_unique:
        n_chembl_dup += 1
        continue
    chembl_unique.add(c)
print(f"chembl dedup: {len(chembl_unique)} unique (dups removed {n_chembl_dup})")
n_zinc_dup = 0
zinc_unique = set()
for c in z_canon:
    if c in zinc_unique:
        n_zinc_dup += 1
        continue
    zinc_unique.add(c)
print(f"zinc dedup: {len(zinc_unique)} unique (dups removed {n_zinc_dup})")
# remove train exact matches
chembl_kept = [c for c in chembl_unique if c not in train_set]
zinc_kept = [c for c in zinc_unique if c not in train_set]
print(f"chembl after train-exact removal: {len(chembl_kept)} (removed {len(chembl_unique)-len(chembl_kept)})")
print(f"zinc after train-exact removal: {len(zinc_kept)} (removed {len(zinc_unique)-len(zinc_kept)})")
# cross-source dedup (chembl priority)
zinc_set = set(zinc_kept)
overlap_cz = len(set(chembl_kept) & zinc_set)
zinc_kept2 = [c for c in zinc_kept if c not in set(chembl_kept)]
print(f"chembl-zinc overlap: {overlap_cz}, zinc after cross-dedup: {len(zinc_kept2)}")
for c in chembl_kept:
    pool.append((c, "chembl_mdm2"))
for c in zinc_kept2:
    pool.append((c, "zinc250k"))
print(f"combined pool: {len(pool)}")

print("\n== 5) Morgan maxT vs 645 train (BulkTanimoto) ==", flush=True)
pool_smiles = [p[0] for p in pool]
pool_sources = [p[1] for p in pool]
pool_fps = []
valid_idx = []
for i, smi in enumerate(tqdm(pool_smiles, desc="fp pool")):
    f = fp_of(smi)
    if f is not None:
        pool_fps.append(f)
        valid_idx.append(i)
print(f"pool fps valid: {len(pool_fps)}/{len(pool_smiles)}")
pool_smiles = [pool_smiles[i] for i in valid_idx]
pool_sources = [pool_sources[i] for i in valid_idx]
maxTs = np.zeros(len(pool_fps), dtype=np.float32)
for i, f in enumerate(tqdm(pool_fps, desc="BulkTanimoto max")):
    sims = DataStructs.BulkTanimotoSimilarity(f, train_fps)
    maxTs[i] = max(sims) if len(sims) else 0.0
print(f"maxT stats all: min {maxTs.min():.3f} max {maxTs.max():.3f} mean {maxTs.mean():.3f} med {np.median(maxTs):.3f}")

print("\n== 6) Filter 0.4<maxT<0.85, target 15-25k ==", flush=True)
mask = (maxTs > 0.4) & (maxTs < 0.85)
print(f"in-window: {mask.sum()} / {len(maxTs)} ({mask.sum()/len(maxTs)*100:.2f}%)")
print(f"  <=0.4: {(maxTs<=0.4).sum()}, >=0.85: {(maxTs>=0.85).sum()}")
filt_smi = [pool_smiles[i] for i in np.where(mask)[0]]
filt_src = [pool_sources[i] for i in np.where(mask)[0]]
filt_maxT = maxTs[mask]
import collections
print("filtered per-source:", dict(collections.Counter(filt_src)))
n_filt = len(filt_smi)
if n_filt > 25000:
    print(f"downsampling {n_filt} -> 20000 (seed {SEED})")
    sel = rng.choice(n_filt, 20000, replace=False)
    filt_smi = [filt_smi[i] for i in sel]
    filt_src = [filt_src[i] for i in sel]
    filt_maxT = filt_maxT[sel]
elif n_filt < 15000:
    print(f"WARNING: only {n_filt} in window (<15k target). Keeping all.")
else:
    print(f"keeping all {n_filt} (within 15-25k target)")
# sort by maxT desc for nicer file
order = np.argsort(-filt_maxT)
filt_smi = [filt_smi[i] for i in order]
filt_src = [filt_src[i] for i in order]
filt_maxT = filt_maxT[order]

out_df = pd.DataFrame({"canonical_smiles": filt_smi, "maxT_to_train": filt_maxT, "source": filt_src})
os.makedirs(os.path.dirname(OUT), exist_ok=True)
out_df.to_csv(OUT, index=False)
print(f"\nsaved {OUT} shape {out_df.shape}")

print("\n== 7) Report stats ==", flush=True)
print(f"FILE: {os.path.abspath(OUT)}")
print("per-source:", out_df["source"].value_counts().to_dict())
print(out_df["maxT_to_train"].describe(percentiles=[0.05,0.25,0.5,0.75,0.95]).to_string())
hist, edges = np.histogram(out_df["maxT_to_train"], bins=10)
for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
    print(f"  [{lo:.2f},{hi:.2f}): {h}")
bg_rate = mask.sum()/len(maxTs)*100
print(f"background AD-rate (0.4<maxT<0.85 in raw pool): {bg_rate:.2f}%")
print(f"filtered pool AD-rate: 100.00% by construction; lift {100/bg_rate:.1f}x, +{100-bg_rate:.2f}pp")
print("first 3 rows:")
print(out_df.head(3).to_string(index=False))
