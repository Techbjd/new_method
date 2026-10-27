
# =============================================================================
# MDM2 Part_20 - KAGGLE SINGLE-SOURCE RUNNER
# =============================================================================
# ONE INPUT DATASET (name it mdm2-single-source). Put these four files in it:
#   ro5_properties_filtered.csv
#   screening_results.csv
#   cuts.json
#   common.py
#
# ONE OUTPUT FOLDER:
#   /kaggle/working/mdm2_outputs/
#
# This file works as a Kaggle script and when pasted into a notebook cell.
# The setup stage copies the four inputs into a private work tree, changes into
# that tree, then runs the pipeline in this same process (the kernel is not
# replaced). A try/finally collects every CSV/JSON/PNG/TXT/MD/PDF/HTML output
# into the single output folder, even if the pipeline raises. It writes
# KAGGLE_OUTPUT_MANIFEST.json with SHA256 hashes of inputs and outputs.
#
# Optional overrides:
#   MDM2_INPUT_DIR=/kaggle/input/mdm2-single-source
#   MDM2_OUTPUT_DIR=/kaggle/working/mdm2_outputs
#   MDM2_DRY_RUN=1   # setup check only; do not start training
# =============================================================================
import os as _os
import sys as _sys
import shutil as _shutil
import hashlib as _hashlib
import json as _json
import time as _time
import tempfile as _tempfile
import urllib.request as _urllib_request
from pathlib import Path as _Path

INPUT_DIR = _Path(_os.environ.get("MDM2_INPUT_DIR", "/kaggle/input/mdm2-single-source"))
OUTPUT_DIR = _Path(_os.environ.get("MDM2_OUTPUT_DIR", "/kaggle/working/mdm2_outputs"))
WORK_DIR = _Path(_os.environ.get("MDM2_WORK_DIR", str(OUTPUT_DIR / "_work")))
STAGE = _os.environ.get("MDM2_STAGE", "setup")
_RUN_DIR = _Path.cwd()
_ACTIVE_RUN_DIR = None
REQUIRED_FILES = [
    "ro5_properties_filtered.csv",
    "screening_results.csv",
    "cuts.json",
    "common.py",
]
GH_BASE = _os.environ.get(
    "MDM2_GH_BASE",
    "https://raw.githubusercontent.com/Techbjd/new_method/kaggle-single-source",
)
INPUT_GH_FILES = {
    "ro5_properties_filtered.csv": "00_data/training/ro5_properties_filtered.csv",
    "screening_results.csv": "00_data/screening/screening_results.csv",
    "cuts.json": "00_data/config/cuts.json",
    "common.py": "02_gnn_models/part18_gnn_triage/common.py",
}
INPUT_MIN_BYTES = {
    "ro5_properties_filtered.csv": 10_000,
    "screening_results.csv": 100_000,
    "cuts.json": 1,
    "common.py": 1_024,
}
OUTPUT_SUFFIXES = {".csv", ".json", ".png", ".txt", ".md", ".pdf", ".html"}


def _find_input(name):
    candidates = [
        INPUT_DIR / name,
        INPUT_DIR / "data" / name,
        INPUT_DIR / "inputs" / name,
        INPUT_DIR / "part20" / name,
    ]
    for p in candidates:
        if p.is_file():
            return p
    if INPUT_DIR.exists():
        hits = sorted(INPUT_DIR.rglob(name))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        "Missing required input: %s\n"
        "Put exactly these four files in ONE Kaggle dataset:\n"
        "  ro5_properties_filtered.csv\n"
        "  screening_results.csv\n"
        "  cuts.json\n"
        "  common.py\n"
        "Dataset path: %s" % (name, INPUT_DIR)
    )


def _resolve_input(name):
    """Use the Kaggle dataset when present; otherwise download from the repo branch."""
    try:
        return _find_input(name)
    except FileNotFoundError:
        rel = INPUT_GH_FILES.get(name)
        if not rel:
            raise
        dest = _Path(_tempfile.gettempdir()) / "mdm2_single_source_inputs" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        min_bytes = INPUT_MIN_BYTES.get(name, 1)
        if dest.is_file() and dest.stat().st_size >= min_bytes:
            print("[kaggle] GitHub fallback cached:", dest)
            return dest
        url = GH_BASE + "/" + rel
        tmp = str(dest) + ".part"
        try:
            print("[kaggle] input dataset missing; downloading", url)
            _urllib_request.urlretrieve(url, tmp)
            _os.replace(tmp, dest)
        except Exception as exc:
            try:
                _os.remove(tmp)
            except OSError:
                pass
            raise RuntimeError(
                "Input file %r is not in %s and GitHub fallback failed: %s\n"
                "Attach ONE Kaggle dataset named 'mdm2-single-source' with the four files, "
                "or check internet access." % (name, INPUT_DIR, exc)
            ) from exc
        print("[kaggle] fallback %s (%.2f MB)" % (dest, dest.stat().st_size / 1e6))
        return dest


def _sha256(path, chunk=1 << 20):
    h = _hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _collect_outputs(run_dir=None):
    run_dir = _Path(run_dir) if run_dir is not None else _RUN_DIR
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        outputs = []
        for p in sorted(run_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in OUTPUT_SUFFIXES and p.name != "run_pipeline.py":
                dst = OUTPUT_DIR / p.name
                _shutil.copy2(p, dst)
                outputs.append({
                    "file": p.name,
                    "bytes": dst.stat().st_size,
                    "sha256": _sha256(dst),
                })
        inputs = {}
        for name in REQUIRED_FILES:
            try:
                p = _resolve_input(name)
                inputs[name] = {
                    "path": str(p),
                    "bytes": p.stat().st_size,
                    "sha256": _sha256(p),
                }
            except Exception as exc:
                inputs[name] = {"error": str(exc)}
        manifest = {
            "created_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
            "input_dir": str(INPUT_DIR),
            "output_dir": str(OUTPUT_DIR),
            "run_dir": str(run_dir),
            "inputs": inputs,
            "outputs": outputs,
        }
        (OUTPUT_DIR / "KAGGLE_OUTPUT_MANIFEST.json").write_text(
            _json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print("[kaggle] %d output files -> %s" % (len(outputs), OUTPUT_DIR))
    except Exception as exc:
        print("[kaggle] output collection failed:", repr(exc))


def _setup_and_prepare():
    train = _resolve_input("ro5_properties_filtered.csv")
    library = _resolve_input("screening_results.csv")
    cuts = _resolve_input("cuts.json")
    common = _resolve_input("common.py")

    if not INPUT_DIR.exists():
        print("[kaggle] input dataset not found; using GitHub fallback:", GH_BASE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _os.chdir(str(OUTPUT_DIR))
    if WORK_DIR.exists():
        _shutil.rmtree(WORK_DIR)
    for old in OUTPUT_DIR.iterdir():
        if old.is_file():
            old.unlink()

    train_dir = WORK_DIR / "00_data" / "training"
    lib_dir = WORK_DIR / "00_data" / "screening"
    cfg_dir = WORK_DIR / "00_data" / "config"
    common_dir = WORK_DIR / "02_gnn_models" / "part18_gnn_triage"
    run_dir = WORK_DIR / "03_screening_pipeline" / "part20_fixed"
    cache_dir = run_dir / "cache"
    for d in (train_dir, lib_dir, cfg_dir, common_dir, run_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    _shutil.copy2(train, train_dir / "ro5_properties_filtered.csv")
    _shutil.copy2(library, lib_dir / "screening_results.csv")
    _shutil.copy2(cuts, cfg_dir / "cuts.json")
    _shutil.copy2(common, common_dir / "common.py")
    _shutil.copy2(common, cache_dir / "gnn_common.py")

    print("[kaggle] input:  ", INPUT_DIR)
    print("[kaggle] train:  ", train, "->", train_dir / "ro5_properties_filtered.csv")
    print("[kaggle] library:", library, "->", lib_dir / "screening_results.csv")
    print("[kaggle] cuts:   ", cuts)
    print("[kaggle] common: ", common)
    print("[kaggle] work:   ", WORK_DIR)
    print("[kaggle] output: ", OUTPUT_DIR)

    if _os.environ.get("MDM2_DRY_RUN") == "1":
        print("[kaggle] dry run: setup complete; pipeline not started")
        return None

    return run_dir


if STAGE != "run":
    _ACTIVE_RUN_DIR = _setup_and_prepare()
    if _ACTIVE_RUN_DIR is not None:
        _os.chdir(str(_ACTIVE_RUN_DIR))
        STAGE = "run"
else:
    _ACTIVE_RUN_DIR = _RUN_DIR

if STAGE == "run":
    try:
        # ===== CELL 00 (markdown) =====
        # # MDM2 — Dual-GPU Regression Pipeline (train → screen), resume-safe
        #
        # **What this notebook does (03_screening_pipeline/part20_fixed upgrade):**
        #
        # 1. **Regression, not classification.** Predicts continuous pIC50 with a delta-ML setup: Morgan/descriptor baseline + GNN residual, so the GNN only has to learn the correction.
        # 2. **Uses both Kaggle T4 GPUs.** GCN and GIN train/screen in parallel threads (cuda:0 / cuda:1); AttentiveFP and HFooladi-GIN also use the second GPU.
        # 3. **Idempotent and resume-safe.** Downloads, fingerprints, graphs, model weights, library predictions and AD caches are reused unless `FORCE_RETRAIN=True`.
        # 4. **GitHub-first data.** Training table `00_data/training/ro5_properties_filtered.csv`, 154k screening library `00_data/screening/screening_results.csv`, and the 78-dim graph featurizer `02_gnn_models/part18_gnn_triage/common.py`; local copies win when present.
        # 5. **Strict applicability-domain evidence.** Same Morgan r3/512 fingerprint for train and screen. Reports nearest neighbours to all 645, to the 464 train compounds, to the train-only actives, and to the train-only inactives. The legacy `in_AD` flag uses `maxT_464 >= 0.40` only for comparison. Shortlist filtering uses `ad_pass`: strict AD `maxT_464 >= AD_CUT_STRICT (0.52)`, nearest active `maxT_train_act >= 0.40`, and positive active–inactive margin `>= 0.02`.
        # 6. **Honest accuracy.** Scaffold split 464/52/129 with canonical-overlap and shared-scaffold leak checks. Early stopping, linear calibration, and primary-model selection use validation only; test is scored once. The ensemble is the simple mean of calibrated GCN + GIN + AttentiveFP (simple mean on 52 validation compounds is not used). An 80% confidence band comes from validation residuals only.
        # 7. **Deduplicated shortlist, not a hit list.** One row per parent compound and per Murcko scaffold cluster, capped at 12 diverse parents. Core-model agreement requires range <= 0.70 and std <= 0.40 pIC50. The deliverable is `shortlist_low_confidence.csv`; these are **low-confidence hypotheses, not validated hits**. Old top-100 and T1/T2/T3 files are retained only as diagnostic/threshold artifacts.
        # 8. **Full audit.** Rebuilds graphs and re-predicts a sample, checks split sizes, AD flags, model agreement, calibration, finite predictions, and output schemas.
        #
        # **Outputs:** `screening_full_regression.csv`, `shortlist_low_confidence.csv`, `threshold_T1/T2/T3_not_hits.csv`, `diagnostic_top100_AD/raw.csv`, `model_comparison.csv`, `screening_regression_QC.png`, `README_SHORTLIST.txt`, `repeated_scaffold_cv.csv`, `official_test_enrichment.csv`, `model_selection_stability.csv`, `robust_evaluation.json`, `run_manifest.json`, `dl_full_predictions.csv`, `dl_training_history.csv`, `dl_audit.json`.

        # ===== CELL 01 (code) =====
        # 0 — Setup: verify/install ALL packages (safe to re-run)
        import os, sys, subprocess, importlib, json, time, hashlib, warnings, urllib.request

        REQUIRED = [
            ('numpy', 'numpy'),
            ('pandas', 'pandas'),
            ('sklearn', 'scikit-learn>=1.3'),
            ('rdkit', 'rdkit'),
            ('matplotlib', 'matplotlib'),
            ('joblib', 'joblib'),
            ('gdown', 'gdown'),
            ('torch', 'torch'),
            ('torch_geometric', 'torch_geometric'),
        ]
        missing = []
        for mod, pkg in REQUIRED:
            try:
                importlib.import_module(mod)
            except Exception:
                missing.append(pkg)
        if missing:
            print('[setup] installing missing packages:', missing)
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet'] + missing, check=False)
        else:
            print('[setup] all required packages already installed — nothing to download')

        IN_COLAB = os.path.isdir('/content')
        IN_KAGGLE = os.path.isdir('/kaggle/working')
        print('IN_COLAB =', IN_COLAB, '| IN_KAGGLE =', IN_KAGGLE)

        import torch, sklearn, numpy as np, pandas as pd, rdkit
        warnings.filterwarnings('ignore')

        def torch_load_compat(path):
            try:
                return torch.load(path, weights_only=False)
            except TypeError:
                return torch.load(path)

        print('sklearn', sklearn.__version__, '| rdkit', rdkit.__version__, '| torch', torch.__version__)
        print('cuda available', torch.cuda.is_available(), '| n_gpu', torch.cuda.device_count(), '| cpu_count', os.cpu_count())
        for i in range(torch.cuda.device_count()):
            print('  GPU', i, torch.cuda.get_device_name(i))

        # ===== CELL 02 (code) =====
        # 1 — Config (single source of truth) + resume flags
        import os, json, numpy as np
        RANDOM_SEED = 57
        np.random.seed(RANDOM_SEED)

        MORGAN_RADIUS = 3
        MORGAN_NBITS = 512              # SAME fingerprint for train + screen (AD axis)

        GH_BASE = 'https://raw.githubusercontent.com/Techbjd/new_method/kaggle-single-source'
        GH_FILES = {
            'train_csv':   '00_data/training/ro5_properties_filtered.csv',
            'library_csv': '00_data/screening/screening_results.csv',      # filtered COCONUT (paper parity)
            'common_py':   '02_gnn_models/part18_gnn_triage/common.py',      # 78-dim graph featurizer
        }

        DATA_DIR = 'data'
        CACHE_DIR = 'cache'
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)

        RESUME = True                   # skip finished downloads / featurization / training / AD
        FORCE_RETRAIN = False           # True -> ignore cached GCN/GIN models
        USE_DUAL_GPU = True             # GCN on cuda:0, GIN on cuda:1 (train + screen)

        GNN_BATCH = 64
        SCREEN_BATCH = 512
        GNN_EPOCHS = 120
        GNN_LR = 1e-3
        GNN_WD = 1e-4
        GNN_HIDDEN = 128
        GNN_LAYERS = 3
        GNN_PATIENCE = 15

        TEST_SIZE = 0.20
        VAL_FRAC = 0.10

        AD_CUT = 0.40
        AD_CUT_STRICT = 0.52        # strict AD for the shortlist (0.50-0.55)
        ACTIVE_MIN_T = 0.40         # nearest active train compound must reach this
        ACTIVE_MARGIN_MIN = 0.02    # active neighbor must beat inactive neighbor
        MAX_MODEL_RANGE = 0.70      # GCN/GIN/AFP max-min agreement
        MAX_MODEL_STD = 0.40        # core-model std agreement
        SHORTLIST_MAX = 12          # diverse parent compounds, not 100
        CONF_ALPHA = 0.20           # 80% validation-residual interval
        RANK_LAMBDA = 1.0           # penalty on core-model std for conservative ranking
        T1_CUT = 7.0
        T2_CUT = 6.5

        for _p in ['../../00_data/config/cuts.json', '00_data/config/cuts.json', 'cuts.json', '../cuts.json',
                   os.path.join(DATA_DIR, 'cuts.json'), '/kaggle/input/mdm2-cuts/cuts.json',
                   os.path.join(str(INPUT_DIR), 'cuts.json'), os.path.join(str(INPUT_DIR), 'data', 'cuts.json')]:
            if os.path.exists(_p):
                try:
                    _c = json.load(open(_p))
                    AD_CUT = float(_c.get('ad_cut', AD_CUT))
                    AD_CUT_STRICT = float(_c.get('ad_cut_strict', AD_CUT_STRICT))
                    T1_CUT = float(_c.get('gnn_hit_cut', T1_CUT))
                    print('cuts.json loaded from', _p, '| ad_cut', AD_CUT, '| gnn_hit_cut', T1_CUT)
                except Exception as _e:
                    print('cuts.json read failed:', _e)
                break

        print('CONFIG | seed', RANDOM_SEED, '| morgan r%d/%d' % (MORGAN_RADIUS, MORGAN_NBITS),
              '| AD_CUT', AD_CUT, '| AD_STRICT', AD_CUT_STRICT, '| T1_CUT', T1_CUT, '| resume', RESUME, '| dual_gpu', USE_DUAL_GPU)

        # ---- accuracy upgrade flags (extra models + calibration) ----
        USE_EXTRA_MODELS = True          # AttentiveFP (DeepChem lineage) + HFooladi GIN (researcher)
        RUN_CHEMBERTA = True             # only runs when `transformers` is importable (Kaggle has it)
        CHEMBERTA_MODEL = 'seyonec/ChemBERTa-zinc-base-v1'
        CHEMBERTA_MAXLEN = 128
        CHEMBERTA_BATCH = 64
        CHEMBERTA_EPOCHS = 200
        CHEMBERTA_PATIENCE = 25
        AFP_HIDDEN = 128
        AFP_LAYERS = 2
        AFP_TIMESTEPS = 2
        HF_GIN_HIDDEN = 64
        EXTRA_EPOCHS = 120
        EXTRA_PATIENCE = 20
        CALIBRATE = True                 # linear calibration fitted on VALIDATION only

        try:
            import transformers as _tr
            HAVE_TRANSFORMERS = True
            print('[optional] transformers', _tr.__version__, '-> ChemBERTa enabled')
        except Exception:
            HAVE_TRANSFORMERS = False
            print('[optional] transformers not installed -> ChemBERTa skipped here (Kaggle images already include it)')

        # ===== CELL 03 (code) =====
        # 2 — GitHub-first data acquisition (idempotent; skips finished downloads)
        import os, time, urllib.request

        def gh_status(url):
            try:
                req = urllib.request.Request(url, method='HEAD')
                with urllib.request.urlopen(req, timeout=20) as r:
                    return r.status
            except Exception as e:
                return 'ERR ' + type(e).__name__

        def ensure_file(url, dest, min_bytes=1024):
            if RESUME and os.path.exists(dest) and os.path.getsize(dest) >= min_bytes:
                print('[resume] %s (%.2f MB) — skip download' % (dest, os.path.getsize(dest) / 1e6))
                return dest
            os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
            tmp = dest + '.part'
            t0 = time.time()
            print('downloading', url)
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, dest)
            print('[ok] %s (%.2f MB, %.1fs)' % (dest, os.path.getsize(dest) / 1e6, time.time() - t0))
            return dest

        print('--- GitHub availability check ---')
        for _key in ['train_csv', 'library_csv', 'common_py']:
            _url = GH_BASE + '/' + GH_FILES[_key]
            print('%-11s %-5s %s' % (_key, str(gh_status(_url)), _url))
        print('cuts.json    %-5s %s' % (str(gh_status(GH_BASE + '/00_data/config/cuts.json')), '(optional; defaults used if 404)'))

        print('--- training data: local first, then GitHub ---')
        TRAIN_CANDIDATES = [
            os.path.join(DATA_DIR, 'ro5_properties_filtered.csv'),
            'ro5_properties_filtered.csv',
            '../../00_data/training/ro5_properties_filtered.csv',
            '00_data/training/ro5_properties_filtered.csv',
            '../../00_data/training/ro5_properties_filtered.csv',
            '/kaggle/input/mdm2-data/ro5_properties_filtered.csv',
            os.path.join(str(INPUT_DIR), 'ro5_properties_filtered.csv'),
            os.path.join(str(INPUT_DIR), 'data', 'ro5_properties_filtered.csv'),
            os.path.join(str(INPUT_DIR), 'inputs', 'ro5_properties_filtered.csv'),
        ]
        TRAIN_PATH = next((p for p in TRAIN_CANDIDATES if os.path.exists(p)), None)
        if TRAIN_PATH is None:
            TRAIN_PATH = ensure_file(GH_BASE + '/' + GH_FILES['train_csv'],
                                     os.path.join(DATA_DIR, 'ro5_properties_filtered.csv'), min_bytes=10_000)
        else:
            print('[local] train =', TRAIN_PATH)
        print('TRAIN_PATH =', TRAIN_PATH)

        # ===== CELL 04 (code) =====
        # 3 — Load 645 training compounds (pIC50 regression target)
        import pandas as pd
        df = pd.read_csv(TRAIN_PATH)
        print('TRAIN =', TRAIN_PATH, '| shape', df.shape)
        assert 'pIC50' in df.columns, 'pIC50 column missing: %s' % list(df.columns)

        SMILES_COL = None
        for _c in ['canonical_smiles', 'smiles', 'smiles_canonical']:
            if _c in df.columns:
                SMILES_COL = _c
                break
        assert SMILES_COL is not None, 'no SMILES column: %s' % list(df.columns)

        if 'bioactivity_class' in df.columns:
            df['label'] = (df['bioactivity_class'].astype(str).str.lower() == 'active').astype(int)
        else:
            df['label'] = (df['pIC50'].astype(float) >= 7.0).astype(int)

        print('pIC50 range %.2f..%.2f | mean %.2f' % (df['pIC50'].min(), df['pIC50'].max(), df['pIC50'].mean()))
        print('labels: %d active / %d inactive' % (int(df['label'].sum()), int((df['label'] == 0).sum())))
        df.head(3)

        # ===== CELL 05 (code) =====
        # 4 — ONE Morgan featurizer (r3/512) + parallel training fingerprints
        import numpy as np, os, time
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
        from multiprocessing import Pool

        mpg = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_NBITS)

        def smiles_to_fp(s):
            m = Chem.MolFromSmiles(str(s))
            if m is None:
                return None
            return np.array(mpg.GetFingerprint(m), dtype=np.uint8)

        N_WORKERS = max(1, min(4, os.cpu_count() or 2))

        def _fp_job_train(s):
            try:
                return smiles_to_fp(s)
            except Exception:
                return None

        fps = None
        try:
            with Pool(N_WORKERS) as pool:
                fps = list(pool.imap(_fp_job_train, df[SMILES_COL].tolist(), chunksize=64))
            print('[mp] train fingerprints via', N_WORKERS, 'workers')
        except Exception as e:
            print('[mp] fallback serial:', e)
            fps = [_fp_job_train(s) for s in df[SMILES_COL].tolist()]

        keep = [i for i, f in enumerate(fps) if f is not None]
        assert keep, 'all training SMILES failed RDKit parsing'
        X = np.stack([fps[i] for i in keep])
        y = df['label'].values[keep].astype(int)
        smiles_train = df[SMILES_COL].values[keep]
        raw_targets = df['pIC50'].astype(float).values[keep]
        if 'molecule_chembl_id' in df.columns:
            train_ids = df['molecule_chembl_id'].astype(str).values[keep]
        else:
            train_ids = np.array(['TRAIN_%d' % i for i in keep])
        print('kept', X.shape, '| dropped invalid', len(fps) - len(keep), '| actives', int(y.sum()))

        # ===== CELL 06 (code) =====
        # 5 — Scaffold split -> 464/52/129 (cumulative 20% of molecules; TEST touched once)
        import numpy as np
        from collections import defaultdict
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from sklearn.model_selection import train_test_split

        def _murcko(s):
            try:
                m = Chem.MolFromSmiles(str(s))
                if m is None:
                    return 'INVALID'
                return MurckoScaffold.MurckoScaffoldSmiles(mol=m)   # RDKit 2026 API: keyword required
            except Exception:
                return 'INVALID'

        scaffolds = np.array([_murcko(s) for s in smiles_train])
        n_uniq = len(set(scaffolds))
        print('unique Murcko scaffolds:', n_uniq, '| INVALID:', int((scaffolds == 'INVALID').sum()))
        if n_uniq < 50:
            print('[warn] scaffold collapse detected — Murcko API may have changed!')

        groups = defaultdict(list)
        for _i, _s in enumerate(scaffolds):
            groups[_s].append(_i)
        n_test_target = int(round(TEST_SIZE * len(scaffolds)))
        order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        te_list = []
        for _s, _ii in order:
            if len(te_list) + len(_ii) <= n_test_target:
                te_list.extend(_ii)
        te = np.array(sorted(te_list), dtype=int)
        tr_all = np.array(sorted(set(range(len(scaffolds))) - set(te_list)), dtype=int)
        assert len(te) > 0 and len(tr_all) > 0
        tr, va = train_test_split(tr_all, test_size=VAL_FRAC, random_state=RANDOM_SEED, stratify=y[tr_all])

        assert len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0 and len(set(va) & set(te)) == 0

        def _canon(s):
            m = Chem.MolFromSmiles(str(s))
            return Chem.MolToSmiles(m) if m is not None else str(s)

        canon_tr = {_canon(s) for s in smiles_train[tr]}
        canon_te = {_canon(s) for s in smiles_train[te]}
        assert not (canon_tr & canon_te), 'canonical SMILES overlap between train and test!'

        print('[split] train %d / val %d / test %d | canonical overlap train-test = %d (leak-check PASS)'
              % (len(tr), len(va), len(te), len(canon_tr & canon_te)))
        print('scaffold groups shared train/test:', len({scaffolds[i] for i in tr} & {scaffolds[i] for i in te}))

        # ===== CELL 07 (code) =====
        # 6 — Delta-ML baseline: Ridge(6 RDKit descriptors) fitted on TRAIN only
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        def rdkit_desc(s):
            m = Chem.MolFromSmiles(str(s))
            if m is None:
                return None
            return [Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m),
                    rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m), rdMolDescriptors.CalcNumRings(m)]

        DESC_NAMES = ['MolWt', 'MolLogP', 'TPSA', 'HBD', 'HBA', 'RingCount']

        def desc_matrix(smiles):
            rows = []
            for s in smiles:
                d = rdkit_desc(s)
                rows.append(d if d is not None else [0.0] * 6)
            return np.asarray(rows, dtype=np.float64)

        def pic50_metrics(pred, true):
            return dict(r2=float(r2_score(true, pred)),
                        rmse=float(mean_squared_error(true, pred) ** 0.5),
                        mae=float(mean_absolute_error(true, pred)))

        D = desc_matrix(smiles_train)
        baseline_model = Ridge(alpha=1.0).fit(D[tr], raw_targets[tr])
        base_all = baseline_model.predict(D)
        delta_all = raw_targets - base_all
        scaler = StandardScaler().fit(delta_all[tr].reshape(-1, 1))
        y_all_scaled = scaler.transform(delta_all.reshape(-1, 1)).ravel()
        y_train_scaled = y_all_scaled[tr]
        y_val_scaled = y_all_scaled[va]
        y_test_scaled = y_all_scaled[te]

        BASE_METRICS = {
            'train': pic50_metrics(base_all[tr], raw_targets[tr]),
            'val': pic50_metrics(base_all[va], raw_targets[va]),
            'test': pic50_metrics(base_all[te], raw_targets[te]),
        }
        print('baseline Ridge (pIC50): train R2=%.3f | val R2=%.3f RMSE=%.3f | test R2=%.3f RMSE=%.3f'
              % (BASE_METRICS['train']['r2'], BASE_METRICS['val']['r2'], BASE_METRICS['val']['rmse'],
                 BASE_METRICS['test']['r2'], BASE_METRICS['test']['rmse']))
        print('delta scaler fitted on train only | delta mean %.3f std %.3f'
              % (float(delta_all[tr].mean()), float(delta_all[tr].std())))

        # ===== CELL 08 (code) =====
        # 7 — Graph featurizer: GitHub common.py (78-dim), cached + verified
        import os, sys, importlib.util, shutil
        COMMON = os.path.join(CACHE_DIR, 'gnn_common.py')
        LOCAL_COMMON = ['../../02_gnn_models/part18_gnn_triage/common.py', '02_gnn_models/part18_gnn_triage/common.py',
                        '/kaggle/input/mdm2-gnn/common.py',
                        os.path.join(str(INPUT_DIR), 'common.py'),
                        os.path.join(str(INPUT_DIR), 'data', 'common.py')]

        if not (RESUME and os.path.exists(COMMON) and os.path.getsize(COMMON) > 1024):
            got = False
            try:
                ensure_file(GH_BASE + '/' + GH_FILES['common_py'], COMMON, min_bytes=1024)
                got = True
            except Exception as e:
                print('[warn] GitHub common.py failed:', e)
            if not got:
                for p in LOCAL_COMMON:
                    if os.path.exists(p):
                        shutil.copy2(p, COMMON)
                        got = True
                        print('[local] common.py copied from', p)
                        break
            assert got, 'common.py unavailable (enable Internet or provide 02_gnn_models/part18_gnn_triage/common.py)'
        else:
            print('[resume] common.py cached')

        spec = importlib.util.spec_from_file_location('part18_common', COMMON)
        _mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mod)
        mol_to_graph = _mod.mol_to_graph
        sys.modules['part18_common'] = _mod

        _g = mol_to_graph('CC(=O)Oc1ccccc1C(=O)O', 0.0)
        assert _g is not None and _g.x.shape[1] == 78, 'unexpected graph feature dim: %s' % (None if _g is None else _g.x.shape)
        print('graph featurizer OK | node dim', _g.x.shape[1], '| edges', _g.edge_index.shape[1])

        # ===== CELL 09 (code) =====
        # 8 — Build train/val/test graphs (y = scaled delta); cached
        import os, torch, numpy as np
        PATH_GRAPHS = os.path.join(CACHE_DIR, 'train_val_test_graphs.pt')

        def _build_graphs():
            graphs = {}
            idx_out = {}
            for nm, ii in [('train', tr), ('val', va), ('test', te)]:
                gs, used = [], []
                for i in ii:
                    g = mol_to_graph(smiles_train[i], 0.0)
                    if g is None:
                        continue
                    g.y = torch.tensor([float(y_all_scaled[i])], dtype=torch.float)  # float target (common.py defaults to long)
                    gs.append(g)
                    used.append(int(i))
                graphs[nm] = gs
                idx_out[nm] = used
            return {'graphs': graphs, 'idx': idx_out}

        if RESUME and os.path.exists(PATH_GRAPHS):
            try:
                cached = torch_load_compat(PATH_GRAPHS)
                graphs, idx_out = cached['graphs'], cached['idx']
                if set(graphs) == {'train', 'val', 'test'} and len(graphs['train']) == len(tr):
                    print('[resume] split graphs loaded')
                else:
                    raise ValueError('cache shape mismatch')
            except Exception as e:
                print('[resume] graph cache invalid (%s) — rebuilding' % e)
                cached = _build_graphs()
                graphs, idx_out = cached['graphs'], cached['idx']
                torch.save(cached, PATH_GRAPHS)
        else:
            cached = _build_graphs()
            graphs, idx_out = cached['graphs'], cached['idx']
            torch.save(cached, PATH_GRAPHS)

        train_graphs, val_graphs, test_graphs = graphs['train'], graphs['val'], graphs['test']
        print('graphs:', len(train_graphs), len(val_graphs), len(test_graphs))

        from torch_geometric.loader import DataLoader
        train_loader = DataLoader(train_graphs, batch_size=GNN_BATCH, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_graphs, batch_size=GNN_BATCH, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_graphs, batch_size=GNN_BATCH, shuffle=False, num_workers=0)

        val_pos = np.asarray(idx_out['val'], dtype=int)
        test_pos = np.asarray(idx_out['test'], dtype=int)
        y_val_raw = raw_targets[val_pos]
        y_test_raw = raw_targets[test_pos]
        val_base = base_all[val_pos]
        test_base = base_all[test_pos]
        print('aligned labels: val', y_val_raw.shape, '| test', y_test_raw.shape)

        # ===== CELL 10 (code) =====
        # 9 — Regression GNNs (GCN / GIN, 78-dim graphs, delta output)
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv, GINConv, BatchNorm, global_mean_pool, global_max_pool

        class GNNReg(nn.Module):
            def __init__(self, in_dim=78, hidden=GNN_HIDDEN, layers=GNN_LAYERS, kind='gcn', dropout=0.2):
                super().__init__()
                self.kind = kind
                self.dropout = dropout
                self.convs = nn.ModuleList()
                self.bns = nn.ModuleList()
                for i in range(layers):
                    ci = in_dim if i == 0 else hidden
                    if kind == 'gcn':
                        conv = GCNConv(ci, hidden)
                    else:
                        mlp = nn.Sequential(nn.Linear(ci, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
                        conv = GINConv(mlp)
                    self.convs.append(conv)
                    self.bns.append(BatchNorm(hidden))
                self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(),
                                          nn.Dropout(dropout), nn.Linear(hidden, 1))

            def forward(self, x, edge_index, batch):
                h = x
                for conv, bn in zip(self.convs, self.bns):
                    h = conv(h, edge_index)
                    h = bn(h)
                    h = F.relu(h)
                    h = F.dropout(h, self.dropout, self.training)
                h = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)
                return self.head(h).view(-1)

        def count_params(m):
            return sum(p.numel() for p in m.parameters())

        print('GNNReg ready | GCN params', count_params(GNNReg(kind='gcn')),
              '| GIN params', count_params(GNNReg(kind='gin')))

        # ===== CELL 11 (code) =====
        # 9B — Extra models: AttentiveFP (DeepChem lineage), HFooladi GIN (researcher), ChemBERTa head
        import torch, torch.nn as nn, torch.nn.functional as F
        from torch_geometric.data import Data
        from torch_geometric.nn import GINConv, BatchNorm, global_mean_pool
        from torch_geometric.utils import softmax as pyg_softmax, scatter
        from part18_common import atom_features as _afp_atom_features

        # ---- AttentiveFP graph: proven 78-dim atoms + 6-dim bonds (both directions) ----
        def afp_graph(smiles, label=0.0):
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None or mol.GetNumAtoms() == 0:
                return None
            xs = [_afp_atom_features(a) for a in mol.GetAtoms()]
            src, dst, eas = [], [], []
            for b in mol.GetBonds():
                i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                bt = b.GetBondType()
                f = [float(bt == Chem.rdchem.BondType.SINGLE), float(bt == Chem.rdchem.BondType.DOUBLE),
                     float(bt == Chem.rdchem.BondType.TRIPLE), float(bt == Chem.rdchem.BondType.AROMATIC),
                     float(b.GetIsConjugated()), float(b.IsInRing())]
                src += [i, j]; dst += [j, i]; eas += [f, f]
            if src:
                ei = torch.tensor([src, dst], dtype=torch.long)
                ea = torch.tensor(eas, dtype=torch.float)
            else:
                ei = torch.tensor([[0], [0]], dtype=torch.long)
                ea = torch.zeros(1, 6)
            return Data(x=torch.tensor(xs, dtype=torch.float), edge_index=ei, edge_attr=ea,
                        y=torch.tensor([float(label)], dtype=torch.float))

        # ---- HFooladi (researcher) graph: 9-dim atoms, 3-dim bonds, edges both directions ----
        def hf_graph(smiles, label=0.0):
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None or mol.GetNumAtoms() == 0:
                return None
            xs = []
            for a in mol.GetAtoms():
                xs.append([float(a.GetAtomicNum()), float(int(a.GetChiralTag())), float(a.GetDegree()),
                           float(a.GetFormalCharge()), float(a.GetTotalNumHs()), float(a.GetNumRadicalElectrons()),
                           float(int(a.GetHybridization())), float(a.GetIsAromatic()), float(a.IsInRing())])
            src, dst = [], []
            for b in mol.GetBonds():
                i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                src += [i, j]; dst += [j, i]
            ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)
            return Data(x=torch.tensor(xs, dtype=torch.float), edge_index=ei,
                        y=torch.tensor([float(label)], dtype=torch.float))

        # ---- AttentiveFP (Xiong et al. 2020; DeepChem / DGL-LifeSci lineage; PyG port) ----
        class AttentiveLayer(nn.Module):
            def __init__(self, hidden, dropout=0.2):
                super().__init__()
                self.align = nn.Linear(3 * hidden, 1)
                self.value = nn.Linear(hidden, hidden)
                self.gru = nn.GRUCell(hidden, hidden)
                self.drop = nn.Dropout(dropout)

            def forward(self, x, edge_index, edge_attr):
                src, dst = edge_index
                score = F.leaky_relu(self.align(torch.cat([x[dst], x[src], edge_attr], dim=1)), 0.2)
                alpha = pyg_softmax(score, dst)
                ctx = scatter(self.value(x[src]) * alpha, dst, dim=0, dim_size=x.size(0), reduce='sum')
                return self.gru(self.drop(F.elu(ctx)), x)

        class AttentiveReadout(nn.Module):
            def __init__(self, hidden, timesteps=2, dropout=0.2):
                super().__init__()
                self.timesteps = timesteps
                self.align = nn.Linear(2 * hidden, 1)
                self.gru = nn.GRUCell(hidden, hidden)
                self.drop = nn.Dropout(dropout)

            def forward(self, x, batch):
                q = global_mean_pool(x, batch)
                for _ in range(self.timesteps):
                    score = F.leaky_relu(self.align(torch.cat([x, q[batch]], dim=1)), 0.2)
                    beta = pyg_softmax(score, batch)
                    ctx = scatter(x * beta, batch, dim=0, dim_size=q.size(0), reduce='sum')
                    q = self.gru(self.drop(F.elu(ctx)), q)
                return q

        class AttentiveFPReg(nn.Module):
            def __init__(self, node_dim=78, edge_dim=6, hidden=AFP_HIDDEN, layers=AFP_LAYERS,
                         timesteps=AFP_TIMESTEPS, dropout=0.2):
                super().__init__()
                self.atom_embed = nn.Linear(node_dim, hidden)
                self.edge_embed = nn.Linear(edge_dim, hidden)
                self.layers = nn.ModuleList([AttentiveLayer(hidden, dropout) for _ in range(layers)])
                self.readout = AttentiveReadout(hidden, timesteps, dropout)
                self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

            def forward(self, x, edge_index, edge_attr, batch):
                h = self.atom_embed(x)
                e = self.edge_embed(edge_attr)
                for layer in self.layers:
                    h = layer(h, edge_index, e)
                q = self.readout(h, batch)
                return self.head(q).view(-1)

        # ---- HFooladi GIN (researcher's Part_10 architecture, regression head) ----
        class _HFMLP(nn.Module):
            def __init__(self, in_ch, hidden, out_ch):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_ch, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                    nn.Linear(hidden, out_ch),
                )

            def forward(self, x):
                return self.net(x)

        class HFGINReg(nn.Module):
            def __init__(self, in_dim=9, hidden=HF_GIN_HIDDEN, layers=3, dropout=0.5):
                super().__init__()
                self.node_encoder = nn.Linear(in_dim, hidden)
                self.convs = nn.ModuleList()
                self.bns = nn.ModuleList()
                for _ in range(layers):
                    self.convs.append(GINConv(_HFMLP(hidden, hidden, hidden), train_eps=True))
                    self.bns.append(nn.BatchNorm1d(hidden))
                self.dropout = dropout
                self.head = _HFMLP(hidden, hidden, 1)

            def forward(self, x, edge_index, batch):
                h = self.node_encoder(x)
                for conv, bn in zip(self.convs, self.bns):
                    h = F.relu(bn(conv(h, edge_index)))
                    h = F.dropout(h, self.dropout, self.training)
                h = global_mean_pool(h, batch)
                return self.head(h).view(-1)

        # ---- ChemBERTa head (frozen LM embeddings -> small MLP) ----
        class ChemBERTaMLP(nn.Module):
            def __init__(self, in_dim=768, hidden=64, dropout=0.3):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

            def forward(self, x):
                return self.net(x).view(-1)

        print('extra models ready | AFP params', sum(p.numel() for p in AttentiveFPReg().parameters()),
              '| HFooladi-GIN params', sum(p.numel() for p in HFGINReg().parameters()),
              '| transformers', HAVE_TRANSFORMERS)

        # ===== CELL 12 (code) =====
        # 10 — Dual-GPU training: GCN-reg on cuda:0 + GIN-reg on cuda:1 (val early-stop; TEST once)
        import os, json, copy, time, numpy as np, torch
        from concurrent.futures import ThreadPoolExecutor

        dev0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        dev1 = torch.device('cuda:1' if (USE_DUAL_GPU and torch.cuda.device_count() > 1) else dev0)

        PATH_GCN = os.path.join(CACHE_DIR, 'gcn_reg.pt')
        PATH_GIN = os.path.join(CACHE_DIR, 'gin_reg.pt')
        PATH_METRICS = os.path.join(CACHE_DIR, 'reg_metrics.json')
        PATH_PREDS = os.path.join(CACHE_DIR, 'reg_val_test_preds.npz')

        def evaluate_reg(model, loader, device):
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for b in loader:
                    b = b.to(device)
                    p = model(b.x, b.edge_index, b.batch)
                    preds.append(p.detach().cpu())
                    trues.append(b.y.view(-1).detach().cpu())
            return torch.cat(preds).numpy(), torch.cat(trues).numpy()

        def _inv_delta(p_scaled, base):
            d = scaler.inverse_transform(np.asarray(p_scaled).reshape(-1, 1)).ravel()
            return np.clip(d + base, 0.0, 12.0)

        def train_one(kind, device, seed):
            torch.manual_seed(seed)
            model = GNNReg(kind=kind).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=GNN_LR, weight_decay=GNN_WD)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=8)
            lossf = torch.nn.SmoothL1Loss(beta=0.5)
            best, best_state, wait, hist = -1e9, None, 0, []
            t0 = time.time()
            for ep in range(1, GNN_EPOCHS + 1):
                model.train()
                tot, nseen = 0.0, 0
                for b in train_loader:
                    b = b.to(device)
                    opt.zero_grad()
                    p = model(b.x, b.edge_index, b.batch)
                    loss = lossf(p, b.y.view(-1))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    tot += float(loss) * b.num_graphs
                    nseen += b.num_graphs
                pv, tv = evaluate_reg(model, val_loader, device)
                r2 = 1.0 - float(((pv - tv) ** 2).sum() / max(float(((tv - tv.mean()) ** 2).sum()), 1e-9))
                hist.append(round(r2, 4))
                sched.step(r2)
                if r2 > best + 1e-4:
                    best = r2
                    best_state = copy.deepcopy({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                    wait = 0
                else:
                    wait += 1
                    if wait >= GNN_PATIENCE:
                        break
                print('  [%s@%s] ep %03d loss %.4f val_R2 %.4f' % (kind, device, ep, tot / max(nseen, 1), r2))
            model.load_state_dict(best_state)
            model.to(device).eval()
            pv, tv = evaluate_reg(model, val_loader, device)
            pt, tt = evaluate_reg(model, test_loader, device)
            return dict(kind=kind, model=model, val_scaled=(pv, tv), test_scaled=(pt, tt),
                        best_val_r2=best, epochs=len(hist), sec=time.time() - t0,
                        params=count_params(model), hist=hist)

        def _run_training():
            jobs = [('gcn', dev0, 57), ('gin', dev1, 58)]
            use_dual = USE_DUAL_GPU and torch.cuda.device_count() > 1
            print(('[dual-gpu] ' if use_dual else '[single-device] ') + 'training gcn@%s | gin@%s' % (dev0, dev1))
            t0 = time.time()
            if use_dual:
                with ThreadPoolExecutor(max_workers=2) as ex:
                    futs = [ex.submit(train_one, k, d, s) for k, d, s in jobs]
                    results = [f.result() for f in futs]
            else:
                results = [train_one(k, d, s) for k, d, s in jobs]
            print('[train] wall time %.1fs' % (time.time() - t0))

            metrics = {'baseline': BASE_METRICS,
                       'split': {'train': int(len(tr)), 'val': int(len(va)), 'test': int(len(te))}}
            preds = {'y_val': y_val_raw, 'y_test': y_test_raw}
            for r in results:
                k = r['kind']
                pv, tv = r['val_scaled']
                pt, tt = r['test_scaled']
                vp = _inv_delta(pv, val_base)
                tp = _inv_delta(pt, test_base)
                metrics[k] = dict(params=int(r['params']), epochs=int(r['epochs']), sec=round(r['sec'], 1),
                                  val=pic50_metrics(vp, y_val_raw), test=pic50_metrics(tp, y_test_raw),
                                  hist=r['hist'])
                preds[k + '_val'] = vp
                preds[k + '_test'] = tp
                torch.save({kk: vv.detach().cpu() for kk, vv in r['model'].state_dict().items()},
                           os.path.join(CACHE_DIR, k + '_reg.pt'))
            np.savez(PATH_PREDS, **preds)
            json.dump(metrics, open(PATH_METRICS, 'w'), indent=2)
            return metrics, preds

        HAVE_CACHE = all(os.path.exists(p) for p in [PATH_GCN, PATH_GIN, PATH_METRICS, PATH_PREDS])
        if RESUME and HAVE_CACHE and not FORCE_RETRAIN:
            METRICS = json.load(open(PATH_METRICS))
            _z = np.load(PATH_PREDS)
            preds_cache = {k: _z[k] for k in _z.files}
            print('[resume] trained GCN/GIN + metrics loaded — skip training')
        else:
            METRICS, preds_cache = _run_training()

        print('')
        print('=== Regression accuracy (pIC50 space) ===')
        print('%-16s %8s %9s %8s %10s %9s' % ('model', 'val R2', 'val RMSE', 'test R2', 'test RMSE', 'test MAE'))
        for name, key in [('Ridge baseline', 'baseline'), ('GCN-reg (GNN)', 'gcn'), ('GIN-reg (GNN)', 'gin')]:
            v, t = METRICS[key]['val'], METRICS[key]['test']
            print('%-16s %8.3f %9.3f %8.3f %10.3f %9.3f' % (name, v['r2'], v['rmse'], t['r2'], t['rmse'], t['mae']))
        print('test evaluated ONCE (after early stopping on val) — no test-based tuning')

        # ===== CELL 13 (code) =====
        # 10B — Train extra models: AttentiveFP (cuda:0) + HFooladi GIN (cuda:1); ChemBERTa if available
        import os, json, copy, time, numpy as np, torch
        from concurrent.futures import ThreadPoolExecutor
        from torch_geometric.loader import DataLoader as PyGLoader

        PATH_AFP = os.path.join(CACHE_DIR, 'afp_reg.pt')
        PATH_HFGIN = os.path.join(CACHE_DIR, 'hfgin_reg.pt')
        PATH_EXT_METRICS = os.path.join(CACHE_DIR, 'reg_metrics_ext.json')
        PATH_EXT_PREDS = os.path.join(CACHE_DIR, 'reg_val_test_preds_ext.npz')
        PATH_CB_HEAD = os.path.join(CACHE_DIR, 'chemberta_head.pt')
        PATH_CB_EMB = os.path.join(CACHE_DIR, 'chemberta_645.pt')
        PATH_CB_PREDS = os.path.join(CACHE_DIR, 'chemberta_preds.npz')
        PATH_CB_METRICS = os.path.join(CACHE_DIR, 'chemberta_metrics.json')

        def _build_extra_graphs(fn, path):
            if RESUME and os.path.exists(path):
                d = torch_load_compat(path)
                print('[resume]', os.path.basename(path))
                return d['train'], d['val'], d['test']
            out = {}
            for nm, ii in [('train', tr), ('val', va), ('test', te)]:
                gs = []
                for i in ii:
                    g = fn(smiles_train[i], float(y_all_scaled[i]))
                    if g is not None:
                        gs.append(g)
                assert len(gs) == len(ii), 'graph build dropped rows in %s split' % nm
                out[nm] = gs
            torch.save(out, path)
            print('[build]', os.path.basename(path), {k: len(v) for k, v in out.items()})
            return out['train'], out['val'], out['test']

        afp_tr, afp_va, afp_te = _build_extra_graphs(afp_graph, os.path.join(CACHE_DIR, 'afp_graphs.pt'))
        hf_tr, hf_va, hf_te = _build_extra_graphs(hf_graph, os.path.join(CACHE_DIR, 'hf_graphs.pt'))
        afp_tr_ld = PyGLoader(afp_tr, batch_size=GNN_BATCH, shuffle=True, num_workers=0)
        afp_va_ld = PyGLoader(afp_va, batch_size=GNN_BATCH, shuffle=False, num_workers=0)
        afp_te_ld = PyGLoader(afp_te, batch_size=GNN_BATCH, shuffle=False, num_workers=0)
        hf_tr_ld = PyGLoader(hf_tr, batch_size=32, shuffle=True, num_workers=0)
        hf_va_ld = PyGLoader(hf_va, batch_size=32, shuffle=False, num_workers=0)
        hf_te_ld = PyGLoader(hf_te, batch_size=32, shuffle=False, num_workers=0)

        def _eval_g(model, loader, device, fwd):
            model.eval(); ps, ts = [], []
            with torch.no_grad():
                for b in loader:
                    b = b.to(device)
                    ps.append(fwd(model, b).detach().cpu())
                    ts.append(b.y.view(-1).detach().cpu())
            return torch.cat(ps).numpy(), torch.cat(ts).numpy()

        def _train_g(model, tr_ld, va_ld, te_ld, device, tag, lr, wd, epochs, patience, seed, fwd):
            torch.manual_seed(seed)
            model = model.to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=8)
            lossf = torch.nn.SmoothL1Loss(beta=0.5)
            best, best_state, wait, hist = -1e9, None, 0, []
            t0 = time.time()
            for ep in range(1, epochs + 1):
                model.train(); tot, nseen = 0.0, 0
                for b in tr_ld:
                    b = b.to(device)
                    opt.zero_grad()
                    loss = lossf(fwd(model, b), b.y.view(-1))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    tot += float(loss) * b.num_graphs; nseen += b.num_graphs
                pv, tv = _eval_g(model, va_ld, device, fwd)
                r2 = 1.0 - float(((pv - tv) ** 2).sum() / max(float(((tv - tv.mean()) ** 2).sum()), 1e-9))
                hist.append(round(r2, 4)); sched.step(r2)
                if r2 > best + 1e-4:
                    best = r2
                    best_state = copy.deepcopy({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break
                if ep == 1 or ep % 5 == 0:
                    print('  [%s@%s] ep %03d loss %.4f val_R2 %.4f' % (tag, device, ep, tot / max(nseen, 1), r2))
            model.load_state_dict(best_state)
            model.to(device).eval()
            pv, tv = _eval_g(model, va_ld, device, fwd)
            pt, tt = _eval_g(model, te_ld, device, fwd)
            return dict(model=model, val=(pv, tv), test=(pt, tt), best=best, hist=hist,
                        epochs=len(hist), sec=time.time() - t0, params=sum(p.numel() for p in model.parameters()))

        def _fwd_afp(m, b):
            return m(b.x, b.edge_index, b.edge_attr, b.batch)

        def _fwd_hf(m, b):
            return m(b.x, b.edge_index, b.batch)

        def _train_extra_graphs():
            jobs = [(AttentiveFPReg(), afp_tr_ld, afp_va_ld, afp_te_ld, dev0, 'afp', GNN_LR, GNN_WD, EXTRA_EPOCHS, EXTRA_PATIENCE, 59, _fwd_afp),
                    (HFGINReg(), hf_tr_ld, hf_va_ld, hf_te_ld, dev1, 'hfgin', GNN_LR, GNN_WD, EXTRA_EPOCHS, EXTRA_PATIENCE, 60, _fwd_hf)]
            use_dual = USE_DUAL_GPU and torch.cuda.device_count() > 1
            print(('[dual-gpu] ' if use_dual else '[single-device] ') + 'training afp@%s | hfgin@%s' % (dev0, dev1))
            t0 = time.time()
            if use_dual:
                with ThreadPoolExecutor(max_workers=2) as ex:
                    futs = [ex.submit(_train_g, *j) for j in jobs]
                    res = [f.result() for f in futs]
            else:
                res = [_train_g(*j) for j in jobs]
            print('[extra-train] wall time %.1fs' % (time.time() - t0))
            metrics = {'split': {'train': int(len(tr)), 'val': int(len(va)), 'test': int(len(te))}}
            preds = {'y_val': y_val_raw, 'y_test': y_test_raw}
            for tag, r in zip(['afp', 'hfgin'], res):
                vp = _inv_delta(r['val'][0], val_base)
                tp = _inv_delta(r['test'][0], test_base)
                metrics[tag] = dict(params=int(r['params']), epochs=int(r['epochs']), sec=round(r['sec'], 1),
                                    val=pic50_metrics(vp, y_val_raw), test=pic50_metrics(tp, y_test_raw),
                                    hist=r['hist'], best_scaled_delta_val_r2=round(float(r['best']), 4))
                preds[tag + '_val'] = vp
                preds[tag + '_test'] = tp
                torch.save({k: v.detach().cpu() for k, v in r['model'].state_dict().items()},
                           PATH_AFP if tag == 'afp' else PATH_HFGIN)
            np.savez(PATH_EXT_PREDS, **preds)
            json.dump(metrics, open(PATH_EXT_METRICS, 'w'), indent=2)
            return metrics, preds

        if RESUME and all(os.path.exists(p) for p in [PATH_AFP, PATH_HFGIN, PATH_EXT_METRICS, PATH_EXT_PREDS]) and not FORCE_RETRAIN:
            EXT_METRICS = json.load(open(PATH_EXT_METRICS))
            _z = np.load(PATH_EXT_PREDS)
            ext_preds = {k: _z[k] for k in _z.files}
            print('[resume] AttentiveFP + HFooladi GIN loaded — skip training')
        else:
            EXT_METRICS, ext_preds = _train_extra_graphs()

        # ---- ChemBERTa: frozen embeddings + small regression head (optional) ----
        CB_PREDS = {}
        CB_METRICS = {}
        if HAVE_TRANSFORMERS and RUN_CHEMBERTA and RESUME and os.path.exists(PATH_CB_PREDS) and os.path.exists(PATH_CB_METRICS):
            _z = np.load(PATH_CB_PREDS)
            CB_PREDS = {k: _z[k] for k in _z.files}
            CB_METRICS = json.load(open(PATH_CB_METRICS))
            print('[resume] ChemBERTa head + predictions loaded')
        elif HAVE_TRANSFORMERS and RUN_CHEMBERTA:
            try:
                from transformers import AutoTokenizer, AutoModel
                t0 = time.time()
                if RESUME and os.path.exists(PATH_CB_EMB):
                    _e = torch_load_compat(PATH_CB_EMB)
                    emb = {'tr': _e['tr'], 'va': _e['va'], 'te': _e['te']}
                    print('[resume] ChemBERTa embeddings loaded')
                else:
                    tok = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL)
                    lm = AutoModel.from_pretrained(CHEMBERTA_MODEL).eval()
                    for p in lm.parameters():
                        p.requires_grad_(False)
                    def _embed(smis):
                        out = []
                        with torch.no_grad():
                            for s in range(0, len(smis), CHEMBERTA_BATCH):
                                enc = tok(smis[s:s + CHEMBERTA_BATCH], padding=True, truncation=True,
                                          max_length=CHEMBERTA_MAXLEN, return_tensors='pt')
                                h = lm(**enc).last_hidden_state
                                m = enc['attention_mask'].unsqueeze(-1).float()
                                out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
                        return torch.cat(out)
                    emb = {'tr': _embed([smiles_train[i] for i in tr]),
                           'va': _embed([smiles_train[i] for i in va]),
                           'te': _embed([smiles_train[i] for i in te])}
                    torch.save(emb, PATH_CB_EMB)
                    print('[chemberta] embeddings built %.1fs' % (time.time() - t0))
                emb = {k: v.to(dev0) for k, v in emb.items()}
                head = ChemBERTaMLP().to(dev0)
                opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
                ytr_s = torch.tensor(y_train_scaled, dtype=torch.float, device=dev0)
                yva_s = torch.tensor(y_val_scaled, dtype=torch.float, device=dev0)
                best, best_state, wait, hist = -1e9, None, 0, []
                for ep in range(1, CHEMBERTA_EPOCHS + 1):
                    head.train()
                    perm = torch.randperm(len(ytr_s), device=dev0)
                    for s in range(0, len(perm), 64):
                        idx = perm[s:s + 64]
                        opt.zero_grad()
                        loss = torch.nn.functional.smooth_l1_loss(head(emb['tr'][idx]), ytr_s[idx], beta=0.5)
                        loss.backward(); opt.step()
                    head.eval()
                    with torch.no_grad():
                        pv = head(emb['va']).detach().cpu().numpy()
                    r2 = 1.0 - float(((pv - y_val_scaled) ** 2).sum() / max(float(((y_val_scaled - y_val_scaled.mean()) ** 2).sum()), 1e-9))
                    hist.append(round(r2, 4))
                    if r2 > best + 1e-4:
                        best = r2
                        best_state = copy.deepcopy({k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
                        wait = 0
                    else:
                        wait += 1
                        if wait >= CHEMBERTA_PATIENCE:
                            break
                head.load_state_dict(best_state); head.to(dev0).eval()
                with torch.no_grad():
                    pv = head(emb['va']).cpu().numpy(); pt = head(emb['te']).cpu().numpy()
                vp = _inv_delta(pv, val_base); tp = _inv_delta(pt, test_base)
                CB_METRICS = {'params': int(sum(p.numel() for p in head.parameters())), 'epochs': len(hist),
                              'hist': hist, 'val': pic50_metrics(vp, y_val_raw), 'test': pic50_metrics(tp, y_test_raw)}
                CB_PREDS = {'cb_val': vp, 'cb_test': tp, 'y_val': y_val_raw, 'y_test': y_test_raw}
                torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, PATH_CB_HEAD)
                np.savez(PATH_CB_PREDS, **CB_PREDS)
                json.dump(CB_METRICS, open(PATH_CB_METRICS, 'w'), indent=2)
                EXT_METRICS['chemberta'] = CB_METRICS
                print('[chemberta] head trained %.1fs | val R2 %.3f | test R2 %.3f'
                      % (time.time() - t0, CB_METRICS['val']['r2'], CB_METRICS['test']['r2']))
            except Exception as _e:
                print('[chemberta] skipped:', type(_e).__name__, str(_e)[:200])
                HAVE_TRANSFORMERS = False
        else:
            print('[chemberta] skipped (transformers not installed) — Kaggle will run it automatically')

        print('')
        print('=== Extra-model accuracy (pIC50 space) ===')
        print('%-18s %8s %9s %8s %10s %9s' % ('model', 'val R2', 'val RMSE', 'test R2', 'test RMSE', 'test MAE'))
        for nm, key in [('AttentiveFP', 'afp'), ('HFooladi GIN', 'hfgin')]:
            if key in EXT_METRICS:
                v, t = EXT_METRICS[key]['val'], EXT_METRICS[key]['test']
                print('%-18s %8.3f %9.3f %8.3f %10.3f %9.3f' % (nm, v['r2'], v['rmse'], t['r2'], t['rmse'], t['mae']))
        if CB_METRICS:
            v, t = CB_METRICS['val'], CB_METRICS['test']
            print('%-18s %8.3f %9.3f %8.3f %10.3f %9.3f' % ('ChemBERTa', v['r2'], v['rmse'], t['r2'], t['rmse'], t['mae']))
        print('extra models: test scored ONCE after validation-only early stopping')

        # ===== CELL 14 (code) =====
        # 11 — Load screening library: GitHub-first filtered COCONUT (idempotent)
        import os, hashlib, pandas as pd
        LIB_CANDIDATES = [
            '../../00_data/screening/screening_results.csv',
            '00_data/screening/screening_results.csv',
            os.path.join(DATA_DIR, 'screening_results.csv'),
            'screening_results.csv',
            'coconut_reduced.csv', 'coconut_filtered.csv', 'ro5_coconut_filtered.csv',
            '/kaggle/input/mdm2-data/screening_results.csv',
            os.path.join(str(INPUT_DIR), 'screening_results.csv'),
            os.path.join(str(INPUT_DIR), 'data', 'screening_results.csv'),
            os.path.join(str(INPUT_DIR), 'inputs', 'screening_results.csv'),
        ]
        LIB_PATH = next((p for p in LIB_CANDIDATES if os.path.exists(p)), None)

        if LIB_PATH is None:
            try:
                LIB_PATH = ensure_file(GH_BASE + '/' + GH_FILES['library_csv'],
                                       os.path.join(DATA_DIR, 'screening_results.csv'), min_bytes=100_000)
            except Exception as e:
                print('[warn] GitHub library failed (%s) — falling back to raw COCONUT gdown' % e)
                import gdown
                LIB_PATH = os.path.join(DATA_DIR, 'coconut_csv-03-2025.csv')
                if RESUME and os.path.exists(LIB_PATH) and os.path.getsize(LIB_PATH) > 1e8:
                    print('[resume] raw COCONUT already downloaded (%.0f MB)' % (os.path.getsize(LIB_PATH) / 1e6))
                else:
                    gdown.download(id='1-DFc6lMf6maNAWPwZFA8uY6951SokoNY', output=LIB_PATH, quiet=False, resume=True)
        else:
            print('[local] library =', LIB_PATH)

        print('LIB =', LIB_PATH, '(%.1f MB)' % (os.path.getsize(LIB_PATH) / 1e6))
        lib = pd.read_csv(LIB_PATH, low_memory=False)

        _cols = {c.lower(): c for c in lib.columns}
        _smi = None
        for _k in ['canonical_smiles', 'smiles', 'smiles_canonical']:
            if _k in _cols:
                _smi = _cols[_k]
                break
        assert _smi is not None, 'no SMILES column: %s' % list(lib.columns)[:25]
        lib = lib.rename(columns={_smi: 'canonical_smiles'})

        _id = None
        for _k in ['identifier', 'molecule_chembl_id', 'chembl_id', 'coconut_id', 'id']:
            if _k in _cols and _cols[_k] != _smi:
                _id = _cols[_k]
                break
        if _id is not None:
            lib['identifier'] = lib[_id].astype(str)
        else:
            lib['identifier'] = ['LIB_%d' % i for i in range(len(lib))]

        lib = lib[['identifier', 'canonical_smiles']].dropna(subset=['canonical_smiles'])
        lib = lib.drop_duplicates('canonical_smiles').reset_index(drop=True)

        LIB_HASH = hashlib.md5((str(len(lib)) + lib['canonical_smiles'].iloc[0] + lib['canonical_smiles'].iloc[-1]).encode()).hexdigest()[:12]
        print('library:', lib.shape, '| hash', LIB_HASH)
        if len(lib) > 300000:
            print('[warn] raw COCONUT (mostly far out-of-domain). Prefer the GitHub filtered screening_results.csv for paper parity.')
        lib.head(3)

        # ===== CELL 15 (code) =====
        # 12 — Library Morgan fingerprints (AD axis), parallel + cached
        import os, time, numpy as np
        from multiprocessing import Pool

        def _fp_job_lib(s):
            return smiles_to_fp(s)

        n_lib = len(lib)
        PATH_LIBFP = os.path.join(CACHE_DIR, 'lib_fp_%s.npy' % LIB_HASH)

        if RESUME and os.path.exists(PATH_LIBFP):
            LIB_FP = np.load(PATH_LIBFP)
            print('[resume] library fingerprints', LIB_FP.shape)
        else:
            t0 = time.time()
            try:
                with Pool(N_WORKERS) as p:
                    fps = list(p.imap(_fp_job_lib, lib['canonical_smiles'].tolist(), chunksize=256))
                print('[mp] library fingerprints via %d workers in %.0fs' % (N_WORKERS, time.time() - t0))
            except Exception as e:
                print('[mp] fallback serial:', e)
                fps = [_fp_job_lib(s) for s in lib['canonical_smiles']]
            LIB_FP = np.stack([f if f is not None else np.zeros(MORGAN_NBITS, dtype=np.uint8) for f in fps])
            np.save(PATH_LIBFP, LIB_FP)
            print('LIB_FP', LIB_FP.shape, '| cached for AD')

        print('library rows:', n_lib, '| fingerprint matrix:', LIB_FP.shape)

        # ===== CELL 16 (code) =====
        # 13 — Dual-GPU screening in memory-safe chunks (GCN on cuda:0, GIN on cuda:1)
        import os, time, gc, numpy as np, torch
        from concurrent.futures import ThreadPoolExecutor
        from torch_geometric.loader import DataLoader as PyGLoader

        PATH_GCN = os.path.join(CACHE_DIR, 'gcn_reg.pt')
        PATH_GIN = os.path.join(CACHE_DIR, 'gin_reg.pt')
        MODEL_SIG = '%d_%d' % (int(os.path.getmtime(PATH_GCN)), int(os.path.getmtime(PATH_GIN)))
        PATH_LIBPRED = os.path.join(CACHE_DIR, 'screen_preds_%s_%s.npz' % (LIB_HASH, MODEL_SIG))

        def predict_graphs(model, graphs, device, batch=SCREEN_BATCH):
            model.eval()
            if not graphs:
                return np.zeros(0, dtype=np.float32)
            out = []
            loader = PyGLoader(graphs, batch_size=batch, shuffle=False, num_workers=0)
            with torch.no_grad():
                for b in loader:
                    b = b.to(device)
                    out.append(model(b.x, b.edge_index, b.batch).detach().cpu())
            return torch.cat(out).numpy()

        if RESUME and os.path.exists(PATH_LIBPRED):
            _z = np.load(PATH_LIBPRED)
            pred_gcn_scaled, pred_gin_scaled = _z['gcn'], _z['gin']
            screen_idx = _z['idx']
            print('[resume] library predictions loaded', pred_gcn_scaled.shape)
        else:
            gcn = GNNReg(kind='gcn')
            gcn.load_state_dict(torch_load_compat(PATH_GCN))
            gcn.to(dev0).eval()
            gin = GNNReg(kind='gin')
            gin.load_state_dict(torch_load_compat(PATH_GIN))
            gin.to(dev1).eval()

            CHUNK = 2000
            smis = lib['canonical_smiles'].tolist()
            idx_all, p_gcn, p_gin = [], [], []
            t0 = time.time()

            for s in range(0, n_lib, CHUNK):
                e = min(n_lib, s + CHUNK)
                chunk = []
                for j in range(s, e):
                    try:
                        g = mol_to_graph(smis[j], 0.0)
                    except Exception:
                        g = None
                    if g is not None:
                        chunk.append((j, g))
                if not chunk:
                    continue
                cidx = [j for j, _ in chunk]
                cgraphs = [g for _, g in chunk]
                if USE_DUAL_GPU and torch.cuda.device_count() > 1:
                    with ThreadPoolExecutor(max_workers=2) as ex:
                        f0 = ex.submit(predict_graphs, gcn, cgraphs, dev0)
                        f1 = ex.submit(predict_graphs, gin, cgraphs, dev1)
                        pc, pg = f0.result(), f1.result()
                else:
                    pc = predict_graphs(gcn, cgraphs, dev0)
                    pg = predict_graphs(gin, cgraphs, dev1)
                idx_all.extend(cidx)
                p_gcn.append(pc)
                p_gin.append(pg)
                del chunk, cgraphs, pc, pg
                if (e % 10000) == 0 or e == n_lib:
                    print('  screened %d/%d  %.0fs' % (e, n_lib, time.time() - t0), flush=True)
                gc.collect()

            screen_idx = np.asarray(idx_all, dtype=np.int32)
            pred_gcn_scaled = np.concatenate(p_gcn).astype(np.float32)
            pred_gin_scaled = np.concatenate(p_gin).astype(np.float32)
            np.savez_compressed(PATH_LIBPRED, gcn=pred_gcn_scaled, gin=pred_gin_scaled, idx=screen_idx)
            print('[screen] %d/%d valid graphs scored in %.0fs' % (len(screen_idx), n_lib, time.time() - t0))

        print('screen preds:', pred_gcn_scaled.shape, pred_gin_scaled.shape, '| valid idx', len(screen_idx))

        # ===== CELL 17 (code) =====
        # 13B — Screen the extra models (AFP cuda:0, HFooladi GIN cuda:1; ChemBERTa if head exists)
        import os, gc, time, numpy as np, torch
        from concurrent.futures import ThreadPoolExecutor
        from torch_geometric.loader import DataLoader as PyGLoader

        PATH_AFP = os.path.join(CACHE_DIR, 'afp_reg.pt')
        PATH_HFGIN = os.path.join(CACHE_DIR, 'hfgin_reg.pt')
        PATH_CB_HEAD = os.path.join(CACHE_DIR, 'chemberta_head.pt')
        _sig = ['%d' % int(os.path.getmtime(PATH_AFP)), '%d' % int(os.path.getmtime(PATH_HFGIN))]
        if os.path.exists(PATH_CB_HEAD):
            _sig.append('%d' % int(os.path.getmtime(PATH_CB_HEAD)))
        MODEL_SIG_EXT = '_'.join(_sig)
        PATH_EXT_SCREEN = os.path.join(CACHE_DIR, 'screen_preds_ext_%s_%s.npz' % (LIB_HASH, MODEL_SIG_EXT))

        def _pred_afp(model, graphs, device, batch=SCREEN_BATCH):
            model.eval(); out = []
            if not graphs:
                return np.zeros(0, dtype=np.float32)
            with torch.no_grad():
                for b in PyGLoader(graphs, batch_size=batch, shuffle=False, num_workers=0):
                    b = b.to(device)
                    out.append(model(b.x, b.edge_index, b.edge_attr, b.batch).detach().cpu())
            return torch.cat(out).numpy()

        def _pred_hf(model, graphs, device, batch=SCREEN_BATCH):
            model.eval(); out = []
            if not graphs:
                return np.zeros(0, dtype=np.float32)
            with torch.no_grad():
                for b in PyGLoader(graphs, batch_size=batch, shuffle=False, num_workers=0):
                    b = b.to(device)
                    out.append(model(b.x, b.edge_index, b.batch).detach().cpu())
            return torch.cat(out).numpy()

        if RESUME and os.path.exists(PATH_EXT_SCREEN):
            _z = np.load(PATH_EXT_SCREEN)
            pred_afp_scaled, pred_hfgin_scaled, ext_idx = _z['afp'], _z['hfgin'], _z['idx']
            pred_cb_scaled = _z['cb'] if 'cb' in _z.files else None
            print('[resume] extra-model screening loaded', pred_afp_scaled.shape)
        else:
            afp = AttentiveFPReg(); afp.load_state_dict(torch_load_compat(PATH_AFP)); afp.to(dev0).eval()
            hfgin = HFGINReg(); hfgin.load_state_dict(torch_load_compat(PATH_HFGIN)); hfgin.to(dev1).eval()
            CHUNK = 2000
            smis = lib['canonical_smiles'].tolist()
            idx_all, p_afp, p_hf = [], [], []
            t0 = time.time()
            for s in range(0, n_lib, CHUNK):
                e = min(n_lib, s + CHUNK)
                chunk, cidx = [], []
                for j in range(s, e):
                    try:
                        ga = afp_graph(smis[j], 0.0); gh = hf_graph(smis[j], 0.0)
                    except Exception:
                        ga, gh = None, None
                    if ga is not None and gh is not None:
                        chunk.append((ga, gh)); cidx.append(j)
                if not chunk:
                    continue
                ga_list = [x[0] for x in chunk]; gh_list = [x[1] for x in chunk]
                if USE_DUAL_GPU and torch.cuda.device_count() > 1:
                    with ThreadPoolExecutor(max_workers=2) as ex:
                        f0 = ex.submit(_pred_afp, afp, ga_list, dev0)
                        f1 = ex.submit(_pred_hf, hfgin, gh_list, dev1)
                        pa, ph = f0.result(), f1.result()
                else:
                    pa = _pred_afp(afp, ga_list, dev0)
                    ph = _pred_hf(hfgin, gh_list, dev1)
                idx_all.extend(cidx); p_afp.append(pa); p_hf.append(ph)
                del chunk, ga_list, gh_list, pa, ph
                if (e % 10000) == 0 or e == n_lib:
                    print('  [extra] screened %d/%d  %.0fs' % (e, n_lib, time.time() - t0), flush=True)
                gc.collect()
            ext_idx = np.asarray(idx_all, dtype=np.int32)
            pred_afp_scaled = np.concatenate(p_afp).astype(np.float32)
            pred_hfgin_scaled = np.concatenate(p_hf).astype(np.float32)
            print('[extra-screen] %d/%d valid in %.0fs' % (len(ext_idx), n_lib, time.time() - t0))
            pred_cb_scaled = None
            if HAVE_TRANSFORMERS and os.path.exists(PATH_CB_HEAD):
                try:
                    from transformers import AutoTokenizer, AutoModel
                    head = ChemBERTaMLP(); head.load_state_dict(torch_load_compat(PATH_CB_HEAD)); head.to(dev0).eval()
                    tok = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL)
                    lm = AutoModel.from_pretrained(CHEMBERTA_MODEL).eval()
                    for p in lm.parameters():
                        p.requires_grad_(False)
                    PATH_CB_LIB = os.path.join(CACHE_DIR, 'chemberta_lib_%s.npy' % LIB_HASH)
                    if RESUME and os.path.exists(PATH_CB_LIB):
                        cb_all = np.load(PATH_CB_LIB)
                        print('[resume] ChemBERTa library embeddings loaded')
                    else:
                        cb_all = []
                        t0 = time.time()
                        with torch.no_grad():
                            for s in range(0, n_lib, 256):
                                enc = tok(smis[s:s + 256], padding=True, truncation=True,
                                          max_length=CHEMBERTA_MAXLEN, return_tensors='pt')
                                h = lm(**enc).last_hidden_state
                                m = enc['attention_mask'].unsqueeze(-1).float()
                                e = ((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu()
                                cb_all.append(head(e).cpu().numpy())
                                if (s % 20000) == 0:
                                    print('  [chemberta] %d/%d %.0fs' % (s, n_lib, time.time() - t0), flush=True)
                        cb_all = np.concatenate(cb_all).astype(np.float32)
                        np.save(PATH_CB_LIB, cb_all)
                        print('[chemberta] library scored %.0fs' % (time.time() - t0))
                    pred_cb_scaled = cb_all[ext_idx]
                except Exception as _e:
                    print('[chemberta-screen] skipped:', type(_e).__name__, str(_e)[:200])
            payload = dict(afp=pred_afp_scaled, hfgin=pred_hfgin_scaled, idx=ext_idx)
            if pred_cb_scaled is not None:
                payload['cb'] = pred_cb_scaled
            np.savez_compressed(PATH_EXT_SCREEN, **payload)
        print('extra screen preds:', pred_afp_scaled.shape, pred_hfgin_scaled.shape,
              '| chemberta', None if pred_cb_scaled is None else pred_cb_scaled.shape)

        # ===== CELL 18 (code) =====
        # 14 — Exact train-screen similarity, strict AD, active/inactive evidence
        import os, json, time
        import numpy as np

        TR_FP = X.astype(np.float32)
        TRAIN_FP = TR_FP[tr]
        TRAIN_IDS = np.asarray(train_ids)[tr]
        _yy = np.asarray(y).ravel()
        ACT_MASK = (_yy[tr] == 1)
        INACT_MASK = ~ACT_MASK
        ACT_FP = TRAIN_FP[ACT_MASK]
        INACT_FP = TRAIN_FP[INACT_MASK]
        ACT_IDS = TRAIN_IDS[ACT_MASK]
        INACT_IDS = TRAIN_IDS[INACT_MASK]
        print('[ad] train-only evidence: %d active / %d inactive' % (int(ACT_MASK.sum()), int(INACT_MASK.sum())))
        TE_FP = TR_FP[te]

        def max_tanimoto_chunked(Q, T, chunk=4000):
            """Exact Tanimoto max + argmax, chunked to keep memory bounded."""
            Q = np.asarray(Q, dtype=np.float32)
            T = np.asarray(T, dtype=np.float32)
            qc = (Q * Q).sum(1)
            tc = (T * T).sum(1)
            mx = np.zeros(len(Q), dtype=np.float32)
            ix = np.full(len(Q), -1, dtype=np.int32)
            for s in range(0, len(Q), chunk):
                e = min(s + chunk, len(Q))
                qf = Q[s:e]
                S = qf @ T.T
                D = qc[s:e, None] + tc[None, :] - S
                Tn = S / np.maximum(D, 1e-9)
                j = Tn.argmax(1)
                ix[s:e] = j
                mx[s:e] = Tn[np.arange(e - s), j]
            return mx, ix

        def loo_max_tanimoto(Q, chunk=4000):
            """Leave-one-out max similarity inside one set (diagonal excluded)."""
            Q = np.asarray(Q, dtype=np.float32)
            qc = (Q * Q).sum(1)
            mx = np.zeros(len(Q), dtype=np.float32)
            for s in range(0, len(Q), chunk):
                e = min(s + chunk, len(Q))
                S = Q[s:e] @ Q.T
                D = qc[s:e, None] + qc[None, :] - S
                Tn = S / np.maximum(D, 1e-9)
                Tn[np.arange(e - s), np.arange(s, e)] = -1.0
                mx[s:e] = Tn.max(1)
            return mx

        def _ids(nn, ref_ids):
            ref_ids = np.asarray(ref_ids)
            return np.array([str(ref_ids[j]) if j >= 0 else '' for j in nn], dtype=object)

        cache_path = os.path.join(CACHE_DIR, 'ad_v2_trainonly_%s.npz' % LIB_HASH)
        _keys = ['m645', 'i645', 'm464', 'i464', 'm_train_act', 'i_train_act', 'm_train_inact', 'i_train_inact']
        ad = None
        if RESUME and os.path.exists(cache_path):
            try:
                z = np.load(cache_path, allow_pickle=True)
                if all(k in z for k in _keys):
                    ad = {k: z[k] for k in _keys}
                    print('[ad] loaded cache', cache_path)
            except Exception as _e:
                print('[ad] cache read failed:', _e)

        if ad is None:
            print('[ad] computing exact Tanimoto to 645 / 464 / train-only active / train-only inactive ...')
            m645, i645 = max_tanimoto_chunked(LIB_FP, TR_FP)
            m464, i464 = max_tanimoto_chunked(LIB_FP, TRAIN_FP)
            m_train_act, i_train_act = max_tanimoto_chunked(LIB_FP, ACT_FP)
            m_train_inact, i_train_inact = max_tanimoto_chunked(LIB_FP, INACT_FP)
            ad = dict(m645=m645, i645=i645, m464=m464, i464=i464,
                      m_train_act=m_train_act, i_train_act=i_train_act, m_train_inact=m_train_inact, i_train_inact=i_train_inact)
            np.savez_compressed(cache_path, **ad)
            print('[ad] saved', cache_path)

        lib['maxT_645'] = ad['m645']
        lib['nn_645_id'] = _ids(ad['i645'], train_ids)
        lib['maxT_464'] = ad['m464']
        lib['nn_464_id'] = _ids(ad['i464'], TRAIN_IDS)
        lib['maxT_train_act'] = ad['m_train_act']
        lib['nn_train_act_id'] = _ids(ad['i_train_act'], ACT_IDS)
        lib['maxT_train_inact'] = ad['m_train_inact']
        lib['nn_train_inact_id'] = _ids(ad['i_train_inact'], INACT_IDS)
        lib['active_margin'] = lib['maxT_train_act'] - lib['maxT_train_inact']
        lib['ad_score'] = lib['maxT_train_act']
        lib['novelty_score'] = 1.0 - lib['maxT_train_act']

        # Legacy 0.40 flag retained for comparison; strict flag is used for the shortlist.
        lib['in_AD'] = lib['maxT_464'] >= AD_CUT
        lib['in_AD_all645'] = lib['maxT_645'] >= AD_CUT
        lib['in_AD_strict'] = lib['maxT_464'] >= AD_CUT_STRICT
        lib['active_supported'] = lib['maxT_train_act'] >= ACTIVE_MIN_T
        lib['active_margin_ok'] = lib['active_margin'] >= ACTIVE_MARGIN_MIN
        lib['ad_pass'] = lib['in_AD_strict'] & lib['active_supported'] & lib['active_margin_ok']
        lib['rank_scope'] = np.where(lib['ad_pass'], 'strict_AD_eligible', 'outside_strict_AD')
        N_ELIGIBLE = int(lib['ad_pass'].sum())
        print('[rank] strict AD eligible:', N_ELIGIBLE, 'of', len(lib))
        lib['parent_id'] = lib['identifier'].astype(str).str.replace(r'\.\d+$', '', regex=True)

        def _summ(v):
            v = np.asarray(v, dtype=float)
            return dict(n=int(len(v)), mean=round(float(v.mean()), 3),
                        median=round(float(np.median(v)), 3),
                        p10=round(float(np.percentile(v, 10)), 3),
                        p90=round(float(np.percentile(v, 90)), 3))

        train_loo = loo_max_tanimoto(TRAIN_FP)
        te_to_train, _ = max_tanimoto_chunked(TE_FP, TRAIN_FP)
        te_to_act, _ = max_tanimoto_chunked(TE_FP, ACT_FP)
        te_to_inact, _ = max_tanimoto_chunked(TE_FP, INACT_FP)
        te_strict = (te_to_train >= AD_CUT_STRICT)
        te_margin = (te_to_act - te_to_inact) >= ACTIVE_MARGIN_MIN
        te_ad_pass = te_strict & (te_to_act >= ACTIVE_MIN_T) & te_margin

        report = dict(
            reference='464 training compounds',
            ad_cut=float(AD_CUT),
            ad_cut_strict=float(AD_CUT_STRICT),
            active_min_t=float(ACTIVE_MIN_T),
            active_margin_min=float(ACTIVE_MARGIN_MIN),
            train_loo=_summ(train_loo),
            test_to_train=_summ(te_to_train),
            test_to_active=_summ(te_to_act),
            test_to_inactive=_summ(te_to_inact),
            screen_to_train=_summ(lib['maxT_464'].values),
            screen_to_active=_summ(lib['maxT_train_act'].values),
            screen_to_inactive=_summ(lib['maxT_train_inact'].values),
            counts=dict(in_AD_040=int(lib['in_AD'].sum()),
                        in_AD_strict=int(lib['in_AD_strict'].sum()),
                        active_supported=int(lib['active_supported'].sum()),
                        ad_pass=int(lib['ad_pass'].sum())),
            test=dict(n=int(len(te)), strict=int(te_strict.sum()), ad_pass=int(te_ad_pass.sum())),
        )
        json.dump(report, open(os.path.join(CACHE_DIR, 'domain_shift.json'), 'w'), indent=2)

        print('=== Domain-shift / evidence report (reference = 464 train) ===')
        print('%-20s %8s %8s %8s %8s %8s' % ('pair', 'n', 'mean', 'median', 'p10', 'p90'))
        for nm, vals in [('train_loo', train_loo), ('test_to_train', te_to_train),
                         ('screen_to_train', lib['maxT_464'].values),
                         ('screen_to_active', lib['maxT_train_act'].values),
                         ('screen_to_inactive', lib['maxT_train_inact'].values)]:
            s = _summ(vals)
            print('%-20s %8d %8.3f %8.3f %8.3f %8.3f' % (nm, s['n'], s['mean'], s['median'], s['p10'], s['p90']))
        print('counts: in_AD(0.40)=%d | strict_AD(%.2f)=%d | active_supported=%d | ad_pass=%d'
              % (int(lib['in_AD'].sum()), AD_CUT_STRICT, int(lib['in_AD_strict'].sum()),
                 int(lib['active_supported'].sum()), int(lib['ad_pass'].sum())))
        print('test coverage: strict_AD %d/%d | ad_pass %d/%d'
              % (int(te_strict.sum()), len(te), int(te_ad_pass.sum()), len(te)))
        if float(np.median(lib['maxT_464'])) < 0.30:
            print('[warn] library is chemically far from training (median maxT < 0.30): ranking is extrapolation.')

        # ===== CELL 19 (code) =====
        # 15 — Accuracy-first ensemble: validation-only calibration, primary model, tiers, TOP-100 tables
        import os, json, numpy as np, pandas as pd
        from multiprocessing import Pool

        def _desc_job_lib(s):
            try:
                return rdkit_desc(s)
            except Exception:
                return None

        PATH_LIBDESC = os.path.join(CACHE_DIR, 'lib_desc_%s.npy' % LIB_HASH)
        if RESUME and os.path.exists(PATH_LIBDESC):
            D_lib = np.load(PATH_LIBDESC)
            print('[resume] library descriptors loaded')
        else:
            try:
                with Pool(N_WORKERS) as p:
                    rows = list(p.imap(_desc_job_lib, lib['canonical_smiles'].tolist(), chunksize=512))
                print('[mp] library descriptors built')
            except Exception as e:
                print('[mp] fallback serial:', e)
                rows = [_desc_job_lib(s) for s in lib['canonical_smiles']]
            D_lib = np.asarray([r if r is not None else [0.0] * 6 for r in rows], dtype=np.float64)
            np.save(PATH_LIBDESC, D_lib)
        base_lib = baseline_model.predict(D_lib)

        def _inv(p_scaled, base):
            d = scaler.inverse_transform(np.asarray(p_scaled).reshape(-1, 1)).ravel()
            return np.clip(d + base, 0.0, 12.0)

        # ---- collect OOF-free validation/test predictions from every model ----
        VAL, TEST, LIBP = {}, {}, {}
        VAL['gcn'], TEST['gcn'] = preds_cache['gcn_val'], preds_cache['gcn_test']
        VAL['gin'], TEST['gin'] = preds_cache['gin_val'], preds_cache['gin_test']
        LIBP['gcn'] = _inv(pred_gcn_scaled, base_lib[np.asarray(screen_idx)])
        LIBP['gin'] = _inv(pred_gin_scaled, base_lib[np.asarray(screen_idx)])
        try:
            _e = np.load(os.path.join(CACHE_DIR, 'reg_val_test_preds_ext.npz'))
            for k in ['afp', 'hfgin']:
                if k + '_val' in _e.files:
                    VAL[k], TEST[k] = _e[k + '_val'], _e[k + '_test']
        except Exception as _x:
            print('[extra] graph-model predictions missing:', _x)
        if os.path.exists(os.path.join(CACHE_DIR, 'screen_preds_ext_%s.npz' % LIB_HASH)):
            pass
        try:
            _s = np.load(PATH_EXT_SCREEN if 'PATH_EXT_SCREEN' in globals() else
                         os.path.join(CACHE_DIR, 'screen_preds_ext_%s_%s.npz' % (LIB_HASH, MODEL_SIG_EXT)))
            ext_idx = _s['idx']
            if 'afp' in _s.files:
                LIBP['afp'] = _inv(_s['afp'], base_lib[ext_idx])
            if 'hfgin' in _s.files:
                LIBP['hfgin'] = _inv(_s['hfgin'], base_lib[ext_idx])
            if 'cb' in _s.files:
                LIBP['cb'] = _inv(_s['cb'], base_lib[ext_idx])
        except Exception as _x:
            print('[extra] screen predictions missing:', _x)
        try:
            _c = np.load(os.path.join(CACHE_DIR, 'chemberta_preds.npz'))
            VAL['cb'], TEST['cb'] = _c['cb_val'], _c['cb_test']
        except Exception:
            pass

        LABEL = {'gcn': 'GCN-reg', 'gin': 'GIN-reg', 'afp': 'AttentiveFP', 'hfgin': 'HFooladi-GIN', 'cb': 'ChemBERTa'}
        ORDER = [k for k in ['gcn', 'gin', 'afp', 'hfgin', 'cb'] if k in VAL and k in TEST]

        # ---- calibration fitted on VALIDATION only (then frozen) ----
        CAL, MC = {}, []
        for k in ORDER:
            try:
                slope, intercept = np.polyfit(VAL[k], y_val_raw, 1)
            except Exception:
                slope, intercept = 1.0, 0.0
            if not np.isfinite(slope) or not (0.2 <= slope <= 2.0):
                slope, intercept = 1.0, 0.0
            CAL[k] = (float(slope), float(intercept))
            vcal = VAL[k] * slope + intercept
            tcal = TEST[k] * slope + intercept
            mv, mt = pic50_metrics(vcal, y_val_raw), pic50_metrics(tcal, y_test_raw)
            MC.append(dict(model=LABEL[k], key=k, cal_slope=round(slope, 4), cal_intercept=round(intercept, 4),
                           val_r2=mv['r2'], val_rmse=mv['rmse'], test_r2=mt['r2'], test_rmse=mt['rmse'], test_mae=mt['mae']))

        primary = max(ORDER, key=lambda k: pic50_metrics(VAL[k] * CAL[k][0] + CAL[k][1], y_val_raw)['r2'])
        print('primary model selected on validation only:', LABEL[primary])

        # ---- fixed simple mean of pre-specified core models (no fitted weights) ----
        ENS_KEYS = [k for k in ['gcn', 'gin', 'afp'] if k in ORDER]
        if not ENS_KEYS:
            ENS_KEYS = [ORDER[0]]
        ens_val = np.mean([VAL[k] * CAL[k][0] + CAL[k][1] for k in ENS_KEYS], axis=0)
        ens_test = np.mean([TEST[k] * CAL[k][0] + CAL[k][1] for k in ENS_KEYS], axis=0)
        print('ensemble: simple mean of', [LABEL[k] for k in ENS_KEYS])

        # ---- write calibrated library predictions on a common index ----
        # gcn/gin are indexed by screen_idx; afp/hfgin/cb by ext_idx. Use gcn/gin index as master.
        master = np.asarray(screen_idx, dtype=int)
        lib['pred_gcn_pic50'] = np.nan; lib['pred_gin_pic50'] = np.nan
        lib['pred_afp_pic50'] = np.nan; lib['pred_hfgin_pic50'] = np.nan; lib['pred_chemberta_pic50'] = np.nan
        for k, col in [('gcn', 'pred_gcn_pic50'), ('gin', 'pred_gin_pic50')]:
            lib.loc[master, col] = LIBP[k]
        for k, col in [('afp', 'pred_afp_pic50'), ('hfgin', 'pred_hfgin_pic50'), ('cb', 'pred_chemberta_pic50')]:
            if k in LIBP:
                lib.loc[ext_idx, col] = LIBP[k]

        # calibrated versions for ranking
        for k, col in [('gcn', 'pred_gcn_pic50'), ('gin', 'pred_gin_pic50'), ('afp', 'pred_afp_pic50'),
                       ('hfgin', 'pred_hfgin_pic50'), ('cb', 'pred_chemberta_pic50')]:
            if k in CAL:
                lib[col] = lib[col] * CAL[k][0] + CAL[k][1]
                lib[col] = lib[col].clip(0.0, 12.0)

        # primary and ensemble on their native indices
        lib['pred_primary_pic50'] = np.nan
        _p_idx = master if primary in ('gcn', 'gin') else ext_idx
        lib.loc[_p_idx, 'pred_primary_pic50'] = (LIBP[primary] * CAL[primary][0] + CAL[primary][1]).clip(0.0, 12.0)
        # ensemble from the already-calibrated core model columns (fixed simple mean)
        cols_avail = [c for c in ['pred_gcn_pic50', 'pred_gin_pic50', 'pred_afp_pic50', 'pred_hfgin_pic50', 'pred_chemberta_pic50']
                      if lib[c].notna().any()]
        ENS_COLS = ['pred_%s_pic50' % k for k in ENS_KEYS]
        lib['pred_ensemble_pic50'] = lib[ENS_COLS].mean(axis=1)
        lib['avg_pred_pic50'] = lib[['pred_gcn_pic50', 'pred_gin_pic50']].mean(axis=1)
        lib['std_pred_pic50'] = lib[cols_avail].std(axis=1)
        lib['conservative_pic50'] = lib['pred_primary_pic50'] - lib['std_pred_pic50']
        med_std = float(lib['std_pred_pic50'].median())

        def _tier(r):
            if pd.isna(r['pred_primary_pic50']) or not bool(r['ad_pass']):
                return 'outside_strict_AD'
            if r['pred_primary_pic50'] >= T1_CUT and r['std_pred_pic50'] <= med_std:
                return 'T1'
            if r['pred_primary_pic50'] >= T2_CUT:
                return 'T2'
            return 'T3'

        lib['tier'] = lib.apply(_tier, axis=1)
        lib['rank_primary'] = lib['pred_primary_pic50'].rank(ascending=False, method='first')
        lib['schema_version'] = 4

        # all-model comparison (honest: test scored once)
        mv = pic50_metrics(ens_val, y_val_raw); mt = pic50_metrics(ens_test, y_test_raw)
        _mpv = pic50_metrics(VAL[primary] * CAL[primary][0] + CAL[primary][1], y_val_raw)
        _mpt = pic50_metrics(TEST[primary] * CAL[primary][0] + CAL[primary][1], y_test_raw)
        MC.append(dict(model='Primary (val-selected)', key=primary, cal_slope=CAL[primary][0],
                       cal_intercept=CAL[primary][1], val_r2=_mpv['r2'], val_rmse=_mpv['rmse'],
                       test_r2=_mpt['r2'], test_rmse=_mpt['rmse'], test_mae=_mpt['mae']))
        MCDF = pd.DataFrame(MC)
        MCDF.to_csv('model_comparison.csv', index=False)
        print(MCDF.round(3).to_string(index=False))

        OUT_COLS = ['identifier', 'canonical_smiles',
                    'pred_gcn_pic50', 'pred_gin_pic50', 'pred_afp_pic50', 'pred_hfgin_pic50', 'pred_chemberta_pic50',
                    'pred_primary_pic50', 'pred_ensemble_pic50', 'avg_pred_pic50', 'std_pred_pic50', 'conservative_pic50',
                    'maxT_464', 'nn_464_id', 'maxT_645', 'nn_645_id', 'maxT_train_act', 'nn_train_act_id',
                    'in_AD', 'in_AD_all645', 'rank_scope', 'tier', 'rank_primary', 'schema_version']
        OUT_COLS = [c for c in OUT_COLS if c in lib.columns]
        lib[OUT_COLS].to_csv('screening_full_regression.csv', index=False)
        for t in ['T1', 'T2', 'T3']:
            lib[lib['tier'] == t].nlargest(100, 'pred_primary_pic50')[OUT_COLS].to_csv('hits_%s.csv' % t, index=False)

        # ---- TOP-100 with full information (AD-aware and raw rankings) ----
        TOP_VIEW = ['identifier', 'canonical_smiles', 'pred_primary_pic50', 'pred_ensemble_pic50',
                    'pred_gcn_pic50', 'pred_gin_pic50', 'pred_afp_pic50', 'pred_hfgin_pic50', 'pred_chemberta_pic50',
                    'std_pred_pic50', 'maxT_464', 'nn_464_id', 'maxT_645', 'maxT_train_act', 'in_AD', 'in_AD_all645', 'tier']
        TOP_VIEW = [c for c in TOP_VIEW if c in lib.columns]
        ad = lib[lib['in_AD']].dropna(subset=['pred_primary_pic50']).nlargest(100, 'pred_primary_pic50')
        raw = lib.dropna(subset=['pred_primary_pic50']).nlargest(100, 'pred_primary_pic50')
        ad.to_csv('top100_screening_AD.csv', index=False)
        raw.to_csv('top100_screening_raw.csv', index=False)
        pd.concat([ad.assign(selection='AD'), raw.assign(selection='raw')]).to_csv('top100_screening_full.csv', index=False)

        print('tier counts:', lib['tier'].value_counts().to_dict())
        print('saved screening_full_regression.csv + hits_T1/T2/T3.csv + top100_screening_*.csv')
        print('')
        print('TOP 100 — AD-aware (in_AD=True), ranked by validation-selected primary model')
        print(ad[TOP_VIEW].round(3).to_string(index=False))
        print('')
        print('TOP 100 — raw ranking (no AD filter; %d of top 100 are in_AD)' % int(raw['in_AD'].sum()))
        print(raw[TOP_VIEW].round(3).to_string(index=False))

        # ===== CELL 20 (code) =====
        # 16 — Strict evidence shortlist, confidence bands, honest outputs, QC
        import os
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        # ---- 1. Stable ensemble: simple mean of pre-specified core models (no NNLS) ----
        CORE = [k for k in ['gcn', 'gin', 'afp'] if k in ORDER]
        if not CORE:
            CORE = [ORDER[0]]
        CORE_COLS = ['pred_%s_pic50' % k for k in CORE]

        def _cal_vec(k, arr):
            return np.clip(np.asarray(arr, dtype=float) * CAL[k][0] + CAL[k][1], 0.0, 12.0)

        ENS_VAL = np.mean([_cal_vec(k, VAL[k]) for k in CORE], axis=0)
        ENS_TEST = np.mean([_cal_vec(k, TEST[k]) for k in CORE], axis=0)
        lib['pred_ensemble_pic50'] = np.clip(
            np.mean([lib[c].values.astype(float) for c in CORE_COLS], axis=0), 0.0, 12.0)

        # Validation-only conformal-style interval (80%); not a guarantee under domain shift.
        resid_val = np.asarray(y_val_raw, dtype=float) - ENS_VAL
        q_low, q_high = np.quantile(resid_val, [CONF_ALPHA / 2.0, 1.0 - CONF_ALPHA / 2.0])
        lib['conf_low_pic50'] = lib['pred_ensemble_pic50'] + q_low
        lib['conf_high_pic50'] = lib['pred_ensemble_pic50'] + q_high
        lib['conf_width_pic50'] = q_high - q_low

        # ---- 2. Model agreement (core models only) ----
        lib['model_range_pic50'] = lib[CORE_COLS].max(axis=1) - lib[CORE_COLS].min(axis=1)
        lib['model_std_core'] = lib[CORE_COLS].std(axis=1, ddof=0)
        lib['rank_score_pic50'] = lib['pred_ensemble_pic50'] - RANK_LAMBDA * lib['model_std_core']
        lib['model_agree'] = ((lib['model_range_pic50'] <= MAX_MODEL_RANGE) &
                              (lib['model_std_core'] <= MAX_MODEL_STD))

        # ---- 3. Strict, diverse, deduplicated shortlist ----
        def _scaf(smi):
            try:
                m = Chem.MolFromSmiles(str(smi))
                if m is None:
                    return ''
                s = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
                return s if s else Chem.MolToSmiles(m, canonical=True)
            except Exception:
                return ''

        base = lib[lib['ad_pass'] & lib['model_agree']].copy()
        base = base.sort_values('rank_score_pic50', ascending=False)
        selected, seen_parent, seen_scaf = [], set(), set()
        for cutoff, level in [(T1_CUT, 'T1_strict'), (T2_CUT, 'T2_strict')]:
            pool = base[base['pred_ensemble_pic50'] >= cutoff]
            for _, r in pool.head(2500).iterrows():
                if len(selected) >= SHORTLIST_MAX:
                    break
                pid = str(r['parent_id'])
                scaf = _scaf(r['canonical_smiles']) or ('PARENT:' + pid)
                if pid in seen_parent or scaf in seen_scaf:
                    continue
                seen_parent.add(pid)
                seen_scaf.add(scaf)
                rec = r.to_dict()
                rec['_idx'] = r.name
                rec['evidence_level'] = level
                rec['shortlist_rank'] = len(selected) + 1
                selected.append(rec)
            if len(selected) >= SHORTLIST_MAX:
                break

        SHORT_COLS = ['identifier', 'parent_id', 'canonical_smiles', 'pred_ensemble_pic50',
                      'conf_low_pic50', 'conf_high_pic50', 'conf_width_pic50',
                      'model_range_pic50', 'model_std_core', 'rank_score_pic50', 'maxT_464', 'maxT_train_act',
                      'maxT_train_inact', 'active_margin', 'nn_train_act_id', 'nn_train_inact_id',
                      'in_AD_strict', 'active_supported', 'active_margin_ok', 'ad_pass',
                      'novelty_score', 'tier', 'evidence_level', 'shortlist_rank']
        short_idx = [r['_idx'] for r in selected]
        SHORTLIST = pd.DataFrame([{k: v for k, v in r.items() if k != '_idx'} for r in selected],
                                 columns=SHORT_COLS)
        SHORTLIST.to_csv('shortlist_low_confidence.csv', index=False)

        lib['shortlist_flag'] = lib.index.isin(short_idx)
        lib['candidate_status'] = np.where(lib['shortlist_flag'],
                                           'low_confidence_hypothesis', 'not_a_hit')
        lib['evidence_level'] = np.where(lib['shortlist_flag'], 'shortlist', 'insufficient')

        # ---- 4. Rewrite outputs so old tier/top-100 labels are not mistaken for hits ----
        for _name in ['hits_T1.csv', 'hits_T2.csv', 'hits_T3.csv',
                      'top100_screening_AD.csv', 'top100_screening_raw.csv',
                      'top100_screening_full.csv']:
            if os.path.exists(_name):
                os.replace(_name, 'legacy_' + _name)

        NEW_COLS = ['pred_ensemble_pic50', 'conf_low_pic50', 'conf_high_pic50', 'conf_width_pic50',
                    'model_range_pic50', 'model_std_core', 'model_agree', 'rank_score_pic50', 'rank_scope', 'maxT_train_inact',
                    'nn_train_inact_id', 'active_margin', 'active_supported', 'active_margin_ok',
                    'in_AD_strict', 'ad_pass', 'ad_score', 'novelty_score', 'parent_id',
                    'shortlist_flag', 'candidate_status', 'evidence_level']
        out_cols = [c for c in list(dict.fromkeys(list(OUT_COLS) + NEW_COLS)) if c in lib.columns]
        lib[out_cols].to_csv('screening_full_regression.csv', index=False)

        for _t in ['T1', 'T2', 'T3']:
            _sub = lib[lib['tier'] == _t]
            _sub[out_cols].to_csv('threshold_%s_not_hits.csv' % _t, index=False)

        _diag_ad = lib[lib['ad_pass']].nlargest(100, 'rank_score_pic50')
        _diag_raw = lib.nlargest(100, 'pred_ensemble_pic50')
        _diag_ad[out_cols].to_csv('diagnostic_top100_AD.csv', index=False)
        _diag_raw[out_cols].to_csv('diagnostic_top100_raw.csv', index=False)

        # ---- 5. Nearest-training / novelty annotation (post-processing, no model change) ----
        def _annotate_similarity(frame):
            """Attach explicit nearest-training evidence and novelty status to every row."""
            f = frame.copy()
            if 'maxT_464' not in f.columns or 'nn_464_id' not in f.columns:
                return f
            f['nearest_train_id'] = f['nn_464_id'].astype(str).replace({'nan': '', 'None': ''})
            f['nearest_train_similarity'] = pd.to_numeric(
                f['maxT_464'], errors='coerce').clip(0.0, 1.0)
            f.loc[f['nearest_train_id'].eq(''), 'nearest_train_similarity'] = np.nan
            _meta = pd.DataFrame({
                'train_id': TRAIN_IDS,
                'train_class': np.where(_yy[tr] == 1, 'Active', 'Inactive'),
                'train_pIC50': raw_targets[tr],
            }).drop_duplicates('train_id').set_index('train_id')
            f['nearest_train_class'] = f['nearest_train_id'].map(_meta['train_class'])
            f['nearest_train_pIC50'] = f['nearest_train_id'].map(_meta['train_pIC50'])
            f['similarity_informative'] = (
                f['nearest_train_similarity'].notna() & f['nearest_train_id'].ne(''))
            f['near_training'] = np.select(
                [~f['similarity_informative'],
                 f['nearest_train_similarity'] >= 0.99,
                 f['nearest_train_similarity'] >= AD_CUT_STRICT,
                 f['nearest_train_similarity'] >= AD_CUT],
                ['unknown', 'exact_or_near_exact', 'near_strict', 'borderline_legacy'],
                default='far')
            f['known_training_compound'] = (
                f['similarity_informative'] & (f['nearest_train_similarity'] >= 0.99))
            _ad_pass = f['ad_pass'] if 'ad_pass' in f.columns else pd.Series(False, index=f.index)
            _ad_pass = _ad_pass.fillna(False).astype(bool)
            f['novelty_claim_allowed'] = _ad_pass & ~f['known_training_compound']
            f['prediction_status'] = np.select(
                [f['known_training_compound'], _ad_pass],
                ['known_training_compound', 'AD_supported_prediction'],
                default='extrapolation_no_novelty_claim')
            if 'candidate_status' in f.columns:
                f.loc[f['known_training_compound'], 'candidate_status'] = 'known_training_compound'
            if 'evidence_level' in f.columns:
                f.loc[f['known_training_compound'], 'evidence_level'] = 'training_match'
            return f

        for _fname in ['screening_full_regression.csv', 'shortlist_low_confidence.csv',
                       'diagnostic_top100_AD.csv', 'diagnostic_top100_raw.csv',
                       'threshold_T1_not_hits.csv', 'threshold_T2_not_hits.csv',
                       'threshold_T3_not_hits.csv']:
            if not os.path.exists(_fname):
                continue
            try:
                _frame = pd.read_csv(_fname)
            except Exception:
                continue
            if 'maxT_464' not in _frame.columns or 'nn_464_id' not in _frame.columns:
                continue
            _frame = _annotate_similarity(_frame)
            if _fname.startswith('shortlist') or _fname.startswith('diagnostic'):
                _frame = _frame[~_frame['known_training_compound']].copy()
            _frame.to_csv(_fname, index=False)

        _full = pd.read_csv('screening_full_regression.csv')
        _overlap = _full[_full['known_training_compound']].copy()
        _overlap.to_csv('training_overlap.csv', index=False)
        print('[similarity] known/near-exact training compounds: %d | near_strict: %d | borderline_legacy: %d'
              % (int(_overlap.shape[0]), int((_full['near_training'] == 'near_strict').sum()),
                 int((_full['near_training'] == 'borderline_legacy').sum())))
        print('[similarity] training_overlap.csv written; exact training compounds excluded from shortlist/diagnostic tables')

        with open('README_SHORTLIST.txt', 'w') as _fh:
            _fh.write(
                'MDM2 Part_20 screening: low-confidence hypothesis shortlist\n'
                '===========================================================\n'
                'These are NOT validated hits and NOT assay recommendations.\n\n'
                'Filters applied:\n'
                '  strict AD: maxT to 464 train >= %.2f\n'
                '  nearest active train compound: maxT >= %.2f\n'
                '  active-inactive margin: maxT(active) - maxT(inactive) >= %.2f\n'
                '  core-model agreement: range <= %.2f and std <= %.2f pIC50\n'
                '  ranking score: pred_ensemble - %.1f x core-model std\n'
                '  strict eligible (ad_pass): %d compounds\n'
                '  one row per parent compound and per Murcko scaffold cluster\n'
                '  cap: %d diverse parents; shortlisted: %d\n\n'
                'Confidence band: %.0f%% interval from validation residuals only.\n'
                'The old top-100 and T1/T2/T3 files are diagnostic/threshold artifacts.\n'
                'Do not treat them as hits.\n'
                % (AD_CUT_STRICT, ACTIVE_MIN_T, ACTIVE_MARGIN_MIN, MAX_MODEL_RANGE,
                   MAX_MODEL_STD, RANK_LAMBDA, N_ELIGIBLE, SHORTLIST_MAX, len(SHORTLIST),
                   100 * (1 - CONF_ALPHA)))

        # ---- 5. QC ----
        fig, ax = plt.subplots(2, 3, figsize=(16, 9))
        v = lib['pred_ensemble_pic50'].dropna()
        ax[0, 0].hist(v, bins=60, color='steelblue')
        ax[0, 0].axvline(T1_CUT, color='red', ls='--')
        ax[0, 0].set_title('Simple ensemble predicted pIC50')

        ax[0, 1].hist(lib['model_range_pic50'].dropna(), bins=60, color='darkorange')
        ax[0, 1].axvline(MAX_MODEL_RANGE, color='red', ls='--')
        ax[0, 1].set_title('Core-model range (pIC50)')

        ax[0, 2].hist(lib['maxT_464'].dropna(), bins=60, color='seagreen')
        ax[0, 2].axvline(AD_CUT, color='orange', ls=':', label='legacy 0.40')
        ax[0, 2].axvline(AD_CUT_STRICT, color='red', ls='--', label='strict %.2f' % AD_CUT_STRICT)
        ax[0, 2].legend(fontsize=7)
        ax[0, 2].set_title('max Tanimoto to 464 training')

        ax[1, 0].scatter(lib['maxT_464'], lib['pred_ensemble_pic50'],
                         c=lib['ad_pass'].astype(int), s=3, alpha=0.3, cmap='coolwarm')
        ax[1, 0].axvline(AD_CUT_STRICT, color='red', ls='--')
        ax[1, 0].set_xlabel('maxT to 464 train')
        ax[1, 0].set_ylabel('pred pIC50')
        ax[1, 0].set_title('Similarity vs prediction (red = ad_pass)')

        for k, lab in [('gcn', 'GCN'), ('gin', 'GIN'), ('afp', 'AttentiveFP'), ('hfgin', 'HFooladi-GIN')]:
            if k in METRICS and METRICS[k].get('hist'):
                ax[1, 1].plot(METRICS[k]['hist'], label=lab)
        if EXT_METRICS.get('afp', {}).get('hist'):
            ax[1, 1].plot(EXT_METRICS['afp']['hist'], label='AttentiveFP')
        if EXT_METRICS.get('hfgin', {}).get('hist'):
            ax[1, 1].plot(EXT_METRICS['hfgin']['hist'], label='HFooladi-GIN')
        ax[1, 1].set_title('Validation R2 during training')
        ax[1, 1].set_xlabel('epoch')
        ax[1, 1].legend(fontsize=7)

        for k, lab, mk in [('gcn', 'GCN', 'o'), ('gin', 'GIN', 'o'), ('afp', 'AttentiveFP', '^'), ('hfgin', 'HFooladi-GIN', 's')]:
            if k in TEST:
                ax[1, 2].scatter(y_test_raw, TEST[k], s=14, alpha=0.45, label=lab, marker=mk)
        if 'cb' in TEST:
            ax[1, 2].scatter(y_test_raw, TEST['cb'], s=14, alpha=0.45, label='ChemBERTa', marker='d')
        ax[1, 2].scatter(y_test_raw, ENS_TEST, s=20, alpha=0.9, label='simple ensemble', marker='x', color='black')
        ax[1, 2].scatter(y_test_raw, TEST[primary], s=22, alpha=0.9, label='primary', marker='+', color='red')
        _lo = min(float(y_test_raw.min()), 4.0); _hi = max(float(y_test_raw.max()), 10.5)
        ax[1, 2].plot([_lo, _hi], [_lo, _hi], 'k--')
        ax[1, 2].set_xlabel('true pIC50'); ax[1, 2].set_ylabel('pred pIC50')
        ax[1, 2].set_title('Test parity (scored once)')
        ax[1, 2].legend(fontsize=7)

        plt.tight_layout()
        plt.savefig('screening_regression_QC.png', dpi=120)
        plt.show()
        print('saved screening_regression_QC.png')

        print('strict AD: %d | active_supported: %d | ad_pass: %d | model_agree: %d'
              % (int(lib['in_AD_strict'].sum()), int(lib['active_supported'].sum()),
                 int(lib['ad_pass'].sum()), int(lib['model_agree'].sum())))
        print('shortlist: %d diverse parents (cap %d) | parents=%d | scaffolds=%d'
              % (len(SHORTLIST), SHORTLIST_MAX, SHORTLIST['parent_id'].nunique() if len(SHORTLIST) else 0,
                 SHORTLIST['canonical_smiles'].map(_scaf).nunique() if len(SHORTLIST) else 0))
        print('conformal interval: q_low=%.3f q_high=%.3f (validation residuals only)' % (q_low, q_high))
        print('outputs: shortlist_low_confidence.csv | threshold_T*_not_hits.csv | diagnostic_top100_*.csv')

        # ===== CELL 21 (code) =====
        # 17 — Acceptance report + run manifest (resume-safe)
        import os, json, time, torch

        def _met(y, p):
            y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
            r2 = 1.0 - float(((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
            return dict(r2=round(r2, 4), rmse=round(float(np.sqrt(((y - p) ** 2).mean())), 4),
                        mae=round(float(np.abs(y - p).mean()), 4))

        manifest = {
            'run_utc': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
            'config': dict(seed=RANDOM_SEED, morgan_radius=MORGAN_RADIUS, morgan_nbits=MORGAN_NBITS,
                           ad_cut=AD_CUT, ad_cut_strict=AD_CUT_STRICT, active_min_t=ACTIVE_MIN_T,
                           active_margin_min=ACTIVE_MARGIN_MIN, max_model_range=MAX_MODEL_RANGE,
                           max_model_std=MAX_MODEL_STD, shortlist_max=SHORTLIST_MAX,
                           t1_cut=T1_CUT, t2_cut=T2_CUT, hidden=GNN_HIDDEN,
                           layers=GNN_LAYERS, epochs_cap=GNN_EPOCHS, patience=GNN_PATIENCE,
                           batch=GNN_BATCH, resume=RESUME, dual_gpu=bool(USE_DUAL_GPU),
                           extra_models=bool(USE_EXTRA_MODELS), chemberta=bool('cb' in TEST),
                           calibration='validation-only linear; simple_mean_gcn_gin_afp'),
            'split': dict(train=int(len(tr)), val=int(len(va)), test=int(len(te))),
            'library': dict(path=LIB_PATH, rows=int(len(lib)), hash=LIB_HASH),
            'metrics': {k: {'val': _met(y_val_raw, VAL[k] * CAL[k][0] + CAL[k][1]),
                            'test': _met(y_test_raw, TEST[k] * CAL[k][0] + CAL[k][1])} for k in ORDER},
            'primary': primary,
            'primary_label': LABEL[primary],
            'calibration': {k: dict(slope=CAL[k][0], intercept=CAL[k][1]) for k in ORDER},
            'ensemble_method': 'simple_mean_%s' % '+'.join(CORE),
            'ensemble_weights': {k: 1.0 / len(CORE) for k in CORE},
            'ad': dict(reference='maxT_464 >= AD_CUT', in_AD=int(lib['in_AD'].sum()),
                       fraction=round(float(lib['in_AD'].mean()), 4),
                       in_AD_all645=int(lib['in_AD_all645'].sum()),
                       in_AD_strict=int(lib['in_AD_strict'].sum()),
                       active_supported=int(lib['active_supported'].sum()),
                       ad_pass=int(lib['ad_pass'].sum()),
                       strict_cut=float(AD_CUT_STRICT)),
            'tiers': lib['tier'].value_counts().to_dict(),
            'gpu': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'pretrained_classifier_used': False,
            'chemberta_available': bool(HAVE_TRANSFORMERS),
        }
        manifest['metrics']['primary'] = _met(y_val_raw, VAL[primary] * CAL[primary][0] + CAL[primary][1])
        manifest['metrics']['primary']['test'] = _met(y_test_raw, TEST[primary] * CAL[primary][0] + CAL[primary][1])
        manifest['metrics']['ensemble'] = {'val': _met(y_val_raw, ENS_VAL), 'test': _met(y_test_raw, ENS_TEST)}
        manifest['shortlist'] = dict(n=int(len(SHORTLIST)), n_eligible=int(N_ELIGIBLE), cap=int(SHORTLIST_MAX),
                                    status='low-confidence hypotheses, not validated hits',
                                    file='shortlist_low_confidence.csv')
        json.dump(manifest, open('run_manifest.json', 'w'), indent=2)

        print('=== ACCEPTANCE ===')
        print('split train/val/test = %d/%d/%d | scaffold split + canonical-overlap assert (leak-check)' % (len(tr), len(va), len(te)))
        print('library rows: %d | hash %s | source %s' % (len(lib), LIB_HASH, LIB_PATH))
        print('coverage: GCN %d | GIN %d | AFP %d | HFooladi %d | ChemBERTa %d | maxT_464 %d'
              % (int(lib['pred_gcn_pic50'].notna().sum()), int(lib['pred_gin_pic50'].notna().sum()),
                 int(lib['pred_afp_pic50'].notna().sum()), int(lib['pred_hfgin_pic50'].notna().sum()),
                 int(lib['pred_chemberta_pic50'].notna().sum()) if 'pred_chemberta_pic50' in lib else 0,
                 int(lib['maxT_464'].notna().sum())))
        print('AD: legacy %d | strict %d | active-supported %d | ad_pass %d | shortlist %d'
              % (int(lib['in_AD'].sum()), int(lib['in_AD_strict'].sum()),
                 int(lib['active_supported'].sum()), int(lib['ad_pass'].sum()), int(len(SHORTLIST))))
        print('AD reference = 464 training compounds | legacy in_AD_all645 retained')
        print('T1/T2/T3 are thresholds, not validated hits; use shortlist_low_confidence.csv')
        print('test evaluated ONCE | early stopping + calibration + model selection only on val')
        print('pretrained Part_18/HFooladi .pth NOT used (trained/calibrated on 645 => leakage)')
        print('GPU:', manifest['gpu'], '| dual-gpu threads used:', bool(USE_DUAL_GPU and torch.cuda.device_count() > 1))
        print('primary (validation-selected):', LABEL[primary], '| ChemBERTa available:', HAVE_TRANSFORMERS)
        print('')
        _ev = _met(y_val_raw, ENS_VAL); _et = _met(y_test_raw, ENS_TEST)
        MCDF = pd.concat([MCDF, pd.DataFrame([dict(model='Simple ensemble', key='ensemble',
            cal_slope=np.nan, cal_intercept=np.nan, val_r2=_ev['r2'], val_rmse=_ev['rmse'],
            test_r2=_et['r2'], test_rmse=_et['rmse'], test_mae=_et['mae'])])], ignore_index=True)
        MCDF.to_csv('model_comparison.csv', index=False)
        print('Regression metrics (pIC50; calibrated on val only):')
        for row in MCDF.itertuples():
            print('  %-22s val R2=%.3f RMSE=%.3f | test R2=%.3f RMSE=%.3f MAE=%.3f'
                  % (row.model, row.val_r2, row.val_rmse, row.test_r2, row.test_rmse, row.test_mae))
        print('')
        for f in ['screening_full_regression.csv', 'shortlist_low_confidence.csv',
                  'threshold_T1_not_hits.csv', 'threshold_T2_not_hits.csv', 'threshold_T3_not_hits.csv',
                  'diagnostic_top100_AD.csv', 'diagnostic_top100_raw.csv', 'model_comparison.csv',
                  'screening_regression_QC.png', 'README_SHORTLIST.txt', 'run_manifest.json']:
            if os.path.exists(f):
                print('  [out] %s (%.2f MB)' % (f, os.path.getsize(f) / 1e6))
        print('cache:', sorted(os.listdir(CACHE_DIR)))
        print('[done] re-run is safe: downloads, fingerprints, models, predictions and AD are cached and skipped.')

        # ===== CELL 22 (code) =====
        # 18 — FULL INPUT/OUTPUT AUDIT: every DL input, prediction, residual and error
        import os, json, numpy as np, pandas as pd, torch
        from torch_geometric.loader import DataLoader as _AuditLoader

        AUDIT = {'data': {}, 'models': {}, 'metrics': {}, 'checks': {}, 'screening': {}}
        SPLITS = [('train', tr, train_graphs), ('val', va, val_graphs), ('test', te, test_graphs)]

        print('=' * 82)
        print('DEEP-LEARNING INPUT AUDIT (features, targets, model config, devices)')
        print('=' * 82)
        for nm, ii, gs in SPLITS:
            yt = np.asarray(raw_targets)[ii]
            ns = np.array([int(g.num_nodes) for g in gs])
            es = np.array([int(g.num_edges) for g in gs])
            print('%-5s n=%3d | pIC50 min/mean/max %.3f / %.3f / %.3f | nodes %.1f+-%.1f | edges %.1f+-%.1f'
                  % (nm, len(gs), yt.min(), yt.mean(), yt.max(), ns.mean(), ns.std(), es.mean(), es.std()))
            AUDIT['data'][nm] = dict(n=int(len(gs)), pic50_min=float(yt.min()), pic50_mean=float(yt.mean()),
                                     pic50_max=float(yt.max()), nodes_mean=float(ns.mean()), edges_mean=float(es.mean()))

        g0 = train_graphs[0]
        print('node feature dim: %d | edge_index: 2 x %d | target = scaled delta'
              % (int(g0.x.shape[1]), int(g0.edge_index.shape[1])))
        print('sample INPUT (train graph 0) y_scaled = %.4f' % float(g0.y.view(-1)[0]))
        print('  x.shape', tuple(g0.x.shape), '| first 3 atom feature rows:')
        print(g0.x[:3].numpy())
        print('  edge_index[:, :5]:')
        print(g0.edge_index[:, :5].numpy())
        print('delta scaler: mean %.4f | scale %.4f (fit on 464 train only)'
              % (float(scaler.mean_[0]), float(scaler.scale_[0])))
        print('model config: hidden=%d layers=%d batch=%d lr=%g wd=%g patience=%d'
              % (GNN_HIDDEN, GNN_LAYERS, GNN_BATCH, GNN_LR, GNN_WD, GNN_PATIENCE))
        print('devices: dev0=%s | dev1=%s | cuda=%s | n_gpu=%d'
              % (dev0, dev1, torch.cuda.is_available(), torch.cuda.device_count()))
        print('policy: 464 train / 52 val / 129 test | early stopping = val only | test scored once')

        print('')
        print('=' * 82)
        print('MODEL LOAD AUDIT (strict load, finite weights, parameter count)')
        print('=' * 82)
        MODELS, RAW_PRED = {}, {}
        for kind in ['gcn', 'gin']:
            path = os.path.join(CACHE_DIR, kind + '_reg.pt')
            model = GNNReg(kind=kind)
            sd = torch_load_compat(path)
            model.load_state_dict(sd)
            dev = dev0 if kind == 'gcn' else dev1
            model.to(dev).eval()
            nonfinite = sum(int((~torch.isfinite(v)).sum().item()) for v in model.state_dict().values())
            MODELS[kind] = (model, dev)
            print('%s: params %d | state_dict tensors %d | non-finite weights %d | device %s'
                  % (kind.upper(), count_params(model), len(sd), nonfinite, dev))
            print(model)
            RAW_PRED[kind] = {}
            for nm, ii, gs in SPLITS:
                dl = _AuditLoader(gs, batch_size=SCREEN_BATCH, shuffle=False, num_workers=0)
                out = []
                with torch.no_grad():
                    for b in dl:
                        b = b.to(dev)
                        out.append(model(b.x, b.edge_index, b.batch).detach().cpu().numpy())
                RAW_PRED[kind][nm] = np.concatenate(out)
            AUDIT['models'][kind] = dict(params=int(count_params(model)), state_tensors=int(len(sd)),
                                          nonfinite_weights=int(nonfinite), device=str(dev))

        def _audit_inv(p_scaled, idx):
            d = scaler.inverse_transform(np.asarray(p_scaled).reshape(-1, 1)).ravel()
            return np.clip(d + base_all[np.asarray(idx)], 0.0, 12.0)

        PIC50 = {k: {nm: _audit_inv(RAW_PRED[k][nm], ii) for nm, ii, _ in SPLITS} for k in ['gcn', 'gin']}
        max_saved_diff = 0.0
        for kind in ['gcn', 'gin']:
            for nm in ['val', 'test']:
                saved = preds_cache.get(kind + '_' + nm)
                if saved is not None:
                    d = float(np.max(np.abs(np.asarray(saved) - PIC50[kind][nm])))
                    max_saved_diff = max(max_saved_diff, d)
                    print('recompute-vs-saved %s %s: max|diff| = %.2e' % (kind, nm, d))

        rows = []
        for nm, ii, gs in SPLITS:
            for j, i in enumerate(ii):
                i = int(i)
                rows.append(dict(split=nm, row_index=i, identifier=str(train_ids[i]), smiles=str(smiles_train[i]),
                                 pIC50_true=float(raw_targets[i]), pIC50_baseline=float(base_all[i]),
                                 pIC50_gcn=float(PIC50['gcn'][nm][j]), pIC50_gin=float(PIC50['gin'][nm][j])))
        audit = pd.DataFrame(rows)
        audit['pIC50_ensemble'] = (audit.pIC50_gcn + audit.pIC50_gin) / 2.0
        for c in ['gcn', 'gin', 'ensemble']:
            audit['resid_' + c] = audit.pIC50_true - audit['pIC50_' + c]
            audit['abs_err_' + c] = audit['resid_' + c].abs()
        audit.to_csv('dl_full_predictions.csv', index=False)
        print('full prediction table:', audit.shape, '-> dl_full_predictions.csv')

        def _audit_metrics(t, p):
            t = np.asarray(t, float)
            p = np.asarray(p, float)
            resid = t - p
            ss_res = float((resid ** 2).sum())
            ss_tot = float(((t - t.mean()) ** 2).sum())
            slope, intercept = np.polyfit(t, p, 1)
            return dict(n=int(len(t)), r2=float(1 - ss_res / max(ss_tot, 1e-12)),
                        rmse=float(np.sqrt((resid ** 2).mean())), mae=float(np.abs(resid).mean()),
                        bias=float(resid.mean()), resid_std=float(resid.std()),
                        cal_slope=float(slope), cal_intercept=float(intercept),
                        max_abs=float(np.abs(resid).max()), pred_min=float(p.min()), pred_max=float(p.max()))

        print('')
        print('=' * 82)
        print('OUTPUT / ERROR AUDIT (pIC50; residual = true - predicted)')
        print('=' * 82)
        print('%-6s %-8s %4s %7s %7s %7s %8s %9s %9s %9s %8s'
              % ('split', 'model', 'n', 'R2', 'RMSE', 'MAE', 'bias', 'resid_sd', 'max|err|', 'cal_slope', 'cal_int'))
        for nm, ii, gs in SPLITS:
            t = raw_targets[ii]
            ens = (PIC50['gcn'][nm] + PIC50['gin'][nm]) / 2.0
            for label, p in [('Ridge', base_all[ii]), ('GCN', PIC50['gcn'][nm]),
                             ('GIN', PIC50['gin'][nm]), ('ENS', ens)]:
                m = _audit_metrics(t, p)
                AUDIT['metrics']['%s_%s' % (nm, label)] = m
                print('%-6s %-8s %4d %7.3f %7.3f %7.3f %8.3f %9.3f %9.3f %9.3f %8.2f'
                      % (nm, label, m['n'], m['r2'], m['rmse'], m['mae'], m['bias'], m['resid_std'],
                         m['max_abs'], m['cal_slope'], m['cal_intercept']))

        print('')
        print('TRAINING HISTORY — every epoch (model selected by best val R2)')
        max_ep = max(len(METRICS['gcn']['hist']), len(METRICS['gin']['hist']))
        hdf = pd.DataFrame({'epoch': np.arange(1, max_ep + 1)})
        for kind in ['gcn', 'gin']:
            h = METRICS[kind]['hist']
            hdf[kind + '_val_r2'] = pd.Series(h)
            best_ep = int(np.argmax(h)) + 1
            print('%s: %d epochs | best epoch %d | best val R2 %.4f | params %d'
                  % (kind.upper(), len(h), best_ep, max(h), METRICS[kind]['params']))
            for ep, r2 in enumerate(h, 1):
                print('  %s ep %03d  val_R2 %.4f%s' % (kind, ep, r2, '  <= BEST' if ep == best_ep else ''))
        hdf.to_csv('dl_training_history.csv', index=False)

        pd.set_option('display.max_rows', 400)
        pd.set_option('display.width', 250)
        pd.set_option('display.max_colwidth', 46)
        show = audit.copy()
        show['smiles'] = show['smiles'].str.slice(0, 46)
        cols = ['split', 'identifier', 'smiles', 'pIC50_true', 'pIC50_baseline', 'pIC50_gcn',
                'pIC50_gin', 'pIC50_ensemble', 'resid_gcn', 'resid_gin', 'resid_ensemble']
        print('')
        print('FULL 645-ROW OUTPUT TABLE (train + val + test; every compound, every prediction)')
        print('NOTE: train rows are in-sample (fit data), not accuracy estimates; val/test rows are held out.')
        print(show[cols].round(3).to_string(index=False))

        print('')
        print('WORST 8 ABSOLUTE ERRORS PER SPLIT (ensemble)')
        for nm in ['train', 'val', 'test']:
            sub = audit[audit.split == nm].nlargest(8, 'abs_err_ensemble').copy()
            sub['smiles'] = sub['smiles'].str.slice(0, 42)
            print('-- %s --' % nm)
            print(sub[['identifier', 'smiles', 'pIC50_true', 'pIC50_gcn', 'pIC50_gin',
                       'pIC50_ensemble', 'resid_ensemble']].round(3).to_string(index=False))

        print('')
        print('SIGNED BIAS BY TRUE pIC50 BIN (ensemble; negative = over-prediction)')
        audit['true_bin'] = pd.cut(audit.pIC50_true, bins=[0, 6, 7, 8, 9, 12])
        print(audit.groupby(['split', 'true_bin'], observed=True)['resid_ensemble']
              .agg(['count', 'mean']).round(3).to_string())

        print('')
        print('=' * 82)
        print('SCREENING INPUT/OUTPUT AUDIT (SMILES -> graph -> prediction -> saved CSV)')
        print('=' * 82)
        SPATH = 'screening_full_regression.csv'
        if os.path.exists(SPATH):
            scr = pd.read_csv(SPATH)
            print('rows %d | columns %d | pred_gcn non-null %d | pred_gin non-null %d | maxT_464 non-null %d'
                  % (len(scr), scr.shape[1], int(scr.pred_gcn_pic50.notna().sum()),
                     int(scr.pred_gin_pic50.notna().sum()), int(scr.maxT_464.notna().sum())))
            print('range: GCN %.2f..%.2f | GIN %.2f..%.2f | ENS %.2f..%.2f | std %.3f..%.3f'
                  % (scr.pred_gcn_pic50.min(), scr.pred_gcn_pic50.max(), scr.pred_gin_pic50.min(),
                     scr.pred_gin_pic50.max(), scr.avg_pred_pic50.min(), scr.avg_pred_pic50.max(),
                     scr.std_pred_pic50.min(), scr.std_pred_pic50.max()))
            print('in_AD(0.40) %d (%.2f%%) | strict_AD %d | ad_pass %d | tiers %s'
                  % (int(scr.in_AD.sum()), 100.0 * float(scr.in_AD.mean()),
                     int(scr.in_AD_strict.sum()), int(scr.ad_pass.sum()),
                     {str(k): int(v) for k, v in scr.tier.value_counts().items()}))

            chk = {}
            chk['screening_coverage'] = bool(scr.pred_gcn_pic50.notna().all() and scr.pred_gin_pic50.notna().all())
            chk['screening_predictions_finite'] = bool(np.isfinite(scr[['pred_gcn_pic50', 'pred_gin_pic50']].values).all())
            chk['screening_pred_in_0_12'] = bool(((scr.pred_gcn_pic50 >= 0) & (scr.pred_gcn_pic50 <= 12)).all()
                                                 and ((scr.pred_gin_pic50 >= 0) & (scr.pred_gin_pic50 <= 12)).all())
            chk['avg_is_mean_of_two'] = bool(np.allclose(scr.avg_pred_pic50,
                                                         (scr.pred_gcn_pic50 + scr.pred_gin_pic50) / 2.0, atol=1e-6))
            chk['in_AD_flag_consistent'] = bool((scr.in_AD == (scr.maxT_464 >= AD_CUT)).all())
            chk['strict_ad_flag_consistent'] = bool((scr.in_AD_strict == (scr.maxT_464 >= AD_CUT_STRICT)).all())
            chk['active_support_flag_consistent'] = bool((scr.active_supported == (scr.maxT_train_act >= ACTIVE_MIN_T)).all())
            chk['active_margin_flag_consistent'] = bool((scr.active_margin_ok == (scr.active_margin >= ACTIVE_MARGIN_MIN)).all())
            chk['ad_pass_flag_consistent'] = bool((scr.ad_pass == (scr.in_AD_strict & scr.active_supported & scr.active_margin_ok)).all())
            chk['model_agree_flag_consistent'] = bool((scr.model_agree == ((scr.model_range_pic50 <= MAX_MODEL_RANGE) & (scr.model_std_core <= MAX_MODEL_STD))).all())
            chk['tier_values_valid'] = bool(scr.tier.isin(['T1', 'T2', 'T3', 'outside_strict_AD']).all())

            pick = pd.concat([
                scr.nlargest(2, 'avg_pred_pic50'),
                scr[scr.in_AD].nlargest(2, 'avg_pred_pic50'),
                scr[scr.in_AD].sample(1, random_state=RANDOM_SEED),
            ]).drop_duplicates(subset=['identifier', 'canonical_smiles']).head(5)
            print('spot-check: rebuilding graph + re-predicting %d saved rows' % len(pick))
            spot_rows = []
            for _, r in pick.iterrows():
                g = mol_to_graph(r['canonical_smiles'], 0.0)
                d = rdkit_desc(r['canonical_smiles'])
                base = 0.0 if d is None else float(baseline_model.predict(np.asarray([d], dtype=float))[0])
                vals = {}
                for kind in ['gcn', 'gin']:
                    model, dev = MODELS[kind]
                    if g is None:
                        vals[kind] = float('nan')
                        continue
                    gg = g.to(dev)
                    batch = torch.zeros(gg.num_nodes, dtype=torch.long, device=dev)
                    with torch.no_grad():
                        ps = float(model(gg.x, gg.edge_index, batch).item())
                    vals[kind] = float(np.clip(scaler.inverse_transform([[ps]])[0, 0] + base, 0.0, 12.0))
                for kind in ['gcn', 'gin']:
                    if kind in CAL:
                        vals[kind] = float(vals[kind] * CAL[kind][0] + CAL[kind][1])
                spot_rows.append(dict(identifier=r['identifier'], smiles=str(r['canonical_smiles'])[:40],
                                      saved_gcn=float(r['pred_gcn_pic50']), recomputed_gcn=vals['gcn'],
                                      saved_gin=float(r['pred_gin_pic50']), recomputed_gin=vals['gin'],
                                      max_abs_diff=max(abs(vals['gcn'] - float(r['pred_gcn_pic50'])),
                                                       abs(vals['gin'] - float(r['pred_gin_pic50'])))))
            spot = pd.DataFrame(spot_rows)
            print(spot.round(3).to_string(index=False))
            chk['screening_recompute_match'] = bool(spot.max_abs_diff.max() < 0.05)  # device/graph rebuild tolerance
            AUDIT['checks'].update(chk)
            AUDIT['screening'] = dict(rows=int(len(scr)), in_AD=int(scr.in_AD.sum()),
                                      in_AD_strict=int(scr.in_AD_strict.sum()),
                                      active_supported=int(scr.active_supported.sum()),
                                      ad_pass=int(scr.ad_pass.sum()),
                                      tiers={str(k): int(v) for k, v in scr.tier.value_counts().items()},
                                      spotcheck=json.loads(spot.round(6).to_json(orient='records')))
        else:
            print('[skip] screening_full_regression.csv not found')

        AUDIT['checks']['graph_counts_match_split'] = bool(len(train_graphs) == len(tr)
                                                           and len(val_graphs) == len(va)
                                                           and len(test_graphs) == len(te))
        AUDIT['checks']['split_464_52_129'] = bool(len(tr) == 464 and len(va) == 52 and len(te) == 129)
        AUDIT['checks']['full_table_rows_645'] = bool(len(audit) == 645)
        AUDIT['checks']['all_predictions_finite'] = bool(np.isfinite(audit[['pIC50_gcn', 'pIC50_gin']].values).all())
        AUDIT['checks']['model_weights_finite'] = bool(all(AUDIT['models'][k]['nonfinite_weights'] == 0
                                                           for k in ['gcn', 'gin']))
        AUDIT['checks']['recompute_matches_saved'] = bool(max_saved_diff < 1e-6)
        try:
            _mc = pd.read_csv('model_comparison.csv')
            AUDIT['model_comparison'] = _mc.to_dict('records')
            AUDIT['checks']['model_comparison_rows'] = bool(len(_mc) >= 4)
            AUDIT['checks']['primary_column_present'] = bool('pred_primary_pic50' in scr.columns)
            AUDIT['checks']['chemberta_column_present'] = bool('pred_chemberta_pic50' in scr.columns)
            AUDIT['checks']['diagnostic_top100_ad_file'] = bool(os.path.exists('diagnostic_top100_AD.csv'))
            AUDIT['checks']['diagnostic_top100_raw_file'] = bool(os.path.exists('diagnostic_top100_raw.csv'))
            _pm = scr['pred_primary_pic50']
            AUDIT['checks']['primary_predictions_finite'] = bool(np.isfinite(_pm.dropna().values).all())
            AUDIT['checks']['diagnostic_top100_ad_rows'] = bool(len(pd.read_csv('diagnostic_top100_AD.csv')) >= 1)
            _sh = pd.read_csv('shortlist_low_confidence.csv')
            AUDIT['checks']['shortlist_file'] = bool(os.path.exists('shortlist_low_confidence.csv'))
            AUDIT['checks']['shortlist_rows_le_max'] = bool(len(_sh) <= SHORTLIST_MAX)
            AUDIT['checks']['shortlist_parent_unique'] = bool(len(_sh) == 0 or _sh.parent_id.nunique() == len(_sh))
            AUDIT['checks']['shortlist_has_confidence'] = bool(all(c in _sh.columns for c in ['conf_low_pic50', 'conf_high_pic50', 'active_margin']))
        except Exception as _e:
            print('[audit] extra-model audit skipped:', _e)
            AUDIT['checks']['model_comparison_rows'] = False


        print('')
        print('=' * 82)
        print('AUDIT SUMMARY (PASS/FAIL)')
        print('=' * 82)
        for k, v in AUDIT['checks'].items():
            print('%-32s %s' % (k, 'PASS' if v else 'FAIL'))
        print('')
        print('INTERPRETATION: R2/RMSE/MAE above are the honest test errors (test scored once).')
        print('Residual bias near 0 = no systematic over/under-prediction; cal_slope near 1 = calibrated.')
        print('Library predictions are extrapolation for most rows — use ad_pass (strict AD + active support + margin), not in_AD alone.')
        json.dump(AUDIT, open('dl_audit.json', 'w'), indent=2)
        for f in ['dl_full_predictions.csv', 'dl_training_history.csv', 'dl_audit.json']:
            print('  [out] %s (%.2f MB)' % (f, os.path.getsize(f) / 1e6))
        print('[done] full DL input/output audit saved. Every row is in dl_full_predictions.csv.')

        # ===== CELL 23 (code) =====
        # 19 — Robust evaluation: repeated scaffold CV, enrichment, bootstrap stability
        import json
        import numpy as np
        import pandas as pd
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from sklearn.linear_model import Ridge

        def _eval_metrics(y, p):
            y = np.asarray(y, dtype=float).ravel()
            p = np.asarray(p, dtype=float).ravel()
            ok = np.isfinite(y) & np.isfinite(p)
            y, p = y[ok], p[ok]
            if len(y) < 2:
                return dict(n=int(len(y)), r2=np.nan, rmse=np.nan, mae=np.nan)
            denom = float(((y - y.mean()) ** 2).sum())
            return dict(n=int(len(y)),
                        r2=float(1.0 - ((y - p) ** 2).sum() / max(denom, 1e-12)),
                        rmse=float(np.sqrt(((y - p) ** 2).mean())),
                        mae=float(np.abs(y - p).mean()))

        def _active(y):
            return (np.asarray(y, dtype=float).ravel() >= T1_CUT).astype(int)

        def _precision_at_10(y, score):
            y = np.asarray(y, dtype=int).ravel()
            score = np.asarray(score, dtype=float).ravel()
            k = min(10, len(y))
            if k <= 0:
                return np.nan
            idx = np.argsort(-score)[:k]
            return float(y[idx].sum() / k)

        def _ef1(y, score):
            y = np.asarray(y, dtype=int).ravel()
            score = np.asarray(score, dtype=float).ravel()
            n = len(y)
            npos = int(y.sum())
            k = max(1, int(np.ceil(0.01 * n)))
            if npos == 0:
                return np.nan
            idx = np.argsort(-score)[:k]
            return float((y[idx].sum() / k) / (npos / n))

        def _bedroc(y, score, alpha=20.0):
            y = np.asarray(y, dtype=int).ravel()
            score = np.asarray(score, dtype=float).ravel()
            n = len(y)
            npos = int(y.sum())
            if npos == 0 or npos == n:
                return np.nan
            order = np.argsort(-score)
            yy = y[order]
            ranks = np.where(yy == 1)[0] + 1
            rie = float(np.exp(-alpha * ranks / n).mean())
            rie_max = float(np.exp(-alpha * np.arange(1, npos + 1) / n).mean())
            return float(rie / rie_max) if rie_max > 0 else np.nan

        def _murcko(smi):
            try:
                m = Chem.MolFromSmiles(str(smi))
                if m is None:
                    return ''
                s = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
                return s if s else Chem.MolToSmiles(m, canonical=True)
            except Exception:
                return ''

        def _make_scaffold_split(groups, rng, test_frac=0.20, val_frac=0.10):
            keys = list(groups.keys())
            rng.shuffle(keys)
            total = sum(len(groups[k]) for k in keys)
            n_test = max(1, int(round(test_frac * total)))
            n_val = max(1, int(round(val_frac * total)))
            test, val = [], []
            for k in keys:
                g = groups[k]
                if len(test) < n_test:
                    test.extend(g)
                elif len(val) < n_val:
                    val.extend(g)
            test, val = set(test), set(val)
            train = [i for i in range(total) if i not in test and i not in val]
            if not train:
                raise RuntimeError('scaffold split left no training rows')
            if not val:
                val.add(train.pop())
            if not test:
                test.add(train.pop())
            return (np.asarray(train, dtype=int),
                    np.asarray(sorted(val), dtype=int),
                    np.asarray(sorted(test), dtype=int))

        # ---- repeated scaffold CV on all 645 compounds (fast, deterministic baselines) ----
        _smiles_col = next((c for c in ['canonical_smiles', 'smiles', 'smiles_canonical'] if c in df.columns), None)
        if _smiles_col is None:
            raise RuntimeError('no SMILES column found for scaffold CV')
        _smiles_all = df[_smiles_col].astype(str).tolist()
        _y_all = np.asarray(df['pIC50'].values, dtype=float) if 'pIC50' in df.columns else np.asarray(raw_targets, dtype=float)
        _D_all = np.asarray(D, dtype=float)
        _X_all = np.asarray(X, dtype=float)
        _scaf_all = np.asarray([_murcko(s) for s in _smiles_all], dtype=object)
        _groups = {}
        for _i, _s in enumerate(_scaf_all):
            _groups.setdefault(_s, []).append(_i)

        _feature_sets = [('Ridge(6 descriptors)', _D_all), ('Ridge(Morgan r3/512)', _X_all)]
        _cv_rows = []
        for _rep in range(5):
            _rng = np.random.default_rng(RANDOM_SEED + _rep)
            _tr, _va, _te = _make_scaffold_split(_groups, _rng)
            for _name, _mat in _feature_sets:
                _model = Ridge(alpha=1.0).fit(_mat[_tr], _y_all[_tr])
                _pva = _model.predict(_mat[_va])
                _pte = _model.predict(_mat[_te])
                _mv = _eval_metrics(_y_all[_va], _pva)
                _mt = _eval_metrics(_y_all[_te], _pte)
                _act = _active(_y_all[_te])
                _cv_rows.append(dict(
                    repeat=_rep + 1, model=_name,
                    n_train=int(len(_tr)), n_val=int(len(_va)), n_test=int(len(_te)),
                    val_r2=_mv['r2'], val_rmse=_mv['rmse'], val_mae=_mv['mae'],
                    test_r2=_mt['r2'], test_rmse=_mt['rmse'], test_mae=_mt['mae'],
                    precision_at_10=_precision_at_10(_act, _pte),
                    ef1=_ef1(_act, _pte), bedroc=_bedroc(_act, _pte)))
        CVDF = pd.DataFrame(_cv_rows)
        CVDF.to_csv('repeated_scaffold_cv.csv', index=False)

        # ---- official test metrics + enrichment + bootstrap intervals ----
        def _bootstrap_ci(y, p, B=1000, seed=RANDOM_SEED + 101):
            y = np.asarray(y, dtype=float).ravel()
            p = np.asarray(p, dtype=float).ravel()
            rng = np.random.default_rng(seed)
            vals = []
            n = len(y)
            for _ in range(B):
                idx = rng.integers(0, n, n)
                vals.append(_eval_metrics(y[idx], p[idx]))
            out = {}
            for key in ['r2', 'rmse', 'mae']:
                arr = np.asarray([v[key] for v in vals], dtype=float)
                out[key + '_lo'] = float(np.nanquantile(arr, 0.025))
                out[key + '_hi'] = float(np.nanquantile(arr, 0.975))
            return out

        _y_test = np.asarray(y_test_raw, dtype=float).ravel()
        _act_test = _active(_y_test)
        _official_rows = []
        for _k in ORDER:
            _p = np.asarray(TEST[_k], dtype=float).ravel() * CAL[_k][0] + CAL[_k][1]
            _m = _eval_metrics(_y_test, _p)
            _official_rows.append(dict(
                model=_k,
                role='sensitivity' if _k in ('hfgin', 'cb') else 'primary_candidate',
                **_m,
                precision_at_10=_precision_at_10(_act_test, _p),
                ef1=_ef1(_act_test, _p),
                bedroc=_bedroc(_act_test, _p),
                **_bootstrap_ci(_y_test, _p)))
        _p_ens = np.asarray(ENS_TEST, dtype=float).ravel()
        _m_ens = _eval_metrics(_y_test, _p_ens)
        _official_rows.append(dict(
            model='simple_ensemble', role='reported_ensemble',
            **_m_ens,
            precision_at_10=_precision_at_10(_act_test, _p_ens),
            ef1=_ef1(_act_test, _p_ens),
            bedroc=_bedroc(_act_test, _p_ens),
            **_bootstrap_ci(_y_test, _p_ens)))
        OFFICIAL = pd.DataFrame(_official_rows)
        OFFICIAL.to_csv('official_test_enrichment.csv', index=False)

        # ---- validation-only bootstrap model-selection stability ----
        def _bootstrap_selection(B=2000, seed=RANDOM_SEED + 202):
            y = np.asarray(y_val_raw, dtype=float).ravel()
            rng = np.random.default_rng(seed)
            keys = list(ORDER)
            wins = {k: 0 for k in keys}
            r2s = {k: [] for k in keys}
            n = len(y)
            for _ in range(B):
                idx = rng.integers(0, n, n)
                yy = y[idx]
                if float(np.std(yy)) < 1e-9:
                    continue
                best = None
                for k in keys:
                    pp = (np.asarray(VAL[k], dtype=float).ravel() * CAL[k][0] + CAL[k][1])[idx]
                    r2 = _eval_metrics(yy, pp)['r2']
                    r2s[k].append(r2)
                    if best is None or r2 > best[1]:
                        best = (k, r2)
                wins[best[0]] += 1
            total = sum(wins.values())
            rows = []
            for k in keys:
                arr = np.asarray(r2s[k], dtype=float)
                rows.append(dict(
                    model=k,
                    role='sensitivity' if k in ('hfgin', 'cb') else 'primary_candidate',
                    wins=int(wins[k]),
                    win_rate=float(wins[k] / total) if total else 0.0,
                    mean_val_r2=float(np.nanmean(arr)) if len(arr) else np.nan,
                    val_r2_lo=float(np.nanquantile(arr, 0.025)) if len(arr) else np.nan,
                    val_r2_hi=float(np.nanquantile(arr, 0.975)) if len(arr) else np.nan))
            return pd.DataFrame(rows)

        SELDF = _bootstrap_selection()
        SELDF.to_csv('model_selection_stability.csv', index=False)

        json.dump(dict(
            active_definition='pIC50 >= T1_CUT',
            t1_cut=float(T1_CUT),
            repeated_scaffold_cv=dict(n_repeats=5, rows=CVDF.to_dict(orient='records')),
            official_test=OFFICIAL.to_dict(orient='records'),
            bootstrap_selection=SELDF.to_dict(orient='records')),
            open('robust_evaluation.json', 'w'), indent=2)

        print('=== Repeated scaffold CV (all 645; mean over 5 repeats) ===')
        print(CVDF.groupby('model')[['test_r2', 'test_rmse', 'test_mae', 'precision_at_10', 'ef1', 'bedroc']]
              .agg(['mean', 'std']).round(3).to_string())
        print('\n=== Official test: regression + enrichment ===')
        print(OFFICIAL[['model', 'role', 'r2', 'rmse', 'mae', 'precision_at_10', 'ef1', 'bedroc']]
              .round(3).to_string(index=False))
        print('\n=== Validation-only bootstrap model selection ===')
        print(SELDF[['model', 'role', 'win_rate', 'mean_val_r2', 'val_r2_lo', 'val_r2_hi']]
              .round(3).to_string(index=False))
        print('\noutputs: repeated_scaffold_cv.csv | official_test_enrichment.csv | model_selection_stability.csv | robust_evaluation.json')

    finally:
        if _ACTIVE_RUN_DIR is not None:
            _collect_outputs(_ACTIVE_RUN_DIR)
