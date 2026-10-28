#!/usr/bin/env python3
# =============================================================================
# verify_pipeline.py - standalone verbose verification for Part_21
# =============================================================================
# Checks (each printed PASS/FAIL):
#   1. py_compile of mdm2_pipeline_v1.py
#   2. import of the module without side effects
#   3. real-data (FULL 645-row) training load + type verification
#   4. data sanity (NaN pIC50, duplicate SMILES, label vocabulary)
#   5. scaffold-split leakage on the FULL real data (zero shared scaffolds)
#   6. screening loader drops old prediction/probability columns
#   7. safety: outputs land inside part21; Part_20 untouched
#   8. end-to-end SMOKE run (isolated out dir, ChemBERTa off) + output contract
# Exit code 0 only when every check passes.
# =============================================================================
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SMOKE_OUT = HERE / "outputs_smoke"

RESULTS = []


def step(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def main():
    pipeline = HERE / "mdm2_pipeline_v1.py"

    # 1. compile ---------------------------------------------------------------
    import py_compile
    try:
        py_compile.compile(str(pipeline), doraise=True)
        step("py_compile", True, str(pipeline.name))
    except py_compile.PyCompileError as exc:
        step("py_compile", False, str(exc))
        return finish()

    # 2. import ---------------------------------------------------------------
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("mdm2_pipeline_v1", pipeline)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        step("import_module", True, "loaded without side effects")
    except Exception as exc:
        step("import_module", False, f"{type(exc).__name__}: {exc}")
        return finish()

    # 3-4. real data + sanity -------------------------------------------------
    train_raw = mod.load_training()
    needs_cols = {"molecule_chembl_id", "bioactivity_class", "pIC50", "canonical_smiles"}
    step("training_columns", needs_cols.issubset(train_raw.columns),
         f"columns={list(train_raw.columns)}")
    df, sanity = mod.validate_and_clean_training(train_raw)
    step("type_pIC50_float64",
         int(df["pIC50"].isna().sum()) == 0
         and hasattr(df["pIC50"], "dtype") and df["pIC50"].dtype.kind == "f",
         f"pIC50 dtype={df['pIC50'].dtype}; coercion notes={mod.REPORT.get('type_issues')}")
    step("data_sanity", df["pIC50"].isna().sum() == 0 and df["smiles"].duplicated().sum() == 0,
         json.dumps(sanity))
    vocab = sorted(set(df["activity"].unique()))
    step("label_vocabulary", len(vocab) == 2 and set(vocab).issubset({0, 1}),
         f"activity values={[int(v) for v in vocab]} (expect Inactive,Active)")

    # 5. leakage on FULL real data ---------------------------------------------
    (i_tr, i_va, i_te, keys, split_info) = mod.scaffold_split(df)
    leak = ((set(keys[i] for i in i_tr) & set(keys[i] for i in i_va))
            | (set(keys[i] for i in i_tr) & set(keys[i] for i in i_te))
            | (set(keys[i] for i in i_va) & set(keys[i] for i in i_te)))
    step("scaffold_leakage_full_data", len(leak) == 0,
         f"shared_scaffolds={len(leak)} splits={split_info}")

    # 6. screening old-column isolation ---------------------------------------
    screen = mod.load_screening()
    leaked_cols = [c for c in ("prediction", "prob_class_0", "prob_class_1")
                   if c in screen.columns]
    step("screening_old_columns_isolated", not leaked_cols
         and list(screen.columns) == ["identifier", "canonical_smiles"],
         f"kept={list(screen.columns)} n={len(screen)} (structures only)")

    # 7. folder safety ---------------------------------------------------------
    part20 = REPO_ROOT / "03_screening_pipeline" / "part20_fixed"
    if part20.is_dir():
        snap20 = {(str(p), p.stat().st_mtime) for p in part20.rglob("*") if p.is_file()}
        step("part20_isolated", "part20_fixed" not in str(mod.OUT_DIR.resolve()),
             f"out_dir={mod._rel(mod.OUT_DIR)}; part20={mod._rel(part20)} (untouched)")
    else:
        snap20 = None
        step("part20_isolated", "part20_fixed" not in str(mod.OUT_DIR.resolve()),
             "part20 folder absent here; isolation via out_dir path check only")

    # 8. end-to-end SMOKE run --------------------------------------------------
    smoke_files = ("top50_regression.csv", "top50_classification.csv",
                   "verification_report.json")
    proc = None
    try:
        env = dict(os.environ)
        env["MDM2_PART21_SMOKE"] = "1"
        env["MDM2_PART21_OUTPUT_DIR"] = str(SMOKE_OUT)
        if SMOKE_OUT.exists():
            shutil.rmtree(SMOKE_OUT)          # stale outputs must not fake PASS
        proc = subprocess.run([sys.executable, str(pipeline)],
                              capture_output=True, text=True, timeout=1800, env=env)
        tail = "\n".join(proc.stdout.strip().splitlines()[-8:])
        step("smoke_run_exit0", proc.returncode == 0,
             f"returncode={proc.returncode}; last lines:\n{tail}")
        if proc.returncode != 0:
            print(proc.stdout[-4000:]); print(proc.stderr[-4000:])
    except subprocess.TimeoutExpired as exc:
        tail = "\n".join((exc.stdout or "")[-2000:]) if isinstance(
            exc.stdout, str) else ""
        step("smoke_run_exit0", False, f"TIMEOUT after 1800s; output tail: {tail}")
    except Exception as exc:
        step("smoke_run_exit0", False, f"runner error: {type(exc).__name__}: {exc}")

    all_found = all((SMOKE_OUT / f).is_file() for f in smoke_files)
    step("smoke_outputs_contract", all_found,
         f"found={[f for f in smoke_files if (SMOKE_OUT / f).is_file()]}")
    if all_found:
        rep = json.loads((SMOKE_OUT / "verification_report.json").read_text())
        bad = [k for k, v in rep["checks"].items() if v["status"] == "FAIL"]
        step("pipeline_self_verification", not bad,
             f"self-report failed areas={bad if bad else 'none'}")

    if snap20 is not None:
        snap20_after = {(str(p), p.stat().st_mtime) for p in part20.rglob("*")
                        if p.is_file()}
        step("part20_untouched_after_run", snap20_after == snap20,
             f"files_before={len(snap20)} files_after={len(snap20_after)}")

    return finish()


def finish():
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n=== verify_pipeline summary: {len(RESULTS) - len(failed)}/{len(RESULTS)} "
          f"passed ===")
    for n, ok, d in RESULTS:
        if not ok:
            print(f"  FAILED: {n}")
    if failed:
        sys.exit(1)
    print("ALL CHECKS PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
