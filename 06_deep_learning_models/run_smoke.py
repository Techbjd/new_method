#!/usr/bin/env python3
# run_smoke.py — SMOKE-validate all five DL notebooks end-to-end on CPU (DL_SMOKE=1),
# save stream logs, then clear cell outputs so the delivered notebooks are clean.
# Provenance: [GENERATED] validation tooling.
import os
import sys
import time
from pathlib import Path

os.environ["DL_SMOKE"] = "1"
os.environ["DL_FAST"] = "0"
HERE = Path(__file__).resolve().parent

import nbformat
from nbclient import NotebookClient

NOTEBOOKS = [
    "DL_01_Graph_Models_No_Fingerprints.ipynb",
    "DL_02_Morgan_Fingerprint_DL.ipynb",
    "DL_03_MACCS_Keys_DL.ipynb",
    "DL_04_Tier1_Advanced_Architectures.ipynb",
    "DL_05_MolCLR_Pretrain_Finetune.ipynb",
]

only = sys.argv[1] if len(sys.argv) > 1 else None
failures = 0
for fname in NOTEBOOKS:
    if only and only not in fname:
        continue
    nb = nbformat.read(HERE / fname, as_version=4)
    client = NotebookClient(nb, timeout=2400, kernel_name="python3",
                            resources={"metadata": {"path": str(HERE)}},
                            allow_errors=False)
    t0 = time.time()
    err = ""
    try:
        client.execute()
        nbformat.validate(nb)
        status = "OK"
    except Exception as e:
        status = "FAIL"
        err = str(e)[-4000:]
        failures += 1
    dt = time.time() - t0
    print(f"[smoke] {fname}: {status} in {dt / 60:.1f} min", flush=True)
    log = HERE / "smoke_logs" / (Path(fname).stem + ".log")
    log.parent.mkdir(exist_ok=True)
    with open(log, "w") as fh:
        for c in nb.cells:
            if c.cell_type == "code":
                for o in c.get("outputs", []):
                    if o.get("output_type") == "stream":
                        fh.write(o.get("text", ""))
                    elif o.get("output_type") in ("execute_result", "display_data"):
                        txt = o.get("data", {}).get("text/plain")
                        if txt:
                            fh.write(str(txt) + "\n")
        if err:
            fh.write("\n\n==== ERROR ====\n" + err)
    if status == "FAIL":
        print(err, flush=True)
        continue
    if os.environ.get("DL_CLEAR_OUTPUTS", "0") == "1":
        for c in nb.cells:                       # clean deliverable: no stored outputs
            if c.cell_type == "code":
                c["outputs"] = []
                c["execution_count"] = None
        nbformat.write(nb, HERE / fname)
        print(f"[smoke] outputs cleared -> {fname}", flush=True)
    else:
        print(f"[smoke] outputs kept -> {fname}", flush=True)

if failures:
    print(f"[smoke] {failures} notebook(s) FAILED")
    sys.exit(1)
print("[smoke] ALL NOTEBOOKS PASSED")
