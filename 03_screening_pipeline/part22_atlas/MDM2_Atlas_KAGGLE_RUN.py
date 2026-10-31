#!/usr/bin/env python3
"""
MDM2 Atlas Part 22 — one-command Kaggle runner.

What it does
------------
1. Downloads the current flat pipeline from GitHub:
      03_screening_pipeline/part22_atlas/MDM2_Atlas_KAGGLE_FLAT.py
2. Downloads the three input files from GitHub:
      00_data/training/ro5_properties_filtered.csv
      00_data/screening/screening_results.csv
      00_data/config/cuts.json
3. Installs missing packages (torch_geometric, xgboost, lightgbm, catboost).
4. Runs the full pipeline with SMOKE=0 and DL_EPOCHS=60.
5. Prints every stage and writes:
      <workdir>/atlas_run.log
      <workdir>/atlas_outputs.zip

Kaggle one-cell command
-----------------------
!wget -q https://raw.githubusercontent.com/Techbjd/new_method/main/03_screening_pipeline/part22_atlas/MDM2_Atlas_KAGGLE_RUN.py -O /kaggle/working/run_atlas.py
!python /kaggle/working/run_atlas.py

Optional environment overrides
------------------------------
SMOKE=1                 quick smoke test instead of the full run
DL_EPOCHS=30            shorter DL training
MDM2_ATLAS_WORKDIR=...  output/work directory (default: /kaggle/working)
MDM2_ATLAS_PIPELINE=... use a local flat-pipeline file instead of downloading
"""

from __future__ import annotations

import glob
import importlib
import os
import runpy
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request

REPO_RAW = "https://raw.githubusercontent.com/Techbjd/new_method/main"
PIPELINE_REL = "03_screening_pipeline/part22_atlas/MDM2_Atlas_KAGGLE_FLAT.py"
DATA_FILES = [
    "00_data/training/ro5_properties_filtered.csv",
    "00_data/screening/screening_results.csv",
    "00_data/config/cuts.json",
]


class Tee:
    """Write to several streams at once (console + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _default_workdir() -> str:
    return "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()


def _download(url: str, dst: str, timeout: int = 300) -> None:
    """Download a URL to dst with a browser-like User-Agent."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        with open(dst, "wb") as handle:
            shutil.copyfileobj(response, handle)


def _ensure_packages() -> None:
    packages = [
        ("torch_geometric", "torch_geometric"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("catboost", "catboost"),
    ]
    for package, module in packages:
        try:
            __import__(module)
            print("OK:", package)
        except Exception:
            print("installing:", package)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", package],
                check=False,
            )

    importlib.invalidate_caches()

    for package in ("torch_geometric", "xgboost", "lightgbm", "catboost"):
        try:
            __import__(package)
            print("READY:", package)
        except Exception as exc:
            print("MISSING:", package, exc)


def _format_metrics(values: dict) -> dict:
    clean = {}
    for key, value in values.items():
        if isinstance(value, (int, float)):
            clean[key] = round(float(value), 4)
    return clean


def main() -> None:
    workdir = os.environ.get("MDM2_ATLAS_WORKDIR", _default_workdir())
    os.makedirs(workdir, exist_ok=True)

    out_dir = os.path.join(workdir, "atlas_outputs")
    log_path = os.path.join(workdir, "atlas_run.log")
    os.makedirs(out_dir, exist_ok=True)

    # Full run by default. Override with env vars only for a deliberate smoke test.
    os.environ["SMOKE"] = os.environ.get("SMOKE", "0")
    os.environ["SCAN_FOLDS"] = os.environ.get("SCAN_FOLDS", "5")
    os.environ["DL_EPOCHS"] = os.environ.get("DL_EPOCHS", "60")
    os.environ["MDM2_ATLAS_INPUT_DIR"] = workdir
    os.environ["MDM2_ATLAS_OUTPUT_DIR"] = out_dir

    # Optional heavy models stay off. They are load-checks, not training.
    os.environ["USE_DEEPCHEM"] = os.environ.get("USE_DEEPCHEM", "0")
    os.environ["USE_CHEMPROP"] = os.environ.get("USE_CHEMPROP", "0")
    os.environ["USE_CHEMBERTA"] = os.environ.get("USE_CHEMBERTA", "0")

    log_file = open(log_path, "w", encoding="utf-8")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(old_stdout, log_file)
    sys.stderr = Tee(old_stderr, log_file)

    namespace = None
    try:
        print("=== MDM2 ATLAS FULL RUN ===")
        print("start:", time.strftime("%Y-%m-%d %H:%M:%S"))
        print("workdir:", workdir)

        # 1) Data
        for relative in DATA_FILES:
            destination = os.path.join(workdir, relative)
            if not os.path.exists(destination) or os.path.getsize(destination) == 0:
                print("downloading data:", relative)
                _download(f"{REPO_RAW}/{relative}", destination)
            print("data:", destination, os.path.getsize(destination), "bytes")

        # 2) Packages
        _ensure_packages()

        # 3) Pipeline
        local_pipeline = os.environ.get("MDM2_ATLAS_PIPELINE", "")
        if local_pipeline and os.path.exists(local_pipeline):
            pipeline_path = os.path.abspath(local_pipeline)
            print("pipeline (local):", pipeline_path)
        else:
            pipeline_path = os.path.join(workdir, "MDM2_Atlas_KAGGLE_FLAT.py")
            print("downloading pipeline:", PIPELINE_REL)
            _download(f"{REPO_RAW}/{PIPELINE_REL}", pipeline_path)
            print("pipeline:", pipeline_path, os.path.getsize(pipeline_path), "bytes")

        # 4) Run every pipeline stage in one namespace.
        print("\n=== RUN PIPELINE ===")
        namespace = runpy.run_path(pipeline_path, run_name="__main__")

        # 5) Compact summary
        dl_results = namespace.get("DL_RESULTS", {})
        print("\n=== DL TEST METRICS ===")
        for key, value in sorted(dl_results.items()):
            if isinstance(value, dict) and "test" in value:
                print(key, _format_metrics(value["test"]))

        print("\n=== OUTPUT FILES ===")
        for path in sorted(glob.glob(os.path.join(out_dir, "*"))):
            print(path, os.path.getsize(path), "bytes")

        manifest_path = os.path.join(out_dir, "run_manifest.json")
        if os.path.exists(manifest_path):
            print("\n=== MANIFEST ===")
            with open(manifest_path, encoding="utf-8") as handle:
                print(handle.read())

        print("\n=== DONE ===")
        print("end:", time.strftime("%Y-%m-%d %H:%M:%S"))
        print("log:", log_path)

    except Exception:
        traceback.print_exc()
        raise
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        log_file.close()

    if namespace is not None:
        shutil.copy2(log_path, os.path.join(out_dir, "atlas_run.log"))
        shutil.make_archive(out_dir, "zip", out_dir)
        print("Saved log:", log_path)
        print("Download:", out_dir + ".zip")


if __name__ == "__main__":
    main()
