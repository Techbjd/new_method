"""Build part22 atlas notebook + flat Kaggle script from a single cell source of truth.

Usage:  python build_notebook.py
Writes: MDM2_Atlas_5Fold_ML_DL_Colab.ipynb, MDM2_Atlas_KAGGLE_FLAT.py
"""
from __future__ import annotations

import json
import os

from atlas_cells_a import CELLS_A
from atlas_cells_b import CELLS_B

CELLS = CELLS_A + CELLS_B

NB_NAME = "MDM2_Atlas_5Fold_ML_DL_Colab.ipynb"
FLAT_NAME = "MDM2_Atlas_KAGGLE_FLAT.py"
BANNER = "=" * 78


def build_notebook() -> str:
    nb_cells = []
    for kind, cid, src in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        else:
            cell["metadata"] = {"id": cid if cid.startswith("md") else f"md_{cid}"}
        nb_cells.append(cell)
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {"provenance": [], "name": NB_NAME},
        },
        "cells": nb_cells,
    }
    return json.dumps(nb, indent=1, ensure_ascii=False)


def build_flat() -> str:
    lines = [
        f'"""{BANNER}',
        "MDM2 Atlas 5-Fold ML+DL pipeline — flat Kaggle script generated from the notebook.",
        "Edit atlas_cells_*.py and re-run build_notebook.py; never edit this file.",
        "Env contract: SMOKE (default 1), MDM2_ATLAS_INPUT_DIR, MDM2_ATLAS_OUTPUT_DIR,",
        "USE_CHEMBERTA, USE_DEEPCHEM, USE_CHEMPROP, SCAN_FOLDS, DL_EPOCHS.",
        f"{BANNER}\"\"\"", "",
        "import warnings, os, sys",
        "warnings.filterwarnings('ignore')",
        "",
    ]
    for kind, cid, src in CELLS:
        if kind != "code":
            continue
        lines.append(f"# {BANNER}")
        lines.append(f"# CELL {cid}")
        lines.append(f"# {BANNER}")
        lines.append(src.rstrip())
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, NB_NAME), "w", encoding="utf-8") as f:
        f.write(build_notebook())
    with open(os.path.join(here, FLAT_NAME), "w", encoding="utf-8") as f:
        f.write(build_flat())
    print(f"built {len(CELLS)} cells -> {NB_NAME}, {FLAT_NAME}")


if __name__ == "__main__":
    main()
