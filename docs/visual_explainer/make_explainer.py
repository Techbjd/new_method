#!/usr/bin/env python3
"""Build a visual, non-expert explainer deck for the MDM2 screening pipeline.

Run from anywhere:
    python docs/visual_explainer/make_explainer.py

Outputs:
    docs/visual_explainer/figures/*.png
    docs/visual_explainer/MDM2_ML_Root_Cause_Explainer.pptx
    docs/visual_explainer/MDM2_ML_Root_Cause_Explainer.odp   (via LibreOffice)
    docs/visual_explainer/MDM2_ML_Root_Cause_Explainer.pdf   (via LibreOffice)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt
from lxml import etree
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
NAVY = "#0B1F3A"
INK = "#16213E"
PAPER = "#F7F5F0"
WHITE = "#FFFFFF"
RED = "#D62828"
ORANGE = "#E76F51"
AMBER = "#F4A261"
GREEN = "#2A9D8F"
GRAY = "#8D99AE"
LIGHT = "#E9E4D8"
MID = "#C9C2B4"
FONT = "Liberation Sans"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": MID,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
})

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
full = pd.read_csv(ROOT / "03_screening_pipeline/part20_fixed/screening_full_regression.csv")
train = pd.read_csv(ROOT / "00_data/training/ro5_properties_filtered.csv")
canonical = pd.read_csv(ROOT / "00_data/screening/screening_results.csv", usecols=["identifier"])
domain = json.loads((ROOT / "03_screening_pipeline/part20_fixed/cache/domain_shift.json").read_text())
run_manifest = json.loads((ROOT / "03_screening_pipeline/part20_fixed/run_manifest.json").read_text())
metrics_ext = json.loads((ROOT / "03_screening_pipeline/part20_fixed/cache/reg_metrics_ext.json").read_text())

N_LIB = len(full)
N_CANON = len(canonical)
N_TOP = 100
top = full.nlargest(N_TOP, "avg_pred_pic50")
top_mean = float(top["maxT_464"].mean())
top_median = float(top["maxT_464"].median())
top_min = float(top["maxT_464"].min())
top_max = float(top["maxT_464"].max())
top_in_ad = int((top["maxT_464"] >= 0.40).sum())
top_in_strict = int((top["maxT_464"] >= 0.52).sum())
lib_median = float(full["maxT_464"].median())
lib_mean = float(full["maxT_464"].mean())
n_ad = int((full["maxT_464"] >= 0.40).sum())
n_strict = int((full["maxT_464"] >= 0.52).sum())
n_ge_03 = int((full["maxT_464"] >= 0.30).sum())
train_loo = domain["train_loo"]
test_dom = domain["test_to_train"]
screen_dom = domain["screen_to_train"]

print(f"Library rows: {N_LIB:,}; canonical: {N_CANON:,}; diff: {N_CANON - N_LIB:,}")
print(f"AD >=0.40: {n_ad:,} ({100*n_ad/N_LIB:.3f}%)")
print(f"AD >=0.52: {n_strict:,} ({100*n_strict/N_LIB:.3f}%)")
print(f"Top100 maxT mean/median/min/max: {top_mean:.3f}/{top_median:.3f}/{top_min:.3f}/{top_max:.3f}")
print(f"Top100 in AD 0.40 / 0.52: {top_in_ad}/{top_in_strict}")

# ---------------------------------------------------------------------------
# Figure 1: similarity to training
# ---------------------------------------------------------------------------
labels = ["Train (leave-one-out)", "Test set", "Screening library", "Top-100 predicted"]
values = [train_loo["mean"], test_dom["mean"], screen_dom["mean"], top_mean]
colors = [GREEN, AMBER, GRAY, RED]
fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=200)
y = np.arange(len(labels))[::-1]
ax.axvspan(0, 0.40, color=RED, alpha=0.06, zorder=0)
ax.axvspan(0.40, 0.52, color=AMBER, alpha=0.10, zorder=0)
ax.axvspan(0.52, 1.0, color=GREEN, alpha=0.06, zorder=0)
bars = ax.barh(y, values, color=colors, height=0.55, zorder=3)
for yy, v in zip(y, values):
    ax.text(v + 0.015, yy, f"{v:.3f}", va="center", ha="left", fontsize=13, fontweight="bold", color=INK)
ax.axvline(0.40, color=ORANGE, ls="--", lw=1.8, zorder=4)
ax.axvline(0.52, color=GREEN, ls="--", lw=1.8, zorder=4)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=12)
ax.set_xlim(0, 0.95)
ax.set_xlabel("Maximum Tanimoto similarity to the 464 training compounds", fontsize=11)
ax.set_title("How close is each compound to the training chemistry?", fontsize=15, fontweight="bold", loc="left", pad=12)
ax.grid(axis="x", color=MID, alpha=0.35, lw=0.7)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "domain_gap.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: screening funnel
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=200)
counts = [N_LIB, n_ad, n_strict]
names = ["All library compounds", "Similarity ≥ 0.40", "Similarity ≥ 0.52 (strict)"]
cols = [GRAY, AMBER, GREEN]
y = np.arange(len(counts))[::-1]
bars = ax.barh(y, counts, color=cols, height=0.55, log=True, zorder=3)
for yy, c in zip(y, counts):
    ax.text(c * 1.25, yy, f"{c:,}  ({100*c/N_LIB:.3f}%)", va="center", ha="left", fontsize=13, fontweight="bold", color=INK)
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=12)
ax.set_xlim(1, N_LIB * 5)
ax.set_xlabel("Number of compounds (log scale)", fontsize=11)
ax.set_title("The strict-domain pool is 96 compounds, not 154k", fontsize=15, fontweight="bold", loc="left", pad=12)
ax.grid(axis="x", color=MID, alpha=0.35, lw=0.7)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "screening_funnel.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: predicted pIC50 vs similarity
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
sample = full.sample(min(22000, len(full)), random_state=42)
fig, ax = plt.subplots(figsize=(8.8, 4.9), dpi=200)
ax.axvspan(0, 0.40, color=RED, alpha=0.05, zorder=0)
ax.axvspan(0.40, 0.52, color=AMBER, alpha=0.09, zorder=0)
ax.axvline(0.40, color=ORANGE, ls="--", lw=1.5, zorder=2)
ax.axvline(0.52, color=GREEN, ls="--", lw=1.5, zorder=2)
ax.axhline(7.0, color=NAVY, ls=":", lw=1.2, zorder=2)
ax.scatter(sample["maxT_464"], sample["avg_pred_pic50"], s=5, color=GRAY, alpha=0.18, linewidths=0, zorder=3, label="Library sample")
ax.scatter(top["maxT_464"], top["avg_pred_pic50"], s=26, color=RED, edgecolor=WHITE, linewidth=0.4, zorder=5, label="Top-100 predicted")
ax.text(0.405, 11.45, "AD 0.40", color=ORANGE, fontsize=9, fontweight="bold")
ax.text(0.525, 11.45, "strict 0.52", color=GREEN, fontsize=9, fontweight="bold")
ax.text(0.02, 7.18, "pIC50 = 7.0", color=NAVY, fontsize=9, fontweight="bold")
ax.set_xlim(0, 0.95)
ax.set_ylim(0, 12.2)
ax.set_xlabel("Maximum Tanimoto similarity to training", fontsize=11)
ax.set_ylabel("Predicted pIC50 (GCN/GIN mean)", fontsize=11)
ax.set_title("The highest predictions sit in the least-similar region", fontsize=15, fontweight="bold", loc="left", pad=12)
ax.legend(loc="upper right", frameon=False, fontsize=9, markerscale=1.4)
ax.grid(color=MID, alpha=0.25, lw=0.6)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "pred_vs_similarity.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 4: validation vs test R2
# ---------------------------------------------------------------------------
# Cached run metrics: GCN/GIN/ensemble; AFP/HF-GIN from reg_metrics_ext.
model_rows = [
    ("Ridge", 0.4855, 0.4104),
    ("GCN", 0.861, 0.649),
    ("GIN", 0.866, 0.671),
    ("Ensemble", 0.871, 0.666),
    ("AFP", 0.7705, 0.4532),
    ("HF-GIN", 0.8608, 0.5836),
]
fig, ax = plt.subplots(figsize=(8.8, 4.7), dpi=200)
x = np.arange(len(model_rows))
w = 0.36
val = [r[1] for r in model_rows]
test = [r[2] for r in model_rows]
ax.bar(x - w/2, val, w, color="#8FB3C9", label="Validation R²", zorder=3)
ax.bar(x + w/2, test, w, color=RED, label="Test R²", zorder=3)
for i, (v, t) in enumerate(zip(val, test)):
    ax.text(i - w/2, v + 0.018, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=INK)
    ax.text(i + w/2, t + 0.018, f"{t:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=INK)
ax.set_xticks(x)
ax.set_xticklabels([r[0] for r in model_rows], fontsize=11)
ax.set_ylim(0, 1.14)
ax.set_ylabel("R²", fontsize=11)
ax.set_title("Validation R² is not deployment performance", fontsize=15, fontweight="bold", loc="left", pad=12)
ax.grid(axis="y", color=MID, alpha=0.3, lw=0.7)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9, loc="upper right")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "val_test_gap.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# PPTX helpers
# ---------------------------------------------------------------------------
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

def rgb(hexstr: str) -> RGBColor:
    return RGBColor.from_string(hexstr.replace("#", "").upper())

def set_bg(slide, color: str):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = rgb(color)

def add_fade(slide):
    """Add a simple fade transition (LibreOffice/PowerPoint may honor it)."""
    sld = slide._element
    for e in sld.findall(qn("p:transition")):
        sld.remove(e)
    trans = etree.SubElement(sld, qn("p:transition"))
    trans.set("spd", "med")
    etree.SubElement(trans, qn("p:fade"))

def textbox(slide, x, y, w, h, text, size=18, bold=False, color=INK,
            align=PP_ALIGN.LEFT, font=FONT, anchor=MSO_ANCHOR.TOP, italic=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = rgb(color)
    r.font.name = font
    tb.shadow.inherit = False
    return tb

def bullets(slide, x, y, w, h, items, size=17, color=INK, space=8, bullet_color=RED):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space)
        p.level = 0
        r1 = p.add_run()
        r1.text = "•  "
        r1.font.size = Pt(size)
        r1.font.bold = True
        r1.font.color.rgb = rgb(bullet_color)
        r1.font.name = FONT
        r2 = p.add_run()
        r2.text = item
        r2.font.size = Pt(size)
        r2.font.color.rgb = rgb(color)
        r2.font.name = FONT
    tb.shadow.inherit = False
    return tb

def card(slide, x, y, w, h, title, body, accent=RED, fill=WHITE, title_size=16, body_size=12.5):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fill)
    shp.line.color.rgb = rgb(accent)
    shp.line.width = Pt(1.2)
    shp.shadow.inherit = False
    shp.adjustments[0] = 0.08
    textbox(slide, x + 0.18, y + 0.14, w - 0.36, 0.35, title, size=title_size, bold=True, color=accent)
    textbox(slide, x + 0.18, y + 0.52, w - 0.36, h - 0.62, body, size=body_size, color=INK)

def add_image_fit(slide, path, x, y, w, h):
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    nw, nh = iw * scale, ih * scale
    left = x + (w - nw) / 2
    top = y + (h - nh) / 2
    slide.shapes.add_picture(str(path), Inches(left), Inches(top), width=Inches(nw), height=Inches(nh))

def header(slide, kicker, title, subtitle=None, dark=False):
    fg = WHITE if dark else NAVY
    muted = "#C9D3E0" if dark else GRAY
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, 13.333, 0.10)
    bar.fill.solid(); bar.fill.fore_color.rgb = rgb(RED); bar.line.fill.background()
    bar.shadow.inherit = False
    textbox(slide, 0.65, 0.28, 12.0, 0.28, kicker.upper(), size=11, bold=True, color=RED)
    textbox(slide, 0.65, 0.55, 12.0, 0.72, title, size=29, bold=True, color=fg)
    if subtitle:
        textbox(slide, 0.67, 1.28, 11.8, 0.42, subtitle, size=14.5, color=muted)

def footer(slide, n, total, dark=False):
    color = "#9FB0C4" if dark else GRAY
    textbox(slide, 0.65, 7.13, 11.0, 0.22,
            "MDM2 pipeline review  ·  645 training compounds / 154,587 screening compounds  ·  data: 2026-10-26",
            size=8.5, color=color)
    textbox(slide, 12.2, 7.13, 0.5, 0.22, f"{n}/{total}", size=8.5, color=color, align=PP_ALIGN.RIGHT)

def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text

TOTAL = 10
# ---------------------------------------------------------------------------
# Slide 1 — title
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(s, NAVY)
add_fade(s)
# decorative thin lines
for x in [0.65, 12.25]:
    ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(1.65), Inches(0.03), Inches(4.4))
    ln.fill.solid(); ln.fill.fore_color.rgb = rgb(RED); ln.line.fill.background()
    ln.shadow.inherit = False
textbox(s, 0.95, 1.25, 11.4, 0.35, "MDM2 SCREENING PIPELINE  ·  ROOT-CAUSE REVIEW", size=12, bold=True, color=AMBER)
textbox(s, 0.95, 1.85, 11.4, 1.65, "Why the model looks confident —\nbut picks compounds outside its range", size=38, bold=True, color=WHITE)
textbox(s, 0.98, 3.75, 10.4, 0.75, "A visual explanation for a non-specialist, built from the code, the saved metrics, and the cached 154,587-row screening table.", size=16, color="#C9D3E0")
textbox(s, 0.98, 5.0, 10.5, 0.9, "645 training compounds  →  154,587 library compounds\n0 of the top 100 are inside the applicability domain.", size=19, bold=True, color=AMBER)
textbox(s, 0.98, 6.55, 10.5, 0.3, "Prepared 2026-10-26  ·  local review  ·  cached run: 154,587 rows (canonical 154,647; 60-row difference)", size=9, color="#8FA3B8")
notes(s, "This deck separates the root cause from the code bugs. The root cause is distribution mismatch plus selection at the extreme. The code bugs make the bias worse, but fixing them alone will not create evidence outside the training chemistry.")

# ---------------------------------------------------------------------------
# Slide 2 — one sentence
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(s, PAPER); add_fade(s)
header(s, "The single point", "The model is solving a different problem than the one you need")
textbox(s, 0.65, 1.72, 12.0, 1.0, "It is not mainly one wrong line of code. It is a mismatch between the chemistry it learned and the chemistry it is asked to rank — made worse by selecting the most extreme predictions.", size=20, color=INK)
card(s, 0.65, 3.1, 3.75, 2.05, "It learned", "645 MDM2 compounds from a limited chemical neighborhood. 204 Murcko scaffolds; top 10 cover 39%.", accent=GREEN, title_size=18)
card(s, 4.80, 3.1, 3.75, 2.05, "It is asked", "Rank 154,587 diverse library compounds. 99.4% are below the 0.40 similarity cut; 99.94% are below the strict 0.52 cut.", accent=ORANGE, title_size=18)
card(s, 8.95, 3.1, 3.75, 2.05, "It does", "Takes the top predicted pIC50 values. Those values are highest where the model has the least evidence.", accent=RED, title_size=18)
textbox(s, 0.65, 5.55, 12.0, 0.9, "A model can output a number everywhere. That does not mean it has evidence everywhere.", size=25, bold=True, color=RED, align=PP_ALIGN.CENTER)
textbox(s, 0.65, 6.35, 12.0, 0.35, "The algorithm is secondary. The data coverage and the selection rule are primary.", size=13, color=GRAY, align=PP_ALIGN.CENTER)
footer(s, 2, TOTAL)
notes(s, "Key line: one model, two different worlds. Training/validation chemistry is analog-rich; the library is mostly outside the training manifold. The model's top predictions are selected precisely from the extrapolation zone.")

# ---------------------------------------------------------------------------
# Slide 3 — what out of range means
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(s, PAPER); add_fade(s)
header(s, "What 'out of range' means", "Similarity collapses outside the training set", "Max Tanimoto similarity to the 464 training compounds. Higher = more evidence; lower = more extrapolation.")
add_image_fit(s, FIG / "domain_gap.png", 0.45, 1.82, 8.1, 4.82)
bullets(s, 8.75, 1.95, 4.0, 4.2, [
    f"Train LOO mean: {train_loo['mean']:.3f}",
    f"Test mean: {test_dom['mean']:.3f}",
    f"Library mean: {screen_dom['mean']:.3f}",
    f"Top-100 mean: {top_mean:.3f}",
    f"Library median: {lib_median:.3f}",
    f"Top-100 max: {top_max:.3f}",
    "Dashed lines: 0.40 AD / 0.52 strict",
], size=15, space=9)
textbox(s, 8.75, 5.55, 4.0, 1.15, "Only 0.57% of the library is inside the 0.40 cut.\nOnly 0.062% is inside the strict 0.52 cut.", size=15, bold=True, color=RED)
textbox(s, 0.65, 6.72, 7.8, 0.22, "Red zone = extrapolation (<0.40)  ·  amber = caution (0.40–0.52)  ·  green = in domain (≥0.52).", size=9.2, color=GRAY)
footer(s, 3, TOTAL)
notes(s, "The test set is not the library. Test mean similarity is 0.608 and 97.7% of test compounds pass 0.40. The library mean is 0.219, and only 886 of 154,587 pass 0.40. The top-100 predicted compounds are at 0.186 mean — even farther out than the library median.")

# ---------------------------------------------------------------------------
# Slide 4 — funnel
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(s, PAPER); add_fade(s)
header(s, "The real search space", "The strict-domain pool is only 96 compounds", "The library is not a 154k-choice buffet. Strict applicability-domain filtering leaves almost nothing to rank.")
add_image_fit(s, FIG / "screening_funnel.png", 0.45, 1.9, 8.1, 4.8)
bullets(s, 8.75, 2.0, 4.0, 4.5, [
    f"{N_LIB:,} total library compounds",
    f"{n_ad:,} pass similarity 0.40 ({100*n_ad/N_LIB:.2f}%)",
    f"{n_strict:,} pass strict 0.52 ({100*n_strict/N_LIB:.3f}%)",
    f"{top_in_ad} of the top 100 pass 0.40",
    f"{top_in_strict} of the top 100 pass 0.52",
], size=14.5, space=10)
textbox(s, 8.75, 5.62, 4.0, 1.0, "If active support and model agreement are also required, the pool may be empty. An empty shortlist is a valid scientific result.", size=13.5, bold=True, color=RED)
footer(s, 4, TOTAL)
notes(s, "This is the most important practical consequence. Strict AD does not produce a 12-compound shortlist from 154k candidates; it produces a pool of 96. If that pool has no active-supported, model-agreeing compounds, the honest answer is 'no reliable hits in this library'.")

# ---------------------------------------------------------------------------
# Slide 5 — smoking gun
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(s, PAPER); add_fade(s)
header(s, "The smoking gun", "Top predictions are the least similar compounds", "Predicted pIC50 versus maximum similarity to training. The red points are the top 100 by prediction.")
add_image_fit(s, FIG / "pred_vs_similarity.png", 0.45, 1.9, 8.35, 4.85)
card(s, 8.95, 1.98, 3.8, 1.55, "Top-100 signal", f"Mean similarity {top_mean:.3f}\nLibrary median {lib_median:.3f}\nAll 100 outside AD 0.40", accent=RED, body_size=13)
card(s, 8.95, 3.70, 3.8, 1.55, "What this means", "Its most extreme predictions occur exactly where evidence is weakest. This is extrapolation, not discovery.", accent=ORANGE, body_size=13)
card(s, 8.95, 5.42, 3.8, 1.20, "Selection bias", "Choosing the maximum of 154k noisy predictions enriches for over-predictions. This is the winner's curse.", accent=NAVY, body_size=12.5)
footer(s, 5, TOTAL)
notes(s, "Top-100 mean maxT is 0.186, median 0.186, max 0.239. Library median is 0.217. So the top predictions are not just outside the domain; they are slightly farther out than a random library compound. The model is not finding hidden analogs; it is selecting the extrapolation tail.")

# ---------------------------------------------------------------------------
# Slide 6 — validation
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(s, PAPER); add_fade(s)
header(s, "Why the numbers looked good", "Validation looked good because it was analog-rich", "The test is scaffold-disjoint — good — but 97.7% of test compounds still pass the 0.40 similarity cut.")
add_image_fit(s, FIG / "val_test_gap.png", 0.45, 1.9, 8.2, 4.8)
bullets(s, 8.85, 2.0, 3.9, 4.3, [
    "42 of 52 validation compounds share a scaffold with training.",
    "30 scaffolds are shared between train and validation.",
    "Train–test and validation–test scaffold overlap is zero.",
    "Val R² ≈ 0.87; test R² ≈ 0.65.",
    "The test is cleaner than validation, but it is not the library.",
], size=13.5, space=9)
textbox(s, 8.85, 5.8, 3.9, 0.8, "Model selection, early stopping, and calibration were all chosen on the analog-rich validation set.", size=12.5, bold=True, color=RED)
footer(s, 6, TOTAL)
notes(s, "The test set is genuinely scaffold-disjoint, so it is not cheating. But scaffold-disjoint does not mean out-of-domain: 97.7% of test compounds still have a training neighbor at Tanimoto >= 0.40. The library does not. That is why test R² 0.65 does not transfer to a 0.22-similarity library.")

# ---------------------------------------------------------------------------
# Slide 7 — causal chain
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(s, PAPER); add_fade(s)
header(s, "The causal chain", "One mismatch, amplified by one selection rule")
# flow boxes
boxes = [
    (0.55, 2.0, 2.25, 1.15, "645 analogs", "small, biased chemical neighborhood", GREEN),
    (3.05, 2.0, 2.25, 1.15, "Analog validation", "42/52 val share a train scaffold", RED),
    (5.55, 2.0, 2.25, 1.15, "Select + calibrate", "choose the best on analogs", RED),
    (8.05, 2.0, 2.25, 1.15, "154k library", "99.4% outside AD 0.40", RED),
    (10.55, 2.0, 2.25, 1.15, "Top-100", "all outside AD; mean 0.186", RED),
]
for x, y, w, h, t, b, c in boxes:
    card(s, x, y, w, h, t, b, accent=c, title_size=15, body_size=11.5)
for x in [2.82, 5.32, 7.82, 10.32]:
    ar = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(2.37), Inches(0.20), Inches(0.38))
    ar.fill.solid(); ar.fill.fore_color.rgb = rgb(GRAY); ar.line.fill.background()
    ar.shadow.inherit = False
textbox(s, 0.65, 3.45, 12.0, 0.65, "Two red links: the validation does not represent the library, and the shortlist selects the extrapolation tail.", size=17, bold=True, color=RED)
# amplifiers
textbox(s, 0.65, 4.35, 12.0, 0.35, "Code-level amplifiers (not the root cause)", size=15, bold=True, color=NAVY)
amps = [
    ("Active evidence", "all 645 actives used, including test; 52 test actives self-match"),
    ("NNLS label", "‘NNLS ensemble’ row is actually the simple mean"),
    ("Tier mismatch", "tier uses 0.40 while the shortlist uses 0.52"),
    ("Stale cache", "154,587 rows vs 154,647 canonical; final shortlist missing"),
]
for i, (t, b) in enumerate(amps):
    card(s, 0.65 + i*3.12, 4.8, 2.95, 1.45, t, b, accent=ORANGE, title_size=13.5, body_size=10.8)
footer(s, 7, TOTAL)
notes(s, "The root cause is the top chain, not the bottom row. The code bugs are real and should be fixed, but even a perfect NNLS label and perfect active evidence would not make a 0.186-similarity compound reliably predictable. The root is the mismatch between the validation/deployment chemistry and the selection rule.")

# ---------------------------------------------------------------------------
# Slide 8 — proper training
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(s, PAPER); add_fade(s)
header(s, "The proper way", "Train for the chemistry you will actually screen", "A protocol that matches the deployment question, not a random benchmark.")
steps = [
    ("1", "Define the domain", "Is the goal analog optimization or diverse-library discovery? They need different data and metrics."),
    ("2", "Split by scaffold", "GroupShuffleSplit / GroupKFold by Murcko scaffold. Never let analogs leak into validation."),
    ("3", "Nested CV", "Select and calibrate inside the training folds. Keep the test set frozen."),
    ("4", "Freeze test", "Compute test predictions once, at the end. No repeated peeking."),
    ("5", "Train-only evidence", "Active support and AD reference must use training data for held-out evaluation."),
    ("6", "Report uncertainty", "AD coverage, calibration slope, prediction intervals, enrichment with bootstrap CIs."),
    ("7", "Prospective assay", "A computational shortlist is a hypothesis. Only a wet-lab assay makes it a hit."),
]
for i, (num, t, b) in enumerate(steps):
    col = i % 4
    row = i // 4
    x = 0.65 + col * 3.12
    y = 1.85 + row * 2.05
    w = 2.9 if i < 4 else 2.9
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(1.72))
    shp.fill.solid(); shp.fill.fore_color.rgb = rgb(WHITE); shp.line.color.rgb = rgb(GREEN); shp.line.width = Pt(1.1)
    shp.shadow.inherit = False
    textbox(s, x + 0.15, y + 0.12, 0.5, 0.4, num, size=20, bold=True, color=GREEN)
    textbox(s, x + 0.55, y + 0.15, w - 0.65, 0.35, t, size=14, bold=True, color=NAVY)
    textbox(s, x + 0.15, y + 0.62, w - 0.3, 1.0, b, size=10.5, color=INK)
# bottom callout spanning last row
textbox(s, 0.65, 5.95, 12.0, 0.7, "If the goal is a diverse library: use a classifier for enrichment + AD, then assay. Do not rank pIC50 outside the domain.", size=18, bold=True, color=RED, align=PP_ALIGN.CENTER)
textbox(s, 0.65, 6.62, 12.0, 0.3, "A lower, honest score is better than a high score that does not transfer.", size=11.5, color=GRAY, align=PP_ALIGN.CENTER)
footer(s, 8, TOTAL)
notes(s, "The key change is not a new model. It is a new evaluation contract: split by scaffold, nest all selection, freeze the test, use train-only evidence, and report AD coverage. If the deployment domain is a diverse library, a pIC50 regressor is the wrong primary tool; a classifier plus AD plus assay is more honest.")

# ---------------------------------------------------------------------------
# Slide 9 — fix order
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(s, PAPER); add_fade(s)
header(s, "What I would fix first", "P0 is not a new model. P0 is an honest evaluation.")
card(s, 0.65, 1.85, 6.05, 2.15, "P0 — before any more claims", "• Scaffold-grouped validation\n• Train-only active evidence\n• Freeze test predictions\n• Fix/relabel NNLS result\n• Align tier with 0.52\n• Rebuild cache with content hashes\n• Rerun the final pipeline", accent=RED, title_size=17, body_size=12.5)
card(s, 6.95, 1.85, 5.75, 2.15, "P1 — robustness", "• Deterministic seeds / generators\n• Parent + salt + stereoisomer collapse\n• Explicit model availability\n• Nested CV for every compared model\n• Bootstrap CIs for enrichment\n• Per-compound uncertainty", accent=ORANGE, title_size=17, body_size=12.5)
card(s, 0.65, 4.25, 6.05, 1.85, "P2 — engineering", "• Split the 2,423-line script into modules\n• Add smoke tests and a pinned environment\n• One config source + schema version\n• Pin data by SHA256, not ‘main’\n• Remove silent exception swallowing", accent=GREEN, title_size=17, body_size=12.5)
card(s, 6.95, 4.25, 5.75, 1.85, "Expected result", "Lower validation/test numbers.\nA smaller — possibly empty — shortlist.\nThat is success, not failure: it tells you what the data can support.", accent=NAVY, title_size=17, body_size=12.5)
textbox(s, 0.65, 6.35, 12.0, 0.35, "Do not start by tuning the model. Start by making the evaluation match the screening question.", size=13, bold=True, color=RED, align=PP_ALIGN.CENTER)
footer(s, 9, TOTAL)
notes(s, "P0 is about truth, not performance. If the scaffold-grouped validation drops R² to 0.3, that is the honest number for new chemotypes. If the strict shortlist is empty, that is a finding: the library does not contain reliable analogs. P1/P2 make the result reproducible and maintainable.")

# ---------------------------------------------------------------------------
# Slide 10 — bottom line
# ---------------------------------------------------------------------------
s = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(s, NAVY); add_fade(s)
textbox(s, 0.85, 1.25, 11.7, 0.35, "BOTTOM LINE", size=12, bold=True, color=AMBER)
textbox(s, 0.85, 1.8, 11.6, 1.6, "A model is not a microscope.\nIt cannot see chemistry it was never trained on.", size=33, bold=True, color=WHITE)
textbox(s, 0.88, 3.65, 11.2, 0.85, "Your pipeline is a potentially useful analog-ranking tool, once validation is fixed. It is not yet a discovery engine for a diverse library.", size=19, color="#C9D3E0")
textbox(s, 0.88, 4.75, 11.2, 0.8, "The correct answer may be: “No reliable hits in this library.”", size=26, bold=True, color=AMBER)
textbox(s, 0.88, 5.85, 11.2, 0.55, "Fix the validation contract first. Then the model has a chance to be believed.", size=15, color="#C9D3E0")
footer(s, 10, TOTAL, dark=True)
notes(s, "Close with the distinction between a ranking tool and a discovery engine. The current shortlist is a hypothesis generator, not evidence. The most valuable next step is an honest, scaffold-grouped evaluation and a prospective assay, not a more complex model.")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
pptx_path = OUT / "MDM2_ML_Root_Cause_Explainer.pptx"
prs.save(pptx_path)
print("Saved", pptx_path)
