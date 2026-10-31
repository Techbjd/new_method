"""Pretrained Transformer (HuggingFace) — ChemBERTa / MoLFormer fine-tuning.

[ChemBERTa: Chithrananda et al., "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction", arXiv:2010.09891 (2020)]
  - HF: https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1
  - Code: github.com/seyonechithrananda/bert-loves-chemistry
[MoLFormer-XL: Ross et al., Nature Machine Intelligence 2022, 4:1256–1264]
  - HF: https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct

GUARDED: skipped with a clear message if `transformers` is missing.
Project rule: these pretrained weights stay SENSITIVITY-ONLY unless a
validation-only protocol re-trains them; never reuse checkpoints that saw the
645 labels (see docs/ERRORS_AND_FIXES.md).

Run:  python models/pretrained_hf_model.py --smoke
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn as nn

import dl_utils

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_OK = True
except Exception:
    TRANSFORMERS_OK = False

MODELS = {
    "chemberta": "seyonec/ChemBERTa-zinc-base-v1",
    "chemberta-mlm": "seyonec/ChemBERTa-77M-MLM",
    "molformer": "ibm-research/MoLFormer-XL-both-10pct",
}


class HFForSequence(nn.Module):
    def __init__(self, base_model, hidden: int, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.base = base_model
        out_dim = num_classes
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(64, out_dim))

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0] if hasattr(out, "last_hidden_state") else out[0][:, 0]
        return self.head(cls)


def tokenize_smiles(smiles_list, tokenizer, max_len=128):
    enc = tokenizer(list(map(str, smiles_list)), truncation=True, padding="max_length",
                    max_length=max_len, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def train_hf(model, train_dl, val_dl, task="cls", epochs=30, lr=2e-5, device=dl_utils.DEVICE):
    from torch.nn.functional import cross_entropy, mse_loss
    from sklearn.metrics import roc_auc_score, mean_squared_error
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best, state, bad = (-np.inf if task == "cls" else np.inf), None, 0
    for ep in range(epochs):
        model.train()
        for ids, mask, yb in train_dl:
            ids, mask, yb = ids.to(device), mask.to(device), yb.to(device)
            opt.zero_grad()
            out = model(ids, mask)
            loss = cross_entropy(out, yb.long()) if task == "cls" else mse_loss(out.squeeze(-1), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        outs, ys = [], []
        with torch.no_grad():
            for ids, mask, yb in val_dl:
                outs.append(model(ids.to(device), mask.to(device)).cpu())
                ys.append(yb)
        outs = torch.cat(outs); ys = torch.cat(ys)
        if task == "cls":
            val = roc_auc_score(ys.int(), torch.softmax(outs, dim=1)[:, 1].numpy())
            best_cond = val > best
        else:
            val = float(np.sqrt(mean_squared_error(ys.numpy(), outs.squeeze(-1).numpy())))
            best_cond = val < best
        if best_cond:
            best, bad, state = val, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                break
    if state:
        model.load_state_dict(state)
    return model, best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="chemberta", choices=list(MODELS))
    ap.add_argument("--task", default="cls", choices=["cls", "reg"])
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_known_args(argv)[0]

    if not TRANSFORMERS_OK:
        print(f"[skipped] pretrained_hf_model: transformers not installed "
              f"(pip install transformers; model card {MODELS[args.model]})")
        return

    from torch.utils.data import DataLoader, TensorDataset
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "..", "01_traditional_ml", "models"))
    import featurize as fz
    df = fz.load_training_table()
    smiles = fz.smiles_column(df).tolist()
    y_cls = fz.label_encode(df)
    y_reg = df["pIC50"].astype(float).to_numpy()
    tr, va, te = fz.scaffold_split(smiles, y_reg, y_cls)
    if args.smoke:
        tr, va, te = tr[:40], va[:20], te[:20]

    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model])
    base = AutoModel.from_pretrained(MODELS[args.model])
    ids_tr, mask_tr = tokenize_smiles(list(np.array(smiles)[tr]), tokenizer)
    ids_va, mask_va = tokenize_smiles(list(np.array(smiles)[va]), tokenizer)
    ytr = torch.tensor(y_cls[tr] if args.task == "cls" else y_reg[tr], dtype=torch.float)
    yva = torch.tensor(y_cls[va] if args.task == "cls" else y_reg[va], dtype=torch.float)
    train_dl = DataLoader(TensorDataset(ids_tr, mask_tr, ytr), batch_size=16)
    val_dl = DataLoader(TensorDataset(ids_va, mask_va, yva), batch_size=32)

    dl_utils.set_seed()
    hidden = base.config.hidden_size
    model = HFForSequence(base, hidden).to(dl_utils.DEVICE)
    model, best = train_hf(model, train_dl, val_dl, task=args.task, epochs=args.epochs)
    print(f"[HF-pretrained {args.model}] val={'AUC' if args.task == 'cls' else 'RMSE'}={best:.4f}")


if __name__ == "__main__":
    main()

