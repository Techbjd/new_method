"""Shared DL training/eval utilities (seed 57; softmax classification per project convention)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

RANDOM_SEED = 57
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = RANDOM_SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(graphs, tr, va, te, batch=32):
    from torch.utils.data import Subset
    dl = lambda idx: DataLoader(Subset(graphs, idx), batch_size=batch, shuffle=False)
    return dl(tr), dl(va), dl(te)


def softmax_probs(logits: torch.Tensor) -> np.ndarray:
    """Project convention: classification probabilities are softmax(logits)[:,1]."""
    return torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()


def evaluate(model, loader, task="cls", device=DEVICE):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            if task == "cls":
                probs = softmax_probs(out)
            else:
                probs = out.squeeze(-1).detach().cpu().numpy()
            ys.append(batch.y.detach().float().cpu().numpy() if task == "cls"
                      else batch.y.float().cpu().numpy())
            ps.append(probs)
    ys = np.concatenate([np.atleast_1d(a) for a in ys])
    ps = np.concatenate([np.atleast_1d(a) for a in ps])
    if task == "cls":
        from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
        yb = ys.astype(int)
        return {"AUC": float(roc_auc_score(yb, ps)),
                "AAP": float(average_precision_score(yb, ps)),
                "balanced_acc": float(balanced_accuracy_score(yb, (ps >= 0.5).astype(int)))}
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {"RMSE": float(np.sqrt(mean_squared_error(ys, ps))),
            "MAE": float(mean_absolute_error(ys, ps)),
            "R2": float(r2_score(ys, ps))}


def train_loop(model, train_loader, val_loader, task="cls", epochs=30, lr=1e-3,
               patience=10, device=DEVICE, verbose=False):
    """Validation-only early stopping (never touch test split here)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = (lambda o, b: F.cross_entropy(o, b.y) if task == "cls"
               else F.mse_loss(o.squeeze(-1), b.y.float()))
    best_state, best_val, bad = None, (-np.inf if task == "cls" else np.inf), 0
    key = "AUC" if task == "cls" else "RMSE"
    history = []
    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            opt.step()
        val_m = evaluate(model, val_loader, task=task, device=device)[key]
        history.append({"epoch": ep, "val_" + key: round(float(val_m), 5)})
        improved = (val_m > best_val) if task == "cls" else (val_m < best_val)
        if improved:
            best_val, bad = val_m, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and (ep + 1) % 5 == 0:
            print(f"  epoch {ep + 1}: val_{key}={val_m:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
