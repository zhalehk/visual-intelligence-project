"""
train_scatnet.py — ScatNet training with 5-fold cross-validation.

Outputs:
  outputs/scatnet_best.pth              model weights
  outputs/scatnet_kfold_results.json    per-fold metrics
  outputs/figures/scatnet_learning_curves.png
  outputs/figures/scatnet_filters.png
"""

import sys, json
from pathlib import Path

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from models import CatDogScatNet

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
OUT_DIR      = PROJECT_ROOT / "outputs"
FIG_DIR      = OUT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# ── device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── hyperparameters ──────────────────────────────────────────────────────────
IMG_SIZE   = 128
BATCH_SIZE = 64
EPOCHS     = 20
K_FOLDS    = 5
LR         = 3e-4
PATIENCE   = 5

# ── pre-flight: remove images that cannot be loaded at all ───────────────────

def clean_dataset():
    removed = 0
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        for path in DATA_DIR.rglob(pattern):
            try:
                with Image.open(path) as img:
                    img.load()
                    if img.mode != "RGB":
                        img.convert("RGB").save(path)
            except Exception:
                try:
                    path.unlink()
                    removed += 1
                except Exception:
                    pass
    if removed:
        print(f"Removed {removed} truly corrupt images.")
    else:
        print("Dataset clean — no corrupt images found.")

clean_dataset()

# ── transforms ───────────────────────────────────────────────────────────────
# ScatNet is already translation-invariant by design, so only light augmentation
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── datasets ─────────────────────────────────────────────────────────────────
train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
val_ds   = datasets.ImageFolder(DATA_DIR / "val",   transform=eval_tf)
test_ds  = datasets.ImageFolder(DATA_DIR / "test",  transform=eval_tf)

train_ds_eval = datasets.ImageFolder(DATA_DIR / "train", transform=eval_tf)
val_ds_eval   = datasets.ImageFolder(DATA_DIR / "val",   transform=eval_tf)

full_ds_aug  = ConcatDataset([train_ds,      val_ds])
full_ds_eval = ConcatDataset([train_ds_eval, val_ds_eval])

full_labels = (
    [s[1] for s in train_ds.samples] +
    [s[1] for s in val_ds.samples]
)

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"K-fold pool: {len(full_ds_aug)} images  |  Test: {len(test_ds)} images")
print(f"Classes: {train_ds.class_to_idx}")


# ── training helpers ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scheduler=None):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        preds   = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        total_loss += loss.item() * labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader):
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs     = imgs.to(DEVICE)
            labels_f = labels.float().unsqueeze(1).to(DEVICE)
            logits   = model(imgs)
            total_loss += criterion(logits, labels_f).item() * labels.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long().squeeze(1).cpu()
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())
    n = len(y_true)
    return total_loss / n, accuracy_score(y_true, y_pred), \
           f1_score(y_true, y_pred, average="binary", pos_label=1)


# ── k-fold cross-validation ──────────────────────────────────────────────────

def run_kfold():
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(range(len(full_ds_aug)), full_labels), start=1):

        print(f"\n{'='*50}")
        print(f"  Fold {fold}/{K_FOLDS}")
        print(f"{'='*50}")

        fold_train = Subset(full_ds_aug,  train_idx)
        fold_val   = Subset(full_ds_eval, val_idx)

        fold_train_loader = DataLoader(fold_train, batch_size=BATCH_SIZE,
                                       shuffle=True,  num_workers=0)
        fold_val_loader   = DataLoader(fold_val,   batch_size=BATCH_SIZE,
                                       shuffle=False, num_workers=0)

        model     = CatDogScatNet().to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LR * 5,
            steps_per_epoch=len(fold_train_loader),
            epochs=EPOCHS,
            pct_start=0.1,
        )

        best_val_acc   = 0.0
        patience_count = 0
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, EPOCHS + 1):
            tr_loss, tr_acc = train_one_epoch(model, fold_train_loader,
                                              optimizer, criterion, scheduler)
            vl_loss, vl_acc, _ = evaluate(model, fold_val_loader)

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(vl_loss)
            history["val_acc"].append(vl_acc)

            print(f"  Epoch {epoch:02d}/{EPOCHS}  "
                  f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
                  f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}")

            if vl_acc > best_val_acc:
                best_val_acc   = vl_acc
                best_weights   = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        model.load_state_dict(best_weights)
        _, fold_acc, fold_f1 = evaluate(model, fold_val_loader)
        print(f"  Fold {fold} best → acc={fold_acc:.4f}  F1={fold_f1:.4f}")

        fold_results.append({
            "fold":    fold,
            "val_acc": fold_acc,
            "val_f1":  fold_f1,
            "history": history,
        })

    mean_acc = np.mean([r["val_acc"] for r in fold_results])
    mean_f1  = np.mean([r["val_f1"]  for r in fold_results])
    print(f"\n{'='*50}")
    print(f"  K-Fold Results  (k={K_FOLDS})")
    print(f"  Mean Accuracy : {mean_acc:.4f}")
    print(f"  Mean F1 Score : {mean_f1:.4f}")
    print(f"{'='*50}\n")

    summary = {
        "mean_acc": mean_acc,
        "mean_f1":  mean_f1,
        "folds": [{k: v for k, v in r.items() if k != "history"}
                  for r in fold_results],
    }
    with open(OUT_DIR / "scatnet_kfold_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    return fold_results, mean_acc, mean_f1


# ── learning curves ───────────────────────────────────────────────────────────

def plot_learning_curves(fold_results, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10.colors

    for ax, metric, title in zip(
            axes,
            [("train_loss", "val_loss"), ("train_acc", "val_acc")],
            ["Loss", "Accuracy"]):

        for i, r in enumerate(fold_results):
            h   = r["history"]
            ep  = range(1, len(h["train_loss"]) + 1)
            col = colors[i % len(colors)]
            ax.plot(ep, h[metric[0]], color=col, lw=1.2, alpha=0.7,
                    label=f"Fold {r['fold']} train" if metric[0] == "train_loss" else "_")
            ax.plot(ep, h[metric[1]], color=col, lw=1.2, alpha=0.7, linestyle="--")

        max_ep = max(len(r["history"]["train_loss"]) for r in fold_results)
        def pad(arr, length):
            return arr + [arr[-1]] * (length - len(arr))

        mean_train = np.mean(
            [pad(r["history"][metric[0]], max_ep) for r in fold_results], axis=0)
        mean_val   = np.mean(
            [pad(r["history"][metric[1]], max_ep) for r in fold_results], axis=0)
        ep_all = range(1, max_ep + 1)
        ax.plot(ep_all, mean_train, "k-",  lw=2.5, label="Mean train")
        ax.plot(ep_all, mean_val,   "k--", lw=2.5, label="Mean val")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"ScatNet {title} — 5-Fold CV")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.text(0.5, 0.01, "Solid lines = train  |  Dashed lines = validation",
             ha="center", fontsize=9, style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved learning curves → {save_path}")


# ── scattering filter visualisation ──────────────────────────────────────────

def visualize_scatnet_filters(model, save_path):
    """
    Visualise the first-order Morlet wavelet filters (psi) from
    the Scattering2D module.  These are the fixed analysis filters —
    J=2 scales × L=8 orientations = 16 filters.
    """
    scat  = model.scat_wrapper.scat
    psi   = scat.psi             # list of filter info dicts

    filters_spatial = []
    for psi_info in psi:
        # Each entry is a dict with keys like 'levels', 'xi', 'sigma', etc.
        # The actual filter is stored at psi_info['levels'][0]
        if 'levels' in psi_info and len(psi_info['levels']) > 0:
            f = psi_info['levels'][0]        # complex tensor or ndarray (H, W)
            f = torch.as_tensor(f)
            f_spatial = torch.fft.ifftshift(
                torch.fft.ifft2(f).real
            )
            filters_spatial.append(f_spatial.cpu())

    n = len(filters_spatial)
    cols = 8
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = np.array(axes).reshape(rows, cols)

    for idx in range(rows * cols):
        ax = axes.flat[idx]
        if idx < n:
            f = filters_spatial[idx].numpy()
            vmax = np.abs(f).max()
            ax.imshow(f, cmap="RdBu", vmin=-vmax, vmax=vmax,
                      interpolation="nearest")
            ax.set_title(f"ψ {idx}", fontsize=7)
        ax.axis("off")

    fig.suptitle(
        f"ScatNet — First-order Morlet Wavelets  (J=2, L=8, {n} filters)",
        fontsize=11
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ScatNet filters → {save_path}")


# ── final model training ──────────────────────────────────────────────────────

def train_final_model():
    print("\nTraining final ScatNet on full train+val data...")
    full_loader = DataLoader(full_ds_aug, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=0)

    model     = CatDogScatNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR * 5,
        steps_per_epoch=len(full_loader),
        epochs=EPOCHS,
        pct_start=0.1,
    )

    best_loss      = float("inf")
    patience_count = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, full_loader, optimizer, criterion, scheduler)
        print(f"  Epoch {epoch:02d}/{EPOCHS}  train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}")

        if tr_loss < best_loss:
            best_loss    = tr_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), OUT_DIR / "scatnet_best.pth")
    print(f"Saved model → {OUT_DIR / 'scatnet_best.pth'}")
    return model


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fold_results, mean_acc, mean_f1 = run_kfold()
    plot_learning_curves(fold_results, FIG_DIR / "scatnet_learning_curves.png")
    final_model = train_final_model()
    visualize_scatnet_filters(final_model, FIG_DIR / "scatnet_filters.png")
    test_loss, test_acc, test_f1 = evaluate(final_model, test_loader)
    print(f"\nTest accuracy : {test_acc:.4f}")
    print(f"Test F1 score : {test_f1:.4f}")
    print(f"Test loss     : {test_loss:.4f}")
