"""
xai_analysis.py — Explainability (XAI) analysis for CatDogCNN and CatDogScatNet.

6 XAI methods applied to both models on 4 test images (2 cats, 2 dogs):
  1. GradCAM             — from scratch via models.py GradCAM class
  2. Integrated Gradients — Captum
  3. Saliency             — Captum
  4. DeepLIFT            — Captum  (skipped gracefully if unsupported)
  5. GuidedBackprop      — Captum
  6. Occlusion           — Captum

Outputs saved to outputs/figures/:
  xai_cnn_{label}_{idx}.png       CNN attributions for each test image  (8 total)
  xai_scatnet_{label}_{idx}.png   ScatNet attributions for each test image (8 total)
  xai_gradcam_comparison.png      GradCAM scratch vs Captum side-by-side  (1 total)

GradCAM target layers:
  CNN     : model.features[3]   (block 3, last double-conv block)
  ScatNet : model.conv_head     (trainable conv block after scattering)
"""

import sys
import cv2
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from models import CatDogCNN, CatDogScatNet, GradCAM

# ── Captum imports ────────────────────────────────────────────────────────────
from captum.attr import (
    IntegratedGradients,
    Saliency,
    DeepLift,
    GuidedBackprop,
    Occlusion,
    LayerGradCam,
)

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
OUT_DIR      = PROJECT_ROOT / "outputs"
FIG_DIR      = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── constants ─────────────────────────────────────────────────────────────────
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]
IMG_SIZE    = 128
BLEND_ALPHA = 0.5   # heatmap opacity when blending over the original image

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_models():
    """Load pretrained CNN and ScatNet weights from outputs/."""
    cnn = CatDogCNN().to(DEVICE)
    cnn.load_state_dict(
        torch.load(OUT_DIR / "cnn_best.pth", map_location=DEVICE, weights_only=True)
    )
    cnn.eval()

    scat = CatDogScatNet().to(DEVICE)
    scat.load_state_dict(
        torch.load(OUT_DIR / "scatnet_best.pth", map_location=DEVICE, weights_only=True)
    )
    scat.eval()

    return cnn, scat


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_test_images(n_per_class: int = 2) -> list:
    """
    Load `n_per_class` images from data/test/cats and data/test/dogs.
    Returns list of (PIL.Image, label_str, local_index).
    Order: cat0, cat1, dog0, dog1.
    """
    images = []
    for label in ["cats", "dogs"]:
        folder = DATA_DIR / "test" / label
        files  = sorted(folder.glob("*.jpg"))[:n_per_class]
        for idx, f in enumerate(files):
            images.append((Image.open(f).convert("RGB"), label, idx))
    return images


def img_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Return normalised (1, 3, IMG_SIZE, IMG_SIZE) tensor on DEVICE."""
    return eval_tf(pil_img).unsqueeze(0).to(DEVICE)


def img_to_numpy(pil_img: Image.Image) -> np.ndarray:
    """Return H×W×3 float32 array in [0, 1] at IMG_SIZE resolution."""
    return np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# Attribution helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_heatmap(arr: np.ndarray) -> np.ndarray:
    """Shift and scale a 2-D float array to [0, 1]."""
    arr = arr - arr.min()
    return arr / (arr.max() + 1e-8)


def tensor_attr_to_heatmap(attr: torch.Tensor) -> np.ndarray:
    """
    Convert a Captum attribution tensor to a normalised 2-D heatmap.

    Accepts shapes: (1, C, H', W')  or  (1, 1, H', W').
    Aggregates over channels using mean of absolute values, resizes to
    IMG_SIZE × IMG_SIZE, and normalises to [0, 1].
    """
    a = attr.squeeze(0).detach().cpu()       # (C, H', W') or (H', W')
    if a.dim() == 3:
        a = torch.mean(torch.abs(a), dim=0)  # → (H', W')
    else:
        a = torch.abs(a)

    arr = a.numpy().astype(np.float32)
    arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE))
    return normalize_heatmap(arr)


def overlay_heatmap(
    heatmap: np.ndarray,
    image:   np.ndarray,
    alpha:   float = BLEND_ALPHA,
) -> np.ndarray:
    """
    Blend a normalised 2-D heatmap over an H×W×3 image using JET colormap.
    Returns H×W×3 float32 in [0, 1].
    """
    h, w    = image.shape[:2]
    cam_up  = cv2.resize(heatmap.astype(np.float32), (w, h))
    colored = cv2.applyColorMap(
        (cam_up * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip(alpha * colored + (1.0 - alpha) * image, 0.0, 1.0)


def get_prediction(model: torch.nn.Module, img_tensor: torch.Tensor):
    """Return (predicted_label: str, confidence_pct: float) for one image."""
    with torch.no_grad():
        prob = torch.sigmoid(model(img_tensor)).item()   # P(dog)
    pred = "dog" if prob >= 0.5 else "cat"
    conf = (prob if prob >= 0.5 else 1.0 - prob) * 100.0
    return pred, conf


# ─────────────────────────────────────────────────────────────────────────────
# Captum LayerGradCam helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_captum_gradcam(
    model:        torch.nn.Module,
    img_tensor:   torch.Tensor,
    target_layer: torch.nn.Module,
) -> np.ndarray:
    """
    Compute GradCAM using Captum's LayerGradCam.
    Returns a normalised 2-D heatmap at IMG_SIZE × IMG_SIZE.
    """
    lgc  = LayerGradCam(model, target_layer)
    attr = lgc.attribute(img_tensor, target=0)   # (1, C, H', W')
    return tensor_attr_to_heatmap(attr)


# ─────────────────────────────────────────────────────────────────────────────
# Core: compute all 6 attributions for one model + one image
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_attributions(
    model:        torch.nn.Module,
    img_tensor:   torch.Tensor,
    raw_img:      np.ndarray,
    target_layer: torch.nn.Module,
    model_name:   str,
) -> list:
    """
    Compute all 6 XAI attributions and return:
        list of (method_label: str, overlay: np.ndarray, heatmap_2d: np.ndarray)

    overlay    : H×W×3 float32 — attribution blended on original image
    heatmap_2d : H×W  float32 in [0, 1] — raw normalised heatmap

    DeepLIFT is skipped gracefully if unsupported (e.g. ScatNet's fixed
    scattering extractor). The slot is kept with the original image as the
    overlay and a blank heatmap, so the figure layout is always 7 panels.
    """
    results  = []
    baseline = torch.zeros_like(img_tensor)   # black-image baseline

    # 1 ── GradCAM (from scratch) ─────────────────────────────────────────────
    print(f"      GradCAM (scratch) ... ", end="", flush=True)
    cam     = GradCAM(model, target_layer)
    heatmap = cam.compute(img_tensor.clone())
    cam.remove_hooks()
    results.append(("GradCAM\n(scratch)", overlay_heatmap(heatmap, raw_img), heatmap))
    print("done")

    # 2 ── Integrated Gradients ───────────────────────────────────────────────
    print(f"      Integrated Gradients ... ", end="", flush=True)
    ig   = IntegratedGradients(model)
    attr = ig.attribute(
        img_tensor.clone(),
        baselines=baseline,
        target=0,
        n_steps=50,
    )
    hm = tensor_attr_to_heatmap(attr)
    results.append(("Integrated\nGradients", overlay_heatmap(hm, raw_img), hm))
    print("done")

    # 3 ── Saliency ───────────────────────────────────────────────────────────
    print(f"      Saliency ... ", end="", flush=True)
    sal  = Saliency(model)
    attr = sal.attribute(img_tensor.clone(), target=0, abs=True)
    hm   = tensor_attr_to_heatmap(attr)
    results.append(("Saliency", overlay_heatmap(hm, raw_img), hm))
    print("done")

    # 4 ── DeepLIFT (may be unsupported on ScatNet — handled gracefully) ──────
    print(f"      DeepLIFT ... ", end="", flush=True)
    try:
        dl   = DeepLift(model)
        attr = dl.attribute(img_tensor.clone(), baselines=baseline, target=0)
        hm   = tensor_attr_to_heatmap(attr)
        results.append(("DeepLIFT", overlay_heatmap(hm, raw_img), hm))
        print("done")
    except Exception as exc:
        print(f"skipped ({type(exc).__name__})")
        print(f"        [INFO] DeepLIFT not supported for {model_name}: {exc}")
        blank_hm = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        results.append(("DeepLIFT\n(N/A)", raw_img.copy(), blank_hm))

    # 5 ── GuidedBackprop ─────────────────────────────────────────────────────
    print(f"      GuidedBackprop ... ", end="", flush=True)
    gbp  = GuidedBackprop(model)
    attr = gbp.attribute(img_tensor.clone(), target=0)
    hm   = tensor_attr_to_heatmap(attr)
    results.append(("Guided\nBackprop", overlay_heatmap(hm, raw_img), hm))
    print("done")

    # 6 ── Occlusion ──────────────────────────────────────────────────────────
    print(f"      Occlusion ... ", end="", flush=True)
    occ  = Occlusion(model)
    attr = occ.attribute(
        img_tensor,                         # perturbation-based; no grad needed
        target=0,
        sliding_window_shapes=(3, 15, 15),  # occlude 15×15 patch across all 3 channels
        strides=(3, 8, 8),                  # step 8 px spatially
        baselines=0,
    )
    hm = tensor_attr_to_heatmap(attr)
    results.append(("Occlusion", overlay_heatmap(hm, raw_img), hm))
    print("done")

    return results   # 6 items


# ─────────────────────────────────────────────────────────────────────────────
# Figure type 1 (×8): per-image per-model — all 6 attributions in one row
# ─────────────────────────────────────────────────────────────────────────────

def plot_image_attributions(
    raw_img:      np.ndarray,
    attributions: list,
    title:        str,
    save_path:    Path,
) -> None:
    """
    1 × 7 figure:
        [Original | GradCAM | IG | Saliency | DeepLIFT | GuidedBP | Occlusion]
    Each attribution cell shows the heatmap blended over the original image.
    """
    n_cols = 7
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2.6, 3.2))

    axes[0].imshow(raw_img)
    axes[0].set_title("Original", fontsize=9, fontweight="bold")
    axes[0].axis("off")

    for ax, (name, overlay, _) in zip(axes[1:], attributions):
        ax.imshow(overlay)
        ax.set_title(name, fontsize=7.5, linespacing=1.3)
        ax.axis("off")

    fig.suptitle(title, fontsize=9, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      → Saved: {save_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure type 2 (×1): GradCAM from-scratch vs Captum for both models
# ─────────────────────────────────────────────────────────────────────────────

def plot_gradcam_comparison(
    images:        list,
    cnn:           torch.nn.Module,
    scatnet:       torch.nn.Module,
    cnn_layer:     torch.nn.Module,
    scatnet_layer: torch.nn.Module,
    save_path:     Path,
) -> None:
    """
    4-row × 5-col figure comparing GradCAM implementations for both models.

    Columns: Original | CNN scratch | CNN Captum | ScatNet scratch | ScatNet Captum
    Rows:    one per test image  (cat0, cat1, dog0, dog1)
    """
    col_titles = [
        "Original",
        "CNN\nGradCAM (scratch)",
        "CNN\nGradCAM (Captum)",
        "ScatNet\nGradCAM (scratch)",
        "ScatNet\nGradCAM (Captum)",
    ]
    n_rows, n_cols = len(images), 5
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.0, n_rows * 3.2),
    )

    for r, (pil_img, label, idx) in enumerate(images):
        print(f"      row {r+1}/{n_rows}: {label} #{idx}", flush=True)
        raw_img    = img_to_numpy(pil_img)
        img_tensor = img_to_tensor(pil_img)

        # CNN — from scratch
        cam_s    = GradCAM(cnn, cnn_layer)
        hm_cnn_s = cam_s.compute(img_tensor.clone())
        cam_s.remove_hooks()

        # CNN — Captum LayerGradCam
        hm_cnn_c = compute_captum_gradcam(cnn, img_tensor, cnn_layer)

        # ScatNet — from scratch
        cam_s     = GradCAM(scatnet, scatnet_layer)
        hm_scat_s = cam_s.compute(img_tensor.clone())
        cam_s.remove_hooks()

        # ScatNet — Captum LayerGradCam
        hm_scat_c = compute_captum_gradcam(scatnet, img_tensor, scatnet_layer)

        row_imgs = [
            raw_img,
            overlay_heatmap(hm_cnn_s,  raw_img),
            overlay_heatmap(hm_cnn_c,  raw_img),
            overlay_heatmap(hm_scat_s, raw_img),
            overlay_heatmap(hm_scat_c, raw_img),
        ]

        for c, img_data in enumerate(row_imgs):
            ax = axes[r, c]
            ax.imshow(img_data)
            ax.axis("off")
            if r == 0:
                ax.set_title(col_titles[c], fontsize=8, fontweight="bold",
                             linespacing=1.4)

        # Row label on the leftmost cell
        axes[r, 0].set_ylabel(
            f"{label} #{idx}",
            fontsize=8, rotation=0, labelpad=50, va="center",
        )

    fig.suptitle(
        "GradCAM Comparison: From Scratch vs Captum LayerGradCam\n"
        "CNN: model.features[3]   |   ScatNet: model.conv_head",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      → Saved: {save_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  XAI Analysis — CatDogCNN and CatDogScatNet")
    print("=" * 62)

    # ── load models ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading models ...")
    cnn, scatnet  = load_models()
    cnn_layer     = cnn.features[3]        # block 3 — last double-conv block
    scatnet_layer = scatnet.conv_head   # trainable Conv2d block — hooks fire correctly here
    print("      CNN and ScatNet loaded.")

    # ── load images ──────────────────────────────────────────────────────────
    print("\n[2/4] Loading test images (2 cats + 2 dogs) ...")
    images = load_test_images(n_per_class=2)
    print(f"      Loaded: {[(lbl, idx) for _, lbl, idx in images]}")

    # ── per-image attribution figures (8 total) ───────────────────────────────
    print("\n[3/4] Computing per-image attribution figures (8 total) ...")
    for pil_img, label, idx in images:
        raw_img    = img_to_numpy(pil_img)
        img_tensor = img_to_tensor(pil_img)

        for model, model_name, layer in [
            (cnn,     "CNN",     cnn_layer),
            (scatnet, "ScatNet", scatnet_layer),
        ]:
            print(f"\n    [{model_name}]  {label} #{idx}")
            pred_cls, conf = get_prediction(model, img_tensor)
            print(f"      Prediction: {pred_cls}  ({conf:.1f}%)")

            attrs = compute_all_attributions(
                model, img_tensor, raw_img, layer, model_name
            )

            title = (
                f"{model_name}  —  {label} #{idx}  "
                f"[Pred: {pred_cls}  ({conf:.1f}%)]"
            )
            fname = FIG_DIR / f"xai_{model_name.lower()}_{label}_{idx}.png"
            plot_image_attributions(raw_img, attrs, title, fname)

    # ── GradCAM comparison figure (1 total) ──────────────────────────────────
    print("\n[4/4] GradCAM comparison figure ...")
    plot_gradcam_comparison(
        images,
        cnn, scatnet,
        cnn_layer, scatnet_layer,
        FIG_DIR / "xai_gradcam_comparison.png",
    )

    print(f"\n{'='*62}")
    print(f"  Done.  9 figures saved to:")
    print(f"    {FIG_DIR}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
