"""
models.py — all model definitions for the Visual Intelligence project.

Contains:
  - ClassifierHead   : shared binary classifier head (only in_features differs)
  - CatDogCNN        : 4-block VGG-style custom CNN
  - ScatteringWrapper: thin nn.Module wrapper around Kymatio Scattering2D
  - CatDogScatNet    : ScatteringWrapper + GAP + ClassifierHead
  - GradCAM          : from-scratch GradCAM using PyTorch hooks
"""

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Shared Classifier Head
# ---------------------------------------------------------------------------

class ClassifierHead(nn.Module):
    """
    Identical head used by both CNN and ScatNet.
    Only `in_features` differs (256 for CNN, 243 for ScatNet).
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.fc1     = nn.Linear(in_features, 256)
        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)   # raw logit — use BCEWithLogitsLoss


# ---------------------------------------------------------------------------
# Helper: build a single double-conv block
# ---------------------------------------------------------------------------

def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


# ---------------------------------------------------------------------------
# Custom CNN  (4 double-conv blocks, ~1.1 M trainable parameters)
# ---------------------------------------------------------------------------

class CatDogCNN(nn.Module):
    """
    VGG-style custom CNN for binary cat/dog classification.

    Architecture (input 3×128×128):
      Block 1:  3 →  32 × double-conv + MaxPool  →  32×64×64
      Block 2: 32 →  64 × double-conv + MaxPool  →  64×32×32
      Block 3: 64 → 128 × double-conv + MaxPool  → 128×16×16
      Block 4:128 → 256 × double-conv + MaxPool  → 256×8×8
      AdaptiveAvgPool(1) → Flatten(256) → ClassifierHead(256)

    GradCAM target layer : self.features[3]  (block 4)
    Filter visualisation : self.features[0][0].weight  (32, 3, 3, 3)
    """

    def __init__(self):
        super().__init__()
        self.features = nn.ModuleList([
            _conv_block(3,    32),   # block 0
            _conv_block(32,   64),   # block 1
            _conv_block(64,  128),   # block 2
            _conv_block(128, 256),   # block 3  ← GradCAM target
        ])
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = ClassifierHead(in_features=256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.features:
            x = block(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# ScatNet
# ---------------------------------------------------------------------------

class ScatteringWrapper(nn.Module):
    """
    Thin nn.Module around Kymatio's Scattering2D so we can:
      • move it to GPU with .to(device)
      • register hooks on it for GradCAM
    """
    def __init__(self, J: int = 2, L: int = 8, shape: tuple = (128, 128)):
        super().__init__()
        from kymatio.torch import Scattering2D
        self.scat = Scattering2D(J=J, L=L, shape=shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        # kymatio returns (B, C_in, S, H/2^J, W/2^J) — 5D tensor
        # With J=2, L=8: S=81, output (B, 3, 81, 32, 32)
        # Reshape to (B, 3*S, H/2^J, W/2^J) = (B, 243, 32, 32)
        out = self.scat(x)
        B = out.shape[0]
        return out.reshape(B, -1, out.shape[-2], out.shape[-1])

    def to(self, *args, **kwargs):
        self.scat = self.scat.to(*args, **kwargs)
        return super().to(*args, **kwargs)


class CatDogScatNet(nn.Module):
    """
    ScatNet for binary cat/dog classification.

    Architecture (input 3×128×128):
      ScatteringWrapper(J=3, L=8) → (B, 651, 16, 16)  [fixed, not trainable]
      conv_head: Conv(651→256)→BN→ReLU→Conv(256→128)→BN→ReLU→GAP → (B, 128)
      ClassifierHead(128)         → (B, 1)

    GradCAM target: self.conv_head  (last trainable spatial layer)
    Filter visualisation: self.scat_wrapper.scat.psi  (Morlet wavelets)
    """

    def __init__(self, J: int = 3, L: int = 8, shape: tuple = (128, 128)):
        super().__init__()
        self.scat_wrapper = ScatteringWrapper(J=J, L=L, shape=shape)

        # Number of scattering output channels after reshape: C_in * S
        # S = 1 + J*L + (J*(J-1)//2)*L^2; with J=2,L=8: S=81 → 3*81=243
        scat_channels = 3 * (1 + J * L + (J * (J - 1) // 2) * L * L)

        # Small conv classifier preserving spatial structure before pooling
        self.conv_head = nn.Sequential(
            nn.Conv2d(scat_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = ClassifierHead(in_features=128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.scat_wrapper(x)   # (B, 243, 32, 32)
        x = self.conv_head(x)      # (B, 128, 1, 1)
        x = x.flatten(1)           # (B, 128)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# GradCAM — implemented from scratch
# ---------------------------------------------------------------------------

class GradCAM:
    """
    GradCAM (Selvaraju et al., 2017) implemented from scratch using
    PyTorch forward/backward hooks.

    Usage:
        cam = GradCAM(model, target_layer=model.features[3])
        heatmap = cam.compute(img_tensor)          # numpy array in [0,1]
        overlay = cam.overlay(heatmap, img_array)  # RGB array in [0,1]
        cam.remove_hooks()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model       = model
        self.activations : torch.Tensor | None = None
        self.gradients   : torch.Tensor | None = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output.detach()   # (B, C, H, W)

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()  # (B, C, H, W)

    def compute(self, x: torch.Tensor) -> np.ndarray:
        """
        Returns a GradCAM heatmap as a 2-D numpy array normalised to [0, 1].
        `x` must be a single image tensor of shape (1, C, H, W) on the
        correct device, with requires_grad enabled on the computation graph.
        """
        self.model.eval()
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Forward pass
        logit = self.model(x)       # (1, 1)
        score = logit[0, 0]

        # Backward
        self.model.zero_grad()
        score.backward()

        # alpha_k: global-average-pool the gradients → (C,)
        alpha = self.gradients[0].mean(dim=(1, 2))   # (C,)

        # Weighted combination of activation maps → (H, W)
        cam = (alpha[:, None, None] * self.activations[0]).sum(dim=0)

        # ReLU & normalise
        cam = torch.clamp(cam, min=0)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()

    def overlay(
        self,
        cam: np.ndarray,
        image: np.ndarray,
        alpha: float = 0.45,
    ) -> np.ndarray:
        """
        Upsample `cam` to `image` size and blend as a JET heatmap.

        Args:
            cam   : 2-D float array in [0, 1], any spatial size
            image : H×W×3 float array in [0, 1]
            alpha : heatmap opacity

        Returns:
            H×W×3 float array in [0, 1]
        """
        import cv2
        h, w = image.shape[:2]
        cam_up = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap(
            (cam_up * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        return np.clip(alpha * heatmap + (1 - alpha) * image, 0, 1)

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()
