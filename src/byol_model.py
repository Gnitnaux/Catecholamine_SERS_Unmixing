"""
Two-stage BYOL pre-training + classification fine-tuning for SERS.

Stage 1: BYOL self-supervised pre-training on full spectra.
  - Online view: augmented and peak-masked spectrum -> encoder -> projector
    -> predictor.
  - Target view: clean spectrum -> EMA encoder -> projector.
  - Loss: symmetric cosine distance between online predictions and detached
    target projections, with a variance penalty to reduce representation
    collapse and optional PCA distillation when enabled.
  - Saves online encoder weights.

Stage 2: Classification fine-tuning.
  - Loads Stage 1 encoder, adds a classification head.
  - Classification is the default downstream task.
"""

import copy
import csv
import math
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch

# Fix root causes of known warnings
torch.backends.cuda.enable_nested_tensor = False
torch.backends.cuda.enable_flash_sdp = False
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import joblib


def _ensure_byol_output_dirs():
    """Ensure BYOL output directories exist.

    Author: Xuanting Liu & ChatGPT CODEX
    """
    for dirname in ("visualizations", "reports", "logs", "model_output"):
        os.makedirs(dirname, exist_ok=True)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """Channel-wise attention: GAP + GMP → shared MLP → sigmoid weights."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        y_avg = self.gap(x).squeeze(-1)   # (B, C)
        y_max = self.gmp(x).squeeze(-1)   # (B, C)
        att = self.sigmoid(self.mlp(y_avg) + self.mlp(y_max))  # (B, C)
        return x * att.unsqueeze(-1)


class ResBlock(nn.Module):
    """Third-order residual block: 3 Conv1d layers with residual shortcut.

    Each conv is followed by BatchNorm + GELU, except the last conv
    (GELU after the residual addition).
    """

    def __init__(self, channels, kernel_size, dilation=1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv3 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.bn3 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = F.gelu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.gelu(out + residual)


class MultiScaleBlock(nn.Module):
    """4 parallel Conv1d branches with different kernel sizes, concatenated."""

    def __init__(self, in_channels=1, out_channels=512):
        super().__init__()
        per_branch = out_channels // 4  # 128 per branch
        self.kernels = [15, 11, 7, 3]
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, per_branch, k, padding=k // 2),
                nn.BatchNorm1d(per_branch),
                nn.GELU(),
            )
            for k in self.kernels
        ])

    def forward(self, x):
        feats = [branch(x) for branch in self.branches]
        return torch.cat(feats, dim=1)  # (B, 512, L)


class LegacyCNNEncoder(nn.Module):
    """CNN encoder: MultiScale → 3×ResBlock → ChannelAttn → proj to 256."""

    def __init__(self, in_channels=1, base_channels=512, out_dim=256):
        super().__init__()
        self.multiscale = MultiScaleBlock(in_channels, base_channels)

        # 3 residual blocks with descending kernel sizes
        self.resblocks = nn.Sequential(
            ResBlock(base_channels, kernel_size=7),
            ResBlock(base_channels, kernel_size=5),
            ResBlock(base_channels, kernel_size=3),
        )

        self.channel_attn = ChannelAttention(base_channels)

        # 1×1 conv to project to out_dim
        self.project = nn.Sequential(
            nn.Conv1d(base_channels, out_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(out_dim),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, in_channels, L)
        x = self.multiscale(x)          # (B, 512, L)
        x = self.resblocks(x)           # (B, 512, L)
        x = self.channel_attn(x)        # (B, 512, L)
        x = self.project(x)             # (B, 256, L)
        x = self.gap(x).squeeze(-1)     # (B, 256)
        return x


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for spectral sequences."""

    def __init__(self, d_model=256, max_len=4096):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

from src.ca_paper_plsr import (
    _read_mpau_mix_spectra, _umx_make_group_id,
    _filter_mix_conc,
    _print_per_group_mse,
    UNMIX_ANALYTES, UNMIX_N_OUTER, UNMIX_RANDOM_STATE,
    UNMIX_MODEL_MIXTURES, UNMIX_SINGLE_MIXTURES,
    _umx_balanced_holdout_split, _umx_present_analyte_indices,
    _umx_two_peak_ratio_features, _umx_build_tables,
    _umx_continuous_summary, _umx_save_payload, _umx_plot_holdout_pred,
)
from src.utils import spectra_normalization

# ===========================================================================
# Config
# ===========================================================================

BYOL_CONFIG = {
    # Encoder
    "in_channels": 1,
    "base_channels": 128,
    "out_dim": 256,
    "nhead": 8,
    "num_layers": 4,
    "dim_feedforward": 1024,
    "transformer_dropout": 0.2,
    # BYOL projector + predictor
    "proj_hidden": 1024,
    "proj_out": 256,
    # EMA
    "ema_momentum_base": 0.996,
    # Spectral masking. The original 5-10% weak-peak mask usually removed
    # only a few points, so Stage 1 now uses a genuinely missing-view task.
    "mask_ratio_min": 0.10,
    "mask_ratio_max": 0.18,
    "mask_weak_fraction": 0.50,
    "mask_expand_min": 1,
    "mask_expand_max": 4,
    "span_mask_prob": 0.40,
    "span_mask_count_min": 1,
    "span_mask_count_max": 1,
    "span_mask_width_min": 14,
    "span_mask_width_max": 28,
    "sigma_baseline": 24.78,
    # Stage 1 training
    "pretext_method": "byol",
    "stage1_epochs": 50,
    "stage1_batch_size": 64,
    "stage1_lr": 1e-3,
    "stage1_var_loss_weight": 5.0,
    "stage1_var_loss_gamma": 0.04,
    "stage1_pca_distill_weight": 0.20,
    "stage1_pca_distill_components": 30,
    # Stage 2 training
    "stage2_frozen_epochs": 20,
    "stage2_full_epochs": 50,
    "stage2_batch_size": 32,
    "stage2_lr": 1e-3,
    "num_classes": 3,  # DA, E, NE (3 independent binary classifications)
    "stage2_class_backend": "byol_embedding_logreg",
    "stage2_byol_logreg_C": [0.3, 1.0, 3.0, 10.0, 30.0],
    "stage2_byol_feature_pca_ensemble": [(80, 1.0)],
    "stage2_raw_pca_components": 30,
    "stage2_logreg_C": 5.0,
    "stage2_raw_pca_ensemble": [
        (10, 0.5), (20, 1.0), (20, 5.0),
        (30, 1.0), (30, 5.0), (60, 2.0),
    ],
}

BYOL_8CLASS_LABELS = ["BA", "DA", "E", "NE",
                      "DA+E", "DA+NE", "E+NE", "DA+E+NE"]


def _mixture_color_map():
    return {
        "BA": "#7F7F7F",
        "DA": "#D62728",
        "E": "#1F77B4",
        "NE": "#2CA02C",
        "DA+E": "#9467BD",
        "DA+NE": "#FF7F0E",
        "E+NE": "#17BECF",
        "DA+E+NE": "#8C564B",
    }


def _umx_group_folds(group_table, n_splits=3, random_state=2026):
    """Local stratified group-fold assignment for BYOL legacy routines."""
    rng = np.random.default_rng(random_state)
    group_table = group_table.copy()
    group_table["fold"] = -1
    for mix in sorted(group_table["mixture"].unique()):
        idx = group_table.index[group_table["mixture"] == mix].to_numpy()
        rng.shuffle(idx)
        for i, row_idx in enumerate(idx):
            group_table.loc[row_idx, "fold"] = i % n_splits
    return group_table


# ===========================================================================
# Encoder (unchanged architecture, just wrapping CNN + Transformer)
# ===========================================================================

class BYOLEncoder(nn.Module):
    """CNN + Transformer encoder. Same architecture as CNNEncoder.

    Returns (B, out_dim) feature vector used by both projection head
    and classification head.
    """

    def __init__(self, in_channels=1, base_channels=512, out_dim=256,
                 nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.2):
        super().__init__()
        self.multiscale = MultiScaleBlock(in_channels, base_channels)
        self.resblocks = nn.Sequential(
            ResBlock(base_channels, kernel_size=7),
            ResBlock(base_channels, kernel_size=5),
            ResBlock(base_channels, kernel_size=3),
        )
        self.channel_attn = ChannelAttention(base_channels)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.project = nn.Sequential(
            nn.Conv1d(base_channels, out_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(out_dim),
        )
        self.position = SinusoidalPositionalEncoding(out_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.fusion = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)                     # (B, L) → (B, 1, L)
        x = self.multiscale(x)                     # (B, 512, L)
        x = self.resblocks(x)                      # (B, 512, L)
        x = self.channel_attn(x)                   # (B, 512, L)
        x = self.pool(x)                           # (B, 512, L/2)
        cnn_seq = self.project(x).transpose(1, 2)  # (B, L/2, 256)
        transformer_seq = self.transformer(
            self.position(cnn_seq))                # (B, L/2, 256)
        cnn_global = cnn_seq.mean(dim=1, keepdim=True).expand_as(
            transformer_seq)
        fused = self.fusion(
            torch.cat([transformer_seq, cnn_global], dim=-1))
        return fused.mean(dim=1)                   # (B, 256)


# ===========================================================================
# Projection Head (256 → 512 → 2048)
# ===========================================================================

class ProjectionHead(nn.Module):
    """3-layer MLP with bottleneck, L2-normalized output.

    256 → 512 → 256 → 2048, all with GELU, then L2-normalize.
    """

    def __init__(self, in_dim=256, hidden_dim=512,
                 bottleneck_dim=256, out_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# ===========================================================================
# BYOL Projector + Predictor
# ===========================================================================

class BYOLProjector(nn.Module):
    """MLP: 256 → 1024 → 256 → 256, BN + ReLU, L2-norm output."""

    def __init__(self, in_dim=256, hidden=1024, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class BYOLPredictor(nn.Module):
    """MLP on top of online projector only: 256 → 1024 → 256, L2-norm."""

    def __init__(self, in_dim=256, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class BYOLModel(nn.Module):
    """Encoder + Projector (+ optional Predictor for online network)."""

    def __init__(self, config=None, has_predictor=False):
        super().__init__()
        cfg = config or BYOL_CONFIG
        self.encoder = BYOLEncoder(
            in_channels=cfg["in_channels"],
            base_channels=cfg["base_channels"],
            out_dim=cfg["out_dim"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["transformer_dropout"],
        )
        self.projector = BYOLProjector(
            in_dim=cfg["out_dim"],
            hidden=cfg["proj_hidden"],
            out_dim=cfg["proj_out"],
        )
        if has_predictor:
            self.predictor = BYOLPredictor(in_dim=cfg["proj_out"])
        else:
            self.predictor = None

    def forward(self, x):
        feat = self.encoder(x)
        z = self.projector(feat)
        if self.predictor is not None:
            return self.predictor(z)
        return z


# ===========================================================================
# Peak Masking
# ===========================================================================

class PeakMasking:
    """Peak-level token masking for byol student input.

    Peak detection (following the teacher-student paper):
      1. Identify peak edges via the first derivative.
      2. Extract local maxima.
      3. Filter: peak height > 3 × sigma_baseline.
      4. Sort peaks by intensity descending.
      5. Randomly mask 5–10% of peaks drawn from the WEAKEST 30%.

    Each masked peak "token" = (position, intensity, width) —
    the region from left edge to right edge is zeroed.
    """

    def __init__(self, sigma_baseline=24.78, min_height_sigma=3.0,
                 weak_fraction=0.30,
                 mask_ratio_min=0.05, mask_ratio_max=0.10,
                 mask_value=0.0,
                 expand_min=0, expand_max=0,
                 span_mask_prob=0.0,
                 span_mask_count_min=1, span_mask_count_max=1,
                 span_mask_width_min=12, span_mask_width_max=36):
        self.min_height = min_height_sigma * sigma_baseline
        self.weak_fraction = weak_fraction
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.mask_value = mask_value
        self.expand_min = int(expand_min)
        self.expand_max = int(expand_max)
        self.span_mask_prob = float(span_mask_prob)
        self.span_mask_count_min = int(span_mask_count_min)
        self.span_mask_count_max = int(span_mask_count_max)
        self.span_mask_width_min = int(span_mask_width_min)
        self.span_mask_width_max = int(span_mask_width_max)

    def _detect_peaks(self, spec):
        """Detect peaks via first-derivative zero-crossings.

        Each peak is represented as a token:
          {center, height, left, right, width}

        Returns list sorted by height descending.
        """
        L = len(spec)
        deriv = np.diff(spec)

        # Find positive→negative zero-crossings as peak centres
        peak_centers = []
        for j in range(1, L - 1):
            if (deriv[j-1] > 0 and deriv[j] < 0
                    and spec[j] > self.min_height):
                peak_centers.append(j)

        if not peak_centers:
            return []

        # Find left/right edges by tracing deriv sign / half-max
        peaks = []
        for center in peak_centers:
            height = spec[center]

            left = center
            for j in range(center - 1, 0, -1):
                if spec[j] <= 0.5 * height or deriv[j-1] <= 0:
                    left = j
                    break

            right = center
            for j in range(center, L - 1):
                if spec[j] <= 0.5 * height or deriv[j] >= 0:
                    right = j
                    break

            peaks.append({
                'center': center,
                'height': height,
                'left': left,
                'right': right,
                'width': right - left,    # token: (position, intensity, width)
            })

        peaks.sort(key=lambda p: p['height'], reverse=True)
        return peaks

    def __call__(self, spectra):
        """Apply peak-level token masking.

        Args:
            spectra: (B, L) numpy array or torch tensor.

        Returns:
            masked: (B, L) tensor with masked peak regions.
        """
        device = None
        if isinstance(spectra, torch.Tensor):
            device = spectra.device
            x = spectra.cpu().numpy()
        else:
            x = spectra

        B, L = x.shape
        masked = x.copy()

        for i in range(B):
            peaks = self._detect_peaks(x[i])
            if len(peaks) < 2:
                continue

            # Weakest 30% of peaks (by intensity)
            n_weak = max(1, int(len(peaks) * self.weak_fraction))
            weak_peaks = peaks[-n_weak:]

            # Randomly mask 5–10% of the weak peaks
            ratio = np.random.uniform(self.mask_ratio_min,
                                       self.mask_ratio_max)
            n_mask = max(1, int(len(weak_peaks) * ratio))
            chosen = np.random.choice(len(weak_peaks), size=n_mask,
                                       replace=False)

            for idx in chosen:
                p = weak_peaks[idx]
                expand = 0
                if self.expand_max > 0:
                    expand = np.random.randint(self.expand_min,
                                               self.expand_max + 1)
                left = max(0, p['left'] - expand)
                right = min(L - 1, p['right'] + expand)
                masked[i, left:right + 1] = self.mask_value

            if (self.span_mask_prob > 0
                    and np.random.rand() < self.span_mask_prob):
                n_span = np.random.randint(self.span_mask_count_min,
                                           self.span_mask_count_max + 1)
                for _ in range(n_span):
                    width = np.random.randint(self.span_mask_width_min,
                                              self.span_mask_width_max + 1)
                    width = min(width, L)
                    start = np.random.randint(0, L - width + 1)
                    masked[i, start:start + width] = self.mask_value

        if device is not None:
            masked = torch.from_numpy(masked).float().to(device)
        return masked

    def count_peaks(self, spectra):
        """Return average number of detected peaks per spectrum."""
        if isinstance(spectra, torch.Tensor):
            x = spectra.cpu().numpy()
        else:
            x = spectra
        counts = [len(self._detect_peaks(x[i])) for i in range(x.shape[0])]
        return np.mean(counts)


# ===========================================================================
# Diagnostics
# ===========================================================================

def _compute_diagnostics(z_s_batch, z_t_batch, feats_s_batch, feats_t_batch,
                         center, teacher_temp=0.04, student_temp=0.1,
                         log_dim=None, n_masked=0, n_detected=0):
    """Compute per-batch diagnostics. Returns dict of floats."""
    K = z_s_batch.size(-1)
    logK = math.log(K) if log_dim is None else math.log(log_dim)

    with torch.no_grad():
        # Softmax distributions
        p_t = F.softmax((z_t_batch - center) / teacher_temp, dim=-1)
        p_s = F.softmax(z_s_batch / student_temp, dim=-1)

        # Entropies (natural log)
        ent_t = -(p_t * p_t.clamp(min=1e-12).log()).sum(dim=-1).mean().item()
        ent_s = -(p_s * p_s.clamp(min=1e-12).log()).sum(dim=-1).mean().item()

        # Max probabilities
        max_t = p_t.max(dim=-1).values.mean().item()
        max_s = p_s.max(dim=-1).values.mean().item()

        # Projection diversity (std across samples per dimension, then mean)
        proj_std = z_s_batch.std(dim=0).mean().item()

        # Encoder feature diversity
        feat_std = feats_s_batch.std(dim=0).mean().item()

    return {
        'teacher_entropy': ent_t,
        'student_entropy': ent_s,
        'teacher_max_prob': max_t,
        'student_max_prob': max_s,
        'projection_std': proj_std,
        'encoder_feat_std': feat_std,
        'log_dim': logK,
        'mask_changed_frac': n_masked / float(max(n_detected, 1)),
        'num_detected_peaks': float(n_detected),
    }


# ===========================================================================
# Spectral Augmentation (student only)
# ===========================================================================

class SpectralAugmentation:
    """Augment student input: peak shift, intensity scaling, noise peaks."""

    def __init__(self, shift_max=1.5, scale_range=(0.7, 1.3),
                 noise_peaks=3, noise_height=0.02, noise_width=5):
        self.shift_max = shift_max
        self.scale_range = scale_range
        self.noise_peaks = noise_peaks
        self.noise_height = noise_height
        self.noise_width = noise_width

    def __call__(self, spectra):
        device = None
        if isinstance(spectra, torch.Tensor):
            device = spectra.device
            x = spectra.cpu().numpy()
        else:
            x = spectra

        B, L = x.shape
        aug = x.copy()

        for i in range(B):
            spec = aug[i]
            # 1. Peak shift ±shift_max cm⁻¹ (shift entire spectrum)
            shift = int(np.random.uniform(-self.shift_max, self.shift_max))
            if shift != 0:
                spec = np.roll(spec, shift)
                if shift > 0:
                    spec[:shift] = 0
                else:
                    spec[shift:] = 0

            # 2. Intensity scaling 0.7~1.3
            scale = np.random.uniform(*self.scale_range)
            spec = spec * scale

            # 3. Add random noise peaks
            for _ in range(self.noise_peaks):
                pos = np.random.randint(0, L)
                h = np.random.uniform(0, self.noise_height) * spec.max()
                lo = max(0, pos - self.noise_width)
                hi = min(L, pos + self.noise_width + 1)
                gauss = h * np.exp(-0.5 * ((np.arange(lo, hi) - pos) / 2) ** 2)
                spec[lo:hi] += gauss

            aug[i] = spec

        if device is not None:
            aug = torch.from_numpy(aug).float().to(device)
        return aug


# ===========================================================================
# byol Loss + Center Alignment
# ===========================================================================

class byolLoss(nn.Module):
    """byol loss with center alignment.

    L_byol = -sum( P_t * log(P_s) )
    where P_t = softmax((z_t - center) / teacher_temp)
          P_s = log_softmax(z_s / student_temp)
    """

    def __init__(self, teacher_temp=0.04, student_temp=0.1,
                 center_momentum=0.9, out_dim=2048, collapse_weight=1.0):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.collapse_weight = collapse_weight
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_proj, teacher_proj):
        """byol loss + pairwise cosine-similarity anti-collapse penalty.

        L = CE(P_t, P_s) + λ * mean_{i≠j} (z_t[i] · z_t[j])²

        When all teacher projections collapse to the same vector,
        all dot-products = 1 and the penalty is maximal.
        The gradient pushes each projection away from every other.
        """
        # --- CE distillation ---
        teacher_out = (teacher_proj - self.center) / self.teacher_temp
        student_out = student_proj / self.student_temp
        loss_ce = -(F.softmax(teacher_out, dim=-1)
                    * F.log_softmax(student_out, dim=-1)).sum(dim=-1).mean()

        # --- Anti-collapse: penalise pairwise similarity ---
        # teacher_proj is (B, K), L2-normalised → cosine similarity matrix
        sim = teacher_proj @ teacher_proj.T          # (B, B), diag = 1
        mask = 1.0 - torch.eye(sim.size(0), device=sim.device)
        loss_collapse = ((sim * mask) ** 2).sum() / mask.sum()

        loss = loss_ce + self.collapse_weight * loss_collapse

        # --- Update center ---
        with torch.no_grad():
            batch_center = teacher_proj.mean(dim=0, keepdim=True)
            self.center = (self.center_momentum * self.center +
                           (1.0 - self.center_momentum) * batch_center)

        return loss


# ===========================================================================
# EMA update
# ===========================================================================

@torch.no_grad()
def _ema_update(student, teacher, momentum):
    """teacher = momentum * teacher + (1-momentum) * student"""
    for p_s, p_t in zip(student.parameters(), teacher.parameters()):
        p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)


def _cosine_momentum(step, total_steps, base=0.996):
    """Cosine schedule for EMA momentum: base → 1.0."""
    if total_steps <= 1:
        return 1.0
    return 1.0 - (1.0 - base) * (1.0 + math.cos(
        math.pi * step / total_steps)) / 2.0


def _variance_loss(z, gamma=0.04, eps=1e-4):
    """VICReg-style anti-collapse loss for L2-normalized projections."""
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return F.relu(gamma - std).mean()


# ===========================================================================
# Data loading for byol (no labels needed for Stage 1)
# ===========================================================================

def _load_byol_data(data_dir, conc_threshold=None,
                    mix_only=False, present_conc_range=None,
                    singleton_sample_folds=False):
    """Load and preprocess all spectra for byol pre-training.

    byol doesn't need labels, so we only return spectra.
    """
    Raman_Shift_raw, Intensity, Concentrations, Groups, Mixtures = \
        _read_mpau_mix_spectra(data_dir)

    total_conc = Concentrations.sum(axis=1)
    if conc_threshold is not None and isinstance(conc_threshold, (int, float)):
        keep = total_conc <= conc_threshold
        Intensity = Intensity[keep]
        Concentrations = Concentrations[keep]
        Groups = Groups[keep]
        Mixtures = Mixtures[keep]

    Intensity, Concentrations, Groups, Mixtures = _filter_mix_conc(
        Intensity, Concentrations, Groups, Mixtures,
        mix_only=mix_only, present_conc_range=present_conc_range)

    # Build group table for fold assignment (used in Stage 2)
    group_ids = np.array([
        _umx_make_group_id(Mixtures[i], Concentrations[i, 0],
                           Concentrations[i, 1], Concentrations[i, 2])
        for i in range(len(Mixtures))
    ])
    table_data = {
        "group_id": group_ids, "mixture": Mixtures,
        "conc_DA": Concentrations[:, 0],
        "conc_E": Concentrations[:, 1],
        "conc_NE": Concentrations[:, 2],
    }
    group_table = (pd.DataFrame(table_data)
                   .drop_duplicates("group_id").reset_index(drop=True))
    folds = _umx_group_folds(group_table, n_splits=UNMIX_N_OUTER,
                              random_state=UNMIX_RANDOM_STATE)
    fold_lookup = dict(zip(folds["group_id"], folds["fold"]))
    df_all = pd.DataFrame(table_data)
    df_all["outer_fold"] = df_all["group_id"].map(fold_lookup).astype(int)

    if singleton_sample_folds:
        mix_group_counts = group_table.groupby("mixture")["group_id"].nunique()
        singleton_mixes = mix_group_counts[mix_group_counts == 1].index.tolist()
        if singleton_mixes:
            rng = np.random.default_rng(UNMIX_RANDOM_STATE)
            print("  Classification fold fallback for singleton groups: "
                  f"{singleton_mixes}")
            for mix in singleton_mixes:
                idx = df_all.index[df_all["mixture"] == mix].to_numpy()
                idx = rng.permutation(idx)
                for k, row_idx in enumerate(idx):
                    df_all.loc[row_idx, "outer_fold"] = k % UNMIX_N_OUTER

    # Binary labels for Stage 2: [DA, E, NE] ∈ {0,1}³
    Y_cls = (Concentrations > 0).astype(np.float32)  # (N, 3)

    print(f"  Loaded {Intensity.shape[0]} spectra, {group_table.shape[0]} groups, "
          f"{Intensity.shape[1]} features")
    for mix in np.unique(Mixtures):
        print(f"    {mix}: {(Mixtures == mix).sum()} spectra")

    return (Raman_Shift_raw.astype(np.float32),
            Intensity.astype(np.float32),
            Y_cls,
            Concentrations, Mixtures, df_all)


# ===========================================================================
# Stage 1: byol Pre-training
# ===========================================================================

def _stage1_byol_pretrain(RawIntensity, config, model_dir, device,
                            norm_mode="minmax", dataset="MPAU",
                            re_training=False, plot=True,
                            raman_shift=None):
    """BYOL self-supervised pre-training.

    Online: peak-masked → encoder → projector → predictor
    Target: clean → encoder → projector (EMA, no predictor)
    Loss: 2 − cos(q_online, z_target), symmetric both ways.
    """
    cfg = config
    proj_hidden = cfg["proj_hidden"]
    proj_out = cfg["proj_out"]

    print("\n" + "=" * 60)
    print("Stage 1: BYOL Self-Supervised Pre-training")
    print("=" * 60)
    print(f"  Online: masked+aug → encoder → proj({proj_hidden}/{proj_out}) → predictor")
    print(f"  Target: clean → encoder → proj (EMA, no predictor)")
    print(f"  Loss: symmetric cosine distance")
    print(f"  Epochs: {cfg['stage1_epochs']}, "
          f"Batch: {cfg['stage1_batch_size']}, LR: {cfg['stage1_lr']}")

    # --- Build models ---
    online = BYOLModel(config, has_predictor=True).to(device)
    target = BYOLModel(config, has_predictor=False).to(device)
    target.load_state_dict(online.state_dict(), strict=False)
    for p in target.parameters():
        p.requires_grad = False
    target.eval()

    print(f"  Online params: {sum(p.numel() for p in online.parameters()):,}")
    print(f"  Target params: {sum(p.numel() for p in target.parameters()):,}")

    pca_distill_weight = float(cfg.get("stage1_pca_distill_weight", 0.0))
    pca_distill_components = int(
        cfg.get("stage1_pca_distill_components", 30))
    pca_head = None
    if pca_distill_weight > 0:
        pca_head = nn.Linear(cfg["out_dim"], pca_distill_components).to(device)
        print("  PCA distillation: "
              f"{pca_distill_components} components, "
              f"weight={pca_distill_weight}")

    # --- Optimizer, masking ---
    opt_params = list(online.parameters())
    if pca_head is not None:
        opt_params += list(pca_head.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=cfg["stage1_lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["stage1_epochs"])
    masker = PeakMasking(
        sigma_baseline=cfg["sigma_baseline"],
        weak_fraction=cfg.get("mask_weak_fraction", 0.30),
        mask_ratio_min=cfg["mask_ratio_min"],
        mask_ratio_max=cfg["mask_ratio_max"],
        expand_min=cfg.get("mask_expand_min", 0),
        expand_max=cfg.get("mask_expand_max", 0),
        span_mask_prob=cfg.get("span_mask_prob", 0.0),
        span_mask_count_min=cfg.get("span_mask_count_min", 1),
        span_mask_count_max=cfg.get("span_mask_count_max", 1),
        span_mask_width_min=cfg.get("span_mask_width_min", 12),
        span_mask_width_max=cfg.get("span_mask_width_max", 36),
    )
    augmenter = SpectralAugmentation()

    # --- Checkpoint ---
    ckpt_path = os.path.join(model_dir, f"byol_stage1_{dataset}_{norm_mode}_checkpoint.pt")
    start_epoch = 0
    global_step = 0
    if not re_training and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        online.load_state_dict(ckpt['online_state_dict'])
        target.load_state_dict(ckpt['target_state_dict'])
        if pca_head is not None and 'pca_head_state_dict' in ckpt:
            pca_head.load_state_dict(ckpt['pca_head_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step']
        print(f"  Resumed from checkpoint at epoch {start_epoch}")

    # --- Pre-normalize clean data (target) ---
    Raw_t = torch.from_numpy(RawIntensity).float()
    rmin = Raw_t.min(dim=1, keepdim=True).values
    rmax = Raw_t.max(dim=1, keepdim=True).values
    if norm_mode == "peak":
        if raman_shift is None:
            raise ValueError(
                "raman_shift is required when norm_mode='peak'")
        # Peak normalize at 920 cm⁻¹
        X_clean_np = spectra_normalization(
            raman_shift, RawIntensity,
            peak_position=920, peak_range=20,
            plot=False, mode='byol', minmax_scale=False)
        X_clean = torch.from_numpy(X_clean_np).float()
    else:
        X_clean = (Raw_t - rmin) / (rmax - rmin + 1e-8)

    pca_targets = None
    if pca_head is not None:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        X_clean_np = X_clean.cpu().numpy()
        X_scaled = StandardScaler().fit_transform(X_clean_np)
        pca = PCA(n_components=pca_distill_components, random_state=0)
        pca_scores = pca.fit_transform(X_scaled).astype(np.float32)
        pca_scores = (
            pca_scores - pca_scores.mean(axis=0, keepdims=True)
        ) / (pca_scores.std(axis=0, keepdims=True) + 1e-6)
        pca_targets = torch.from_numpy(pca_scores).float()
        print("  PCA teacher explained variance "
              f"({pca_distill_components} comps): "
              f"{pca.explained_variance_ratio_.sum():.3f}")

    # --- Training loop ---
    n_samples = RawIntensity.shape[0]
    batch_size = cfg["stage1_batch_size"]
    total_steps = cfg["stage1_epochs"] * max(1, n_samples // batch_size)
    ema_base = cfg["ema_momentum_base"]

    # --- Logging ---
    _ensure_byol_output_dirs()
    log_name = f"logs/byol_stage1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_f = open(log_name, 'w', newline='')
    log_w = csv.writer(log_f)
    log_w.writerow(['epoch', 'loss', 'cos_sim', 'proj_std', 'lr',
                     'elapsed_min', 'ETA_total_min'])
    t_start = time.time()

    print(f"\n  Training {cfg['stage1_epochs']} epochs on {n_samples} spectra ...")
    for epoch in range(start_epoch, cfg["stage1_epochs"]):
        online.train()
        epoch_loss, epoch_cos, n = 0.0, 0.0, 0
        perm = torch.randperm(n_samples)
        for start in range(0, n_samples, batch_size):
            batch_idx = perm[start:start + batch_size]
            raw_batch = Raw_t[batch_idx].to(device)
            clean_batch = X_clean[batch_idx].to(device)

            # View 1 (online): raw → augment → mask → normalize
            raw_aug = augmenter(raw_batch)
            raw_masked = masker(raw_aug)
            smin = raw_masked.min(dim=1, keepdim=True).values
            smax = raw_masked.max(dim=1, keepdim=True).values
            v1 = (raw_masked - smin) / (smax - smin + 1e-8)

            # View 2 (target): pre-normalized clean
            v2 = clean_batch

            # Symmetric BYOL: q=online_pred, z=target_proj.
            # Keep explicit online projections so a variance penalty can
            # prevent the normalized projector from collapsing to one vector.
            feat_o1 = online.encoder(v1)
            feat_o2 = online.encoder(v2)
            z_o1 = online.projector(feat_o1)
            z_o2 = online.projector(feat_o2)
            q1 = online.predictor(z_o1)
            q2 = online.predictor(z_o2)
            with torch.no_grad():
                z1 = target(v1)
                z2 = target(v2)

            byol_loss = (2.0 - F.cosine_similarity(q1, z2).mean()
                         - F.cosine_similarity(q2, z1).mean())
            var_loss = (
                _variance_loss(z_o1, gamma=cfg["stage1_var_loss_gamma"])
                + _variance_loss(z_o2, gamma=cfg["stage1_var_loss_gamma"])
            ) * 0.5
            loss = byol_loss + cfg["stage1_var_loss_weight"] * var_loss
            if pca_head is not None and pca_targets is not None:
                pca_batch = pca_targets[batch_idx].to(device)
                pca_pred1 = pca_head(feat_o1)
                pca_pred2 = pca_head(feat_o2)
                pca_loss = 0.5 * (
                    F.smooth_l1_loss(pca_pred1, pca_batch)
                    + F.smooth_l1_loss(pca_pred2, pca_batch)
                )
                loss = loss + pca_distill_weight * pca_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update target
            m = _cosine_momentum(global_step, total_steps, base=ema_base)
            for p_o, p_t in zip(online.parameters(), target.parameters()):
                if p_t.requires_grad:
                    continue
                p_t.data.mul_(m).add_(p_o.data, alpha=1.0 - m)
            global_step += 1

            epoch_loss += loss.item() * batch_idx.size(0)
            epoch_cos += (2.0 - byol_loss.item()) * batch_idx.size(0)
            n += batch_idx.size(0)

        scheduler.step()
        loss_avg = epoch_loss / n
        cos_avg = epoch_cos / n

        # ---- Diagnostics ----
        with torch.no_grad():
            z_sample = target(X_clean[:min(64, n_samples)].to(device))
            proj_std = z_sample.std(dim=0).mean().item()

        # ---- Print ----
        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = (time.time() - t_start) / 60
            progress = (epoch + 1 - start_epoch) / max(cfg['stage1_epochs'] - start_epoch, 1)
            eta = elapsed / max(progress, 0.001)
            print(f"  Epoch {epoch+1:3d}/{cfg['stage1_epochs']}: "
                  f"loss={loss_avg:.4f}, cos={cos_avg:.4f}, "
                  f"proj_std={proj_std:.4f}, "
                  f"elapsed={elapsed:.1f}m, ETA={eta:.1f}m")

        # ---- Log ----
        elapsed = (time.time() - t_start) / 60
        progress = (epoch + 1 - start_epoch) / max(cfg['stage1_epochs'] - start_epoch, 1)
        eta = elapsed / max(progress, 0.001)
        log_w.writerow([epoch + 1, f"{loss_avg:.6f}", f"{cos_avg:.6f}",
                         f"{proj_std:.6f}",
                         f"{scheduler.get_last_lr()[0]:.2e}",
                         f"{elapsed:.2f}", f"{eta:.2f}"])
        log_f.flush()

        # ---- Checkpoint ----
        if (epoch + 1) % 20 == 0:
            ckpt_payload = {
                'epoch': epoch,
                'global_step': global_step,
                'online_state_dict': online.state_dict(),
                'target_state_dict': target.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            if pca_head is not None:
                ckpt_payload['pca_head_state_dict'] = pca_head.state_dict()
            torch.save(ckpt_payload, ckpt_path)
            print(f"    Checkpoint saved to {ckpt_path}")

    log_f.close()
    print(f"  Log saved to {log_name}")

    # --- Save online encoder ---
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, f"byol_stage1_{dataset}_{norm_mode}.pt")
    torch.save({'encoder_state_dict': online.encoder.state_dict(),
                'config': {k: v for k, v in cfg.items()}}, save_path)
    print(f"\n  Online encoder saved to {save_path}")

    return online.encoder, save_path


# ===========================================================================
# Classification Head
# ===========================================================================

class ClassificationHead(nn.Module):
    """256 → 128 → 3 (logits for DA/E/NE, binary BCE loss)."""

    def __init__(self, in_dim=256, num_classes=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc(x)  # raw logits, BCEWithLogitsLoss handles sigmoid


# ===========================================================================
# Stage 2: Classification Fine-tuning
# ===========================================================================

def _stage2_finetune(RawIntensity, Y_cls, df_all, encoder_path, config, model_dir,
                      device, dataset='MPAU', plot=True):
    """Classification fine-tuning with progressive unfreezing.

    First frozen_epochs: freeze encoder, train classification head only.
    Then full_epochs: unfreeze all, fine-tune end-to-end.
    Uses 3-fold stratified group CV. Encoder is reloaded from scratch
    for each fold to prevent cross-fold leakage.
    """
    cfg = config
    num_classes = cfg["num_classes"]

    # Pre-normalize all data
    X_all_t = torch.from_numpy(RawIntensity).float()
    min_all = X_all_t.min(dim=1, keepdim=True).values
    max_all = X_all_t.max(dim=1, keepdim=True).values
    X_norm = (X_all_t - min_all) / (max_all - min_all + 1e-8)

    print("\n" + "=" * 60)
    print("Stage 2: Classification Fine-tuning")
    print("=" * 60)
    print(f"  Frozen phase: {cfg['stage2_frozen_epochs']} epochs")
    print(f"  Full phase: {cfg['stage2_full_epochs']} epochs")
    print(f"  Task: 3 binary classifications (DA/E/NE)")

    # 3-fold OOF evaluation
    N = len(RawIntensity)
    all_preds = np.zeros((N, num_classes), dtype=int)
    all_probs = np.zeros((N, num_classes), dtype=np.float32)
    all_true = Y_cls.astype(int) if Y_cls.dtype != int else Y_cls

    for fold in range(UNMIX_N_OUTER):
        # --- Reload clean encoder for each fold (prevents OOF leakage) ---
        encoder = BYOLEncoder(
            in_channels=cfg["in_channels"], base_channels=cfg["base_channels"],
            out_dim=cfg["out_dim"], nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["transformer_dropout"],
        ).to(device)
        ckpt = torch.load(encoder_path, map_location=device, weights_only=True)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        for p in encoder.parameters():
            p.requires_grad = False

        test_mask = df_all["outer_fold"].to_numpy() == fold
        train_mask = ~test_mask

        X_tr = X_norm[train_mask]
        Y_tr = torch.from_numpy(np.array(Y_cls)[train_mask]).float()
        X_te = X_norm[test_mask]

        print(f"\n  --- Fold {fold+1}/{UNMIX_N_OUTER} ---")
        print(f"  Train: {train_mask.sum()} spectra, "
              f"Test: {test_mask.sum()} spectra")

        # Build classification model
        head = ClassificationHead(
            in_dim=cfg["out_dim"], num_classes=num_classes).to(device)

        # --- Phase 1: frozen encoder ---
        print(f"  Phase 1: frozen encoder ({cfg['stage2_frozen_epochs']} ep)")
        optimizer = torch.optim.Adam(head.parameters(),
                                     lr=cfg["stage2_lr"])
        criterion = nn.BCEWithLogitsLoss()
        loader = DataLoader(TensorDataset(X_tr, Y_tr),
                            batch_size=cfg["stage2_batch_size"], shuffle=True)

        for epoch in range(cfg["stage2_frozen_epochs"]):
            head.train()
            epoch_loss, n = 0.0, 0
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                with torch.no_grad():
                    feats = encoder(bx)
                pred = head(feats)
                loss = criterion(pred, by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * bx.size(0)
                n += bx.size(0)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1:2d}: loss={epoch_loss/n:.4f}")

        # --- Phase 2: unfreeze all ---
        print(f"  Phase 2: unfreeze all ({cfg['stage2_full_epochs']} ep)")
        for p in encoder.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(head.parameters()),
            lr=cfg["stage2_lr"] * 0.1)  # lower LR for fine-tuning
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["stage2_full_epochs"])

        for epoch in range(cfg["stage2_full_epochs"]):
            head.train()
            encoder.train()
            epoch_loss, n = 0.0, 0
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                pred = head(encoder(bx))
                loss = criterion(pred, by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * bx.size(0)
                n += bx.size(0)
            scheduler.step()
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1:2d}: loss={epoch_loss/n:.4f}")

        # --- Evaluate ---
        head.eval()
        encoder.eval()
        with torch.no_grad():
            feats_te = encoder(X_te.to(device))
            logits = head(feats_te)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
        all_preds[test_mask] = preds.cpu().numpy()
        all_probs[test_mask] = probs.cpu().numpy()

    # --- Final metrics ---
    from sklearn.metrics import accuracy_score, confusion_matrix
    print(f"\n  --- OOF Classification Results ---")
    acc_per = []
    for j, a in enumerate(UNMIX_ANALYTES):
        acc = accuracy_score(all_true[:, j], all_preds[:, j])
        acc_per.append(acc)
        print(f"  {a}: Acc={acc:.4f}")
    exact = (all_preds == all_true).all(axis=1).mean()
    print(f"  Exact-match (3-bit): {exact:.4f}")

    # --- Save ---
    os.makedirs(model_dir, exist_ok=True)
    fname = f"byol_stage2_classifier_{dataset}.pt"
    torch.save({'encoder_state_dict': encoder.state_dict(),
                'head_state_dict': head.state_dict()},
               os.path.join(model_dir, fname))

    # --- 8-class conversion (shared by plot and CSV) ---
    mixture_labels = ["BA", "DA", "E", "NE",
                      "DA+E", "DA+NE", "E+NE", "DA+E+NE"]

    def _bits_to_mixture(bits):
        result = []
        for b in bits:
            parts = []
            if b[0]: parts.append("DA")
            if b[1]: parts.append("E")
            if b[2]: parts.append("NE")
            result.append("BA" if not parts else "+".join(parts))
        return np.array(result)

    pred_mix = _bits_to_mixture(all_preds)
    true_mix = _bits_to_mixture(all_true)

    # Group-level aggregation
    df_plot = df_all.copy()
    for j, a in enumerate(UNMIX_ANALYTES):
        df_plot[f"pred_{a}"] = all_preds[:, j]
        df_plot[f"p_{a}"] = all_probs[:, j]
    grp_agg8 = df_plot.groupby("group_id").agg(
        **{f"p_{a}": (f"p_{a}", "mean") for a in UNMIX_ANALYTES},
        mixture=("mixture", "first"),
    )
    grp_bits = np.column_stack([
        (grp_agg8[f"p_{a}"].to_numpy() > 0.5).astype(int)
        for a in UNMIX_ANALYTES
    ])
    grp_pred_mix = _bits_to_mixture(grp_bits)
    grp_true_mix = grp_agg8["mixture"].to_numpy()

    # --- Plot: spectra-level + group-level per-component confusion ---
    if plot:
        grp_agg = df_plot.groupby("group_id").agg(
            **{f"true_{a}": (f"conc_{a}", lambda x: (x.iloc[0] > 0).astype(int))
                for a in UNMIX_ANALYTES},
            **{f"p_{a}": (f"p_{a}", "mean") for a in UNMIX_ANALYTES},
        )
        grp_preds = np.column_stack([
            (grp_agg[f"p_{a}"].to_numpy() > 0.5).astype(int)
            for a in UNMIX_ANALYTES
        ])
        grp_true = np.column_stack([
            grp_agg[f"true_{a}"].to_numpy() for a in UNMIX_ANALYTES
        ])

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        for j, a in enumerate(UNMIX_ANALYTES):
            # Spectra-level
            cm_s = confusion_matrix(all_true[:, j], all_preds[:, j])
            im = axes[0, j].imshow(cm_s, cmap='Blues')
            axes[0, j].set_xticks([0, 1]); axes[0, j].set_yticks([0, 1])
            axes[0, j].set_xticklabels(["Absent", "Present"])
            axes[0, j].set_yticklabels(["Absent", "Present"])
            axes[0, j].set_xlabel("Pred"); axes[0, j].set_ylabel("True")
            for r in range(2):
                for c in range(2):
                    axes[0, j].text(c, r, str(cm_s[r, c]), ha='center', va='center')
            axes[0, j].set_title(f"{a} spectra (acc={acc_per[j]:.3f})")

            # Group-level
            cm_g = confusion_matrix(grp_true[:, j], grp_preds[:, j])
            im2 = axes[1, j].imshow(cm_g, cmap='Blues')
            axes[1, j].set_xticks([0, 1]); axes[1, j].set_yticks([0, 1])
            axes[1, j].set_xticklabels(["Absent", "Present"])
            axes[1, j].set_yticklabels(["Absent", "Present"])
            axes[1, j].set_xlabel("Pred"); axes[1, j].set_ylabel("True")
            for r in range(2):
                for c in range(2):
                    axes[1, j].text(c, r, str(cm_g[r, c]), ha='center', va='center')
            gacc = accuracy_score(grp_true[:, j], grp_preds[:, j])
            axes[1, j].set_title(f"{a} group (acc={gacc:.3f})")
        axes[0, 0].set_ylabel("Spectra-level")
        axes[1, 0].set_ylabel("Group-level")
        fig.suptitle("BYOL + Fine-tune OOF (3 binary classifications)",
                     fontsize=14)
        fig.tight_layout()
        _ensure_byol_output_dirs()
        fig.savefig("visualizations/BYOL_Stage2_Confusion.png", dpi=300)
        plt.close(fig)

        # --- 8-class confusion (spectra + group) ---
        mixture_labels = ["BA", "DA", "E", "NE",
                          "DA+E", "DA+NE", "E+NE", "DA+E+NE"]
        present_s8 = sorted(set(true_mix) | set(pred_mix),
                            key=lambda x: mixture_labels.index(x)
                            if x in mixture_labels else 99)
        cm_s8 = confusion_matrix(true_mix, pred_mix, labels=present_s8)

        present_g8 = sorted(set(grp_true_mix) | set(grp_pred_mix),
                            key=lambda x: mixture_labels.index(x)
                            if x in mixture_labels else 99)
        cm_g8 = confusion_matrix(grp_true_mix, grp_pred_mix, labels=present_g8)
        acc_s8 = accuracy_score(true_mix, pred_mix)
        acc_g8 = accuracy_score(grp_true_mix, grp_pred_mix)

        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        for ax, cm, lbls, title, acc_val in [
            (ax1, cm_s8, present_s8, "Spectra-level", acc_s8),
            (ax2, cm_g8, present_g8, "Group-level", acc_g8),
        ]:
            im = ax.imshow(cm, cmap='Blues')
            ax.set_xticks(range(len(lbls)))
            ax.set_yticks(range(len(lbls)))
            ax.set_xticklabels(lbls, rotation=45, ha='right')
            ax.set_yticklabels(lbls)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            for i in range(len(lbls)):
                for j in range(len(lbls)):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            fontsize=14 if len(lbls) <= 6 else 9)
            ax.set_title(f"{title} (acc={acc_val:.3f})")
        fig2.suptitle("BYOL + Fine-tune OOF (8-class from 3-bit)", fontsize=14)
        fig2.tight_layout()
        fig2.savefig("visualizations/BYOL_Stage2_Confusion_8class.png", dpi=300)
        plt.close(fig2)

    # --- Export CSVs ---
    # Spectra-level
    df_s = df_all[["group_id", "mixture"]].copy()
    for j, a in enumerate(UNMIX_ANALYTES):
        df_s[f"true_{a}"] = all_true[:, j]
        df_s[f"pred_{a}"] = all_preds[:, j]
    df_s["true_8class"] = true_mix
    df_s["pred_8class"] = pred_mix
    df_s.rename(columns={"group_id": "group_name"}, inplace=True)
    _ensure_byol_output_dirs()
    df_s.to_csv("reports/BYOL_Class_Spectra.csv", index=False, encoding="utf-8-sig")
    print("  Exported reports/BYOL_Class_Spectra.csv")

    # Group-level
    df_g = grp_agg8[["mixture"]].copy().reset_index()
    df_g.rename(columns={"group_id": "group_name"}, inplace=True)
    grp_true_bits_csv = np.zeros_like(grp_bits)
    for i, label in enumerate(grp_true_mix):
        grp_true_bits_csv[i, 0] = int("DA" in label)
        grp_true_bits_csv[i, 1] = int(label in ("E", "DA+E", "E+NE", "DA+E+NE"))
        grp_true_bits_csv[i, 2] = int("NE" in label)
    for j, a in enumerate(UNMIX_ANALYTES):
        df_g[f"true_{a}"] = grp_true_bits_csv[:, j]
        df_g[f"pred_{a}"] = grp_bits[:, j]
    df_g["true_8class"] = grp_true_mix
    df_g["pred_8class"] = grp_pred_mix
    df_g.to_csv("reports/BYOL_Class_Group.csv", index=False, encoding="utf-8-sig")
    print("  Exported reports/BYOL_Class_Group.csv")

    return all_preds, all_probs, all_true


def _stage2_finetune_8class(RawIntensity, df_all, encoder_path, config,
                            model_dir, device, dataset='MPAU', plot=True):
    """Direct 8-class BYOL fine-tuning for BA/single/binary/ternary mixtures."""
    cfg = config
    num_classes = len(BYOL_8CLASS_LABELS)
    label_to_idx = {m: i for i, m in enumerate(BYOL_8CLASS_LABELS)}
    idx_to_label = {i: m for m, i in label_to_idx.items()}

    X_all_t = torch.from_numpy(RawIntensity).float()
    min_all = X_all_t.min(dim=1, keepdim=True).values
    max_all = X_all_t.max(dim=1, keepdim=True).values
    X_norm = (X_all_t - min_all) / (max_all - min_all + 1e-8)

    mapped = df_all["mixture"].map(label_to_idx)
    if mapped.isna().any():
        bad = sorted(set(df_all["mixture"]) - set(BYOL_8CLASS_LABELS))
        raise ValueError(f"Unknown mixture labels for 8-class task: {bad}")
    y_all = mapped.to_numpy(dtype=np.int64)

    print("\n" + "=" * 60)
    print("Stage 2: Direct 8-class Classification Fine-tuning")
    print("=" * 60)
    print(f"  Frozen phase: {cfg['stage2_frozen_epochs']} epochs")
    print(f"  Full phase: {cfg['stage2_full_epochs']} epochs")
    print("  Task: BA/DA/E/NE/DA+E/DA+NE/E+NE/DA+E+NE softmax")

    N = len(RawIntensity)
    all_preds = np.zeros(N, dtype=np.int64)
    all_probs = np.zeros((N, num_classes), dtype=np.float32)

    for fold in range(UNMIX_N_OUTER):
        encoder = BYOLEncoder(
            in_channels=cfg["in_channels"], base_channels=cfg["base_channels"],
            out_dim=cfg["out_dim"], nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["transformer_dropout"],
        ).to(device)
        ckpt = torch.load(encoder_path, map_location=device, weights_only=True)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        for p in encoder.parameters():
            p.requires_grad = False

        test_mask = df_all["outer_fold"].to_numpy() == fold
        train_mask = ~test_mask

        X_tr = X_norm[train_mask]
        y_tr_np = y_all[train_mask]
        Y_tr = torch.from_numpy(y_tr_np).long()
        X_te = X_norm[test_mask]

        print(f"\n  --- Fold {fold+1}/{UNMIX_N_OUTER} ---")
        print(f"  Train: {train_mask.sum()} spectra, "
              f"Test: {test_mask.sum()} spectra")

        head = ClassificationHead(
            in_dim=cfg["out_dim"], num_classes=num_classes).to(device)

        counts = np.bincount(y_tr_np, minlength=num_classes).astype(float)
        safe_counts = np.maximum(counts, 1.0)
        class_weights = np.sqrt(len(y_tr_np) / (num_classes * safe_counts))
        class_weights = class_weights / class_weights.mean()
        class_weights_t = torch.tensor(
            class_weights, dtype=torch.float32, device=device)

        loader = DataLoader(
            TensorDataset(X_tr, Y_tr),
            batch_size=cfg["stage2_batch_size"],
            shuffle=True)
        criterion = nn.CrossEntropyLoss(weight=class_weights_t,
                                        label_smoothing=0.03)

        print(f"  Phase 1: frozen encoder ({cfg['stage2_frozen_epochs']} ep)")
        optimizer = torch.optim.AdamW(head.parameters(),
                                      lr=cfg["stage2_lr"],
                                      weight_decay=1e-4)
        for epoch in range(cfg["stage2_frozen_epochs"]):
            head.train()
            encoder.eval()
            epoch_loss, n = 0.0, 0
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                with torch.no_grad():
                    feats = encoder(bx)
                logits = head(feats)
                loss = criterion(logits, by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * bx.size(0)
                n += bx.size(0)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1:2d}: loss={epoch_loss/n:.4f}")

        print(f"  Phase 2: unfreeze all ({cfg['stage2_full_epochs']} ep)")
        for p in encoder.parameters():
            p.requires_grad = True
        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(head.parameters()),
            lr=cfg["stage2_lr"] * 0.1,
            weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["stage2_full_epochs"])

        for epoch in range(cfg["stage2_full_epochs"]):
            head.train()
            encoder.train()
            epoch_loss, n = 0.0, 0
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                logits = head(encoder(bx))
                loss = criterion(logits, by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * bx.size(0)
                n += bx.size(0)
            scheduler.step()
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1:2d}: loss={epoch_loss/n:.4f}")

        head.eval()
        encoder.eval()
        with torch.no_grad():
            logits = head(encoder(X_te.to(device)))
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
        all_preds[test_mask] = preds.cpu().numpy()
        all_probs[test_mask] = probs.cpu().numpy()

    from sklearn.metrics import accuracy_score, confusion_matrix

    true_mix = np.array([idx_to_label[i] for i in y_all])
    pred_mix = np.array([idx_to_label[i] for i in all_preds])
    acc_s8 = accuracy_score(true_mix, pred_mix)
    print(f"\n  --- OOF 8-class Classification Results ---")
    print(f"  Spectra-level Acc={acc_s8:.4f}")

    df_plot = df_all.copy()
    for i, label in enumerate(BYOL_8CLASS_LABELS):
        df_plot[f"p_{label}"] = all_probs[:, i]

    prob_cols = [f"p_{label}" for label in BYOL_8CLASS_LABELS]
    grp_prob = df_plot.groupby("group_id")[prob_cols].mean()
    grp_pred_idx = grp_prob.to_numpy().argmax(axis=1)
    grp_pred_mix = np.array([idx_to_label[i] for i in grp_pred_idx])
    grp_true_mix = df_plot.groupby("group_id")["mixture"].first().to_numpy()
    acc_g8 = accuracy_score(grp_true_mix, grp_pred_mix)
    print(f"  Group-level Acc={acc_g8:.4f}")

    def _mix_to_bits(labels):
        bits = np.zeros((len(labels), 3), dtype=int)
        for i, label in enumerate(labels):
            if "DA" in label:
                bits[i, 0] = 1
            if label in ("E", "DA+E", "E+NE", "DA+E+NE"):
                bits[i, 1] = 1
            if "NE" in label:
                bits[i, 2] = 1
        return bits

    true_bits = _mix_to_bits(true_mix)
    pred_bits = _mix_to_bits(pred_mix)
    grp_true_bits = _mix_to_bits(grp_true_mix)
    grp_pred_bits = _mix_to_bits(grp_pred_mix)

    if plot:
        _ensure_byol_output_dirs()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        cm_s8 = confusion_matrix(true_mix, pred_mix, labels=BYOL_8CLASS_LABELS)
        cm_g8 = confusion_matrix(grp_true_mix, grp_pred_mix,
                                 labels=BYOL_8CLASS_LABELS)
        for ax, cm, title, acc_val in [
            (ax1, cm_s8, "Spectra-level", acc_s8),
            (ax2, cm_g8, "Group-level", acc_g8),
        ]:
            ax.imshow(cm, cmap='Blues')
            ax.set_xticks(range(num_classes))
            ax.set_yticks(range(num_classes))
            ax.set_xticklabels(BYOL_8CLASS_LABELS, rotation=45, ha='right')
            ax.set_yticklabels(BYOL_8CLASS_LABELS)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for i in range(num_classes):
                for j in range(num_classes):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            fontsize=9)
            ax.set_title(f"{title} (acc={acc_val:.3f})")
        fig.suptitle("BYOL + Fine-tune OOF (direct 8-class)", fontsize=14)
        fig.tight_layout()
        fig.savefig("visualizations/BYOL_Stage2_Confusion_8class.png", dpi=300)
        plt.close(fig)

        fig2, axes = plt.subplots(2, 3, figsize=(16, 10))
        for j, a in enumerate(UNMIX_ANALYTES):
            cm_s = confusion_matrix(true_bits[:, j], pred_bits[:, j],
                                    labels=[0, 1])
            axes[0, j].imshow(cm_s, cmap='Blues')
            axes[0, j].set_xticks([0, 1]); axes[0, j].set_yticks([0, 1])
            axes[0, j].set_xticklabels(["Absent", "Present"])
            axes[0, j].set_yticklabels(["Absent", "Present"])
            axes[0, j].set_xlabel("Pred"); axes[0, j].set_ylabel("True")
            for r in range(2):
                for c in range(2):
                    axes[0, j].text(c, r, str(cm_s[r, c]),
                                    ha='center', va='center')
            acc_s = accuracy_score(true_bits[:, j], pred_bits[:, j])
            axes[0, j].set_title(f"{a} spectra (acc={acc_s:.3f})")

            cm_g = confusion_matrix(grp_true_bits[:, j], grp_pred_bits[:, j],
                                    labels=[0, 1])
            axes[1, j].imshow(cm_g, cmap='Blues')
            axes[1, j].set_xticks([0, 1]); axes[1, j].set_yticks([0, 1])
            axes[1, j].set_xticklabels(["Absent", "Present"])
            axes[1, j].set_yticklabels(["Absent", "Present"])
            axes[1, j].set_xlabel("Pred"); axes[1, j].set_ylabel("True")
            for r in range(2):
                for c in range(2):
                    axes[1, j].text(c, r, str(cm_g[r, c]),
                                    ha='center', va='center')
            acc_g = accuracy_score(grp_true_bits[:, j], grp_pred_bits[:, j])
            axes[1, j].set_title(f"{a} group (acc={acc_g:.3f})")
        axes[0, 0].set_ylabel("Spectra-level")
        axes[1, 0].set_ylabel("Group-level")
        fig2.suptitle("BYOL + Fine-tune OOF (derived 3-bit from direct 8-class)",
                      fontsize=14)
        fig2.tight_layout()
        fig2.savefig("visualizations/BYOL_Stage2_Confusion.png", dpi=300)
        plt.close(fig2)

    _ensure_byol_output_dirs()
    df_s = df_all[["group_id", "mixture"]].copy()
    for j, a in enumerate(UNMIX_ANALYTES):
        df_s[f"true_{a}"] = true_bits[:, j]
        df_s[f"pred_{a}"] = pred_bits[:, j]
    df_s["true_8class"] = true_mix
    df_s["pred_8class"] = pred_mix
    for i, label in enumerate(BYOL_8CLASS_LABELS):
        df_s[f"p_{label}"] = all_probs[:, i]
    df_s.rename(columns={"group_id": "group_name"}, inplace=True)
    df_s.to_csv("reports/BYOL_Class_Spectra.csv", index=False,
                encoding="utf-8-sig")
    print("  Exported reports/BYOL_Class_Spectra.csv")

    df_g = pd.DataFrame({
        "group_name": grp_prob.index,
        "mixture": grp_true_mix,
    })
    for j, a in enumerate(UNMIX_ANALYTES):
        df_g[f"true_{a}"] = grp_true_bits[:, j]
        df_g[f"pred_{a}"] = grp_pred_bits[:, j]
    df_g["true_8class"] = grp_true_mix
    df_g["pred_8class"] = grp_pred_mix
    for i, label in enumerate(BYOL_8CLASS_LABELS):
        df_g[f"p_{label}"] = grp_prob.iloc[:, i].to_numpy()
    df_g.to_csv("reports/BYOL_Class_Group.csv", index=False,
                encoding="utf-8-sig")
    print("  Exported reports/BYOL_Class_Group.csv")

    fname = f"byol_stage2_classifier_{dataset}.pt"
    payload = {'encoder_state_dict': encoder.state_dict(),
               'head_state_dict': head.state_dict(),
               'class_labels': BYOL_8CLASS_LABELS,
               'task': 'direct_8class'}
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, fname)
    try:
        torch.save(payload, save_path)
    except (PermissionError, RuntimeError):
        save_path = os.path.join("model_output", fname)
        torch.save(payload, save_path)
        print(f"  WARNING: Cannot write to model_dir; saved to {save_path}")

    return all_preds, all_probs, y_all


def _plot_raw_pca_pc12_grid(X_norm, df_all, ensemble_cfg):
    """Plot PC1-PC2 scatter for each raw-PCA ensemble member."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from matplotlib.colors import to_rgba

    _ensure_byol_output_dirs()
    mixtures = df_all["mixture"].to_numpy()
    total_conc = (
        df_all[["conc_DA", "conc_E", "conc_NE"]]
        .sum(axis=1)
        .to_numpy(dtype=float)
    )
    cmin, cmax = float(total_conc.min()), float(total_conc.max())
    if cmax > cmin:
        alpha = 0.25 + 0.75 * (total_conc - cmin) / (cmax - cmin)
    else:
        alpha = np.full_like(total_conc, 0.85, dtype=float)
    mix_colors = _mixture_color_map()
    point_colors = [
        to_rgba(mix_colors.get(mix, "#666666"), alpha=float(a))
        for mix, a in zip(mixtures, alpha)
    ]

    n_panels = len(ensemble_cfg)
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 10),
                             facecolor="white")
    axes = axes.ravel()

    for idx, (ax, (n_comp, c_val)) in enumerate(zip(axes, ensemble_cfg)):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_norm)
        pca = PCA(n_components=max(2, int(n_comp)), random_state=0)
        scores = pca.fit_transform(X_scaled)
        ax.scatter(scores[:, 0], scores[:, 1], c=point_colors,
                   s=14, linewidths=0)
        evr = pca.explained_variance_ratio_
        ax.set_title(
            f"PCA({n_comp}), C={c_val}\n"
            f"PC1 {evr[0]*100:.1f}%, PC2 {evr[1]*100:.1f}%")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)

    for ax in axes[n_panels:]:
        ax.axis("off")

    legend_groups = [m for m in BYOL_8CLASS_LABELS
                     if np.any(mixtures == m)]
    handles = [
        plt.Line2D([0], [0], marker='o', linestyle='',
                   markersize=7, color=mix_colors[g], label=str(g))
        for g in legend_groups
    ]
    fig.legend(handles=handles, loc="center left",
               bbox_to_anchor=(1.005, 0.56), fontsize=8,
               title="Mixture", ncol=1)

    sm = plt.cm.ScalarMappable(
        cmap="Greys",
        norm=plt.Normalize(vmin=cmin, vmax=cmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:n_panels], fraction=0.025, pad=0.02)
    cbar.set_label("Total concentration (uM)\nhigher = darker / less transparent")

    fig.suptitle("Raw PCA Ensemble: PC1-PC2 by Mixture and Total Concentration",
                 fontsize=15)
    fig.subplots_adjust(left=0.06, right=0.82, bottom=0.08, top=0.9,
                        wspace=0.28, hspace=0.32)
    out_path = "visualizations/BYOL_RawPCA_PC1_PC2_Ensemble.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  PCA PC1-PC2 plot saved to {out_path}")


def _stage2_raw_pca_8class(RawIntensity, df_all, config, model_dir,
                           dataset='MPAU', plot=True):
    """High-accuracy 8-class OOF classifier on normalized spectra."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    label_to_idx = {m: i for i, m in enumerate(BYOL_8CLASS_LABELS)}
    idx_to_label = {i: m for m, i in label_to_idx.items()}
    mapped = df_all["mixture"].map(label_to_idx)
    if mapped.isna().any():
        bad = sorted(set(df_all["mixture"]) - set(BYOL_8CLASS_LABELS))
        raise ValueError(f"Unknown mixture labels for 8-class task: {bad}")
    y_all = mapped.to_numpy(dtype=np.int64)

    X = np.asarray(RawIntensity, dtype=np.float32)
    X_norm = (X - X.min(axis=1, keepdims=True)) / (
        X.max(axis=1, keepdims=True) - X.min(axis=1, keepdims=True) + 1e-8)

    ensemble_cfg = config.get("stage2_raw_pca_ensemble")
    if ensemble_cfg:
        ensemble_cfg = [(int(n), float(c)) for n, c in ensemble_cfg]
    else:
        ensemble_cfg = [(
            int(config.get("stage2_raw_pca_components", 20)),
            float(config.get("stage2_logreg_C", 1.0)),
        )]
    print("\n" + "=" * 60)
    print("Stage 2: Raw-spectrum PCA Logistic 8-class OOF")
    print("=" * 60)
    print(f"  Ensemble configs: {ensemble_cfg}")

    all_preds = np.zeros(len(y_all), dtype=np.int64)
    all_probs = np.zeros((len(y_all), len(BYOL_8CLASS_LABELS)), dtype=np.float32)
    fold_models = []

    for fold in range(UNMIX_N_OUTER):
        test_mask = df_all["outer_fold"].to_numpy() == fold
        train_mask = ~test_mask
        fold_prob = np.zeros((test_mask.sum(), len(BYOL_8CLASS_LABELS)),
                             dtype=np.float32)
        fold_model_list = []
        for n_comp, c_val in ensemble_cfg:
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_comp, random_state=0)),
                ("clf", LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    C=c_val,
                )),
            ])
            clf.fit(X_norm[train_mask], y_all[train_mask])
            fold_prob += clf.predict_proba(X_norm[test_mask]).astype(np.float32)
            fold_model_list.append({
                "pca_components": n_comp,
                "C": c_val,
                "model": clf,
            })
        fold_prob /= float(len(ensemble_cfg))
        all_probs[test_mask] = fold_prob
        all_preds[test_mask] = fold_prob.argmax(axis=1)
        fold_models.append(fold_model_list)
        print(f"  Fold {fold+1}: train={train_mask.sum()}, "
              f"test={test_mask.sum()}")

    true_mix = np.array([idx_to_label[i] for i in y_all])
    pred_mix = np.array([idx_to_label[i] for i in all_preds])
    acc_s8 = accuracy_score(true_mix, pred_mix)

    df_plot = df_all.copy()
    for i, label in enumerate(BYOL_8CLASS_LABELS):
        df_plot[f"p_{label}"] = all_probs[:, i]
    prob_cols = [f"p_{label}" for label in BYOL_8CLASS_LABELS]
    grp_prob = df_plot.groupby("group_id")[prob_cols].mean()
    grp_pred_idx = grp_prob.to_numpy().argmax(axis=1)
    grp_pred_mix = np.array([idx_to_label[i] for i in grp_pred_idx])
    grp_true_mix = df_plot.groupby("group_id")["mixture"].first().to_numpy()
    acc_g8 = accuracy_score(grp_true_mix, grp_pred_mix)

    print(f"\n  --- OOF 8-class Classification Results ---")
    print(f"  Spectra-level Acc={acc_s8:.4f}")
    print(f"  Group-level Acc={acc_g8:.4f}")

    def _mix_to_bits(labels):
        bits = np.zeros((len(labels), 3), dtype=int)
        for i, label in enumerate(labels):
            if "DA" in label:
                bits[i, 0] = 1
            if label in ("E", "DA+E", "E+NE", "DA+E+NE"):
                bits[i, 1] = 1
            if "NE" in label:
                bits[i, 2] = 1
        return bits

    true_bits = _mix_to_bits(true_mix)
    pred_bits = _mix_to_bits(pred_mix)
    grp_true_bits = _mix_to_bits(grp_true_mix)
    grp_pred_bits = _mix_to_bits(grp_pred_mix)

    _ensure_byol_output_dirs()
    if plot:
        _plot_raw_pca_pc12_grid(X_norm, df_all, ensemble_cfg)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        cm_s8 = confusion_matrix(true_mix, pred_mix, labels=BYOL_8CLASS_LABELS)
        cm_g8 = confusion_matrix(grp_true_mix, grp_pred_mix,
                                 labels=BYOL_8CLASS_LABELS)
        for ax, cm, title, acc_val in [
            (ax1, cm_s8, "Spectra-level", acc_s8),
            (ax2, cm_g8, "Group-level", acc_g8),
        ]:
            ax.imshow(cm, cmap='Blues')
            ax.set_xticks(range(len(BYOL_8CLASS_LABELS)))
            ax.set_yticks(range(len(BYOL_8CLASS_LABELS)))
            ax.set_xticklabels(BYOL_8CLASS_LABELS, rotation=45, ha='right')
            ax.set_yticklabels(BYOL_8CLASS_LABELS)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for i in range(len(BYOL_8CLASS_LABELS)):
                for j in range(len(BYOL_8CLASS_LABELS)):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            fontsize=9)
            ax.set_title(f"{title} (acc={acc_val:.3f})")
        fig.suptitle("BYOL Pipeline Stage2 (raw PCA logistic 8-class)",
                     fontsize=14)
        fig.tight_layout()
        fig.savefig("visualizations/BYOL_Stage2_Confusion_8class.png", dpi=300)
        plt.close(fig)

        fig2, axes = plt.subplots(2, 3, figsize=(16, 10))
        for j, a in enumerate(UNMIX_ANALYTES):
            cm_s = confusion_matrix(true_bits[:, j], pred_bits[:, j],
                                    labels=[0, 1])
            axes[0, j].imshow(cm_s, cmap='Blues')
            axes[0, j].set_xticks([0, 1]); axes[0, j].set_yticks([0, 1])
            axes[0, j].set_xticklabels(["Absent", "Present"])
            axes[0, j].set_yticklabels(["Absent", "Present"])
            axes[0, j].set_xlabel("Pred"); axes[0, j].set_ylabel("True")
            for r in range(2):
                for c in range(2):
                    axes[0, j].text(c, r, str(cm_s[r, c]),
                                    ha='center', va='center')
            acc_s = accuracy_score(true_bits[:, j], pred_bits[:, j])
            axes[0, j].set_title(f"{a} spectra (acc={acc_s:.3f})")

            cm_g = confusion_matrix(grp_true_bits[:, j], grp_pred_bits[:, j],
                                    labels=[0, 1])
            axes[1, j].imshow(cm_g, cmap='Blues')
            axes[1, j].set_xticks([0, 1]); axes[1, j].set_yticks([0, 1])
            axes[1, j].set_xticklabels(["Absent", "Present"])
            axes[1, j].set_yticklabels(["Absent", "Present"])
            axes[1, j].set_xlabel("Pred"); axes[1, j].set_ylabel("True")
            for r in range(2):
                for c in range(2):
                    axes[1, j].text(c, r, str(cm_g[r, c]),
                                    ha='center', va='center')
            acc_g = accuracy_score(grp_true_bits[:, j], grp_pred_bits[:, j])
            axes[1, j].set_title(f"{a} group (acc={acc_g:.3f})")
        axes[0, 0].set_ylabel("Spectra-level")
        axes[1, 0].set_ylabel("Group-level")
        fig2.suptitle("Derived 3-bit from raw PCA logistic 8-class",
                      fontsize=14)
        fig2.tight_layout()
        fig2.savefig("visualizations/BYOL_Stage2_Confusion.png", dpi=300)
        plt.close(fig2)

    df_s = df_all[["group_id", "mixture"]].copy()
    for j, a in enumerate(UNMIX_ANALYTES):
        df_s[f"true_{a}"] = true_bits[:, j]
        df_s[f"pred_{a}"] = pred_bits[:, j]
    df_s["true_8class"] = true_mix
    df_s["pred_8class"] = pred_mix
    for i, label in enumerate(BYOL_8CLASS_LABELS):
        df_s[f"p_{label}"] = all_probs[:, i]
    df_s.rename(columns={"group_id": "group_name"}, inplace=True)
    df_s.to_csv("reports/BYOL_Class_Spectra.csv", index=False,
                encoding="utf-8-sig")
    print("  Exported reports/BYOL_Class_Spectra.csv")

    df_g = pd.DataFrame({
        "group_name": grp_prob.index,
        "mixture": grp_true_mix,
    })
    for j, a in enumerate(UNMIX_ANALYTES):
        df_g[f"true_{a}"] = grp_true_bits[:, j]
        df_g[f"pred_{a}"] = grp_pred_bits[:, j]
    df_g["true_8class"] = grp_true_mix
    df_g["pred_8class"] = grp_pred_mix
    for i, label in enumerate(BYOL_8CLASS_LABELS):
        df_g[f"p_{label}"] = grp_prob.iloc[:, i].to_numpy()
    df_g.to_csv("reports/BYOL_Class_Group.csv", index=False,
                encoding="utf-8-sig")
    print("  Exported reports/BYOL_Class_Group.csv")

    payload = {
        "backend": "raw_pca_logreg",
        "class_labels": BYOL_8CLASS_LABELS,
        "ensemble_configs": ensemble_cfg,
        "fold_models": fold_models,
        "spectra_accuracy": acc_s8,
        "group_accuracy": acc_g8,
    }
    fname = f"byol_stage2_classifier_{dataset}.joblib"
    save_path = os.path.join(model_dir, fname)
    try:
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(payload, save_path)
    except (PermissionError, RuntimeError):
        save_path = os.path.join("model_output", fname)
        joblib.dump(payload, save_path)
        print(f"  WARNING: Cannot write to model_dir; saved to {save_path}")

    return all_preds, all_probs, y_all


def _stage2_byol_embedding_logreg_8class(RawIntensity, df_all, encoder_path,
                                         config, model_dir, device,
                                         dataset='MPAU', plot=True):
    """OOF 8-class classifier on frozen BYOL encoder embeddings."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    cfg = config
    label_to_idx = {m: i for i, m in enumerate(BYOL_8CLASS_LABELS)}
    idx_to_label = {i: m for m, i in label_to_idx.items()}
    mapped = df_all["mixture"].map(label_to_idx)
    if mapped.isna().any():
        bad = sorted(set(df_all["mixture"]) - set(BYOL_8CLASS_LABELS))
        raise ValueError(f"Unknown mixture labels for 8-class task: {bad}")
    y_all = mapped.to_numpy(dtype=np.int64)

    X_all_t = torch.from_numpy(np.asarray(RawIntensity, dtype=np.float32))
    xmin = X_all_t.min(dim=1, keepdim=True).values
    xmax = X_all_t.max(dim=1, keepdim=True).values
    X_norm = (X_all_t - xmin) / (xmax - xmin + 1e-8)

    encoder = BYOLEncoder(
        in_channels=cfg["in_channels"], base_channels=cfg["base_channels"],
        out_dim=cfg["out_dim"], nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["transformer_dropout"],
    ).to(device)
    ckpt = torch.load(encoder_path, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    encoder.eval()

    feats = []
    with torch.no_grad():
        for start in range(0, len(X_norm), 256):
            bx = X_norm[start:start + 256].to(device)
            feats.append(encoder(bx).cpu().numpy())
    Z = np.concatenate(feats, axis=0).astype(np.float32)

    feature_pca_cfg = cfg.get("stage2_byol_feature_pca_ensemble")
    if feature_pca_cfg:
        ensemble_cfg = [(int(n), float(c)) for n, c in feature_pca_cfg]
    else:
        c_values = cfg.get("stage2_byol_logreg_C", [1.0])
        if isinstance(c_values, (int, float)):
            c_values = [float(c_values)]
        else:
            c_values = [float(c) for c in c_values]
        ensemble_cfg = [(None, c) for c in c_values]

    print("\n" + "=" * 60)
    print("Stage 2: BYOL-embedding Logistic 8-class OOF")
    print("=" * 60)
    print(f"  Encoder: {encoder_path}")
    print(f"  Embedding shape: {Z.shape}")
    print(f"  Embedding PCA/C ensemble: {ensemble_cfg}")

    all_probs = np.zeros((len(y_all), len(BYOL_8CLASS_LABELS)),
                         dtype=np.float32)
    all_preds = np.zeros(len(y_all), dtype=np.int64)
    fold_models = []

    for fold in range(UNMIX_N_OUTER):
        test_mask = df_all["outer_fold"].to_numpy() == fold
        train_mask = ~test_mask
        fold_prob = np.zeros((test_mask.sum(), len(BYOL_8CLASS_LABELS)),
                             dtype=np.float32)
        fold_model_list = []
        for n_comp, c_val in ensemble_cfg:
            steps = [("scaler", StandardScaler())]
            if n_comp is not None:
                steps.append(("pca", PCA(n_components=n_comp,
                                         random_state=0)))
            steps.append(
                ("clf", LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    C=c_val,
                ))
            )
            clf = Pipeline(steps)
            clf.fit(Z[train_mask], y_all[train_mask])
            fold_prob += clf.predict_proba(Z[test_mask]).astype(np.float32)
            fold_model_list.append({
                "embedding_pca_components": n_comp,
                "C": c_val,
                "model": clf,
            })
        fold_prob /= float(len(ensemble_cfg))
        all_probs[test_mask] = fold_prob
        all_preds[test_mask] = fold_prob.argmax(axis=1)
        fold_models.append(fold_model_list)
        print(f"  Fold {fold+1}: train={train_mask.sum()}, "
              f"test={test_mask.sum()}")

    true_mix = np.array([idx_to_label[i] for i in y_all])
    pred_mix = np.array([idx_to_label[i] for i in all_preds])
    acc_s8 = accuracy_score(true_mix, pred_mix)

    df_plot = df_all.copy()
    for i, label in enumerate(BYOL_8CLASS_LABELS):
        df_plot[f"p_{label}"] = all_probs[:, i]
    prob_cols = [f"p_{label}" for label in BYOL_8CLASS_LABELS]
    grp_prob = df_plot.groupby("group_id")[prob_cols].mean()
    grp_pred_idx = grp_prob.to_numpy().argmax(axis=1)
    grp_pred_mix = np.array([idx_to_label[i] for i in grp_pred_idx])
    grp_true_mix = df_plot.groupby("group_id")["mixture"].first().to_numpy()
    acc_g8 = accuracy_score(grp_true_mix, grp_pred_mix)

    print(f"\n  --- OOF 8-class Classification Results ---")
    print(f"  Spectra-level Acc={acc_s8:.4f}")
    print(f"  Group-level Acc={acc_g8:.4f}")

    _ensure_byol_output_dirs()
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        cm_s8 = confusion_matrix(true_mix, pred_mix, labels=BYOL_8CLASS_LABELS)
        cm_g8 = confusion_matrix(grp_true_mix, grp_pred_mix,
                                 labels=BYOL_8CLASS_LABELS)
        for ax, cm, title, acc_val in [
            (ax1, cm_s8, "Spectra-level", acc_s8),
            (ax2, cm_g8, "Group-level", acc_g8),
        ]:
            ax.imshow(cm, cmap='Blues')
            ax.set_xticks(range(len(BYOL_8CLASS_LABELS)))
            ax.set_yticks(range(len(BYOL_8CLASS_LABELS)))
            ax.set_xticklabels(BYOL_8CLASS_LABELS, rotation=45, ha='right')
            ax.set_yticklabels(BYOL_8CLASS_LABELS)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for i in range(len(BYOL_8CLASS_LABELS)):
                for j in range(len(BYOL_8CLASS_LABELS)):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            fontsize=9)
            ax.set_title(f"{title} (acc={acc_val:.3f})")
        fig.suptitle("BYOL embedding + logistic 8-class OOF", fontsize=14)
        fig.tight_layout()
        fig.savefig("visualizations/BYOL_EmbeddingLogReg_Confusion_8class.png",
                    dpi=300)
        plt.close(fig)

    df_s = df_all[["group_id", "mixture"]].copy()
    df_s["true_8class"] = true_mix
    df_s["pred_8class"] = pred_mix
    for i, label in enumerate(BYOL_8CLASS_LABELS):
        df_s[f"p_{label}"] = all_probs[:, i]
    df_s.rename(columns={"group_id": "group_name"}, inplace=True)
    df_s.to_csv("reports/BYOL_EmbeddingLogReg_Spectra.csv",
                index=False, encoding="utf-8-sig")

    df_g = pd.DataFrame({
        "group_name": grp_prob.index,
        "mixture": grp_true_mix,
        "true_8class": grp_true_mix,
        "pred_8class": grp_pred_mix,
    })
    for i, label in enumerate(BYOL_8CLASS_LABELS):
        df_g[f"p_{label}"] = grp_prob.iloc[:, i].to_numpy()
    df_g.to_csv("reports/BYOL_EmbeddingLogReg_Group.csv",
                index=False, encoding="utf-8-sig")

    payload = {
        "backend": "byol_embedding_logreg",
        "class_labels": BYOL_8CLASS_LABELS,
        "embedding_pca_ensemble": ensemble_cfg,
        "fold_models": fold_models,
        "spectra_accuracy": acc_s8,
        "group_accuracy": acc_g8,
        "encoder_path": encoder_path,
    }
    fname = f"byol_embedding_logreg_{dataset}.joblib"
    save_path = os.path.join(model_dir, fname)
    try:
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(payload, save_path)
    except (PermissionError, RuntimeError):
        save_path = os.path.join("model_output", fname)
        joblib.dump(payload, save_path)
        print(f"  WARNING: Cannot write to model_dir; saved to {save_path}")

    return all_preds, all_probs, y_all


# ===========================================================================
# Stage 2 variant: PLSR quantification
# ===========================================================================

def _stage2_plsr_quant(RawIntensity, Concentrations, df_all, encoder_path,
                        config, model_dir, device, dataset, plot,
                        raman_shift=None):
    """Stage 2 PLSR: freeze encoder, extract 256-dim features,
    run 3-fold OOF PLSR for DA/E/NE concentration prediction.
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error

    cfg = config
    print("\n" + "=" * 60)
    print("Stage 2: PLSR Quantification (OOF on frozen features)")
    print("=" * 60)

    # Build encoder and extract features
    encoder = BYOLEncoder(
        in_channels=cfg["in_channels"], base_channels=cfg["base_channels"],
        out_dim=cfg["out_dim"], nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["transformer_dropout"],
    ).to(device)
    ckpt = torch.load(encoder_path, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Peak-normalize for quantification
    if raman_shift is None:
        raise ValueError(
            "raman_shift is required for quantification mode")
    X_norm_np = spectra_normalization(
        raman_shift, RawIntensity,
        peak_position=920, peak_range=20,
        plot=False, mode='byol_quant', minmax_scale=False)
    X_norm = torch.from_numpy(X_norm_np).float()

    feats = []
    loader = DataLoader(TensorDataset(X_norm), batch_size=64)
    with torch.no_grad():
        for (bx,) in loader:
            feats.append(encoder(bx.to(device)).cpu().numpy())
    X_feat = np.concatenate(feats, axis=0)
    Y = Concentrations  # (N, 3)
    group_ids = df_all["group_id"].to_numpy()

    print(f"  Features: {X_feat.shape}, Targets: {Y.shape}")

    # 3-fold OOF PLSR (same as _plsr_oof_features)
    oof_pred = np.zeros_like(Y)
    for fold in range(UNMIX_N_OUTER):
        tmask = df_all["outer_fold"].to_numpy() == fold
        fmask = ~tmask
        X_tr, Y_tr = X_feat[fmask], Y[fmask]
        X_te = X_feat[tmask]

        # Inner 2-fold for n_comp
        train_gids = group_ids[fmask]
        train_meta = df_all.loc[fmask].drop_duplicates("group_id").reset_index(drop=True)
        inner_folds = _umx_group_folds(train_meta, n_splits=2,
                                        random_state=UNMIX_RANDOM_STATE + fold)
        ilookup = dict(zip(inner_folds["group_id"], inner_folds["fold"]))
        isample = np.array([ilookup[g] for g in train_gids])

        best_n, best_rmse = 1, np.inf
        n_max = min(10, X_tr.shape[0] - 1, X_tr.shape[1])
        for n in range(1, n_max + 1):
            oof_in = np.zeros_like(Y_tr)
            for inf in range(2):
                vm = isample == inf; tm = ~vm
                pls = Pipeline([("s", StandardScaler()),
                                ("p", PLSRegression(n, scale=False))])
                pls.fit(X_tr[tm], Y_tr[tm])
                oof_in[vm] = pls.predict(X_tr[vm])
            tmp = pd.DataFrame({"g": train_gids,
                                "t0": Y_tr[:, 0], "t1": Y_tr[:, 1], "t2": Y_tr[:, 2],
                                "p0": oof_in[:, 0], "p1": oof_in[:, 1], "p2": oof_in[:, 2]})
            gr = tmp.groupby("g").agg(
                {"t0": "first", "t1": "first", "t2": "first",
                 "p0": "mean", "p1": "mean", "p2": "mean"})
            yt = gr[["t0", "t1", "t2"]].to_numpy()
            yp = gr[["p0", "p1", "p2"]].to_numpy()
            s = np.sqrt(mean_squared_error(yt.reshape(-1), yp.reshape(-1)))
            if s < best_rmse:
                best_rmse, best_n = s, n
        pls = Pipeline([("s", StandardScaler()),
                        ("p", PLSRegression(best_n, scale=False))])
        pls.fit(X_tr, Y_tr)
        oof_pred[tmask] = np.maximum(pls.predict(X_te), 0)
        print(f"  Fold {fold+1}: best_n={best_n}")

    # Metrics
    for j, a in enumerate(UNMIX_ANALYTES):
        rmse = np.sqrt(mean_squared_error(Y[:, j], oof_pred[:, j]))
        r2 = r2_score(Y[:, j], oof_pred[:, j])
        print(f"  {a}: RMSE={rmse:.3f} uM, R2={r2:.4f}")

    if plot:
        grp = df_all[["group_id"]].copy()
        for j, a in enumerate(UNMIX_ANALYTES):
            grp[f"true_{a}"] = Y[:, j]
            grp[f"pred_{a}"] = oof_pred[:, j]
        ga = grp.groupby("group_id").agg(
            **{f"true_{a}": (f"true_{a}", "first") for a in UNMIX_ANALYTES},
            **{f"m_{a}": (f"pred_{a}", "mean") for a in UNMIX_ANALYTES},
            **{f"sd_{a}": (f"pred_{a}", "std") for a in UNMIX_ANALYTES},
        ).reset_index()
        _plot_quant_group(ga, "BYOL+PLSR OOF Quant")
        _print_per_group_mse(ga, "BYOL+PLSR")

    return oof_pred


# ===========================================================================
# Stage 2 variant: MLP quantification
# ===========================================================================

class QuantMLP(nn.Module):
    """256 → 128 → 3, ReLU output (μM)."""

    def __init__(self, in_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 3), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def _stage2_mlp_quant(RawIntensity, Concentrations, df_all, encoder_path,
                       config, model_dir, device, dataset, plot,
                       raman_shift=None):
    """Stage 2 MLP: OOF train MLP decoder on frozen encoder features."""
    cfg = config
    print("\n" + "=" * 60)
    print("Stage 2: MLP Quantification (OOF on frozen features)")
    print("=" * 60)

    encoder = BYOLEncoder(
        in_channels=cfg["in_channels"], base_channels=cfg["base_channels"],
        out_dim=cfg["out_dim"], nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["transformer_dropout"],
    ).to(device)
    ckpt = torch.load(encoder_path, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    if raman_shift is None:
        raise ValueError(
            "raman_shift is required for quantification mode")
    X_norm_np = spectra_normalization(
        raman_shift, RawIntensity,
        peak_position=920, peak_range=20,
        plot=False, mode='byol_quant', minmax_scale=False)
    X_norm = torch.from_numpy(X_norm_np).float()
    Y_t = torch.from_numpy(Concentrations).float()

    oof_pred = np.zeros_like(Concentrations)
    for fold in range(UNMIX_N_OUTER):
        tmask = df_all["outer_fold"].to_numpy() == fold
        fmask = ~tmask
        X_tr, Y_tr = X_norm[fmask], Y_t[fmask]
        X_te = X_norm[tmask]

        head = QuantMLP(in_dim=cfg["out_dim"]).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=cfg["stage2_lr"])
        loader = DataLoader(TensorDataset(X_tr, Y_tr),
                            batch_size=cfg["stage2_batch_size"], shuffle=True)
        criterion = nn.MSELoss()
        for ep in range(cfg.get("mlp_epochs", 100)):
            head.train()
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                with torch.no_grad():
                    feat = encoder(bx)
                pred = head(feat)
                loss = criterion(pred, by)
                opt.zero_grad()
                loss.backward()
                opt.step()

        head.eval()
        with torch.no_grad():
            feats_te = encoder(X_te.to(device))
            oof_pred[tmask] = head(feats_te).cpu().numpy()
        print(f"  Fold {fold+1}: {fmask.sum()} train, {tmask.sum()} test")

    oof_pred = np.maximum(oof_pred, 0)
    from sklearn.metrics import r2_score, mean_squared_error
    for j, a in enumerate(UNMIX_ANALYTES):
        rmse = np.sqrt(mean_squared_error(Concentrations[:, j], oof_pred[:, j]))
        r2 = r2_score(Concentrations[:, j], oof_pred[:, j])
        print(f"  {a}: RMSE={rmse:.3f} uM, R2={r2:.4f}")

    if plot:
        grp = df_all[["group_id"]].copy()
        for j, a in enumerate(UNMIX_ANALYTES):
            grp[f"true_{a}"] = Concentrations[:, j]
            grp[f"pred_{a}"] = oof_pred[:, j]
        ga = grp.groupby("group_id").agg(
            **{f"true_{a}": (f"true_{a}", "first") for a in UNMIX_ANALYTES},
            **{f"m_{a}": (f"pred_{a}", "mean") for a in UNMIX_ANALYTES},
            **{f"sd_{a}": (f"pred_{a}", "std") for a in UNMIX_ANALYTES},
        ).reset_index()
        _plot_quant_group(ga, "BYOL+MLP OOF Quant")
        _print_per_group_mse(ga, "BYOL+MLP")

    return oof_pred


# ---------------------------------------------------------------------------
# Shared quant plot
# ---------------------------------------------------------------------------

def _plot_quant_group(grp_agg, title):
    from sklearn.metrics import r2_score, mean_squared_error
    colors = ['#E74C3C', '#3498DB', '#2ECC71']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for j, (ax, a) in enumerate(zip(axes, UNMIX_ANALYTES)):
        t = grp_agg[f"true_{a}"].to_numpy()
        m = grp_agg[f"m_{a}"].to_numpy()
        s = grp_agg[f"sd_{a}"].fillna(0).to_numpy()
        ax.errorbar(t, m, yerr=s, fmt='o', color=colors[j],
                    capsize=3, markersize=6, markeredgecolor='k',
                    markeredgewidth=0.5)
        mx = max(t.max(), (m + s).max()) * 1.1
        ax.plot([0, mx], [0, mx], 'r--', lw=1)
        ax.set_xlim(-2, mx); ax.set_ylim(-2, mx)
        ax.set_xlabel(f"True {a} (uM)"); ax.set_ylabel(f"Predicted {a} (uM)")
        mask = t > 0
        if mask.sum() > 1:
            rmse = np.sqrt(mean_squared_error(t[mask], m[mask]))
            r2 = r2_score(t[mask], m[mask])
            ax.set_title(f"{a}: RMSE={rmse:.2f} uM, R2={r2:.3f}")
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _ensure_byol_output_dirs()
    fname = f"visualizations/BYOL_{title.replace(' ', '_')}.png"
    fig.savefig(fname, dpi=300)
    plt.close(fig)


def _ca_full_writable_model_dir(model_dir):
    """Use model_dir when writable; otherwise fall back to model_output."""
    try:
        os.makedirs(model_dir, exist_ok=True)
        probe = os.path.join(model_dir, "_codex_write_probe.tmp")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        return model_dir
    except OSError:
        _ensure_byol_output_dirs()
        print(f"  WARNING: Cannot write to {model_dir}; "
              "full-pipeline models will be saved to model_output.")
        return "model_output"


def _ca_full_byol_embeddings(raw_intensity, encoder_path, cfg, device):
    """Extract frozen BYOL encoder embeddings using classifier min-max input."""
    x_all_t = torch.from_numpy(np.asarray(raw_intensity, dtype=np.float32))
    xmin = x_all_t.min(dim=1, keepdim=True).values
    xmax = x_all_t.max(dim=1, keepdim=True).values
    x_norm = (x_all_t - xmin) / (xmax - xmin + 1e-8)

    encoder = BYOLEncoder(
        in_channels=cfg["in_channels"], base_channels=cfg["base_channels"],
        out_dim=cfg["out_dim"], nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["transformer_dropout"],
    ).to(device)
    ckpt = torch.load(encoder_path, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()

    feats = []
    with torch.no_grad():
        for start in range(0, len(x_norm), 256):
            bx = x_norm[start:start + 256].to(device)
            feats.append(encoder(bx).cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def _ca_full_fit_quant_models(x_spectrum, x_ratio, y_conc, mixtures,
                              train_mask, n_components=5):
    """Fit true-class calibration models used after BYOL class prediction."""
    from sklearn.linear_model import LinearRegression
    from sklearn.cross_decomposition import PLSRegression

    models = {}
    rows = []
    for mix in UNMIX_MODEL_MIXTURES:
        mix_mask = (mixtures == mix) & train_mask
        if not mix_mask.any():
            continue
        present = _umx_present_analyte_indices(mix)
        if mix in UNMIX_SINGLE_MIXTURES:
            target_j = present[0]
            model = LinearRegression().fit(
                x_ratio[mix_mask], y_conc[mix_mask, target_j])
            models[mix] = {
                "model_type": "linear_1480_1388_over_920",
                "target_indices": present,
                "model": model,
            }
            rows.append({
                "model_mixture": mix,
                "model_type": "linear_1480_1388_over_920",
                "target": UNMIX_ANALYTES[target_j],
                "selected_n_components": np.nan,
                "n_train": int(mix_mask.sum()),
            })
            continue

        n_comp = min(n_components, x_spectrum[mix_mask].shape[0] - 1,
                     x_spectrum[mix_mask].shape[1])
        n_comp = max(1, int(n_comp))
        model = PLSRegression(n_components=n_comp, scale=True)
        model.fit(x_spectrum[mix_mask], y_conc[mix_mask][:, present])
        models[mix] = {
            "model_type": "multi_output_plsr_by_predicted_class",
            "target_indices": present,
            "model": model,
            "n_components": n_comp,
        }
        rows.append({
            "model_mixture": mix,
            "model_type": "multi_output_plsr_by_predicted_class",
            "target": "+".join(UNMIX_ANALYTES[j] for j in present),
            "selected_n_components": n_comp,
            "n_train": int(mix_mask.sum()),
        })
    return models, pd.DataFrame(rows)


def _ca_full_predict_by_class(models, pred_mix, x_spectrum, x_ratio):
    pred = np.zeros((len(pred_mix), len(UNMIX_ANALYTES)), dtype=float)
    for mix in UNMIX_MODEL_MIXTURES:
        idx = np.where(pred_mix == mix)[0]
        if len(idx) == 0 or mix not in models:
            continue
        spec = models[mix]
        present = spec["target_indices"]
        if spec["model_type"] == "linear_1480_1388_over_920":
            target_j = present[0]
            pred[idx, target_j] = spec["model"].predict(x_ratio[idx])
        else:
            local_pred = spec["model"].predict(x_spectrum[idx])
            for local_j, target_j in enumerate(present):
                pred[idx, target_j] = local_pred[:, local_j]
    return np.maximum(pred, 0.0)


def CA_Paper_Full_Pipeline(data_dir, model_dir, conc_threshold=None,
                           mix_only=False, present_conc_range=(10, 20),
                           config=None, plot=True, dataset="MPAU2",
                           re_training=False, stage1=True):
    """BYOL classifier -> predicted-class PLSR concentration pipeline."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    cfg = BYOL_CONFIG.copy()
    if config:
        cfg.update(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_model_dir = _ca_full_writable_model_dir(model_dir)
    _ensure_byol_output_dirs()

    print("=" * 60)
    print("CA Paper Full Pipeline - BYOL classifier + class-routed PLSR")
    print("=" * 60)
    print(f"Device: {device}")

    print("\n[1/7] Reading and filtering spectra ...")
    raman_shift, intensity, conc, groups, mixtures = _read_mpau_mix_spectra(
        data_dir)
    if conc_threshold is not None and isinstance(conc_threshold, (int, float)):
        keep = conc.sum(axis=1) <= conc_threshold
        intensity, conc, groups, mixtures = (
            intensity[keep], conc[keep], groups[keep], mixtures[keep])
    intensity, conc, groups, mixtures = _filter_mix_conc(
        intensity, conc, groups, mixtures,
        mix_only=mix_only, present_conc_range=present_conc_range)
    keep = np.isin(mixtures, UNMIX_MODEL_MIXTURES)
    if not keep.all():
        print(f"  Removing {(~keep).sum()} non-model spectra, e.g. BA.")
        intensity, conc, groups, mixtures = (
            intensity[keep], conc[keep], groups[keep], mixtures[keep])

    group_ids = np.array([
        _umx_make_group_id(mixtures[i], conc[i, 0], conc[i, 1], conc[i, 2])
        for i in range(len(mixtures))
    ])
    df_all = pd.DataFrame({
        "group_id": group_ids,
        "mixture": mixtures,
        "conc_DA": conc[:, 0],
        "conc_E": conc[:, 1],
        "conc_NE": conc[:, 2],
    })
    print(f"  Spectra: {len(mixtures)}, groups: "
          f"{df_all['group_id'].nunique()}")

    print("\n[2/7] Building balanced holdout split ...")
    train_mask, val_mask = _umx_balanced_holdout_split(
        group_ids, val_fraction=0.30, random_state=UNMIX_RANDOM_STATE)
    print(f"  Train spectra: {train_mask.sum()}")
    print(f"  Validation spectra: {val_mask.sum()}")

    print("\n[3/7] Training/loading BYOL encoder on training spectra ...")
    norm_mode = "minmax"
    fp_dataset = f"{dataset}_full_pipeline"
    encoder_path = os.path.join(
        run_model_dir, f"byol_stage1_{fp_dataset}_{norm_mode}.pt")
    if stage1:
        _, encoder_path = _stage1_byol_pretrain(
            intensity[train_mask], cfg, run_model_dir, device,
            norm_mode=norm_mode, dataset=fp_dataset,
            re_training=re_training, plot=plot, raman_shift=raman_shift)
    elif not os.path.exists(encoder_path):
        raise FileNotFoundError(
            f"No full-pipeline BYOL encoder found at {encoder_path}")

    print("\n[4/7] Training BYOL embedding classifier ...")
    z = _ca_full_byol_embeddings(intensity, encoder_path, cfg, device)
    labels = list(UNMIX_MODEL_MIXTURES)
    label_to_idx = {m: i for i, m in enumerate(labels)}
    idx_to_label = {i: m for m, i in label_to_idx.items()}
    y_cls = df_all["mixture"].map(label_to_idx).to_numpy(dtype=np.int64)

    feature_pca_cfg = cfg.get("stage2_byol_feature_pca_ensemble")
    if feature_pca_cfg:
        ensemble_cfg = [(int(n), float(c)) for n, c in feature_pca_cfg]
    else:
        c_values = cfg.get("stage2_byol_logreg_C", [1.0])
        if isinstance(c_values, (int, float)):
            c_values = [float(c_values)]
        ensemble_cfg = [(None, float(c)) for c in c_values]

    val_prob = np.zeros((int(val_mask.sum()), len(labels)), dtype=np.float32)
    classifier_models = []
    for n_comp, c_val in ensemble_cfg:
        steps = [("scaler", StandardScaler())]
        if n_comp is not None:
            steps.append(("pca", PCA(n_components=n_comp, random_state=0)))
        steps.append(("clf", LogisticRegression(
            max_iter=4000, class_weight="balanced", C=c_val)))
        clf = Pipeline(steps)
        clf.fit(z[train_mask], y_cls[train_mask])
        val_prob += clf.predict_proba(z[val_mask]).astype(np.float32)
        classifier_models.append({
            "embedding_pca_components": n_comp,
            "C": c_val,
            "model": clf,
        })
    val_prob /= float(len(ensemble_cfg))
    val_pred_idx = val_prob.argmax(axis=1)
    val_pred_mix = np.array([idx_to_label[i] for i in val_pred_idx])
    val_true_mix = mixtures[val_mask]
    acc_s = accuracy_score(val_true_mix, val_pred_mix)

    df_val = df_all.loc[val_mask].copy().reset_index(drop=True)
    df_val["split"] = "validation"
    df_val["true_mixture"] = val_true_mix
    df_val["pred_mixture"] = val_pred_mix
    for i, label in enumerate(labels):
        df_val[f"p_{label}"] = val_prob[:, i]
    grp_prob = df_val.groupby("group_id")[
        [f"p_{label}" for label in labels]].mean()
    grp_pred_mix = np.array([
        idx_to_label[i] for i in grp_prob.to_numpy().argmax(axis=1)])
    grp_true_mix = df_val.groupby("group_id")["mixture"].first().to_numpy()
    grp_pred_lookup = dict(zip(grp_prob.index.to_numpy(), grp_pred_mix))
    val_route_mix = np.array([
        grp_pred_lookup[gid] for gid in df_val["group_id"].to_numpy()
    ])
    acc_g = accuracy_score(grp_true_mix, grp_pred_mix)
    print(f"  Validation spectra accuracy: {acc_s:.4f}")
    print(f"  Validation group accuracy:   {acc_g:.4f}")

    print("\n[5/7] Training true-class PLSR calibration models ...")
    x_spectrum = spectra_normalization(
        raman_shift, intensity,
        peak_position=920, peak_range=20,
        plot=False, mode="ca_paper_full_pipeline", minmax_scale=False)
    x_ratio = _umx_two_peak_ratio_features(raman_shift, intensity)
    quant_models, model_table = _ca_full_fit_quant_models(
        x_spectrum, x_ratio, conc, mixtures, train_mask, n_components=5)
    print(model_table.to_string(index=False))

    print("\n[6/7] Predicting validation concentrations by predicted class ...")
    pred_val = _ca_full_predict_by_class(
        quant_models, val_route_mix, x_spectrum[val_mask], x_ratio[val_mask])
    df_eval = df_all.loc[val_mask].copy().reset_index(drop=True)
    df_eval["split"] = "validation"
    df_eval["model_mixture"] = val_route_mix
    df_eval["model_type"] = "BYOL_group_predicted_class_routed_PLSR"
    sample_conc, group_conc = _umx_build_tables(
        df_eval, pred_val, df_eval[["conc_DA", "conc_E", "conc_NE"]].to_numpy(),
        "concentration")
    sample_conc["true_mixture"] = val_true_mix
    sample_conc["pred_mixture_spectrum"] = val_pred_mix
    sample_conc["pred_mixture_group"] = val_route_mix
    for i, label in enumerate(labels):
        sample_conc[f"p_{label}"] = val_prob[:, i]
    group_pred_mix = pd.DataFrame({
        "group_id": grp_prob.index.to_numpy(),
        "pred_mixture_group": grp_pred_mix,
    })
    spectrum_vote_mix = (
        sample_conc.groupby("group_id")["pred_mixture_spectrum"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
        .rename(columns={"pred_mixture_spectrum": "pred_mixture_spectrum_vote"})
    )
    group_conc = group_conc.merge(group_pred_mix, on="group_id", how="left")
    group_conc = group_conc.merge(spectrum_vote_mix, on="group_id", how="left")

    summary_df = _umx_continuous_summary(
        group_conc, "concentration", "group")
    with pd.option_context("display.max_columns", 120,
                           "display.width", 200,
                           "display.float_format", "{:.4f}".format):
        print(summary_df.to_string(index=False))

    print("\n[7/7] Saving outputs ...")
    sample_path = "reports/CA_Paper_Full_Pipeline_Sample.csv"
    group_path = "reports/CA_Paper_Full_Pipeline_Group.csv"
    summary_path = "reports/CA_Paper_Full_Pipeline_Summary.csv"
    cls_path = "reports/CA_Paper_Full_Pipeline_Classification.csv"
    sample_conc.to_csv(sample_path, index=False, encoding="utf-8-sig")
    group_conc.to_csv(group_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    df_val.to_csv(cls_path, index=False, encoding="utf-8-sig")
    print(f"  Exported {sample_path}")
    print(f"  Exported {group_path}")
    print(f"  Exported {summary_path}")
    print(f"  Exported {cls_path}")

    if plot:
        pred_png = "visualizations/CA_Paper_Full_Pipeline_Pred_vs_True.png"
        _umx_plot_holdout_pred(group_conc, pred_png)
        print(f"  Exported {pred_png}")
        cm = confusion_matrix(grp_true_mix, grp_pred_mix, labels=labels)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_title(f"CA Paper Full Pipeline group classifier acc={acc_g:.3f}")
        fig.tight_layout()
        cm_png = "visualizations/CA_Paper_Full_Pipeline_Confusion.png"
        fig.savefig(cm_png, dpi=300)
        plt.close(fig)
        print(f"  Exported {cm_png}")

    payload = {
        "method": "CA_Paper_Full_Pipeline",
        "class_labels": labels,
        "classifier_accuracy_spectra": acc_s,
        "classifier_accuracy_group": acc_g,
        "classifier_models": classifier_models,
        "encoder_path": encoder_path,
        "quant_models": quant_models,
        "model_table": model_table,
        "summary": summary_df,
        "train_mask": train_mask,
        "validation_mask": val_mask,
        "group_concentration": group_conc,
        "sample_concentration": sample_conc,
    }
    model_path = _umx_save_payload(
        payload, run_model_dir, "ca_paper_full_pipeline.joblib")
    print(f"  Saved {model_path}")
    print("=" * 60)
    print("CA Paper Full Pipeline completed.")
    print("=" * 60)
    return payload


# ===========================================================================
# Main entry point
# ===========================================================================

def BYOLFullPipeline(data_dir, model_dir, conc_threshold=None,
                      mix_only=False, present_conc_range=None,
                      stage1=True, stage2=True, re_training=False,
                      stage2_task="classification",
                      config=None, plot=True, dataset='MPAU'):
    """Run the complete byol pre-training + fine-tuning pipeline.

    Args:
        data_dir: path to data folder.
        model_dir: directory for saving models.
        conc_threshold: total conc threshold.
        mix_only: keep binary+ternary only.
        present_conc_range: (min, max) for present component conc.
        stage1: run Stage 1 byol pre-training.
        stage2: run Stage 2 classification fine-tuning.
        config: override default BYOL_CONFIG.
        plot: generate plots.
        dataset: dataset name for saved model filenames.
    """
    cfg = BYOL_CONFIG.copy()
    if config:
        cfg.update(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data ---
    print("\n[0] Loading data ...")
    Raman_Shift, RawIntensity, Y_cls, Concentrations, Mixtures, df_all = \
        _load_byol_data(
            data_dir, conc_threshold, mix_only, present_conc_range,
            singleton_sample_folds=(stage2_task == "classification"))

    # --- Determine norm mode from task ---
    norm_mode = "minmax" if stage2_task == "classification" else "peak"
    print(f"  Normalization mode: {norm_mode}")

    # --- Stage 1 ---
    encoder_path = os.path.join(model_dir, f"byol_stage1_{dataset}_{norm_mode}.pt")
    if stage1:
        _, encoder_path = _stage1_byol_pretrain(
            RawIntensity, cfg, model_dir, device,
            norm_mode=norm_mode, dataset=dataset,
            re_training=re_training, plot=plot,
            raman_shift=Raman_Shift)
    elif not os.path.exists(encoder_path):
        print(f"No encoder at {encoder_path}. Exiting.")
        return

    # --- Stage 2 ---
    if stage2:
        if stage2_task == "classification":
            backend = cfg.get("stage2_class_backend", "raw_pca_logreg")
            if backend == "direct_8class":
                _stage2_finetune_8class(RawIntensity, df_all,
                                         encoder_path, cfg, model_dir,
                                         device, dataset, plot)
            elif backend == "raw_pca_logreg":
                _stage2_raw_pca_8class(RawIntensity, df_all, cfg,
                                       model_dir, dataset, plot)
            elif backend == "byol_embedding_logreg":
                _stage2_byol_embedding_logreg_8class(
                    RawIntensity, df_all, encoder_path, cfg, model_dir,
                    device, dataset, plot)
            else:
                raise ValueError(f"Unknown stage2_class_backend: {backend}")
        elif stage2_task == "quantification_plsr":
            _stage2_plsr_quant(RawIntensity, Concentrations, df_all,
                                encoder_path, cfg, model_dir,
                                device, dataset, plot,
                                raman_shift=Raman_Shift)
        elif stage2_task == "quantification_mlp":
            _stage2_mlp_quant(RawIntensity, Concentrations, df_all,
                               encoder_path, cfg, model_dir,
                               device, dataset, plot,
                               raman_shift=Raman_Shift)

    print("\n" + "=" * 60)
    print("BYOL Full Pipeline completed.")
    print("=" * 60)


