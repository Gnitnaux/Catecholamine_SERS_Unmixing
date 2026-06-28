import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


ANALYTES = ["DA", "E", "NE"]
MODEL_MIXTURES = ["DA", "E", "NE", "DA+E", "DA+NE", "E+NE", "DA+E+NE"]
SINGLE_MIXTURES = ["DA", "E", "NE"]
CLASS_LABELS = ["BA"] + MODEL_MIXTURES


def parse_mix_folder(folder):
    """Parse concentration folder names. Author: Xuanting Liu."""
    nums = re.findall(r"([0-9.]+)uM", folder)
    if len(nums) < 3:
        raise ValueError(f"Invalid mixture folder name: {folder}")
    conc = np.array([float(nums[0]), float(nums[1]), float(nums[2])], dtype=float)
    present = [a for a, v in zip(ANALYTES, conc) if v > 0]
    return conc, ("BA" if not present else "+".join(present))


def read_mix_spectra(data_dir, shift_range=(330, 1600)):
    """Read SERS mixture spectra from concentration folders. Author: Xuanting Liu."""
    spectra, concs, groups, mixtures = [], [], [], []
    raman_shift = None
    for folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        conc, mixture = parse_mix_folder(folder)
        for name in sorted(os.listdir(folder_path)):
            if not name.lower().endswith(".csv"):
                continue
            path = os.path.join(folder_path, name)
            df = pd.read_csv(path)
            if df.shape[1] < 2:
                continue
            x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
            y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
            keep = np.isfinite(x) & np.isfinite(y)
            if shift_range is not None:
                keep &= (x >= shift_range[0]) & (x <= shift_range[1])
            x, y = x[keep], y[keep]
            if raman_shift is None:
                raman_shift = x
            elif len(x) != len(raman_shift) or np.max(np.abs(x - raman_shift)) > 1e-6:
                y = np.interp(raman_shift, x, y)
            spectra.append(y.astype(float))
            concs.append(conc)
            groups.append(folder)
            mixtures.append(mixture)
    if not spectra:
        raise ValueError(f"No CSV spectra found in {data_dir}")
    return (
        raman_shift,
        np.vstack(spectra),
        np.vstack(concs),
        np.asarray(groups),
        np.asarray(mixtures),
    )


def normalize_spectra(raman_shift, intensity, peak_position=920, peak_range=20, minmax=False):
    """Normalize spectra by peak window or per-spectrum min-max. Author: Xuanting Liu."""
    x = np.asarray(intensity, dtype=float)
    if minmax:
        lo = x.min(axis=1, keepdims=True)
        hi = x.max(axis=1, keepdims=True)
        return (x - lo) / (hi - lo + 1e-8)
    idx = np.where(
        (raman_shift >= peak_position - peak_range)
        & (raman_shift <= peak_position + peak_range)
    )[0]
    if len(idx) == 0:
        scale = np.max(np.abs(x), axis=1, keepdims=True)
    else:
        scale = np.max(np.abs(x[:, idx]), axis=1, keepdims=True)
    return x / (scale + 1e-8)


def filter_mix_conc(intensity, concentrations, groups, mixtures, mix_only=False, present_conc_range=None):
    """Filter spectra by mixture type and present concentration range. Author: Xuanting Liu."""
    keep = np.ones(len(intensity), dtype=bool)
    if mix_only:
        keep &= np.isin(mixtures, ["DA+E", "DA+NE", "E+NE", "DA+E+NE"])
    if present_conc_range is not None:
        lo, hi = present_conc_range
        present_ok = []
        for row in concentrations:
            vals = row[row > 0]
            present_ok.append(len(vals) == 0 or np.all((vals >= lo) & (vals <= hi)))
        keep &= np.asarray(present_ok, dtype=bool)
    print(f"  Filter kept {keep.sum()}/{len(keep)} spectra ({np.unique(groups[keep]).size} groups)")
    return intensity[keep], concentrations[keep], groups[keep], mixtures[keep]


def make_group_id(mixture, conc):
    """Build a stable concentration-group id. Author: Xuanting Liu."""
    da, e, ne = [float(v) for v in conc]
    return f"{mixture}|{da:g}|{e:g}|{ne:g}"


def balanced_holdout_split(group_ids, val_fraction=0.30, random_state=2026):
    """Sample validation spectra inside every concentration group. Author: Xuanting Liu."""
    rng = np.random.default_rng(random_state)
    train = np.zeros(len(group_ids), dtype=bool)
    val = np.zeros(len(group_ids), dtype=bool)
    for gid in np.unique(group_ids):
        idx = np.where(group_ids == gid)[0]
        idx = rng.permutation(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        if n_val >= len(idx) and len(idx) > 1:
            n_val = len(idx) - 1
        val[idx[:n_val]] = True
        train[idx[n_val:]] = True
    return train, val


def group_folds(group_ids, mixtures, n_splits=3, random_state=2026):
    """Assign stratified folds by mixture labels. Author: Xuanting Liu."""
    rng = np.random.default_rng(random_state)
    table = pd.DataFrame({"group_id": group_ids, "mixture": mixtures}).drop_duplicates()
    table["fold"] = -1
    for mix in sorted(table["mixture"].unique()):
        idx = table.index[table["mixture"] == mix].to_numpy()
        rng.shuffle(idx)
        for i, row_idx in enumerate(idx):
            table.loc[row_idx, "fold"] = i % n_splits
    return dict(zip(table["group_id"], table["fold"]))


def peak_ratio_features(raman_shift, intensity, centers=(1480, 1388), denominator=920, peak_range=20):
    """Compute marker peak ratios against the 920 cm-1 window. Author: Xuanting Liu."""
    den_idx = np.where((raman_shift >= denominator - peak_range) & (raman_shift <= denominator + peak_range))[0]
    if len(den_idx) == 0:
        raise ValueError(f"Cannot find denominator peak near {denominator} cm-1")
    den = np.max(np.abs(intensity[:, den_idx]), axis=1) + 1e-8
    cols = []
    for center in centers:
        idx = np.where((raman_shift >= center - peak_range) & (raman_shift <= center + peak_range))[0]
        if len(idx) == 0:
            raise ValueError(f"Cannot find marker peak near {center} cm-1")
        cols.append(np.max(np.abs(intensity[:, idx]), axis=1) / den)
    return np.column_stack(cols)


def build_result_tables(df, pred):
    """Build sample-level and group-level concentration result tables. Author: Xuanting Liu."""
    sample = df.copy()
    for j, analyte in enumerate(ANALYTES):
        sample[f"pred_conc_{analyte}"] = pred[:, j]
    agg = {"split": "first", "mixture": "first", "conc_DA": "first", "conc_E": "first", "conc_NE": "first"}
    if "model_mixture" in sample.columns:
        agg["model_mixture"] = "first"
    if "pred_mixture" in sample.columns:
        agg["pred_mixture"] = "first"
    for analyte in ANALYTES:
        agg[f"pred_conc_{analyte}"] = ["mean", "std"]
    group = sample.groupby("group_id").agg(agg)
    group.columns = [
        col[0] if isinstance(col, tuple) and col[1] == "first"
        else f"{col[0]}_{'sd' if col[1] == 'std' else col[1]}"
        if isinstance(col, tuple) else col
        for col in group.columns
    ]
    return sample, group.reset_index().fillna(0.0)


def regression_summary(group_df, label="all"):
    """Summarize concentration prediction metrics. Author: Xuanting Liu."""
    rows = []
    for analyte in ANALYTES:
        y = group_df[f"conc_{analyte}"].to_numpy(float)
        p = group_df[f"pred_conc_{analyte}_mean"].to_numpy(float)
        rows.append({
            "level": label,
            "analyte": analyte,
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": float(r2_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def plot_pred_vs_true(group_df, out_png, title):
    """Plot grouped predicted-vs-true concentration panels. Author: Xuanting Liu."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, analyte in zip(axes, ANALYTES):
        y = group_df[f"conc_{analyte}"].to_numpy(float)
        p = group_df[f"pred_conc_{analyte}_mean"].to_numpy(float)
        err_col = f"pred_conc_{analyte}_sd"
        err = group_df[err_col].to_numpy(float) if err_col in group_df else np.zeros_like(p)
        ax.errorbar(y, p, yerr=err, fmt="o", ms=4, capsize=3, alpha=0.85)
        lim = [0, max(1.0, float(np.nanmax([y.max(), p.max()]))) * 1.1]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"True {analyte} (uM)")
        ax.set_ylabel(f"Pred {analyte} (uM)")
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
