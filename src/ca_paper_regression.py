import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

UNMIX_ANALYTES = ["DA", "E", "NE"]
UNMIX_MODEL_MIXTURES = ["DA", "E", "NE", "DA+E", "DA+NE", "E+NE", "DA+E+NE"]
UNMIX_SINGLE_MIXTURES = ["DA", "E", "NE"]


def _parse_folder_name(folder):
    """Parse an MPAU mixture folder name. Author: Xuanting Liu."""
    parts = folder.split("_")
    da = float(parts[0].replace("uM", ""))
    e = float(parts[1].replace("uM", ""))
    ne = float(parts[2].replace("uM", ""))
    present = []
    if da > 0:
        present.append("DA")
    if e > 0:
        present.append("E")
    if ne > 0:
        present.append("NE")
    mixture = "BA" if len(present) == 0 else "+".join(present)
    return da, e, ne, mixture


def _read_mpau_mix_spectra(data_dir):
    """Read all spectra from data_MPAU_mix folders. Author: Xuanting Liu."""
    data_dict = {}
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        spectra = []
        for file in os.listdir(folder_path):
            if not file.endswith(".csv"):
                continue
            path = os.path.join(folder_path, file)
            data = pd.read_csv(path, sep=",", skiprows=[0],
                               names=["Raman Shift", "Intensity"],
                               encoding="GBK")
            data_cut = data[(data["Raman Shift"] >= 330) &
                            (data["Raman Shift"] <= 1600)]
            spectra.append(data_cut)
        if spectra:
            data_dict[folder] = spectra

    intensity, concentrations, groups, mixtures = [], [], [], []
    raman_shift = None
    for folder, spectra in data_dict.items():
        da, e, ne, mixture = _parse_folder_name(folder)
        for sp in spectra:
            if raman_shift is None:
                raman_shift = sp["Raman Shift"].values
            intensity.append(sp["Intensity"].values)
            concentrations.append([da, e, ne])
            groups.append(folder)
            mixtures.append(mixture)
    return (
        raman_shift,
        np.array(intensity),
        np.array(concentrations, dtype=float),
        np.array(groups),
        np.array(mixtures),
    )


def _spectra_normalization(raman_shift, intensity, peak_position=920, peak_range=20):
    """Normalize spectra against the selected peak window. Author: Xuanting Liu."""
    peak_idx = np.where((raman_shift >= peak_position - peak_range) &
                        (raman_shift <= peak_position + peak_range))[0]
    out = intensity.copy()
    for i in range(intensity.shape[0]):
        peak = np.max(intensity[i, peak_idx])
        if peak != 0:
            out[i, :] = intensity[i, :] / peak
    return out


def _filter_mix_conc(intensity, concentrations, groups, mixtures,
                     mix_only=False, present_conc_range=None):
    """Filter spectra by mixture type and present concentration range. Author: Xuanting Liu."""
    keep = np.ones(len(intensity), dtype=bool)
    if mix_only:
        keep &= np.isin(mixtures, ["DA+E", "DA+NE", "E+NE", "DA+E+NE"])
    if present_conc_range is not None:
        lo, hi = present_conc_range
        for i in range(len(concentrations)):
            if not keep[i]:
                continue
            for val in concentrations[i]:
                if val > 0 and not (lo <= val <= hi):
                    keep[i] = False
                    break
    print(f"  mix_filter(mix_only={mix_only}, range={present_conc_range}): "
          f"kept {keep.sum()}/{len(keep)} spectra "
          f"({np.unique(groups[keep]).size} groups)")
    return intensity[keep], concentrations[keep], groups[keep], mixtures[keep]


def _umx_make_group_id(mixture, da, e, ne):
    """Build a concentration group id. Author: Xuanting Liu."""
    return f"{mixture}|{da}|{e}|{ne}"


def _umx_present_analyte_indices(mixture):
    """Return analyte indices present in a mixture label. Author: Xuanting Liu."""
    return [UNMIX_ANALYTES.index(a) for a in mixture.split("+")]


def _umx_marker_ratio_features(raman_shift, intensity, centers,
                               denominator=920, peak_range=20):
    """Build peak marker ratio features. Author: Xuanting Liu."""
    den_idx = np.where((raman_shift >= denominator - peak_range) &
                       (raman_shift <= denominator + peak_range))[0]
    den = np.max(intensity[:, den_idx], axis=1) + 1e-8
    feats = []
    for center in centers:
        idx = np.where((raman_shift >= center - peak_range) &
                       (raman_shift <= center + peak_range))[0]
        feats.append(np.max(intensity[:, idx], axis=1) / den)
    return np.column_stack(feats)


def _umx_two_peak_ratio_features(raman_shift, intensity):
    """Use 1480/920 and 1388/920 marker ratios. Author: Xuanting Liu."""
    return _umx_marker_ratio_features(
        raman_shift, intensity, centers=[1480, 1388],
        denominator=920, peak_range=20)


def _umx_balanced_holdout_split(group_ids, val_fraction=0.30,
                                random_state=2026):
    """Sample-level holdout split inside every concentration group. Author: Xuanting Liu."""
    rng = np.random.default_rng(random_state)
    train_mask = np.zeros(len(group_ids), dtype=bool)
    val_mask = np.zeros(len(group_ids), dtype=bool)
    for gid in sorted(np.unique(group_ids)):
        idx = np.where(group_ids == gid)[0]
        if len(idx) <= 1:
            train_mask[idx] = True
            continue
        idx = rng.permutation(idx)
        n_val = int(round(len(idx) * val_fraction))
        n_val = min(max(1, n_val), len(idx) - 1)
        val_mask[idx[:n_val]] = True
        train_mask[idx[n_val:]] = True
    return train_mask, val_mask


def _umx_rmse(y_true, y_pred):
    """Compute RMSE. Author: Xuanting Liu."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def _umx_safe_r2(y_true, y_pred):
    """Compute R2 unless the target is constant. Author: Xuanting Liu."""
    if len(np.unique(y_true)) <= 1:
        return np.nan
    return r2_score(y_true, y_pred)


def _umx_build_tables(df_model, pred):
    """Build original sample-level and group-level result tables. Author: Xuanting Liu."""
    base_cols = ["split", "group_id", "mixture",
                 "conc_DA", "conc_E", "conc_NE",
                 "model_mixture", "model_type"]
    sample_df = df_model[base_cols].copy()
    for j, analyte in enumerate(UNMIX_ANALYTES):
        sample_df[f"pred_conc_{analyte}"] = pred[:, j]
    agg = {
        "split": "first", "mixture": "first",
        "conc_DA": "first", "conc_E": "first", "conc_NE": "first",
        "model_mixture": "first", "model_type": "first",
    }
    for analyte in UNMIX_ANALYTES:
        agg[f"pred_conc_{analyte}"] = "mean"
    group_df = sample_df.groupby("group_id", as_index=False).agg(agg)
    sd_df = sample_df.groupby("group_id", as_index=False)[
        [f"pred_conc_{a}" for a in UNMIX_ANALYTES]
    ].std(ddof=1)
    sd_df = sd_df.rename(columns={
        f"pred_conc_{a}": f"pred_conc_{a}_sd"
        for a in UNMIX_ANALYTES
    })
    group_df = group_df.merge(sd_df, on="group_id", how="left")
    for analyte in UNMIX_ANALYTES:
        group_df[f"pred_conc_{analyte}_sd"] = (
            group_df[f"pred_conc_{analyte}_sd"].fillna(0.0))
    return sample_df, group_df


def _umx_continuous_summary(result_df, level_name):
    """Compute original MAE/RMSE/bias/R2 summary. Author: Xuanting Liu."""
    true_cols = [f"conc_{a}" for a in UNMIX_ANALYTES]
    pred_cols = [f"pred_conc_{a}" for a in UNMIX_ANALYTES]
    yt = result_df[true_cols].to_numpy(dtype=float)
    yp = result_df[pred_cols].to_numpy(dtype=float)
    row = {
        "response_mode": "concentration",
        "level": level_name,
        "n": len(result_df),
        "global_MAE": np.mean(np.abs(yp - yt)),
        "global_RMSE": _umx_rmse(yt.reshape(-1), yp.reshape(-1)),
    }
    for j, analyte in enumerate(UNMIX_ANALYTES):
        row[f"{analyte}_MAE"] = np.mean(np.abs(yp[:, j] - yt[:, j]))
        row[f"{analyte}_RMSE"] = _umx_rmse(yt[:, j], yp[:, j])
        row[f"{analyte}_bias"] = np.mean(yp[:, j] - yt[:, j])
        present = yt[:, j] > 0
        row[f"{analyte}_R2"] = (
            _umx_safe_r2(yt[present, j], yp[present, j])
            if present.sum() > 1 else np.nan)
    return pd.DataFrame([row])


def _umx_plot_holdout_pred(group_conc, out_png):
    """Plot group-level predicted-vs-true panels. Author: Xuanting Liu."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, analyte in zip(axes, UNMIX_ANALYTES):
        true = group_conc[f"conc_{analyte}"].to_numpy(dtype=float)
        pred = group_conc[f"pred_conc_{analyte}"].to_numpy(dtype=float)
        sd = group_conc[f"pred_conc_{analyte}_sd"].to_numpy(dtype=float)
        ax.errorbar(true, pred, yerr=sd, fmt="o", ms=4.5,
                    alpha=0.85, capsize=3, elinewidth=0.9,
                    markeredgecolor="black", markeredgewidth=0.3)
        lim = [0, max(22, float(np.nanmax([true.max(), pred.max()]))) + 1]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"True {analyte} (uM)")
        ax.set_ylabel(f"Pred {analyte} (uM)")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


class _UnmixingCNN(nn.Module):
    """Original unmixing CNN. Author: Xuanting Liu."""

    def __init__(self, input_len, out_dim):
        """Initialize CNN layers. Author: Xuanting Liu."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(64, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """Run forward inference. Author: Xuanting Liu."""
        return self.fc(self.conv(x))


def _fit_cnn_unmixing(x_train, y_train, out_dim, epochs=500,
                      batch_size=32, lr=1e-3, patience=50,
                      min_delta=1e-4):
    """Fit the original CNN unmixing model. Author: Xuanting Liu."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _UnmixingCNN(x_train.shape[1], out_dim).to(device)
    x_t = torch.from_numpy(x_train).float().unsqueeze(1)
    y_t = torch.from_numpy(y_train.astype(np.float32)).float()
    loader = DataLoader(TensorDataset(x_t, y_t),
                        batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_loss = np.inf
    best_state = None
    stale_epochs = 0
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            pred = model(bx)
            loss = criterion(pred, by)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * bx.size(0)
        epoch_loss = total / len(x_train)
        if epoch_loss < best_loss - min_delta:
            best_loss = epoch_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if (epoch + 1) % 100 == 0:
            print(f"    CNN epoch {epoch + 1}/{epochs}, "
                  f"loss={epoch_loss:.4f}, best={best_loss:.4f}")
        if stale_epochs >= patience:
            print(f"    CNN early stop at epoch {epoch + 1}/{epochs}, "
                  f"best_loss={best_loss:.4f}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict_cnn_unmixing(model, x):
    """Predict with an unmixing CNN. Author: Xuanting Liu."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x).float().unsqueeze(1).to(device))
    return pred.cpu().numpy()


def _fit_predict_unmixing_method(method, x_spectrum, x_ratio, y_conc,
                                 mixtures, train_mask, val_mask):
    """Fit and predict one original per-mixture unmixing method. Author: Xuanting Liu."""
    pred_val = np.zeros((int(val_mask.sum()), len(UNMIX_ANALYTES)), dtype=float)
    pred_train = np.zeros((int(train_mask.sum()), len(UNMIX_ANALYTES)), dtype=float)
    val_pos = {idx: k for k, idx in enumerate(np.where(val_mask)[0])}
    train_pos = {idx: k for k, idx in enumerate(np.where(train_mask)[0])}
    model_rows, final_models = [], {}

    for mix in UNMIX_MODEL_MIXTURES:
        mix_mask = mixtures == mix
        tr = train_mask & mix_mask
        va = val_mask & mix_mask
        if not tr.any() or not va.any():
            continue
        val_rows = np.array([val_pos[i] for i in np.where(va)[0]])
        train_rows = np.array([train_pos[i] for i in np.where(tr)[0]])
        present = _umx_present_analyte_indices(mix)
        absent = [j for j in range(len(UNMIX_ANALYTES)) if j not in present]

        if method == "plsr":
            if mix in UNMIX_SINGLE_MIXTURES:
                target_j = present[0]
                model = LinearRegression()
                model.fit(x_ratio[tr], y_conc[tr, target_j])
                pred_val[val_rows, target_j] = model.predict(x_ratio[va])
                pred_train[train_rows, target_j] = model.predict(x_ratio[tr])
                final = LinearRegression().fit(
                    x_ratio[mix_mask], y_conc[mix_mask, target_j])
                model_type = "linear_1480_1388_over_920"
                selected = np.nan
                final_models[mix] = {
                    "model_type": model_type,
                    "target": UNMIX_ANALYTES[target_j],
                    "model": final,
                }
            else:
                n_comp = min(5, x_spectrum[tr].shape[0] - 1,
                             x_spectrum[tr].shape[1])
                n_comp = max(1, int(n_comp))
                model = PLSRegression(n_components=n_comp, scale=True)
                model.fit(x_spectrum[tr], y_conc[tr][:, present])
                val_local = model.predict(x_spectrum[va])
                train_local = model.predict(x_spectrum[tr])
                for local_j, target_j in enumerate(present):
                    pred_val[val_rows, target_j] = val_local[:, local_j]
                    pred_train[train_rows, target_j] = train_local[:, local_j]
                final = PLSRegression(n_components=n_comp, scale=True)
                final.fit(x_spectrum[mix_mask], y_conc[mix_mask][:, present])
                model_type = "multi_output_plsr_holdout"
                selected = n_comp
                final_models[mix] = {
                    "model_type": model_type,
                    "targets": [UNMIX_ANALYTES[j] for j in present],
                    "model": final,
                    "n_components": n_comp,
                }
        else:
            use_ratio = (mix in UNMIX_SINGLE_MIXTURES and method != "cnn")
            x_tr = x_ratio[tr] if use_ratio else x_spectrum[tr]
            x_va = x_ratio[va] if use_ratio else x_spectrum[va]
            x_all = x_ratio[mix_mask] if use_ratio else x_spectrum[mix_mask]
            y_tr = y_conc[tr][:, present]
            y_all = y_conc[mix_mask][:, present]
            if method == "rf":
                model = RandomForestRegressor(
                    n_estimators=100, max_depth=15, min_samples_split=5,
                    random_state=42, n_jobs=1)
                model_type = "random_forest_unmixing"
                selected = np.nan
            elif method == "svr":
                model = MultiOutputRegressor(
                    SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1))
                model_type = "svr_unmixing"
                selected = np.nan
            elif method == "cnn":
                model = _fit_cnn_unmixing(x_tr, y_tr, len(present))
                model_type = "cnn_unmixing"
                selected = np.nan
            else:
                raise ValueError(f"Unknown unmixing method: {method}")

            if method == "cnn":
                val_local = _predict_cnn_unmixing(model, x_va)
                train_local = _predict_cnn_unmixing(model, x_tr)
                final = model
            else:
                y_fit = y_tr.ravel() if (
                    method == "rf" and y_tr.shape[1] == 1) else y_tr
                model.fit(x_tr, y_fit)
                val_local = model.predict(x_va)
                train_local = model.predict(x_tr)
                if val_local.ndim == 1:
                    val_local = val_local.reshape(-1, 1)
                if train_local.ndim == 1:
                    train_local = train_local.reshape(-1, 1)
                final = clone(model)
                y_all_fit = y_all.ravel() if (
                    method == "rf" and y_all.shape[1] == 1) else y_all
                final.fit(x_all, y_all_fit)

            for local_j, target_j in enumerate(present):
                pred_val[val_rows, target_j] = val_local[:, local_j]
                pred_train[train_rows, target_j] = train_local[:, local_j]
            final_models[mix] = {
                "model_type": model_type,
                "targets": [UNMIX_ANALYTES[j] for j in present],
                "model": final,
            }

        for j in absent:
            pred_val[val_rows, j] = 0.0
            pred_train[train_rows, j] = 0.0
        model_rows.append({
            "method": method.upper(),
            "model_mixture": mix,
            "model_type": model_type,
            "target": "+".join(UNMIX_ANALYTES[j] for j in present),
            "selected_n_components": selected,
            "n_train": int(tr.sum()),
            "n_validation": int(va.sum()),
        })

    return (
        np.maximum(pred_val, 0.0),
        np.maximum(pred_train, 0.0),
        pd.DataFrame(model_rows),
        final_models,
    )


def _radar_metrics_from_groups(train_group, val_group):
    """Build Origin radar metrics from original group-level outputs. Author: Xuanting Liu."""
    rows = {}
    for analyte in UNMIX_ANALYTES:
        y_tr = train_group[f"conc_{analyte}"].to_numpy(float)
        p_tr = train_group[f"pred_conc_{analyte}"].to_numpy(float)
        y_te = val_group[f"conc_{analyte}"].to_numpy(float)
        p_te = val_group[f"pred_conc_{analyte}"].to_numpy(float)
        present_tr = y_tr > 0
        present_te = y_te > 0
        rmse = float(np.sqrt(mean_squared_error(y_te[present_te], p_te[present_te])))
        y_range = float(np.max(y_te[present_te]) - np.min(y_te[present_te]))
        rows[f"{analyte}_1-normalized RMSE"] = 1.0 - rmse / y_range if y_range > 0 else np.nan
        rows[f"{analyte}_R2Train"] = (
            float(r2_score(y_tr[present_tr], p_tr[present_tr]))
            if present_tr.sum() > 1 else np.nan)
        rows[f"{analyte}_R2Test"] = (
            float(r2_score(y_te[present_te], p_te[present_te]))
            if present_te.sum() > 1 else np.nan)
        rows[f"{analyte}_RPD"] = (
            float(np.std(y_te[present_te], ddof=1) / rmse)
            if rmse > 0 and present_te.sum() > 1 else np.nan)
        rows[f"{analyte}_Explained Variance"] = (
            float(explained_variance_score(y_te[present_te], p_te[present_te]))
            if present_te.sum() > 1 else np.nan)
    return rows


def CA_Paper_Unmixing_Models(data_dir, model_dir, methods=None,
                             random_state=2026, plot=True,
                             mix_only=False, present_conc_range=(10, 20)):
    """Compare PLSR/RF/SVR/CNN for component concentration unmixing. Author: Xuanting Liu."""
    if methods is None:
        methods = ["plsr", "rf", "svr", "cnn"]
    methods = [m.lower() for m in methods]
    os.makedirs("visualizations", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 60)
    print("CA Paper Unmixing Models - PLSR/RF/SVR/CNN")
    print("=" * 60)

    raman_shift, intensity, concentrations, groups, mixtures = \
        _read_mpau_mix_spectra(data_dir)
    intensity, concentrations, groups, mixtures = _filter_mix_conc(
        intensity, concentrations, groups, mixtures,
        mix_only=mix_only, present_conc_range=present_conc_range)
    keep = np.isin(mixtures, UNMIX_MODEL_MIXTURES)
    if not keep.all():
        print(f"  Removing {(~keep).sum()} non-model spectra, e.g. BA.")
        intensity = intensity[keep]
        concentrations = concentrations[keep]
        groups = groups[keep]
        mixtures = mixtures[keep]

    x_spectrum = _spectra_normalization(
        raman_shift, intensity, peak_position=920, peak_range=20)
    x_ratio = _umx_two_peak_ratio_features(raman_shift, intensity)
    y_conc = concentrations.copy()
    group_ids = np.array([
        _umx_make_group_id(mixtures[i], y_conc[i, 0], y_conc[i, 1], y_conc[i, 2])
        for i in range(len(mixtures))
    ])
    df_model = pd.DataFrame({
        "group_id": group_ids,
        "mixture": mixtures,
        "conc_DA": y_conc[:, 0],
        "conc_E": y_conc[:, 1],
        "conc_NE": y_conc[:, 2],
    })
    train_mask, val_mask = _umx_balanced_holdout_split(
        group_ids, val_fraction=0.30, random_state=random_state)
    print(f"  Spectra: {len(mixtures)}, groups: {df_model['group_id'].nunique()}")
    print(f"  Train spectra: {train_mask.sum()}")
    print(f"  Validation spectra: {val_mask.sum()}")

    all_summary, all_model_tables, radar_by_method = [], [], {}
    payload = {
        "method": "ca_paper_unmixing_models",
        "methods": methods,
        "train_mask": train_mask,
        "validation_mask": val_mask,
        "raman_shift": raman_shift,
        "results": {},
    }

    for method in methods:
        print(f"\n--- {method.upper()} unmixing ---")
        pred_val, pred_train, model_table, final_models = \
            _fit_predict_unmixing_method(
                method, x_spectrum, x_ratio, y_conc, mixtures,
                train_mask, val_mask)
        df_val = df_model.loc[val_mask].copy().reset_index(drop=True)
        df_val["split"] = "validation"
        df_val["model_mixture"] = df_val["mixture"]
        df_val["model_type"] = method.upper()
        sample_conc, group_conc = _umx_build_tables(df_val, pred_val)

        df_train = df_model.loc[train_mask].copy().reset_index(drop=True)
        df_train["split"] = "train"
        df_train["model_mixture"] = df_train["mixture"]
        df_train["model_type"] = method.upper()
        _sample_train, group_train = _umx_build_tables(df_train, pred_train)

        sample_out = sample_conc[
            ~sample_conc["mixture"].isin(UNMIX_SINGLE_MIXTURES)
        ].copy()
        group_out = group_conc[
            ~group_conc["mixture"].isin(UNMIX_SINGLE_MIXTURES)
        ].copy()
        train_group_out = group_train[
            ~group_train["mixture"].isin(UNMIX_SINGLE_MIXTURES)
        ].copy()
        model_table_out = model_table[
            ~model_table["model_mixture"].isin(UNMIX_SINGLE_MIXTURES)
        ].copy()
        summary = _umx_continuous_summary(group_out, "group")
        summary.insert(0, "method", method.upper())
        all_summary.append(summary)
        all_model_tables.append(model_table_out)
        radar_by_method[method.upper()] = _radar_metrics_from_groups(
            train_group_out, group_out)

        prefix_report = f"reports/CA_Paper_Unmixing_{method.upper()}"
        sample_out.to_csv(
            f"{prefix_report}_Sample.csv", index=False, encoding="utf-8-sig")
        group_out.to_csv(
            f"{prefix_report}_Group.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(
            f"{prefix_report}_Summary.csv", index=False, encoding="utf-8-sig")
        if plot:
            _umx_plot_holdout_pred(
                group_out,
                f"visualizations/CA_Paper_Unmixing_{method.upper()}_Pred_vs_True.png")

        payload["results"][method] = {
            "sample_concentration": sample_out,
            "group_concentration": group_out,
            "summary": summary,
            "model_table": model_table_out,
            "final_models": final_models,
        }
        print(summary.to_string(index=False))

    summary_df = pd.concat(all_summary, ignore_index=True)
    model_table_df = pd.concat(all_model_tables, ignore_index=True)
    summary_path = "reports/CA_Paper_Unmixing_Models_Summary.csv"
    model_table_path = "reports/CA_Paper_Unmixing_Models_ModelTable.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    model_table_df.to_csv(model_table_path, index=False, encoding="utf-8-sig")

    radar_df = pd.DataFrame({"Metric": list(next(iter(radar_by_method.values())).keys())})
    for method in ["PLSR", "RF", "SVR", "CNN"]:
        if method in radar_by_method:
            radar_df[method] = radar_df["Metric"].map(radar_by_method[method])
    radar_path = "reports/CA_Paper_Unmixing_Models_Radar.csv"
    radar_df.to_csv(radar_path, index=False, encoding="utf-8-sig")

    print(f"\nExported {summary_path}")
    print(f"Exported {model_table_path}")
    print(f"Exported {radar_path}")

    payload["summary"] = summary_df
    payload["model_table"] = model_table_df
    payload["radar"] = radar_df
    model_path = os.path.join(model_dir, "ca_paper_unmixing_models.joblib")
    joblib.dump(payload, model_path)
    print(f"Saved {model_path}")
    print("=" * 60)
    print("CA Paper Unmixing Models completed.")
    print("=" * 60)
    return payload
