import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression

from src.utils import (
    ANALYTES,
    MODEL_MIXTURES,
    SINGLE_MIXTURES,
    balanced_holdout_split,
    build_result_tables,
    filter_mix_conc,
    make_group_id,
    normalize_spectra,
    peak_ratio_features,
    plot_pred_vs_true,
    read_mix_spectra,
    regression_summary,
)


def _fit_single_ratio(x_ratio, y_conc, mixture):
    """Fit a single-component linear ratio calibration. Author: Xuanting Liu."""
    target = ANALYTES.index(mixture)
    model = LinearRegression().fit(x_ratio, y_conc[:, target])
    return {"model": model, "target": target, "model_type": "linear_peak_ratio"}


def _predict_single_ratio(info, x_ratio, out):
    """Predict one single-component concentration column. Author: Xuanting Liu."""
    out[:, info["target"]] = info["model"].predict(x_ratio)
    return out


def _fit_plsr_subset(x_spectrum, y_conc, mixture, n_components=5):
    """Fit a chemically constrained multi-output PLSR model. Author: Xuanting Liu."""
    present = [ANALYTES.index(a) for a in mixture.split("+")]
    n_comp = min(n_components, x_spectrum.shape[0] - 1, x_spectrum.shape[1], len(present))
    n_comp = max(1, n_comp)
    model = PLSRegression(n_components=n_comp, scale=True)
    model.fit(x_spectrum, y_conc[:, present])
    return {
        "model": model,
        "targets": present,
        "n_components": n_comp,
        "model_type": "multi_output_plsr",
    }


def _predict_plsr_subset(info, x_spectrum, out):
    """Predict present analyte columns from a PLSR model. Author: Xuanting Liu."""
    pred = np.asarray(info["model"].predict(x_spectrum), dtype=float)
    if pred.ndim == 1:
        pred = pred[:, None]
    for local_j, target_j in enumerate(info["targets"]):
        out[:, target_j] = pred[:, local_j]
    return out


def fit_class_routed_plsr(x_spectrum, x_ratio, y_conc, mixtures, train_mask, n_components=5):
    """Fit per-class calibration models used by unmixing pipelines. Author: Xuanting Liu."""
    models, rows = {}, []
    for mix in MODEL_MIXTURES:
        mask = train_mask & (mixtures == mix)
        if mask.sum() < 2:
            continue
        if mix in SINGLE_MIXTURES:
            info = _fit_single_ratio(x_ratio[mask], y_conc[mask], mix)
            target = mix
            n_comp = np.nan
        else:
            info = _fit_plsr_subset(x_spectrum[mask], y_conc[mask], mix, n_components)
            target = "+".join(ANALYTES[j] for j in info["targets"])
            n_comp = info["n_components"]
        models[mix] = info
        rows.append({
            "model_mixture": mix,
            "model_type": info["model_type"],
            "target": target,
            "n_components": n_comp,
            "n_train": int(mask.sum()),
        })
    return models, pd.DataFrame(rows)


def predict_by_mixture(models, pred_mixtures, x_spectrum, x_ratio):
    """Route spectra to per-class calibration models. Author: Xuanting Liu."""
    pred = np.zeros((len(pred_mixtures), len(ANALYTES)), dtype=float)
    for mix in np.unique(pred_mixtures):
        rows = np.where(pred_mixtures == mix)[0]
        info = models.get(mix)
        if info is None:
            continue
        if mix in SINGLE_MIXTURES:
            pred[rows] = _predict_single_ratio(info, x_ratio[rows], pred[rows])
        else:
            pred[rows] = _predict_plsr_subset(info, x_spectrum[rows], pred[rows])
    return np.maximum(pred, 0.0)


def _prepare_unmixing_data(data_dir, mix_only=False, present_conc_range=None):
    """Load, filter, and featurize mixture spectra. Author: Xuanting Liu."""
    raman_shift, intensity, conc, groups, mixtures = read_mix_spectra(data_dir)
    intensity, conc, groups, mixtures = filter_mix_conc(
        intensity, conc, groups, mixtures, mix_only=mix_only,
        present_conc_range=present_conc_range,
    )
    keep = np.isin(mixtures, MODEL_MIXTURES)
    if not keep.all():
        print(f"  Removing {(~keep).sum()} non-model spectra, e.g. BA.")
    intensity, conc, groups, mixtures = intensity[keep], conc[keep], groups[keep], mixtures[keep]
    x_spectrum = normalize_spectra(raman_shift, intensity, peak_position=920, peak_range=20)
    x_ratio = peak_ratio_features(raman_shift, intensity)
    group_ids = np.array([make_group_id(m, c) for m, c in zip(mixtures, conc)])
    df = pd.DataFrame({
        "group_id": group_ids,
        "mixture": mixtures,
        "conc_DA": conc[:, 0],
        "conc_E": conc[:, 1],
        "conc_NE": conc[:, 2],
    })
    return raman_shift, x_spectrum, x_ratio, conc, groups, mixtures, group_ids, df


def CA_Paper_PLSR_Unmixing(data_dir, model_dir, plot=True, mix_only=False, present_conc_range=None):
    """Plain holdout PLSR unmixing. Author: Xuanting Liu."""
    print("=" * 60)
    print("CA Paper PLSR Unmixing - plain holdout validation")
    print("=" * 60)
    os.makedirs("visualizations", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    raman_shift, x_spectrum, x_ratio, y_conc, groups, mixtures, group_ids, df = _prepare_unmixing_data(
        data_dir, mix_only=mix_only, present_conc_range=present_conc_range)
    train_mask, val_mask = balanced_holdout_split(group_ids, val_fraction=0.30, random_state=2026)
    print(f"  Spectra: {len(mixtures)}, groups: {df['group_id'].nunique()}")
    print(f"  Train spectra: {train_mask.sum()}, validation spectra: {val_mask.sum()}")

    models, model_table = fit_class_routed_plsr(x_spectrum, x_ratio, y_conc, mixtures, train_mask)
    pred_val = predict_by_mixture(models, mixtures[val_mask], x_spectrum[val_mask], x_ratio[val_mask])
    df_val = df.loc[val_mask].copy()
    df_val["split"] = "validation"
    df_val["model_mixture"] = df_val["mixture"]
    sample_df, group_df = build_result_tables(df_val, pred_val)

    summary_parts = [regression_summary(group_df, "all")]
    for mix in MODEL_MIXTURES:
        sub = group_df[group_df["model_mixture"] == mix]
        if len(sub):
            summary_parts.append(regression_summary(sub, mix))
    summary_df = pd.concat(summary_parts, ignore_index=True)

    sample_path = "reports/PLSR_Unmixing_Holdout_Sample.csv"
    group_path = "reports/PLSR_Unmixing_Holdout_Group.csv"
    summary_path = "reports/PLSR_Unmixing_Holdout_Summary.csv"
    table_path = "reports/PLSR_Unmixing_ModelTable.csv"
    sample_df.to_csv(sample_path, index=False, encoding="utf-8-sig")
    group_df.to_csv(group_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    model_table.to_csv(table_path, index=False, encoding="utf-8-sig")
    if plot:
        plot_pred_vs_true(group_df, "visualizations/PLSR_Unmixing_Holdout_Pred_vs_True.png", "PLSR Unmixing Holdout")

    payload = {
        "method": "plain_holdout_plsr_unmixing",
        "models": models,
        "model_table": model_table,
        "summary": summary_df,
        "sample_concentration": sample_df,
        "group_concentration": group_df,
        "raman_shift": raman_shift,
    }
    model_path = os.path.join(model_dir, "ca_paper_plsr_unmixing.joblib")
    joblib.dump(payload, model_path)
    print(summary_df.to_string(index=False))
    print(f"  Saved {model_path}")
    return payload


UNMIX_ANALYTES = ANALYTES
UNMIX_MODEL_MIXTURES = MODEL_MIXTURES
UNMIX_SINGLE_MIXTURES = SINGLE_MIXTURES
UNMIX_N_OUTER = 3
UNMIX_RANDOM_STATE = 2026


def _read_mpau_mix_spectra(data_dir):
    """Read MPAU mixture spectra for legacy BYOL routines. Author: Xuanting Liu."""
    return read_mix_spectra(data_dir)


def _filter_mix_conc(Intensity, Concentrations, Groups, Mixtures,
                     mix_only=False, present_conc_range=None):
    """Filter mixture spectra for legacy BYOL routines. Author: Xuanting Liu."""
    return filter_mix_conc(
        Intensity, Concentrations, Groups, Mixtures,
        mix_only=mix_only, present_conc_range=present_conc_range,
    )


def _umx_make_group_id(mixture, da, e, ne):
    """Build a legacy concentration group id. Author: Xuanting Liu."""
    return f"{mixture}|{da}|{e}|{ne}"


def _umx_present_analyte_indices(mixture):
    """Return analyte indices present in a mixture label. Author: Xuanting Liu."""
    return [UNMIX_ANALYTES.index(a) for a in mixture.split("+")]


def _umx_two_peak_ratio_features(Raman_Shift, Intensity):
    """Build 1480/920 and 1388/920 ratio features. Author: Xuanting Liu."""
    return peak_ratio_features(
        Raman_Shift, Intensity,
        centers=(1480, 1388), denominator=920, peak_range=20,
    )


def _umx_balanced_holdout_split(group_ids, val_fraction=0.30,
                                random_state=2026):
    """Legacy balanced holdout split wrapper. Author: Xuanting Liu."""
    return balanced_holdout_split(
        group_ids, val_fraction=val_fraction, random_state=random_state)


def _umx_make_ratio_targets(y_conc):
    """Convert concentration targets to component ratios. Author: Xuanting Liu."""
    y = np.asarray(y_conc, dtype=float)
    total = y.sum(axis=1, keepdims=True)
    return np.divide(y, total, out=np.zeros_like(y), where=total > 0)


def _umx_normalize_ratio_pred(pred_ratio):
    """Normalize predicted component ratios. Author: Xuanting Liu."""
    z = np.maximum(np.asarray(pred_ratio, dtype=float), 0.0)
    total = z.sum(axis=1, keepdims=True)
    bad = total.squeeze() <= 1e-12
    if np.any(bad):
        z[bad, :] = 1.0 / z.shape[1]
        total = z.sum(axis=1, keepdims=True)
    return z / total


def _umx_build_tables(df_model, pred, Y_conc=None, response_mode="concentration"):
    """Build legacy sample and group result tables. Author: Xuanting Liu."""
    base_cols = ["split", "group_id", "mixture",
                 "conc_DA", "conc_E", "conc_NE"]
    for col in ("model_mixture", "model_type", "pred_mixture"):
        if col in df_model.columns:
            base_cols.append(col)
    sample_df = df_model[base_cols].copy()
    if response_mode == "concentration":
        for j, analyte in enumerate(UNMIX_ANALYTES):
            sample_df[f"pred_conc_{analyte}"] = pred[:, j]
    else:
        true_ratio = _umx_make_ratio_targets(Y_conc)
        pred_ratio = _umx_normalize_ratio_pred(pred)
        for j, analyte in enumerate(UNMIX_ANALYTES):
            sample_df[f"true_ratio_{analyte}"] = true_ratio[:, j]
            sample_df[f"pred_ratio_{analyte}"] = pred_ratio[:, j]

    agg = {
        "split": "first", "mixture": "first",
        "conc_DA": "first", "conc_E": "first", "conc_NE": "first",
    }
    for col in ("model_mixture", "model_type", "pred_mixture"):
        if col in sample_df.columns:
            agg[col] = "first"
    if response_mode == "concentration":
        for analyte in UNMIX_ANALYTES:
            agg[f"pred_conc_{analyte}"] = "mean"
    else:
        for analyte in UNMIX_ANALYTES:
            agg[f"true_ratio_{analyte}"] = "first"
            agg[f"pred_ratio_{analyte}"] = "mean"

    group_df = sample_df.groupby("group_id", as_index=False).agg(agg)
    if response_mode == "concentration":
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


def _umx_continuous_summary(result_df, response_mode, level_name):
    """Compute legacy concentration or ratio summary. Author: Xuanting Liu."""
    if response_mode == "concentration":
        true_cols = [f"conc_{a}" for a in UNMIX_ANALYTES]
        pred_cols = [f"pred_conc_{a}" for a in UNMIX_ANALYTES]
    else:
        true_cols = [f"true_ratio_{a}" for a in UNMIX_ANALYTES]
        pred_cols = [f"pred_ratio_{a}" for a in UNMIX_ANALYTES]
    yt = result_df[true_cols].to_numpy(dtype=float)
    yp = result_df[pred_cols].to_numpy(dtype=float)
    row = {
        "response_mode": response_mode,
        "level": level_name,
        "n": len(result_df),
        "global_MAE": np.mean(np.abs(yp - yt)),
        "global_RMSE": np.sqrt(mean_squared_error(yt.reshape(-1), yp.reshape(-1))),
    }
    for j, analyte in enumerate(UNMIX_ANALYTES):
        row[f"{analyte}_MAE"] = np.mean(np.abs(yp[:, j] - yt[:, j]))
        row[f"{analyte}_RMSE"] = np.sqrt(mean_squared_error(yt[:, j], yp[:, j]))
        row[f"{analyte}_bias"] = np.mean(yp[:, j] - yt[:, j])
        present = yt[:, j] > 0
        row[f"{analyte}_R2"] = (
            r2_score(yt[present, j], yp[present, j])
            if present.sum() > 1 else np.nan)
    return pd.DataFrame([row])


def _umx_save_payload(payload, model_dir, filename):
    """Save a model payload with a visualization fallback. Author: Xuanting Liu."""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, filename)
    try:
        joblib.dump(payload, model_path)
        return model_path
    except PermissionError:
        os.makedirs("visualizations", exist_ok=True)
        fallback = os.path.join("visualizations", filename)
        joblib.dump(payload, fallback)
        return fallback


def _umx_plot_holdout_pred(group_conc, out_png):
    """Plot legacy group-level predicted-vs-true panels. Author: Xuanting Liu."""
    plot_pred_vs_true(group_conc.rename(columns={
        f"pred_conc_{a}": f"pred_conc_{a}_mean"
        for a in UNMIX_ANALYTES
    }), out_png, "Holdout Prediction")


def _print_per_group_mse(grp_df, tag=""):
    """Print per-group MSE ranking for legacy BYOL routines. Author: Xuanting Liu."""
    rows = []
    for _, row in grp_df.iterrows():
        true = np.array([row.get(f"conc_{a}", row.get(f"true_{a}", 0.0))
                         for a in UNMIX_ANALYTES], dtype=float)
        pred = np.array([row.get(f"pred_conc_{a}", row.get(f"m_{a}", 0.0))
                         for a in UNMIX_ANALYTES], dtype=float)
        rows.append((row["group_id"], mean_squared_error(true, pred)))
    rows.sort(key=lambda x: x[1])
    print(f"Per-group MSE ranking ({tag}):")
    for group_id, mse in rows:
        print(f"  {group_id}: {mse:.4f}")
