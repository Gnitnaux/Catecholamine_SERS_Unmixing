import os
import joblib
import numpy as np
import pandas as pd
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
