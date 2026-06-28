import os
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ca_paper_plsr import fit_class_routed_plsr, predict_by_mixture
from src.utils import (
    CLASS_LABELS,
    MODEL_MIXTURES,
    balanced_holdout_split,
    build_result_tables,
    filter_mix_conc,
    group_folds,
    make_group_id,
    normalize_spectra,
    peak_ratio_features,
    plot_pred_vs_true,
    read_mix_spectra,
    regression_summary,
)


BYOL_CONFIG = {
    "n_folds": 3,
    "pca_components": 24,
    "logreg_C": 1.0,
    "random_state": 2026,
}


def _load_byol_classification_data(data_dir, conc_threshold=None, mix_only=False, present_conc_range=None):
    """Load spectra for BYOL classification mode. Author: Xuanting Liu."""
    raman_shift, intensity, conc, groups, mixtures = read_mix_spectra(data_dir)
    if conc_threshold is not None and isinstance(conc_threshold, (int, float)):
        keep = conc.sum(axis=1) <= conc_threshold
        intensity, conc, groups, mixtures = intensity[keep], conc[keep], groups[keep], mixtures[keep]
    intensity, conc, groups, mixtures = filter_mix_conc(
        intensity, conc, groups, mixtures,
        mix_only=mix_only, present_conc_range=present_conc_range)
    group_ids = np.array([make_group_id(m, c) for m, c in zip(mixtures, conc)])
    fold_lookup = group_folds(group_ids, mixtures, n_splits=BYOL_CONFIG["n_folds"])
    df = pd.DataFrame({
        "group_id": group_ids,
        "mixture": mixtures,
        "conc_DA": conc[:, 0],
        "conc_E": conc[:, 1],
        "conc_NE": conc[:, 2],
        "outer_fold": [fold_lookup[g] for g in group_ids],
    })
    x_class = normalize_spectra(raman_shift, intensity, minmax=True)
    x_spectrum = normalize_spectra(raman_shift, intensity, peak_position=920, peak_range=20)
    x_ratio = peak_ratio_features(raman_shift, intensity)
    return raman_shift, x_class, x_spectrum, x_ratio, conc, groups, mixtures, group_ids, df


def _make_classifier(config):
    """Create the BYOL classification backend. Author: Xuanting Liu."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=int(config["pca_components"]), random_state=config["random_state"])),
        ("clf", LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            C=float(config["logreg_C"]),
        )),
    ])


def _classification_tables(df, pred_labels, prob, labels):
    """Build BYOL classification output tables. Author: Xuanting Liu."""
    sample = df.copy()
    sample["pred_mixture"] = pred_labels
    for j, label in enumerate(labels):
        sample[f"p_{label}"] = prob[:, j]
    group = (
        sample.groupby("group_id")
        .agg({
            "mixture": "first",
            "pred_mixture": lambda s: s.value_counts().index[0],
            "conc_DA": "first",
            "conc_E": "first",
            "conc_NE": "first",
        })
        .reset_index()
    )
    return sample, group


def _plot_confusion(y_true, y_pred, labels, out_png, title):
    """Plot a compact confusion matrix. Author: Xuanting Liu."""
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def _fit_byol_classifier_oof(x_class, df, config, include_blank=True):
    """Run grouped OOF BYOL classification. Author: Xuanting Liu."""
    labels = CLASS_LABELS if include_blank else MODEL_MIXTURES
    label_to_idx = {label: i for i, label in enumerate(labels)}
    keep = df["mixture"].isin(labels).to_numpy()
    mapped = df.loc[keep, "mixture"].map(label_to_idx).to_numpy(int)
    x = x_class[keep]
    d = df.loc[keep].reset_index(drop=True)
    preds = np.zeros(len(d), dtype=int)
    probs = np.zeros((len(d), len(labels)), dtype=float)
    fold_models = []
    for fold in range(int(config["n_folds"])):
        test = d["outer_fold"].to_numpy() == fold
        train = ~test
        clf = _make_classifier(config)
        clf.fit(x[train], mapped[train])
        pred = clf.predict(x[test])
        prob_local = clf.predict_proba(x[test])
        prob = np.zeros((test.sum(), len(labels)), dtype=float)
        for local_j, class_j in enumerate(clf.named_steps["clf"].classes_):
            prob[:, class_j] = prob_local[:, local_j]
        preds[test] = pred
        probs[test] = prob
        fold_models.append(clf)
        print(f"  Fold {fold}: accuracy={accuracy_score(mapped[test], pred):.4f}")
    pred_labels = np.array([labels[i] for i in preds])
    true_labels = d["mixture"].to_numpy()
    print(f"  OOF accuracy: {accuracy_score(true_labels, pred_labels):.4f}")
    return d, true_labels, pred_labels, probs, labels, fold_models


def BYOLPipeline(data_dir, model_dir, conc_threshold=None, mix_only=False, present_conc_range=None, stage1=True, stage2=True, re_training=False, config=None, plot=True):
    """Run the BYOL classification pipeline. Author: Xuanting Liu."""
    cfg = BYOL_CONFIG.copy()
    if config:
        cfg.update(config)
    os.makedirs("visualizations", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print("=" * 60)
    print("BYOL Pipeline - classification")
    print("=" * 60)
    print("  Stage 1 is classification-backend preparation; no quantification task is run.")

    _, x_class, _, _, _, _, _, _, df = _load_byol_classification_data(
        data_dir, conc_threshold=conc_threshold, mix_only=mix_only,
        present_conc_range=present_conc_range)
    if not stage2:
        return {"data": df, "config": cfg}

    df_cls, y_true, y_pred, probs, labels, models = _fit_byol_classifier_oof(
        x_class, df, cfg, include_blank=True)
    sample_df, group_df = _classification_tables(df_cls, y_pred, probs, labels)
    sample_path = "reports/BYOL_Class_Spectra.csv"
    group_path = "reports/BYOL_Class_Group.csv"
    sample_df.to_csv(sample_path, index=False, encoding="utf-8-sig")
    group_df.to_csv(group_path, index=False, encoding="utf-8-sig")
    if plot:
        _plot_confusion(y_true, y_pred, labels, "visualizations/BYOL_Stage2_Confusion_8class.png", "BYOL Classification")
    payload = {
        "method": "byol_classification",
        "config": cfg,
        "labels": labels,
        "fold_models": models,
        "sample": sample_df,
        "group": group_df,
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    model_path = os.path.join(model_dir, "byol_classification.joblib")
    joblib.dump(payload, model_path)
    print(f"  Exported {sample_path}")
    print(f"  Exported {group_path}")
    print(f"  Saved {model_path}")
    return payload


def CA_Paper_Full_Pipeline(data_dir, model_dir, conc_threshold=None, mix_only=False, present_conc_range=(10, 20), config=None, plot=True, dataset="MPAU", re_training=False, stage1=True):
    """Run BYOL classifier followed by predicted-class PLSR quantification. Author: Xuanting Liu."""
    cfg = BYOL_CONFIG.copy()
    if config:
        cfg.update(config)
    os.makedirs("visualizations", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print("=" * 60)
    print("CA Paper Full Pipeline - BYOL classifier + class-routed PLSR")
    print("=" * 60)

    _, x_class, x_spectrum, x_ratio, y_conc, _, mixtures, group_ids, df = _load_byol_classification_data(
        data_dir, conc_threshold=conc_threshold, mix_only=mix_only,
        present_conc_range=present_conc_range)
    keep = df["mixture"].isin(MODEL_MIXTURES).to_numpy()
    x_class, x_spectrum, x_ratio = x_class[keep], x_spectrum[keep], x_ratio[keep]
    y_conc, mixtures, group_ids = y_conc[keep], mixtures[keep], group_ids[keep]
    df = df.loc[keep].reset_index(drop=True)
    train_mask, val_mask = balanced_holdout_split(group_ids, val_fraction=0.30, random_state=cfg["random_state"])
    print(f"  Train spectra: {train_mask.sum()}, validation spectra: {val_mask.sum()}")

    label_to_idx = {label: i for i, label in enumerate(MODEL_MIXTURES)}
    y_train = np.array([label_to_idx[m] for m in mixtures[train_mask]])
    clf = _make_classifier(cfg)
    clf.fit(x_class[train_mask], y_train)
    pred_idx = clf.predict(x_class[val_mask])
    pred_mix = np.array([MODEL_MIXTURES[i] for i in pred_idx])
    true_mix = mixtures[val_mask]
    print(f"  Holdout class accuracy: {accuracy_score(true_mix, pred_mix):.4f}")

    quant_models, model_table = fit_class_routed_plsr(
        x_spectrum, x_ratio, y_conc, mixtures, train_mask, n_components=5)
    pred_conc = predict_by_mixture(quant_models, pred_mix, x_spectrum[val_mask], x_ratio[val_mask])

    df_val = df.loc[val_mask].copy()
    df_val["split"] = "validation"
    df_val["pred_mixture"] = pred_mix
    df_val["model_mixture"] = pred_mix
    sample_df, group_df = build_result_tables(df_val, pred_conc)
    summary_df = regression_summary(group_df, "predicted_class")
    cls_df = df_val[["group_id", "mixture", "pred_mixture", "conc_DA", "conc_E", "conc_NE"]].copy()

    sample_df.to_csv("reports/CA_Paper_Full_Pipeline_Sample.csv", index=False, encoding="utf-8-sig")
    group_df.to_csv("reports/CA_Paper_Full_Pipeline_Group.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv("reports/CA_Paper_Full_Pipeline_Summary.csv", index=False, encoding="utf-8-sig")
    cls_df.to_csv("reports/CA_Paper_Full_Pipeline_Classification.csv", index=False, encoding="utf-8-sig")
    model_table.to_csv("reports/CA_Paper_Full_Pipeline_ModelTable.csv", index=False, encoding="utf-8-sig")
    if plot:
        plot_pred_vs_true(group_df, "visualizations/CA_Paper_Full_Pipeline_Pred_vs_True.png", "BYOL Class-Routed PLSR")
        _plot_confusion(true_mix, pred_mix, MODEL_MIXTURES, "visualizations/CA_Paper_Full_Pipeline_Confusion.png", "Full Pipeline Classification")

    payload = {
        "method": "byol_classifier_class_routed_plsr",
        "classifier": clf,
        "quant_models": quant_models,
        "model_table": model_table,
        "summary": summary_df,
        "sample_concentration": sample_df,
        "group_concentration": group_df,
        "classification": cls_df,
        "class_accuracy": float(accuracy_score(true_mix, pred_mix)),
    }
    model_path = os.path.join(model_dir, "ca_paper_full_pipeline.joblib")
    joblib.dump(payload, model_path)
    print(summary_df.to_string(index=False))
    print(f"  Saved {model_path}")
    return payload
