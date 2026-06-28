import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

from src.ca_paper_plsr import _prepare_unmixing_data
from src.utils import (
    ANALYTES,
    MODEL_MIXTURES,
    balanced_holdout_split,
    build_result_tables,
    plot_pred_vs_true,
    regression_summary,
)


class _SmallCNNRegressor:
    """Lazy wrapper for a compact spectral CNN. Author: Xuanting Liu."""

    def __init__(self, n_features, n_outputs, random_state=2026):
        """Initialize wrapper metadata. Author: Xuanting Liu."""
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.random_state = random_state
        self.state_dict = None
        self.device = "cpu"

    def fit_predict(self, x_train, y_train, x_test):
        """Train the CNN and predict validation spectra. Author: Xuanting Liu."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        class Net(nn.Module):
            """Torch network body. Author: Xuanting Liu."""

            def __init__(self, n_outputs):
                """Initialize layers. Author: Xuanting Liu."""
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv1d(1, 24, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
                    nn.Conv1d(24, 48, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(), nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, n_outputs),
                )

            def forward(self, x):
                """Run forward inference. Author: Xuanting Liu."""
                return self.net(x)

        torch.manual_seed(self.random_state)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = str(device)
        model = Net(self.n_outputs).to(device)
        loader = DataLoader(
            TensorDataset(
                torch.tensor(x_train[:, None, :], dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
            ),
            batch_size=64, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(40):
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                loss = loss_fn(model(bx), by)
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(x_test[:, None, :], dtype=torch.float32).to(device)).cpu().numpy()
        self.state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        return np.maximum(pred, 0.0)


def _metric_row(method, y_true, y_pred):
    """Compute compact multi-output metrics. Author: Xuanting Liu."""
    rows = []
    for j, analyte in enumerate(ANALYTES):
        y = y_true[:, j]
        p = y_pred[:, j]
        rows.append({
            "method": method,
            "analyte": analyte,
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": float(r2_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        })
    return rows


def _fit_predict_plsr(x_train, y_train, x_test):
    """Fit direct multi-output PLSR. Author: Xuanting Liu."""
    n_comp = min(10, x_train.shape[0] - 1, x_train.shape[1], y_train.shape[1])
    n_comp = max(1, n_comp)
    model = PLSRegression(n_components=n_comp, scale=True)
    model.fit(x_train, y_train)
    return model, np.maximum(model.predict(x_test), 0.0), {"n_components": n_comp}


def _fit_predict_rf(x_train, y_train, x_test, random_state=2026):
    """Fit direct multi-output random forest. Author: Xuanting Liu."""
    model = RandomForestRegressor(
        n_estimators=120, max_depth=16, min_samples_split=4,
        random_state=random_state, n_jobs=1)
    model.fit(x_train, y_train)
    return model, np.maximum(model.predict(x_test), 0.0), {}


def _fit_predict_svr(x_train, y_train, x_test):
    """Fit direct multi-output SVR. Author: Xuanting Liu."""
    model = MultiOutputRegressor(SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.05))
    model.fit(x_train, y_train)
    return model, np.maximum(model.predict(x_test), 0.0), {}


def _fit_predict_cnn(x_train, y_train, x_test, random_state=2026):
    """Fit a compact 1D CNN regressor. Author: Xuanting Liu."""
    model = _SmallCNNRegressor(x_train.shape[1], y_train.shape[1], random_state=random_state)
    pred = model.fit_predict(x_train, y_train, x_test)
    return model, pred, {"device": model.device}


def CA_Paper_Unmixing_Models(data_dir, model_dir, methods=None, random_state=2026, plot=True, mix_only=False, present_conc_range=None):
    """Compare direct PLSR/RF/SVR/CNN component unmixing models. Author: Xuanting Liu."""
    methods = [m.lower() for m in (methods or ["plsr", "rf", "svr", "cnn"])]
    os.makedirs("visualizations", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    _, x_spectrum, x_ratio, y_conc, groups, mixtures, group_ids, df = _prepare_unmixing_data(
        data_dir, mix_only=mix_only, present_conc_range=present_conc_range)
    x = np.hstack([x_spectrum, x_ratio])
    train_mask, val_mask = balanced_holdout_split(group_ids, val_fraction=0.30, random_state=random_state)
    x_train, y_train = x[train_mask], y_conc[train_mask]
    x_val, y_val = x[val_mask], y_conc[val_mask]
    print(f"  Train spectra: {train_mask.sum()}, validation spectra: {val_mask.sum()}")

    runners = {
        "plsr": _fit_predict_plsr,
        "rf": lambda a, b, c: _fit_predict_rf(a, b, c, random_state=random_state),
        "svr": _fit_predict_svr,
        "cnn": lambda a, b, c: _fit_predict_cnn(a, b, c, random_state=random_state),
    }
    payload, metric_rows = {}, []
    for method in methods:
        if method not in runners:
            raise ValueError(f"Unknown method: {method}")
        print(f"\n--- {method.upper()} ---")
        model, pred, info = runners[method](x_train, y_train, x_val)
        method_df = df.loc[val_mask].copy()
        method_df["split"] = "validation"
        method_df["model_mixture"] = method_df["mixture"]
        sample_df, group_df = build_result_tables(method_df, pred)
        summary_df = regression_summary(group_df, method)
        metric_rows.extend(_metric_row(method, y_val, pred))
        sample_path = f"reports/CA_Paper_Unmixing_{method.upper()}_Sample.csv"
        group_path = f"reports/CA_Paper_Unmixing_{method.upper()}_Group.csv"
        summary_path = f"reports/CA_Paper_Unmixing_{method.upper()}_Summary.csv"
        sample_df.to_csv(sample_path, index=False, encoding="utf-8-sig")
        group_df.to_csv(group_path, index=False, encoding="utf-8-sig")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        if plot:
            plot_pred_vs_true(group_df, f"visualizations/CA_Paper_Unmixing_{method.upper()}.png", method.upper())
        payload[method] = {"model": model, "info": info, "sample": sample_df, "group": group_df, "summary": summary_df}
        print(summary_df.to_string(index=False))

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv("reports/CA_Paper_Unmixing_Models_Summary.csv", index=False, encoding="utf-8-sig")
    model_path = os.path.join(model_dir, "ca_paper_unmixing_models.joblib")
    joblib.dump(payload, model_path)
    print(f"\nSaved {model_path}")
    return payload
