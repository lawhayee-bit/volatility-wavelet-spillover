from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from arch import arch_model
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBClassifier, XGBRegressor


@dataclass
class ModelSpec:
    name: str
    estimator: object
    feature_set: str


class TorchRNNRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible RNN (LSTM / GRU) regressor backed by PyTorch.

    Each 2-D input row is treated as a *single-timestep* sequence so the
    rolling-window Pipeline remains fully tabular.  The temporal structure
    is already encoded in the HAR lags and wavelet features; the RNN acts
    as a learned nonlinear mapping over that feature space.

    Training uses Adam + MSELoss with chronological early-stopping on a
    held-out validation slice.
    """

    def __init__(
        self,
        rnn_type: str = "lstm",
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        lr: float = 1e-3,
        max_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 256,
        val_fraction: float = 0.10,
        random_state: int = 42,
        num_threads: int = 2,
    ):
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.num_threads = num_threads

    def _build_model(self, input_size: int):
        import torch.nn as nn

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        rnn = rnn_cls(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )
        d = 2 if self.bidirectional else 1
        head = nn.Linear(self.hidden_size * d, 1)

        class _Net(nn.Module):
            def __init__(self_, rnn_, head_):  # noqa: N805
                super().__init__()
                self_.rnn = rnn_
                self_.head = head_

            def forward(self_, x):  # noqa: N805
                out, _ = self_.rnn(x)
                return self_.head(out[:, -1, :]).squeeze(-1)

        return _Net(rnn, head)

    def fit(self, X, y):
        import torch

        torch.manual_seed(self.random_state)
        torch.set_num_threads(self.num_threads)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Normalize target to unit scale so gradients are well-conditioned regardless
        # of the target's absolute magnitude (e.g. log1p-transformed realized variance
        # lives near 0 and causes near-zero gradients without this normalization).
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) + 1e-8
        y_scaled = (y - self.y_mean_) / self.y_std_

        # Chronological validation split for early stopping
        n_val = max(1, int(len(X) * self.val_fraction))
        X_tr, X_val = X[:-n_val], X[-n_val:]
        y_tr, y_val = y_scaled[:-n_val], y_scaled[-n_val:]

        self.n_features_in_ = X.shape[1]
        model = self._build_model(self.n_features_in_).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        # Reshape to [n, 1, features] — single-timestep sequence
        Xt = torch.tensor(X_tr[:, None, :]).to(device)
        yt = torch.tensor(y_tr).to(device)
        Xv = torch.tensor(X_val[:, None, :]).to(device)
        yv = torch.tensor(y_val).to(device)

        best_val, best_state, no_improve = float("inf"), None, 0

        for _ in range(self.max_epochs):
            model.train()
            perm = torch.randperm(len(Xt), device=device)
            for i in range(0, len(Xt), self.batch_size):
                idx = perm[i : i + self.batch_size]
                opt.zero_grad()
                loss_fn(model(Xt[idx]), yt[idx]).backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(Xv), yv).item()

            if val_loss < best_val - 1e-7:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        self.model_ = model
        self.device_ = device
        return self

    def predict(self, X):
        import torch

        torch.set_num_threads(self.num_threads)
        X = np.asarray(X, dtype=np.float32)
        Xt = torch.tensor(X[:, None, :]).to(self.device_)
        self.model_.eval()
        with torch.no_grad():
            scaled_pred = self.model_(Xt).cpu().numpy()
        return scaled_pred * self.y_std_ + self.y_mean_


def make_regressor(model_name: str, random_state: int, n_jobs: int) -> ModelSpec:
    if model_name in {"har", "harx"}:
        estimator = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("linear", LinearRegression()),
            ]
        )
        feature_set = "har" if model_name == "har" else "harx"
        return ModelSpec(model_name, estimator, feature_set)

    if model_name == "svr":
        estimator = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("svr", SVR(C=2.0, epsilon=0.05, gamma="scale")),
            ]
        )
        return ModelSpec(model_name, estimator, "raw")

    if model_name == "random_forest":
        estimator = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=500,
                        max_depth=8,
                        min_samples_leaf=5,
                        random_state=random_state,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        )
        return ModelSpec(model_name, estimator, "raw")

    if model_name == "lightgbm":
        estimator = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "lgbm",
                    LGBMRegressor(
                        n_estimators=300,
                        learning_rate=0.03,
                        num_leaves=31,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=random_state,
                        n_jobs=n_jobs,
                        verbose=-1,
                    ),
                ),
            ]
        )
        return ModelSpec(model_name, estimator, "raw")

    if model_name == "wavelet_lightgbm":
        estimator = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "lgbm",
                    LGBMRegressor(
                        n_estimators=350,
                        learning_rate=0.03,
                        num_leaves=31,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=random_state,
                        n_jobs=n_jobs,
                        verbose=-1,
                    ),
                ),
            ]
        )
        return ModelSpec(model_name, estimator, "all")

    if model_name == "spillover_lightgbm":
        # Same architecture as wavelet_lightgbm but uses the all_spillover
        # feature set which additionally includes cross-index wavelet
        # spillover columns (xspill_*). This tests the hypothesis that
        # medium- and long-scale wavelet signals propagate across equity
        # indices and carry incremental predictive information.
        estimator = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "lgbm",
                    LGBMRegressor(
                        n_estimators=350,
                        learning_rate=0.03,
                        num_leaves=31,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=random_state,
                        n_jobs=n_jobs,
                        verbose=-1,
                    ),
                ),
            ]
        )
        return ModelSpec(model_name, estimator, "all_spillover")

    if model_name == "xgboost":
        estimator = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "xgb",
                    XGBRegressor(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.03,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="reg:squarederror",
                        random_state=random_state,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        )
        return ModelSpec(model_name, estimator, "raw")

    if model_name == "mlp":
        estimator = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        max_iter=500,
                        early_stopping=True,
                        validation_fraction=0.10,
                        n_iter_no_change=10,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        return ModelSpec(model_name, estimator, "all")

    if model_name in {"lstm", "gru", "bilstm"}:
        bidirectional = model_name == "bilstm"
        rnn_type = "lstm" if model_name in {"lstm", "bilstm"} else "gru"
        estimator = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "rnn",
                    TorchRNNRegressor(
                        rnn_type=rnn_type,
                        hidden_size=16,
                        num_layers=1,
                        dropout=0.1,
                        bidirectional=bidirectional,
                        lr=1e-3,
                        max_epochs=200,
                        patience=20,
                        batch_size=1260,
                        val_fraction=0.10,
                        random_state=random_state,
                        num_threads=2,
                    ),
                ),
            ]
        )
        return ModelSpec(model_name, estimator, "har")

    raise ValueError(f"Unsupported regressor: {model_name}")


def make_classifier(model_name: str, random_state: int, n_jobs: int) -> ModelSpec:
    if model_name in {"logistic_raw", "main_warning"}:
        estimator = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        feature_set = "raw" if model_name == "logistic_raw" else "warning"
        return ModelSpec(model_name, estimator, feature_set)

    if model_name == "random_forest":
        estimator = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=8,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        )
        return ModelSpec(model_name, estimator, "raw")

    if model_name == "lightgbm":
        estimator = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "lgbm",
                    LGBMClassifier(
                        n_estimators=300,
                        learning_rate=0.03,
                        num_leaves=31,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=n_jobs,
                        verbose=-1,
                    ),
                ),
            ]
        )
        return ModelSpec(model_name, estimator, "raw")

    raise ValueError(f"Unsupported classifier: {model_name}")


def fit_arch_variance_forecast(
    returns_train: pd.Series,
    mean_model: str,
    vol_model: str,
    horizon: int,
) -> float:
    returns_train = returns_train.dropna().astype(float) * 100.0
    if len(returns_train) < 250:
        return np.nan
    fitted = arch_model(
        returns_train,
        mean=mean_model,
        vol=vol_model,
        p=1,
        o=1 if vol_model == "EGARCH" else 0,
        q=1,
        dist="normal",
        rescale=False,
    ).fit(disp="off", show_warning=False)
    forecast_kwargs = {"horizon": horizon, "reindex": False}
    # EGARCH does not provide analytic multi-step variance forecasts in arch.
    # Switch to a reproducible simulation forecast so h>1 remains available.
    if vol_model == "EGARCH" and horizon > 1:
        forecast_kwargs.update(
            {
                "method": "simulation",
                "simulations": 500,
                "random_state": np.random.RandomState(42),
            }
        )
    forecast = fitted.forecast(**forecast_kwargs).variance.values[-1]
    variance = np.nanmean(forecast) / (100.0**2)
    return float(max(variance, 1.0e-10))


def oof_predictions(estimator_factory: Callable[[], object], X: pd.DataFrame, y: pd.Series, n_splits: int) -> np.ndarray:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    preds = np.full(len(X), np.nan, dtype=float)
    for train_idx, val_idx in splitter.split(X):
        estimator = estimator_factory()
        estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds[val_idx] = estimator.predict(X.iloc[val_idx])
    return preds


def oof_probabilities(estimator_factory: Callable[[], object], X: pd.DataFrame, y: pd.Series, n_splits: int) -> np.ndarray:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    preds = np.full(len(X), np.nan, dtype=float)
    for train_idx, val_idx in splitter.split(X):
        estimator = estimator_factory()
        estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds[val_idx] = estimator.predict_proba(X.iloc[val_idx])[:, 1]
    return preds


def clone_estimator(estimator: object) -> object:
    return clone(estimator)
