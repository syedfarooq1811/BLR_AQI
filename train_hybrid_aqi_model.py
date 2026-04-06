from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import NaNLabelEncoder
    from pytorch_forecasting.metrics import QuantileLoss
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler

    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    except ModuleNotFoundError:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
except ModuleNotFoundError as exc:
    missing_package = getattr(exc, "name", "required dependency")
    raise SystemExit(
        f"Missing dependency: {missing_package}. Install required packages first, for example:\n"
        "pip install torch pytorch-forecasting lightning lightgbm scikit-learn matplotlib"
    ) from exc


SEED = 42
TARGET_COL = "PM2.5 (µg/m³)"
GROUP_COL = "Station"
TIME_COL = "Timestamp"


@dataclass
class Config:
    input_path: Path
    output_dir: Path
    target_col: str = TARGET_COL
    group_col: str = GROUP_COL
    time_col: str = TIME_COL
    test_size: float = 0.15
    valid_size: float = 0.15
    batch_size: int = 256
    pinn_epochs: int = 100
    pinn_lr: float = 1e-3
    pinn_patience: int = 12
    tft_max_epochs: int = 30
    tft_encoder_length: int = 24
    tft_prediction_length: int = 1
    tft_hidden_size: int = 48
    tft_attention_heads: int = 4
    meta_top_feature_count: int = 10


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def load_dataset(config: Config) -> pd.DataFrame:
    df = pd.read_csv(config.input_path, parse_dates=[config.time_col])
    required_cols = {config.target_col, config.group_col, config.time_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.sort_values([config.group_col, config.time_col], kind="stable").reset_index(drop=True)
    df = df.dropna(subset=[config.target_col]).copy()
    df["row_id"] = np.arange(len(df))
    return df


def group_train_valid_test_split(df: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = df[config.group_col]
    gss_test = GroupShuffleSplit(n_splits=1, test_size=config.test_size, random_state=SEED)
    train_valid_idx, test_idx = next(gss_test.split(df, groups=groups))

    train_valid_df = df.iloc[train_valid_idx].copy()
    test_df = df.iloc[test_idx].copy()

    remaining_valid_fraction = config.valid_size / (1.0 - config.test_size)
    gss_valid = GroupShuffleSplit(
        n_splits=1,
        test_size=remaining_valid_fraction,
        random_state=SEED,
    )
    train_idx, valid_idx = next(
        gss_valid.split(train_valid_df, groups=train_valid_df[config.group_col])
    )

    train_df = train_valid_df.iloc[train_idx].copy()
    valid_df = train_valid_df.iloc[valid_idx].copy()

    for split_name, left, right in [
        ("train-valid", train_df, valid_df),
        ("train-test", train_df, test_df),
        ("valid-test", valid_df, test_df),
    ]:
        overlap = set(left[config.group_col]).intersection(set(right[config.group_col]))
        if overlap:
            raise RuntimeError(f"Leakage detected in {split_name} split: {sorted(overlap)}")

    return train_df, valid_df, test_df


def print_split_summary(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, config: Config) -> None:
    print("Station split summary")
    print(f"Train_base stations: {train_df[config.group_col].nunique()}")
    print(f"Valid_base stations: {valid_df[config.group_col].nunique()}")
    print(f"Test stations: {test_df[config.group_col].nunique()}")
    print(f"Train_base rows: {len(train_df)}")
    print(f"Valid_base rows: {len(valid_df)}")
    print(f"Test rows: {len(test_df)}")


def inner_group_split(df: pd.DataFrame, config: Config, holdout_size: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_groups = df[config.group_col].nunique()
    if unique_groups < 2:
        return df.copy(), df.iloc[0:0].copy()

    holdout_size = min(max(holdout_size, 1 / unique_groups), 0.5)
    splitter = GroupShuffleSplit(n_splits=1, test_size=holdout_size, random_state=SEED)
    train_idx, val_idx = next(splitter.split(df, groups=df[config.group_col]))
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def build_feature_lists(df: pd.DataFrame, config: Config) -> tuple[list[str], list[str]]:
    excluded = {config.target_col, config.time_col, "row_id"}
    categorical_cols = [config.group_col] if config.group_col in df.columns else []
    numeric_cols = [
        col
        for col in df.columns
        if col not in excluded and col not in categorical_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    return numeric_cols, categorical_cols


class AQIPINN(nn.Module):
    def __init__(self, numeric_dim: int, station_count: int, station_embedding_dim: int) -> None:
        super().__init__()
        self.station_embedding = nn.Embedding(station_count, station_embedding_dim)
        hidden_input_dim = numeric_dim + station_embedding_dim
        self.network = nn.Sequential(
            nn.Linear(hidden_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, numeric_x: torch.Tensor, station_idx: torch.Tensor) -> torch.Tensor:
        station_emb = self.station_embedding(station_idx)
        x = torch.cat([numeric_x, station_emb], dim=1)
        return self.network(x).squeeze(-1)


def encode_station_categories(
    train_df: pd.DataFrame,
    other_frames: list[pd.DataFrame],
    config: Config,
) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, int]]:
    train_df = train_df.copy()
    mapping = {station: idx + 1 for idx, station in enumerate(sorted(train_df[config.group_col].unique()))}
    unknown_idx = 0
    train_df["station_idx"] = train_df[config.group_col].map(mapping).fillna(unknown_idx).astype(int)

    encoded_frames: list[pd.DataFrame] = []
    for frame in other_frames:
        frame = frame.copy()
        frame["station_idx"] = frame[config.group_col].map(mapping).fillna(unknown_idx).astype(int)
        encoded_frames.append(frame)

    return train_df, encoded_frames, mapping


def scale_numeric_features(
    train_df: pd.DataFrame,
    frames: list[pd.DataFrame],
    numeric_cols: list[str],
) -> tuple[pd.DataFrame, list[pd.DataFrame], StandardScaler]:
    scaler = StandardScaler()
    train_df = train_df.copy()
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])

    transformed_frames: list[pd.DataFrame] = []
    for frame in frames:
        frame = frame.copy()
        frame[numeric_cols] = scaler.transform(frame[numeric_cols])
        transformed_frames.append(frame)

    return train_df, transformed_frames, scaler


def prepare_pinn_frames(
    train_base_df: pd.DataFrame,
    valid_base_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list[str],
    config: Config,
) -> dict[str, Any]:
    inner_train_df, inner_val_df = inner_group_split(train_base_df, config, holdout_size=0.15)
    inner_train_df, encoded_frames, station_mapping = encode_station_categories(
        inner_train_df,
        [inner_val_df, train_base_df, valid_base_df, test_df],
        config,
    )
    inner_val_df, train_base_df, valid_base_df, test_df = encoded_frames

    inner_train_df, scaled_frames, scaler = scale_numeric_features(
        inner_train_df,
        [inner_val_df, train_base_df, valid_base_df, test_df],
        numeric_cols,
    )
    inner_val_df, train_base_df, valid_base_df, test_df = scaled_frames

    return {
        "inner_train_df": inner_train_df,
        "inner_val_df": inner_val_df,
        "train_base_df": train_base_df,
        "valid_base_df": valid_base_df,
        "test_df": test_df,
        "station_mapping": station_mapping,
        "scaler": scaler,
    }


def frame_to_tensors(df: pd.DataFrame, numeric_cols: list[str], config: Config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_numeric = torch.tensor(df[numeric_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
    x_station = torch.tensor(df["station_idx"].to_numpy(dtype=np.int64), dtype=torch.long)
    y = torch.tensor(df[config.target_col].to_numpy(dtype=np.float32), dtype=torch.float32)
    return x_numeric, x_station, y


def physics_informed_loss(
    predictions: torch.Tensor,
    target: torch.Tensor,
    numeric_x: torch.Tensor,
    numeric_cols: list[str],
) -> torch.Tensor:
    mse_loss = nn.functional.mse_loss(predictions, target)

    ws_penalty = torch.tensor(0.0, device=predictions.device)
    rh_penalty = torch.tensor(0.0, device=predictions.device)

    if "WS (m/s)" in numeric_cols:
        ws_idx = numeric_cols.index("WS (m/s)")
        wind_speed = numeric_x[:, ws_idx]
        ws_penalty = torch.relu(predictions - target) * torch.relu(wind_speed)
        ws_penalty = ws_penalty.mean()

    if "RH (%)" in numeric_cols:
        rh_idx = numeric_cols.index("RH (%)")
        humidity = numeric_x[:, rh_idx]
        humidity_weight = torch.sigmoid(humidity)
        rh_penalty = ((predictions - target) ** 2 * humidity_weight).mean()

    return mse_loss + 0.02 * ws_penalty + 0.01 * rh_penalty


def train_pinn(
    prepared: dict[str, Any],
    numeric_cols: list[str],
    config: Config,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], AQIPINN]:
    inner_train_df = prepared["inner_train_df"]
    inner_val_df = prepared["inner_val_df"]
    train_base_df = prepared["train_base_df"]
    valid_base_df = prepared["valid_base_df"]
    test_df = prepared["test_df"]
    station_count = len(prepared["station_mapping"]) + 1
    station_embedding_dim = min(16, max(4, math.ceil(math.sqrt(station_count))))

    model = AQIPINN(
        numeric_dim=len(numeric_cols),
        station_count=station_count,
        station_embedding_dim=station_embedding_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.pinn_lr)
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    patience_counter = 0

    train_x_num, train_x_station, train_y = frame_to_tensors(inner_train_df, numeric_cols, config)
    val_x_num, val_x_station, val_y = frame_to_tensors(inner_val_df, numeric_cols, config)

    train_x_num = train_x_num.to(device)
    train_x_station = train_x_station.to(device)
    train_y = train_y.to(device)
    val_x_num = val_x_num.to(device)
    val_x_station = val_x_station.to(device)
    val_y = val_y.to(device)

    for epoch in range(1, config.pinn_epochs + 1):
        model.train()
        permutation = torch.randperm(train_x_num.size(0), device=device)

        epoch_losses: list[float] = []
        for batch_start in range(0, train_x_num.size(0), config.batch_size):
            batch_idx = permutation[batch_start : batch_start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            preds = model(train_x_num[batch_idx], train_x_station[batch_idx])
            loss = physics_informed_loss(preds, train_y[batch_idx], train_x_num[batch_idx], numeric_cols)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_preds = model(val_x_num, val_x_station)
            val_loss = nn.functional.mse_loss(val_preds, val_y).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            train_loss_mean = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            print(f"PINN epoch {epoch:03d} | train_loss={train_loss_mean:.4f} | val_mse={val_loss:.4f}")

        if patience_counter >= config.pinn_patience:
            print(f"PINN early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    def predict_frame(frame: pd.DataFrame) -> np.ndarray:
        x_num, x_station, _ = frame_to_tensors(frame, numeric_cols, config)
        model.eval()
        with torch.no_grad():
            preds = model(x_num.to(device), x_station.to(device)).cpu().numpy()
        return preds

    predictions = {
        "train_base": predict_frame(train_base_df),
        "valid_base": predict_frame(valid_base_df),
        "test": predict_frame(test_df),
    }

    return predictions, model


def build_time_idx(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    df = df.copy()
    df["time_idx"] = df.groupby(config.group_col).cumcount()
    return df


def classify_tft_columns(df: pd.DataFrame, config: Config) -> tuple[list[str], list[str], list[str]]:
    known_candidates = [
        "hour",
        "day",
        "month",
        "day_of_week",
        "day_of_year",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
    ]
    known_reals = [col for col in known_candidates if col in df.columns]

    excluded = {config.target_col, config.time_col, config.group_col, "row_id", "time_idx"}
    unknown_reals = [
        col
        for col in df.columns
        if col not in excluded
        and col not in known_reals
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    static_categoricals = [config.group_col]
    return known_reals, unknown_reals, static_categoricals


def make_tft_dataset(
    df: pd.DataFrame,
    config: Config,
    known_reals: list[str],
    unknown_reals: list[str],
) -> TimeSeriesDataSet:
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=config.target_col,
        group_ids=[config.group_col],
        max_encoder_length=config.tft_encoder_length,
        min_encoder_length=config.tft_encoder_length,
        max_prediction_length=config.tft_prediction_length,
        min_prediction_length=config.tft_prediction_length,
        static_categoricals=[config.group_col],
        time_varying_known_reals=known_reals + ["time_idx"],
        time_varying_unknown_reals=[config.target_col] + unknown_reals,
        target_normalizer=None,
        categorical_encoders={config.group_col: NaNLabelEncoder(add_nan=True)},
        allow_missing_timesteps=False,
    )


def fit_real_feature_scalers(
    train_df: pd.DataFrame,
    columns: list[str],
) -> dict[str, StandardScaler]:
    scalers: dict[str, StandardScaler] = {}
    for col in columns:
        scaler = StandardScaler()
        values = train_df[[col]].to_numpy(dtype=np.float32)
        scalers[col] = scaler.fit(values)
    return scalers


def apply_real_feature_scalers(
    df: pd.DataFrame,
    scalers: dict[str, StandardScaler],
) -> pd.DataFrame:
    df = df.copy()
    for col, scaler in scalers.items():
        if col in df.columns:
            df[col] = scaler.transform(df[[col]].to_numpy(dtype=np.float32)).astype(np.float32)
    return df


def predict_with_tft(
    model: TemporalFusionTransformer,
    dataset_template: TimeSeriesDataSet,
    df: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["row_id", "tft_pred"])

    predict_dataset = TimeSeriesDataSet.from_dataset(
        dataset_template,
        df,
        stop_randomization=True,
        predict=False,
    )
    predict_loader = predict_dataset.to_dataloader(train=False, batch_size=config.batch_size, num_workers=0)
    raw_output = model.predict(predict_loader, mode="prediction", return_index=True)

    if isinstance(raw_output, tuple):
        predictions, index = raw_output
    else:
        predictions = raw_output.output
        index = raw_output.index

    pred_array = predictions.detach().cpu().numpy().reshape(-1)
    index_df = pd.DataFrame(index)
    if "row_id" not in index_df.columns:
        raise RuntimeError("TFT prediction index did not include row_id; cannot align predictions safely.")

    pred_df = index_df[["row_id"]].copy()
    pred_df["tft_pred"] = pred_array
    return pred_df


def train_tft(
    train_base_df: pd.DataFrame,
    valid_base_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Config,
) -> tuple[dict[str, np.ndarray], TemporalFusionTransformer]:
    train_base_df = build_time_idx(train_base_df, config)
    valid_base_df = build_time_idx(valid_base_df, config)
    test_df = build_time_idx(test_df, config)

    known_reals, unknown_reals, _ = classify_tft_columns(train_base_df, config)
    tft_real_cols = list(dict.fromkeys(known_reals + unknown_reals + [config.target_col, "time_idx"]))

    inner_train_df, inner_val_df = inner_group_split(train_base_df, config, holdout_size=0.15)
    inner_train_df = build_time_idx(inner_train_df, config)
    inner_val_df = build_time_idx(inner_val_df, config)

    real_scalers = fit_real_feature_scalers(inner_train_df, tft_real_cols)
    inner_train_scaled = apply_real_feature_scalers(inner_train_df, real_scalers)
    inner_val_scaled = apply_real_feature_scalers(inner_val_df, real_scalers)
    train_base_scaled = apply_real_feature_scalers(train_base_df, real_scalers)
    valid_base_scaled = apply_real_feature_scalers(valid_base_df, real_scalers)
    test_scaled = apply_real_feature_scalers(test_df, real_scalers)

    training = make_tft_dataset(inner_train_scaled, config, known_reals, unknown_reals)
    validation = TimeSeriesDataSet.from_dataset(
        training,
        inner_val_scaled,
        stop_randomization=True,
        predict=False,
    )

    train_loader = training.to_dataloader(train=True, batch_size=config.batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=config.batch_size, num_workers=0)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.output_dir / "checkpoints",
        filename="tft-best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    trainer = pl.Trainer(
        max_epochs=config.tft_max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[early_stopping, checkpoint_callback],
        enable_model_summary=False,
        logger=False,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.01,
        hidden_size=config.tft_hidden_size,
        attention_head_size=config.tft_attention_heads,
        dropout=0.1,
        hidden_continuous_size=config.tft_hidden_size,
        loss=QuantileLoss(),
        optimizer="adam",
    )

    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if checkpoint_callback.best_model_path:
        tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_callback.best_model_path)

    train_pred_df = predict_with_tft(tft, training, train_base_scaled, config)
    valid_pred_df = predict_with_tft(tft, training, valid_base_scaled, config)
    test_pred_df = predict_with_tft(tft, training, test_scaled, config)

    fallback_value = float(inner_train_df[config.target_col].median())
    fallback_col = "pm25_ug_per_m³_lag_1" if "pm25_ug_per_m³_lag_1" in train_base_df.columns else None

    def merge_predictions(frame: pd.DataFrame, pred_df: pd.DataFrame) -> np.ndarray:
        merged = frame[["row_id"]].merge(pred_df, on="row_id", how="left")
        if fallback_col and fallback_col in frame.columns:
            fallback_series = frame[fallback_col].fillna(fallback_value).to_numpy(dtype=np.float32)
            merged["tft_pred"] = merged["tft_pred"].fillna(pd.Series(fallback_series))
        else:
            merged["tft_pred"] = merged["tft_pred"].fillna(fallback_value)
        return merged["tft_pred"].to_numpy(dtype=np.float32)

    predictions = {
        "train_base": merge_predictions(train_base_df, train_pred_df),
        "valid_base": merge_predictions(valid_base_df, valid_pred_df),
        "test": merge_predictions(test_df, test_pred_df),
    }

    return predictions, tft


def select_meta_features(train_base_df: pd.DataFrame, config: Config) -> list[str]:
    candidate_cols = [
        "PM10 (µg/m³)",
        "NO (µg/m³)",
        "NO2 (µg/m³)",
        "NOx (ppb)",
        "NH3 (µg/m³)",
        "SO2 (µg/m³)",
        "CO (mg/m³)",
        "Ozone (µg/m³)",
        "Benzene (µg/m³)",
        "Toluene (µg/m³)",
        "AT (°C)",
        "RH (%)",
        "WS (m/s)",
        "WD (deg)",
        "RF (mm)",
        "BP (mmHg)",
        "SR (W/mt2)",
        "hour",
        "day_of_week",
        "month",
        "pm25_ug_per_m³_lag_1",
        "pm25_ug_per_m³_lag_4",
        "pm25_ug_per_m³_roll_mean_4",
        "pm25_ug_per_m³_roll_mean_16",
        "pm10_ug_per_m³_lag_1",
        "pm10_ug_per_m³_roll_mean_4",
        "no2_ug_per_m³_lag_1",
        "nox_ppb_lag_1",
        "co_mg_per_m³_lag_1",
        "month_cos",
        "hour_cos",
    ]
    available = [col for col in candidate_cols if col in train_base_df.columns]
    if not available:
        return []

    correlations = (
        train_base_df[available + [config.target_col]]
        .corr(numeric_only=True)[config.target_col]
        .drop(labels=[config.target_col])
        .abs()
        .sort_values(ascending=False)
    )
    return correlations.head(config.meta_top_feature_count).index.tolist()


def add_meta_features(
    base_df: pd.DataFrame,
    pinn_pred: np.ndarray,
    tft_pred: np.ndarray,
) -> pd.DataFrame:
    df = base_df.copy()
    df["pinn_pred"] = pinn_pred
    df["tft_pred"] = tft_pred
    return df


def train_meta_model(
    train_base_df: pd.DataFrame,
    valid_base_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Config,
) -> tuple[np.ndarray, lgb.LGBMRegressor, list[str]]:
    meta_original_features = select_meta_features(train_base_df, config)
    meta_features = ["pinn_pred", "tft_pred"] + meta_original_features

    meta_train_df, meta_eval_df = inner_group_split(valid_base_df, config, holdout_size=0.25)
    if meta_eval_df.empty:
        meta_train_df = valid_base_df.copy()
        meta_eval_df = valid_base_df.copy()

    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.015,
        num_leaves=128,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="regression",
        random_state=SEED,
    )

    model.fit(
        meta_train_df[meta_features],
        meta_train_df[config.target_col],
        eval_set=[(meta_eval_df[meta_features], meta_eval_df[config.target_col])],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)],
    )

    best_iteration = model.best_iteration_ or model.n_estimators
    final_model = lgb.LGBMRegressor(
        n_estimators=best_iteration,
        learning_rate=0.015,
        num_leaves=128,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="regression",
        random_state=SEED,
    )
    final_model.fit(valid_base_df[meta_features], valid_base_df[config.target_col])
    test_pred = final_model.predict(test_df[meta_features])

    return test_pred, final_model, meta_features


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def baseline_metrics(test_df: pd.DataFrame, config: Config) -> dict[str, float] | None:
    baseline_col = "pm25_ug_per_m³_lag_1"
    if baseline_col not in test_df.columns:
        return None

    y_true = test_df[config.target_col].to_numpy()
    y_pred = test_df[baseline_col].to_numpy()
    return compute_metrics(y_true, y_pred)


def save_feature_importance_plot(
    model: lgb.LGBMRegressor,
    feature_names: list[str],
    output_path: Path,
) -> None:
    if not hasattr(model, "feature_importances_"):
        return

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.title("LightGBM Meta-Model Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_predictions(
    test_df: pd.DataFrame,
    final_pred: np.ndarray,
    output_path: Path,
    config: Config,
) -> None:
    output = test_df[[config.group_col, "Station ID", "City", config.time_col, config.target_col]].copy()
    output["hybrid_prediction"] = final_pred
    output.to_csv(output_path, index=False)


def main() -> None:
    seed_everything(SEED)

    base_dir = Path(__file__).resolve().parent
    config = Config(
        input_path=base_dir / "aqi_standard_features.csv",
        output_dir=base_dir / "hybrid_model_outputs",
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    df = load_dataset(config)
    train_base_df, valid_base_df, test_df = group_train_valid_test_split(df, config)
    print_split_summary(train_base_df, valid_base_df, test_df, config)

    numeric_cols, _ = build_feature_lists(df, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pinn_prepared = prepare_pinn_frames(train_base_df, valid_base_df, test_df, numeric_cols, config)
    pinn_predictions, _ = train_pinn(pinn_prepared, numeric_cols, config, device)

    tft_predictions, _ = train_tft(train_base_df, valid_base_df, test_df, config)

    train_meta_df = add_meta_features(train_base_df, pinn_predictions["train_base"], tft_predictions["train_base"])
    valid_meta_df = add_meta_features(valid_base_df, pinn_predictions["valid_base"], tft_predictions["valid_base"])
    test_meta_df = add_meta_features(test_df, pinn_predictions["test"], tft_predictions["test"])

    final_pred, meta_model, meta_features = train_meta_model(
        train_meta_df,
        valid_meta_df,
        test_meta_df,
        config,
    )

    metrics = compute_metrics(test_meta_df[config.target_col].to_numpy(), final_pred)
    lag1_metrics = baseline_metrics(test_meta_df, config)

    print("\nFinal evaluation on unseen stations")
    print(f"Train stations count: {train_base_df[config.group_col].nunique()}")
    print(f"Test stations count: {test_df[config.group_col].nunique()}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")

    if lag1_metrics is not None:
        improvement = ((lag1_metrics["rmse"] - metrics["rmse"]) / lag1_metrics["rmse"]) * 100
        print(f"Lag-1 baseline RMSE: {lag1_metrics['rmse']:.4f}")
        print(f"RMSE improvement vs lag-1 baseline: {improvement:.2f}%")

    metrics_payload = {
        "train_stations": int(train_base_df[config.group_col].nunique()),
        "valid_stations": int(valid_base_df[config.group_col].nunique()),
        "test_stations": int(test_df[config.group_col].nunique()),
        "metrics": metrics,
        "lag1_baseline_metrics": lag1_metrics,
        "meta_features": meta_features,
    }

    with open(config.output_dir / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=2)

    save_predictions(
        test_meta_df,
        final_pred,
        config.output_dir / "test_predictions.csv",
        config,
    )
    save_feature_importance_plot(
        meta_model,
        meta_features,
        config.output_dir / "lightgbm_meta_feature_importance.png",
    )

    print(f"Saved outputs to: {config.output_dir}")


if __name__ == "__main__":
    main()
