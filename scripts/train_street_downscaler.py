from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import islice
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.spatial_interpolation import idw_weights_for_targets, linear_weights_for_targets


FEATURES_DEFAULT = Path("data/processed/features.parquet")
LABELS_DEFAULT = Path("data/raw/street_labels.parquet")
OUT_DIR_DEFAULT = Path("models/street_downscaler")
PAPER_MODEL_NAME = "STARLING-AQI"
PAPER_MODEL_FULL_NAME = "Spatial-Temporal Adaptive Residual Learning with Interpolation-Guided Stacking"
TARGET_R2_DEFAULT = 0.94
TARGET_RMSE_DEFAULT = 0.15


@dataclass
class SplitData:
    x: pd.DataFrame
    y: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised street AQI downscalers for 24h and 7-day horizons.")
    parser.add_argument("--labels", type=Path, default=LABELS_DEFAULT)
    parser.add_argument("--features", type=Path, default=FEATURES_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--target-col", type=str, default="AQI")
    parser.add_argument("--max-samples", type=int, default=80000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--max-trials", type=int, default=0, help="Limit hyperparameter trials (0 means full grid).")
    parser.add_argument("--max-iter", type=int, default=800, help="Boosting iterations for each trial.")
    parser.add_argument(
        "--use-station-proxy-labels",
        action="store_true",
        help="Use station observations from features.parquet as proxy street labels when real street labels are unavailable.",
    )
    parser.add_argument(
        "--model-zoo",
        action="store_true",
        help="Train multiple strong regressors (plus stacking) and pick the best on validation RMSE.",
    )
    parser.add_argument(
        "--paper-novelty-model",
        action="store_true",
        help=f"Prioritize the named paper model: {PAPER_MODEL_NAME}.",
    )
    parser.add_argument("--target-r2", type=float, default=TARGET_R2_DEFAULT)
    parser.add_argument("--target-rmse", type=float, default=TARGET_RMSE_DEFAULT)
    return parser.parse_args()


def _coerce_timestamp(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    return ts.dt.tz_convert("Asia/Kolkata")


def load_station_panel(features_path: Path, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["timestamp", "station_id", "lat", "lon", target_col]
    df = pd.read_parquet(features_path, columns=cols).dropna(subset=cols)
    df["timestamp"] = _coerce_timestamp(df["timestamp"])
    panel = df.pivot_table(index="timestamp", columns="station_id", values=target_col, aggfunc="mean").dropna(how="any")
    coords = df.groupby("station_id", as_index=True)[["lat", "lon"]].first().loc[panel.columns]
    return panel.sort_index(), coords


def load_labels(labels_path: Path, features_path: Path, target_col: str, use_station_proxy_labels: bool) -> pd.DataFrame:
    if labels_path.exists():
        df = pd.read_parquet(labels_path)
        required = {"timestamp", "lat", "lon", target_col}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise SystemExit(f"Street labels missing required columns: {missing}")
        df = df.dropna(subset=["timestamp", "lat", "lon", target_col]).copy()
        df["timestamp"] = _coerce_timestamp(df["timestamp"])
        return df

    if not use_station_proxy_labels:
        raise SystemExit(
            f"Street labels missing at {labels_path}. Provide real labeled rows with columns "
            f"`timestamp`, `lat`, `lon`, and `{target_col}`, or pass --use-station-proxy-labels."
        )

    cols = ["timestamp", "station_id", "lat", "lon", target_col]
    df = pd.read_parquet(features_path, columns=cols).dropna(subset=cols).copy()
    df["timestamp"] = _coerce_timestamp(df["timestamp"])
    df["label_source"] = "station_proxy"
    return df


def station_interpolation_features(
    labels: pd.DataFrame,
    panel: pd.DataFrame,
    coords: pd.DataFrame,
    horizon_hours: int,
    target_col: str,
) -> pd.DataFrame:
    station_ids = list(panel.columns)
    source_coords = coords.loc[station_ids].to_numpy(dtype=float)
    target_coords = labels[["lat", "lon"]].to_numpy(dtype=float)
    idw_w = idw_weights_for_targets(source_coords, target_coords, power=1.3)
    lin_w = linear_weights_for_targets(source_coords, target_coords)
    blend_w = 0.75 * idw_w + 0.25 * lin_w

    out = labels.copy()
    out["target_timestamp"] = out["timestamp"] + pd.to_timedelta(horizon_hours, unit="h")
    valid_mask = out["target_timestamp"].isin(panel.index)
    out = out.loc[valid_mask].copy()
    out["origin_aqi"] = out[target_col].astype(float)

    ts_to_idx = {ts: i for i, ts in enumerate(panel.index)}
    idx = np.array([ts_to_idx[t] for t in out["target_timestamp"]], dtype=int)
    station_values = panel.to_numpy(dtype=float)[idx]
    weights = blend_w[valid_mask.to_numpy()].copy()
    if "station_id" in out.columns:
        station_to_col = {station_id: pos for pos, station_id in enumerate(station_ids)}
        target_station_pos = out["station_id"].map(station_to_col).to_numpy()
        valid_station = ~pd.isna(target_station_pos)
        target_station_pos = target_station_pos[valid_station].astype(int)

        # Align proxy labels to the actual forecast horizon and hide the target
        # station from interpolation features to avoid trivial self-leakage.
        row_pos = np.arange(len(out))[valid_station]
        out.loc[out.index[valid_station], target_col] = station_values[row_pos, target_station_pos]
        weights[row_pos, target_station_pos] = 0.0
        row_sums = weights[row_pos].sum(axis=1)
        nonzero = row_sums > 0
        weights[row_pos[nonzero]] = weights[row_pos[nonzero]] / row_sums[nonzero, None]

    station_pred = np.sum(station_values * weights, axis=1)
    station_mean = station_values.mean(axis=1)
    station_std = station_values.std(axis=1)
    station_min = station_values.min(axis=1)
    station_max = station_values.max(axis=1)

    out["station_interp"] = station_pred
    out["station_mean"] = station_mean
    out["station_std"] = station_std
    out["station_min"] = station_min
    out["station_max"] = station_max
    
    # Simple rolling 7-day trend features (fast, no loops)
    out["rolling_7d_mean"] = out["origin_aqi"].rolling(window=168, min_periods=1).mean()
    out["rolling_7d_std"] = out["origin_aqi"].rolling(window=168, min_periods=1).std().fillna(0.1)
    
    out["hour"] = out["target_timestamp"].dt.hour.astype(int)
    out["dow"] = out["target_timestamp"].dt.dayofweek.astype(int)
    out["hour_sin"] = np.sin(2.0 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2.0 * np.pi * out["dow"] / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * out["dow"] / 7.0)
    out["lat_lon_interaction"] = out["lat"] * out["lon"]
    if "station_id" in out.columns:
        # Helpful for station-proxy training; model learns site-specific offsets.
        out["station_code"] = pd.factorize(out["station_id"])[0].astype(float)
    else:
        out["station_code"] = -1.0
    return out


def temporal_split(df: pd.DataFrame, val_frac: float, test_frac: float, target_col: str) -> tuple[SplitData, SplitData, SplitData]:
    df = df.sort_values("target_timestamp").reset_index(drop=True)
    n = len(df)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    n_train = max(1, n - n_val - n_test)
    if n_train + n_val + n_test > n:
        n_train = n - n_val - n_test

    feat_cols = [
        "lat",
        "lon",
        "station_interp",
        "station_mean",
        "station_std",
        "station_min",
        "station_max",
        "rolling_7d_mean",
        "rolling_7d_std",
        "hour",
        "dow",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "lat_lon_interaction",
        "origin_aqi",
        "station_code",
    ]
    x = df[feat_cols].astype(float)
    y = df[target_col].to_numpy(dtype=float)
    train = SplitData(x.iloc[:n_train], y[:n_train])
    val = SplitData(x.iloc[n_train : n_train + n_val], y[n_train : n_train + n_val])
    test = SplitData(x.iloc[n_train + n_val :], y[n_train + n_val :])
    return train, val, test


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
    }


def param_grid() -> Iterable[dict]:
    learning_rates = [0.02, 0.03, 0.05, 0.08]
    depths = [6, 8, 10]
    l2s = [0.0, 0.001, 0.01]
    leafs = [15, 31, 63]
    for lr in learning_rates:
        for depth in depths:
            for l2 in l2s:
                for ml in leafs:
                    yield {
                        "learning_rate": lr,
                        "max_depth": depth,
                        "l2_regularization": l2,
                        "max_leaf_nodes": ml,
                        "min_samples_leaf": 10,
                        "loss": "squared_error",
                    }


def train_best_model(train: SplitData, val: SplitData, seed: int, max_trials: int, max_iter: int) -> tuple[HistGradientBoostingRegressor, dict, list[dict]]:
    best_model = None
    best_params = None
    best_rmse = float("inf")
    trials = []
    grid_iter = param_grid()
    if max_trials and max_trials > 0:
        grid_iter = islice(grid_iter, int(max_trials))
    for trial_idx, params in enumerate(grid_iter, start=1):
        print(f"[HGB] trial={trial_idx} start params={params}", flush=True)
        model = HistGradientBoostingRegressor(
            random_state=seed,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            **params,
        )
        model.fit(train.x, train.y)
        pred = model.predict(val.x)
        m = metrics(val.y, pred)
        trial = dict(params)
        trial.update(m)
        trials.append(trial)
        print(f"[HGB] trial={trial_idx} done val_rmse={m['rmse']:.4f} val_r2={m['r2']:.4f}", flush=True)
        if m["rmse"] < best_rmse:
            best_rmse = m["rmse"]
            best_model = model
            best_params = params
    assert best_model is not None and best_params is not None
    return best_model, best_params, trials


def paper_method_metadata() -> dict[str, object]:
    return {
        "name": PAPER_MODEL_NAME,
        "full_name": PAPER_MODEL_FULL_NAME,
        "claim_scope": (
            "Novel as an implemented combination in this project: interpolation-prior features, "
            "station/site residual encoding, cyclic temporal context, heterogeneous regressors, "
            "and Ridge meta-learning under horizon-specific validation."
        ),
        "components": [
            "IDW plus Delaunay-linear interpolation prior",
            "station distribution features at the forecast horizon",
            "cyclic hour/day temporal encodings",
            "site residual proxy via station_code when proxy labels are used",
            "heterogeneous base learners: HistGradientBoosting, ExtraTrees, RandomForest, GradientBoosting, KNN, MLP",
            "RidgeCV meta learner with passthrough features",
        ],
        "paper_ready_warning": (
            "Do not claim the numerical targets on street-level AQI unless evaluated on real street labels "
            "or an explicitly justified external validation set. Station-proxy labels are useful for "
            "pipeline development, not final street-level evidence."
        ),
    }


def model_zoo(seed: int, max_iter: int, paper_novelty_model: bool = False) -> dict[str, object]:
    bounded_iter = max(100, min(max_iter, 250))
    hgb = HistGradientBoostingRegressor(
        random_state=seed,
        max_iter=bounded_iter,
        learning_rate=0.04,
        max_depth=8,
        l2_regularization=0.001,
        max_leaf_nodes=63,
        min_samples_leaf=10,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
    )
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=seed,
        n_jobs=1,
    )
    et = ExtraTreesRegressor(
        n_estimators=350,
        max_depth=22,
        min_samples_leaf=1,
        max_features=0.85,
        random_state=seed,
        n_jobs=1,
    )
    gbr = GradientBoostingRegressor(random_state=seed, n_estimators=350, learning_rate=0.03, max_depth=5)
    ada = AdaBoostRegressor(
        random_state=seed,
        n_estimators=250,
        learning_rate=0.025,
    )
    knn = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=12, weights="distance"))
    mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-5,
            learning_rate_init=1e-4,
            max_iter=bounded_iter,
            random_state=seed,
        ),
    )
    stack = StackingRegressor(
        estimators=[
            ("hgb", clone(hgb)),
            ("rf", clone(rf)),
            ("et", clone(et)),
            ("gbr", clone(gbr)),
        ],
        final_estimator=RidgeCV(alphas=(0.1, 1.0, 3.0, 10.0)),
        passthrough=True,
        cv=3,
        n_jobs=1,
    )
    paper_stack = StackingRegressor(
        estimators=[
            ("hgb_prior_residual", clone(hgb)),
            ("extra_trees_spatial", clone(et)),
            ("random_forest_site", clone(rf)),
            ("grad_boost_temporal", clone(gbr)),
            ("knn_local_analogue", clone(knn)),
            ("mlp_nonlinear_corrector", clone(mlp)),
        ],
        final_estimator=RidgeCV(alphas=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0)),
        passthrough=True,
        cv=3,
        n_jobs=1,
    )
    models = {
        PAPER_MODEL_NAME: paper_stack,
        "hgb_strong": hgb,
        "extra_trees_strong": et,
        "random_forest_strong": rf,
        "grad_boost_strong": gbr,
        "adaboost_strong": ada,
        "knn_distance": knn,
        "mlp_deep": mlp,
        "stacked_ensemble": stack,
    }
    if paper_novelty_model:
        return {PAPER_MODEL_NAME: paper_stack, **{k: v for k, v in models.items() if k != PAPER_MODEL_NAME}}
    return models


def train_best_model_zoo(
    train: SplitData,
    val: SplitData,
    seed: int,
    max_iter: int,
    paper_novelty_model: bool,
) -> tuple[object, dict, list[dict]]:
    best_model = None
    best_name = None
    best_rmse = float("inf")
    trials = []
    for idx, (name, model) in enumerate(model_zoo(seed=seed, max_iter=max_iter, paper_novelty_model=paper_novelty_model).items(), start=1):
        print(f"[ZOO] {idx} start model={name}", flush=True)
        row = {"model_name": name}
        try:
            model.fit(train.x, train.y)
            pred = model.predict(val.x)
            m = metrics(val.y, pred)
            row.update(m)
            print(f"[ZOO] {idx} done model={name} val_rmse={m['rmse']:.4f} val_r2={m['r2']:.4f}", flush=True)
        except (MemoryError, RuntimeError, PermissionError) as exc:
            row.update({"failed": True, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[ZOO] {idx} failed model={name} error={type(exc).__name__}: {exc}", flush=True)
            trials.append(row)
            continue
        trials.append(row)
        if m["rmse"] < best_rmse:
            best_rmse = m["rmse"]
            best_model = model
            best_name = name
    if best_model is None or best_name is None:
        raise SystemExit("All model-zoo candidates failed. Try lowering --max-samples or run without --model-zoo.")
    return best_model, {"model_name": best_name}, trials


def sample_rows(df: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_samples:
        return df
    return df.sample(n=max_samples, random_state=seed).sort_values("target_timestamp")


def run_horizon(
    labels: pd.DataFrame,
    panel: pd.DataFrame,
    coords: pd.DataFrame,
    horizon_hours: int,
    target_col: str,
    max_samples: int,
    seed: int,
    val_frac: float,
    test_frac: float,
    out_dir: Path,
    max_trials: int,
    max_iter: int,
    use_model_zoo: bool,
    paper_novelty_model: bool,
) -> dict:
    print(f"=== Horizon {horizon_hours}h: feature prep ===", flush=True)
    data = station_interpolation_features(labels, panel, coords, horizon_hours=horizon_hours, target_col=target_col)
    data = sample_rows(data, max_samples=max_samples, seed=seed)
    train, val, test = temporal_split(data, val_frac=val_frac, test_frac=test_frac, target_col=target_col)
    print(
        f"=== Horizon {horizon_hours}h: rows={len(data)} split={len(train.y)}/{len(val.y)}/{len(test.y)} ===",
        flush=True,
    )

    if use_model_zoo or paper_novelty_model:
        model, best_params, trials = train_best_model_zoo(
            train,
            val,
            seed=seed,
            max_iter=max_iter,
            paper_novelty_model=paper_novelty_model,
        )
    else:
        model, best_params, trials = train_best_model(
            train,
            val,
            seed=seed,
            max_trials=max_trials,
            max_iter=max_iter,
        )
    pred_train = model.predict(train.x)
    pred_val = model.predict(val.x)
    pred_test = model.predict(test.x)
    train_m = metrics(train.y, pred_train)
    val_m = metrics(val.y, pred_val)
    test_m = metrics(test.y, pred_test)
    print(
        f"=== Horizon {horizon_hours}h complete: test_rmse={test_m['rmse']:.4f} test_r2={test_m['r2']:.4f} ===",
        flush=True,
    )

    model_path = out_dir / f"street_downscaler_h{horizon_hours}.joblib"
    joblib.dump({"model": model, "features": list(train.x.columns), "horizon_hours": horizon_hours}, model_path)

    trials_df = pd.DataFrame(trials).sort_values(["rmse", "mae"])
    trials_df.to_csv(out_dir / f"street_downscaler_h{horizon_hours}_trials.csv", index=False)
    pd.DataFrame({"y_true": test.y, "y_pred": pred_test}).to_csv(
        out_dir / f"street_downscaler_h{horizon_hours}_test_predictions.csv",
        index=False,
    )

    return {
        "horizon_hours": horizon_hours,
        "rows": int(len(data)),
        "splits": {"train": int(len(train.y)), "val": int(len(val.y)), "test": int(len(test.y))},
        "best_params": best_params,
        "paper_model_name": PAPER_MODEL_NAME if paper_novelty_model else None,
        "train": train_m,
        "val": val_m,
        "test": test_m,
        "model_path": str(model_path),
    }


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(
        labels_path=args.labels,
        features_path=args.features,
        target_col=args.target_col,
        use_station_proxy_labels=args.use_station_proxy_labels,
    )
    panel, coords = load_station_panel(args.features, args.target_col)

    summary = {
        "target_col": args.target_col,
        "features_path": str(args.features),
        "labels_path": str(args.labels),
        "canonical_scale": "IN_CPCB",
        "label_mode": "station_proxy" if args.use_station_proxy_labels and not args.labels.exists() else "street_labels",
        "paper_method": paper_method_metadata() if args.paper_novelty_model else None,
        "target_gates": {"r2": args.target_r2, "rmse": args.target_rmse},
        "results": [],
    }

    for horizon in (24, 168):
        result = run_horizon(
            labels=labels,
            panel=panel,
            coords=coords,
            horizon_hours=horizon,
            target_col=args.target_col,
            max_samples=args.max_samples,
            seed=args.random_seed,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            out_dir=out_dir,
            max_trials=args.max_trials,
            max_iter=args.max_iter,
            use_model_zoo=args.model_zoo,
            paper_novelty_model=args.paper_novelty_model,
        )
        summary["results"].append(result)

    (out_dir / "street_downscaler_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Street Downscaler Training Summary",
        "",
        f"- Labels: `{args.labels}`",
        f"- Features: `{args.features}`",
        f"- Target: `{args.target_col}`",
        "",
    ]
    if args.paper_novelty_model:
        lines += [
            f"## Paper Novelty Model: {PAPER_MODEL_NAME}",
            f"- Full name: {PAPER_MODEL_FULL_NAME}",
            "- Novelty: interpolation-guided heterogeneous stacking with horizon-specific validation.",
            "- Caution: final street-level claims require real street labels, not station-proxy labels alone.",
            "",
        ]
    for item in summary["results"]:
        target_pass = item["test"]["r2"] >= args.target_r2 and item["test"]["rmse"] <= args.target_rmse
        lines += [
            f"## Horizon {item['horizon_hours']}h",
            f"- Rows: {item['rows']}",
            f"- Split train/val/test: {item['splits']['train']}/{item['splits']['val']}/{item['splits']['test']}",
            f"- Test R2: {item['test']['r2']:.4f}",
            f"- Test RMSE: {item['test']['rmse']:.4f}",
            f"- Test MAE: {item['test']['mae']:.4f}",
            f"- Target gate pass: {target_pass}",
            f"- Model path: `{item['model_path']}`",
            "",
        ]
    (out_dir / "street_downscaler_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote training outputs to {out_dir}")


if __name__ == "__main__":
    main()
