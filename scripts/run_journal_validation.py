"""Run empirical validation tables for the Bangalore AQI journal write-up.

The processed feature table stores AQI and lag features in normalized units, so
the generated metrics are normalized AQI errors unless the upstream ETL is
changed to persist raw-target scalers.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = Path("data/processed/features.parquet")
OUT_DIR = Path("reports/journal_validation")
TARGET = "AQI"
STATION = "station_id"
TIME = "timestamp"
TEMPORAL_COLUMNS = ["hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos", "is_weekend"]
SPATIAL_COLUMNS = ["lat", "lon"]


@dataclass(frozen=True)
class FeatureSet:
    name: str
    columns: list[str]
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-iter", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer boosting iterations for a smoke test of the reporting pipeline.",
    )
    return parser.parse_args()


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {TARGET, STATION, TIME, "AQI_lag_1h"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    df = df.sort_values([TIME, STATION]).reset_index(drop=True)
    df = df.dropna(subset=[TARGET, STATION, TIME])
    return df


def make_feature_sets(df: pd.DataFrame) -> dict[str, FeatureSet]:
    lag_roll = [c for c in df.columns if "_lag_" in c or "_rolling_" in c]
    temporal = [c for c in TEMPORAL_COLUMNS if c in df.columns]
    spatial = [c for c in SPATIAL_COLUMNS if c in df.columns]

    if "AQI_lag_1h" not in lag_roll:
        raise ValueError("Expected AQI_lag_1h in the processed feature table.")

    return {
        "full": FeatureSet(
            "full",
            lag_roll + temporal + spatial,
            "Lag, rolling, cyclical time, weekend, and station location features.",
        ),
        "no_spatial": FeatureSet(
            "no_spatial",
            lag_roll + temporal,
            "Full model without latitude/longitude.",
        ),
        "no_temporal": FeatureSet(
            "no_temporal",
            lag_roll + spatial,
            "Full model without cyclical time and weekend features.",
        ),
        "aqi_lag_only": FeatureSet(
            "aqi_lag_only",
            [c for c in lag_roll if c.startswith("AQI_")],
            "Only AQI lag and rolling history.",
        ),
    }


def temporal_holdout_masks(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.Series, pd.Series, pd.Timestamp]:
    unique_times = pd.Series(df[TIME].drop_duplicates().sort_values().to_numpy())
    cutoff_index = int(len(unique_times) * (1.0 - test_fraction))
    cutoff = unique_times.iloc[cutoff_index]
    train_mask = df[TIME] < cutoff
    test_mask = df[TIME] >= cutoff
    return train_mask, test_mask, cutoff


def calibration_split(train_frame: pd.DataFrame, fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    unique_times = pd.Series(train_frame[TIME].drop_duplicates().sort_values().to_numpy())
    if len(unique_times) < 5:
        split = int(len(train_frame) * (1.0 - fraction))
        return train_frame.index[:split].to_numpy(), train_frame.index[split:].to_numpy()

    cutoff = unique_times.iloc[int(len(unique_times) * (1.0 - fraction))]
    fit_idx = train_frame.index[train_frame[TIME] < cutoff].to_numpy()
    cal_idx = train_frame.index[train_frame[TIME] >= cutoff].to_numpy()
    return fit_idx, cal_idx


def model(max_iter: int, learning_rate: float, random_state: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=max_iter,
        learning_rate=learning_rate,
        l2_regularization=0.01,
        max_leaf_nodes=31,
        random_state=random_state,
    )


def gaussian_crps(y_true: np.ndarray, mu: np.ndarray, sigma: float) -> float:
    sigma = max(float(sigma), 1e-6)
    z = (y_true - mu) / sigma
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    crps = sigma * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def metric_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    sigma: float,
) -> dict[str, float]:
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "picp_90": float(np.mean((y_true >= lower) & (y_true <= upper))),
        "mean_interval_width_90": float(np.mean(upper - lower)),
        "crps_gaussian": gaussian_crps(y_true, y_pred, sigma),
    }


def evaluate_regressor(
    df: pd.DataFrame,
    feature_set: FeatureSet,
    train_mask: pd.Series,
    test_mask: pd.Series,
    args: argparse.Namespace,
) -> dict[str, float | str]:
    train = df.loc[train_mask]
    test = df.loc[test_mask]
    fit_idx, cal_idx = calibration_split(train)

    x_fit = df.loc[fit_idx, feature_set.columns].fillna(0)
    y_fit = df.loc[fit_idx, TARGET].to_numpy()
    x_cal = df.loc[cal_idx, feature_set.columns].fillna(0)
    y_cal = df.loc[cal_idx, TARGET].to_numpy()
    x_test = test[feature_set.columns].fillna(0)
    y_test = test[TARGET].to_numpy()

    reg = model(args.max_iter, args.learning_rate, args.random_state)
    reg.fit(x_fit, y_fit)

    cal_pred = reg.predict(x_cal)
    residual = y_cal - cal_pred
    lower_resid, upper_resid = np.quantile(residual, [0.05, 0.95])
    sigma = float(np.std(residual, ddof=1))

    pred = reg.predict(x_test)
    lower = pred + lower_resid
    upper = pred + upper_resid
    row = metric_row(y_test, pred, lower, upper, sigma)
    row.update({"model": "HistGradientBoosting", "feature_set": feature_set.name})
    return row


def evaluate_persistence(df: pd.DataFrame, test_mask: pd.Series) -> dict[str, float | str]:
    test = df.loc[test_mask]
    y_true = test[TARGET].to_numpy()
    y_pred = test["AQI_lag_1h"].fillna(0).to_numpy()
    residual = y_true - y_pred
    lower_resid, upper_resid = np.quantile(residual, [0.05, 0.95])
    sigma = float(np.std(residual, ddof=1))
    row = metric_row(y_true, y_pred, y_pred + lower_resid, y_pred + upper_resid, sigma)
    row.update({"model": "Persistence", "feature_set": "AQI_lag_1h"})
    return row


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    rounded = df.copy()
    for col in rounded.select_dtypes(include=[np.number]).columns:
        if col != "n":
            rounded[col] = rounded[col].map(lambda value: f"{value:.4f}")
    rows = rounded[columns].astype(str).to_numpy().tolist()
    widths = [
        max(len(str(column)), *(len(row[i]) for row in rows)) if rows else len(str(column))
        for i, column in enumerate(columns)
    ]
    header = "| " + " | ".join(str(column).ljust(widths[i]) for i, column in enumerate(columns)) + " |"
    separator = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, separator] + body)


def write_report(
    args: argparse.Namespace,
    dataset_summary: dict[str, object],
    temporal_results: pd.DataFrame,
    loso_results: pd.DataFrame,
    ablation_results: pd.DataFrame,
    feature_sets: dict[str, FeatureSet],
    elapsed: float,
) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    temporal_results.to_csv(args.out_dir / "temporal_holdout_metrics.csv", index=False)
    loso_results.to_csv(args.out_dir / "leave_one_station_out_metrics.csv", index=False)
    ablation_results.to_csv(args.out_dir / "ablation_metrics.csv", index=False)

    summary = {
        "dataset": dataset_summary,
        "feature_sets": {name: fs.description for name, fs in feature_sets.items()},
        "target_column": TARGET,
        "target_units": "normalized AQI units from data/processed/features.parquet",
        "model": {
            "estimator": "sklearn.ensemble.HistGradientBoostingRegressor",
            "max_iter": args.max_iter,
            "learning_rate": args.learning_rate,
            "interval_method": "90% residual-calibrated interval on the final 20% of each training split",
            "crps_method": "Gaussian CRPS using calibration residual standard deviation",
        },
        "elapsed_seconds": round(elapsed, 3),
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    loso_agg = (
        loso_results.groupby(["model", "feature_set"], as_index=False)
        .agg(
            n=("n", "sum"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
            picp_90=("picp_90", "mean"),
            crps_gaussian=("crps_gaussian", "mean"),
        )
        .sort_values("rmse")
    )

    table_cols = ["model", "feature_set", "n", "mae", "rmse", "r2", "picp_90", "crps_gaussian"]
    report = f"""# Journal Validation Results

Generated by `scripts/run_journal_validation.py`.

Metrics are reported in normalized AQI units from `data/processed/features.parquet`.
The prediction target is the `AQI` column only. Pollutant variables such as
PM2.5, PM10, NO2, SO2, CO, and O3 are used only as historical input features.
Prediction intervals are residual-calibrated 90% intervals. CRPS is computed from
a Gaussian approximation whose scale is estimated on the calibration split.

## Dataset

- Target column: `AQI`
- Rows: {dataset_summary["rows"]}
- Stations: {dataset_summary["stations"]}
- Time span: {dataset_summary["start"]} to {dataset_summary["end"]}
- Temporal holdout cutoff: {dataset_summary["temporal_cutoff"]}

## Temporal Holdout

{markdown_table(temporal_results.sort_values("rmse"), table_cols)}

## Leave-One-Station-Out Summary

{markdown_table(loso_agg, table_cols)}

## Leave-One-Station-Out By Station

{markdown_table(loso_results.sort_values(["station_id", "rmse"]), ["station_id"] + table_cols)}

## Ablation Study

{markdown_table(ablation_results.sort_values("rmse"), table_cols)}

## Methods Text

For empirical validation, we used the processed CPCB station feature table and
evaluated one-step-ahead normalized AQI prediction with chronological holdout
and leave-one-station-out spatial validation. Input features were restricted to
lagged pollutant/AQI values, rolling statistics, cyclical time encodings,
weekend flags, and station coordinates. We report MAE, RMSE, R2, empirical
90% prediction interval coverage probability (PICP), mean interval width, and
Gaussian CRPS. Predictive intervals were calibrated from residuals on the final
20% of each training split.
"""
    (args.out_dir / "journal_validation_tables.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.quick:
        args.max_iter = min(args.max_iter, 40)

    started = perf_counter()
    df = load_features(args.data)
    feature_sets = make_feature_sets(df)
    temporal_train, temporal_test, cutoff = temporal_holdout_masks(df)

    dataset_summary = {
        "rows": int(len(df)),
        "stations": int(df[STATION].nunique()),
        "start": str(df[TIME].min()),
        "end": str(df[TIME].max()),
        "temporal_cutoff": str(cutoff),
    }

    temporal_rows: list[dict[str, float | str]] = [evaluate_persistence(df, temporal_test)]
    temporal_rows.append(evaluate_regressor(df, feature_sets["full"], temporal_train, temporal_test, args))

    ablation_rows: list[dict[str, float | str]] = [evaluate_persistence(df, temporal_test)]
    for feature_set in feature_sets.values():
        ablation_rows.append(evaluate_regressor(df, feature_set, temporal_train, temporal_test, args))

    loso_rows: list[dict[str, float | str]] = []
    for station_id in sorted(df[STATION].dropna().unique()):
        test_mask = df[STATION] == station_id
        train_mask = ~test_mask
        baseline = evaluate_persistence(df, test_mask)
        baseline["station_id"] = station_id
        loso_rows.append(baseline)

        row = evaluate_regressor(df, feature_sets["full"], train_mask, test_mask, args)
        row["station_id"] = station_id
        loso_rows.append(row)

    temporal_results = pd.DataFrame(temporal_rows)
    loso_results = pd.DataFrame(loso_rows)
    ablation_results = pd.DataFrame(ablation_rows)
    write_report(
        args,
        dataset_summary,
        temporal_results,
        loso_results,
        ablation_results,
        feature_sets,
        perf_counter() - started,
    )
    print(f"Wrote journal validation outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
