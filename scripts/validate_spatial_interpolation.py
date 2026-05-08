from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.spatial_interpolation import (
    blend_predictions,
    idw_weights_for_targets,
    linear_weights_for_targets,
)


FEATURES_PATH = Path("data/processed/features.parquet")
OUT_DIR = Path("reports/street_level")
TARGET = "AQI"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune station-to-street interpolation with LOSO station validation.")
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--powers", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    parser.add_argument("--blends", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    return parser.parse_args()


def load_station_panel(features_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(features_path, columns=["timestamp", "station_id", "lat", "lon", TARGET])
    df = df.dropna(subset=["timestamp", "station_id", "lat", "lon", TARGET])
    panel = df.pivot_table(index="timestamp", columns="station_id", values=TARGET, aggfunc="mean")
    panel = panel.dropna(axis=0, how="any").sort_index()
    coords = df.groupby("station_id", as_index=True)[["lat", "lon"]].first().loc[panel.columns]
    return panel, coords


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
    }


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


def main() -> None:
    args = parse_args()
    started = perf_counter()
    panel, coords = load_station_panel(args.features)
    station_ids = list(panel.columns)

    rows = []
    for hidden_station in station_ids:
        source_ids = [sid for sid in station_ids if sid != hidden_station]
        source_coords = coords.loc[source_ids].to_numpy(dtype=float)
        target_coords = coords.loc[[hidden_station]].to_numpy(dtype=float)
        source_values = panel[source_ids].to_numpy(dtype=float)
        y_true = panel[hidden_station].to_numpy(dtype=float)
        linear_weights = linear_weights_for_targets(source_coords, target_coords)

        for power in args.powers:
            idw_weights = idw_weights_for_targets(source_coords, target_coords, power=power)
            for blend in args.blends:
                y_pred = blend_predictions(source_values, linear_weights, idw_weights, blend).ravel()
                row = metrics(y_true, y_pred)
                row.update(
                    {
                        "station_id": hidden_station,
                        "idw_power": power,
                        "idw_blend": blend,
                    }
                )
                rows.append(row)

    by_station = pd.DataFrame(rows)
    aggregate = (
        by_station.groupby(["idw_power", "idw_blend"], as_index=False)
        .agg(
            n=("n", "sum"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
            bias=("bias", "mean"),
        )
        .sort_values(["rmse", "mae"])
    )
    best = aggregate.iloc[0].to_dict()
    best_station = by_station[
        (by_station["idw_power"] == best["idw_power"])
        & (by_station["idw_blend"] == best["idw_blend"])
    ].sort_values("rmse")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(args.out_dir / "interpolation_loso_tuning.csv", index=False)
    best_station.to_csv(args.out_dir / "interpolation_loso_by_station.csv", index=False)
    metadata = {
        "target_column": TARGET,
        "target_units": "normalized AQI units from data/processed/features.parquet",
        "rows": int(len(panel)),
        "stations": len(station_ids),
        "best": best,
        "elapsed_seconds": round(perf_counter() - started, 3),
    }
    (args.out_dir / "interpolation_loso_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    cols = ["idw_power", "idw_blend", "n", "mae", "rmse", "r2", "bias"]
    station_cols = ["station_id", "idw_power", "idw_blend", "n", "mae", "rmse", "r2", "bias"]
    report = f"""# Street-Level Interpolation LOSO Validation

The target is the `AQI` column only. Each station is hidden in turn, AQI is
interpolated from the other stations at the hidden station's coordinates, and
errors are measured against the hidden AQI time series.

Metrics are in normalized AQI units.

## Best Settings

- IDW power: {best["idw_power"]:.4f}
- IDW blend: {best["idw_blend"]:.4f}
- Mean station RMSE: {best["rmse"]:.4f}
- Mean station MAE: {best["mae"]:.4f}

## Tuning Grid

{markdown_table(aggregate, cols)}

## Best Setting By Station

{markdown_table(best_station, station_cols)}
"""
    (args.out_dir / "interpolation_loso_report.md").write_text(report, encoding="utf-8")
    print(f"Wrote interpolation LOSO outputs to {args.out_dir}")
    print(f"Best: power={best['idw_power']:.3f}, blend={best['idw_blend']:.3f}, rmse={best['rmse']:.4f}")


if __name__ == "__main__":
    main()
