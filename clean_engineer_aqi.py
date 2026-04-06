from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SPARSE_COLUMN_THRESHOLD = 0.85

NON_NEGATIVE_COLUMNS = [
    "PM2.5 (µg/m³)",
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
    "Xylene (µg/m³)",
    "O Xylene (µg/m³)",
    "Eth-Benzene (µg/m³)",
    "MP-Xylene (µg/m³)",
    "RH (%)",
    "WS (m/s)",
    "WD (deg)",
    "RF (mm)",
    "TOT-RF (mm)",
    "SR (W/mt2)",
    "BP (mmHg)",
    "VWS (m/s)",
]

KEY_FEATURE_COLUMNS = [
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO2 (µg/m³)",
    "NOx (ppb)",
    "CO (mg/m³)",
    "Ozone (µg/m³)",
    "AT (°C)",
    "RH (%)",
]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["Timestamp"]

    df["year"] = ts.dt.year
    df["month"] = ts.dt.month
    df["day"] = ts.dt.day
    df["day_of_week"] = ts.dt.dayofweek
    df["day_of_year"] = ts.dt.dayofyear
    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["quarter"] = ts.dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = ts.dt.is_month_start.astype(int)
    df["is_month_end"] = ts.dt.is_month_end.astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    if {"PM2.5 (µg/m³)", "PM10 (µg/m³)"} <= set(df.columns):
        df["pm25_pm10_ratio"] = safe_divide(df["PM2.5 (µg/m³)"], df["PM10 (µg/m³)"])
        df["pm10_pm25_diff"] = df["PM10 (µg/m³)"] - df["PM2.5 (µg/m³)"]

    if {"NO2 (µg/m³)", "NOx (ppb)"} <= set(df.columns):
        df["no2_nox_ratio"] = safe_divide(df["NO2 (µg/m³)"], df["NOx (ppb)"])

    if {"NO (µg/m³)", "NO2 (µg/m³)"} <= set(df.columns):
        df["no2_no_ratio"] = safe_divide(df["NO2 (µg/m³)"], df["NO (µg/m³)"])

    if {"Benzene (µg/m³)", "Toluene (µg/m³)"} <= set(df.columns):
        df["benzene_toluene_ratio"] = safe_divide(
            df["Benzene (µg/m³)"], df["Toluene (µg/m³)"]
        )

    if {"AT (°C)", "RH (%)"} <= set(df.columns):
        df["temp_humidity_interaction"] = df["AT (°C)"] * df["RH (%)"]

    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [col for col in KEY_FEATURE_COLUMNS if col in df.columns]
    grouped = df.groupby("Station", observed=True, sort=False)

    for col in feature_columns:
        short_name = (
            col.lower()
            .replace(" ", "_")
            .replace(".", "")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_per_")
            .replace("%", "pct")
            .replace("°", "deg")
            .replace("µ", "u")
            .replace("-", "_")
        )

        df[f"{short_name}_lag_1"] = grouped[col].shift(1)
        df[f"{short_name}_lag_4"] = grouped[col].shift(4)
        df[f"{short_name}_diff_1"] = df[col] - df[f"{short_name}_lag_1"]

        rolling_4 = grouped[col].rolling(window=4, min_periods=1)
        rolling_16 = grouped[col].rolling(window=16, min_periods=1)

        df[f"{short_name}_roll_mean_4"] = (
            rolling_4.mean().reset_index(level=0, drop=True)
        )
        df[f"{short_name}_roll_std_4"] = (
            rolling_4.std().reset_index(level=0, drop=True).fillna(0)
        )
        df[f"{short_name}_roll_mean_16"] = (
            rolling_16.mean().reset_index(level=0, drop=True)
        )
        df[f"{short_name}_roll_std_16"] = (
            rolling_16.std().reset_index(level=0, drop=True).fillna(0)
        )

    return df


def fill_categorical(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    for col in categorical_cols:
        if col not in df.columns:
            continue

        df[col] = (
            df.groupby("Station", observed=True, sort=False)[col]
            .transform(lambda s: s.ffill().bfill())
            .fillna("Unknown")
        )

    return df


def fill_numeric(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    if not numeric_cols:
        return df

    df[numeric_cols] = (
        df.groupby("Station", observed=True, sort=False)[numeric_cols]
        .transform(lambda group: group.interpolate(limit_direction="both"))
    )

    station_medians = df.groupby("Station", observed=True, sort=False)[numeric_cols].transform(
        "median"
    )
    df[numeric_cols] = df[numeric_cols].fillna(station_medians)

    global_medians = df[numeric_cols].median(numeric_only=True)
    df[numeric_cols] = df[numeric_cols].fillna(global_medians)

    return df


def clean_and_engineer(input_file: Path, output_file: Path) -> None:
    df = pd.read_csv(input_file, parse_dates=["Timestamp"])
    original_shape = df.shape

    df = df.dropna(subset=["Timestamp", "Station"]).copy()
    df = df.sort_values(["Station", "Timestamp"], kind="stable").reset_index(drop=True)

    removable_columns: list[str] = []
    missing_ratio = df.isna().mean()

    for col, ratio in missing_ratio.items():
        if ratio == 1.0 or ratio > SPARSE_COLUMN_THRESHOLD:
            removable_columns.append(col)

    protected_columns = {"Timestamp", "Station", "Station ID", "Station Name", "City", "State"}
    removable_columns = [col for col in removable_columns if col not in protected_columns]
    df = df.drop(columns=removable_columns)

    for col in NON_NEGATIVE_COLUMNS:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    if "WD (deg)" in df.columns:
        df.loc[~df["WD (deg)"].between(0, 360), "WD (deg)"] = np.nan

    if "RH (%)" in df.columns:
        df.loc[~df["RH (%)"].between(0, 100), "RH (%)"] = np.nan

    categorical_cols = [
        col for col in ["State", "City", "Station Name", "Station ID"] if col in df.columns
    ]
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    df = fill_categorical(df, categorical_cols)
    df = fill_numeric(df, numeric_cols)
    df = add_time_features(df)
    df = add_ratio_features(df)
    df = add_lag_and_rolling_features(df)

    # Final pass so the engineered feature set is fully free of missing values.
    engineered_numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    df = fill_numeric(df, engineered_numeric_cols)
    df[categorical_cols + ["Station"]] = df[categorical_cols + ["Station"]].fillna("Unknown")

    df["Station_Code"] = df["Station"].astype("category").cat.codes

    total_missing = int(df.isna().sum().sum())
    df.to_csv(output_file, index=False)

    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Original shape: {original_shape}")
    print(f"Final shape: {df.shape}")
    print(f"Dropped sparse columns: {removable_columns if removable_columns else 'None'}")
    print(f"Remaining missing values: {total_missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean the merged AQI dataset and create engineered features."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="merged_aqi_data.csv",
        help="Path to the merged AQI CSV file.",
    )
    parser.add_argument(
        "--output",
        default="engineered_aqi_data.csv",
        help="Path for the cleaned and feature-engineered CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_file = Path(args.input_file).expanduser().resolve()
    output_file = Path(args.output).expanduser()

    if not output_file.is_absolute():
        output_file = input_file.parent / output_file

    clean_and_engineer(input_file=input_file, output_file=output_file)


if __name__ == "__main__":
    main()
