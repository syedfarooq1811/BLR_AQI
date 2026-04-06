from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_FILE = "engineered_aqi_data.csv"
OUTPUT_FILE = "aqi_standard_features.csv"


KEEP_FEATURES = [
    "Station",
    "Station ID",
    "City",
    "Timestamp",
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
    "AT (°C)",
    "RH (%)",
    "WS (m/s)",
    "WD (deg)",
    "RF (mm)",
    "BP (mmHg)",
    "SR (W/mt2)",
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
    "pm25_pm10_ratio",
    "pm10_pm25_diff",
    "no2_nox_ratio",
    "no2_no_ratio",
    "benzene_toluene_ratio",
    "temp_humidity_interaction",
    "pm25_ug_per_m³_lag_1",
    "pm25_ug_per_m³_lag_4",
    "pm25_ug_per_m³_diff_1",
    "pm25_ug_per_m³_roll_mean_4",
    "pm25_ug_per_m³_roll_std_4",
    "pm25_ug_per_m³_roll_mean_16",
    "pm10_ug_per_m³_lag_1",
    "pm10_ug_per_m³_lag_4",
    "pm10_ug_per_m³_roll_mean_4",
    "pm10_ug_per_m³_roll_std_4",
    "no2_ug_per_m³_lag_1",
    "no2_ug_per_m³_lag_4",
    "no2_ug_per_m³_roll_mean_4",
    "nox_ppb_lag_1",
    "nox_ppb_roll_mean_4",
    "co_mg_per_m³_lag_1",
    "ozone_ug_per_m³_lag_1",
    "at_degc_lag_1",
    "rh_pct_lag_1",
    "ws_lag_1",
]


DROP_FEATURES = [
    "Station_Code",
    "minute",
    "quarter",
    "Eth-Benzene (µg/m³)",
    "MP-Xylene (µg/m³)",
    "TOT-RF (mm)",
]


def load_dataset(input_path: Path) -> pd.DataFrame:
    return pd.read_csv(input_path, parse_dates=["Timestamp"])


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    available_keep_features = [col for col in KEEP_FEATURES if col in df.columns]
    selected_df = df[available_keep_features].copy()

    removable_columns = [col for col in DROP_FEATURES if col in selected_df.columns]
    if removable_columns:
        selected_df = selected_df.drop(columns=removable_columns)

    return selected_df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    df = df.sort_values(["Station", "Timestamp"], kind="stable").reset_index(drop=True)

    # Fill station-wise in time order so sequential patterns stay intact.
    df = (
        df.groupby("Station", observed=True, sort=False)
        .transform(lambda column: column.ffill().bfill())
        .combine_first(df)
    )

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    if numeric_columns:
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    non_numeric_columns = [col for col in df.columns if col not in numeric_columns]
    for col in non_numeric_columns:
        if df[col].isna().any():
            mode = df[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            df[col] = df[col].fillna(fill_value)

    return df.reset_index(drop=True)


def print_summary(df: pd.DataFrame) -> None:
    print(f"Number of features: {len(df.columns)}")
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values after cleaning: {int(df.isna().sum().sum())}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / INPUT_FILE
    output_path = base_dir / OUTPUT_FILE

    df = load_dataset(input_path)
    df = select_features(df)
    df = clean_dataset(df)
    df.to_csv(output_path, index=False)

    print(f"Saved cleaned dataset to: {output_path}")
    print_summary(df)


if __name__ == "__main__":
    main()
