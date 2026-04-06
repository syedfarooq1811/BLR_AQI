from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
except ModuleNotFoundError:
    print("This script requires pandas. Install it with: pip install pandas")
    sys.exit(1)


def merge_aqi_csvs(input_folder: Path, output_file: Path) -> None:
    output_file = output_file.resolve()
    csv_files = sorted(
        csv_file
        for csv_file in input_folder.glob("*.csv")
        if csv_file.resolve() != output_file
    )

    if not csv_files:
        print(f"No CSV files found in: {input_folder}")
        return

    dataframes: list[pd.DataFrame] = []
    skipped_files: list[str] = []

    for csv_file in csv_files:
        if not csv_file.is_file():
            skipped_files.append(f"{csv_file.name}: file not found")
            continue

        try:
            df = pd.read_csv(csv_file)
        except FileNotFoundError:
            skipped_files.append(f"{csv_file.name}: file not found")
            continue
        except Exception as exc:
            skipped_files.append(f"{csv_file.name}: {exc}")
            continue

        if "Timestamp" not in df.columns:
            skipped_files.append(f"{csv_file.name}: missing 'Timestamp' column")
            continue

        df["Station"] = csv_file.stem
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        dataframes.append(df)

    if not dataframes:
        print("No valid CSV files were merged.")
        if skipped_files:
            print("\nSkipped files:")
            for item in skipped_files:
                print(f"- {item}")
        return

    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df = merged_df.sort_values("Timestamp", kind="stable").reset_index(drop=True)
    merged_df.to_csv(output_file, index=False)

    station_count = merged_df["Station"].nunique(dropna=True)

    print(f"Merged file saved to: {output_file}")
    print(f"Total rows: {len(merged_df)}")
    print(f"Number of stations: {station_count}")

    if skipped_files:
        print("\nSkipped files:")
        for item in skipped_files:
            print(f"- {item}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple AQI CSV files into a single dataset."
    )
    parser.add_argument(
        "input_folder",
        nargs="?",
        default=".",
        help="Folder containing station CSV files. Defaults to the current directory.",
    )
    parser.add_argument(
        "--output",
        default="merged_aqi_data.csv",
        help="Output CSV file name or path. Defaults to merged_aqi_data.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_folder = Path(args.input_folder).expanduser().resolve()
    output_file = Path(args.output).expanduser()

    if not output_file.is_absolute():
        output_file = input_folder / output_file

    if not input_folder.exists() or not input_folder.is_dir():
        print(f"Input folder does not exist or is not a directory: {input_folder}")
        return

    merge_aqi_csvs(input_folder=input_folder, output_file=output_file)


if __name__ == "__main__":
    main()
