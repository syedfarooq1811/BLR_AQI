from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT = Path("data/raw/street_labels.template.csv")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-06T08:00:00+05:30",
                "lat": 12.9716,
                "lon": 77.5946,
                "AQI": 78.0,
                "source": "sensor_or_accuweather",
            }
        ]
    )
    df.to_csv(OUT, index=False)
    print(f"Wrote template -> {OUT}")


if __name__ == "__main__":
    main()
