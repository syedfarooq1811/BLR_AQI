"""Fetch live station AQI snapshot for backend live forecast anchoring.

Usage:
    $env:WAQI_TOKEN="your_token"
    python scripts/fetch_live_station_aqi.py

The API server automatically reads data/processed/live_station_aqi.json when it
is fresh enough, so this script can be run manually or scheduled every 30-60 min.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests


OUT_PATH = Path("data/processed/live_station_aqi.json")
WAQI_URL = "https://api.waqi.info/feed/geo:{lat};{lon}/"
EXCLUDED_STATION_IDS = {"site_1558"}

STATION_META = {
    "site_1553": {"name": "BWSSB Kadabesanahalli", "lat": 12.9279, "lon": 77.6271},
    "site_162": {"name": "Silk Board", "lat": 12.9174, "lon": 77.6235},
    "site_165": {"name": "BTM Layout", "lat": 12.9166, "lon": 77.6101},
    "site_1554": {"name": "Hebbal", "lat": 13.0450, "lon": 77.5966},
    "site_1555": {"name": "Jayanagar", "lat": 12.9250, "lon": 77.5938},
    "site_5729": {"name": "Peenya", "lat": 13.0289, "lon": 77.5199},
    "site_5681": {"name": "Bapuji Nagar", "lat": 12.9634, "lon": 77.5559},
    "site_163": {"name": "Hombegowda Nagar", "lat": 12.9609, "lon": 77.5996},
    "site_5678": {"name": "City Railway Station", "lat": 12.9774, "lon": 77.5713},
    "site_166": {"name": "Saneguruva Halli", "lat": 13.0068, "lon": 77.5090},
    "site_5686": {"name": "T Dasarahalli", "lat": 13.0450, "lon": 77.5116},
}


def fetch_station(session: requests.Session, token: str, sid: str, meta: dict) -> dict | None:
    response = session.get(
        WAQI_URL.format(lat=meta["lat"], lon=meta["lon"]),
        params={"token": token},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "ok":
        return None

    data = payload.get("data") or {}
    try:
        aqi = float(data.get("aqi"))
    except (TypeError, ValueError):
        return None

    observed_at = None
    time_info = data.get("time") or {}
    if isinstance(time_info, dict):
        observed_at = time_info.get("iso") or time_info.get("s")

    city = data.get("city") or {}
    return {
        "station_id": sid,
        "station_name": meta["name"],
        "lat": meta["lat"],
        "lon": meta["lon"],
        "aqi": round(max(0.0, min(500.0, aqi)), 2),
        "source": "waqi_geo_feed",
        "observed_at": observed_at,
        "provider_station": city.get("name"),
        "provider_url": city.get("url"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch live AQI for configured Bengaluru station coordinates.")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--token", default=os.environ.get("WAQI_TOKEN"))
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("Set WAQI_TOKEN or pass --token.")

    stations = {}
    errors = {}
    with requests.Session() as session:
        for sid, meta in STATION_META.items():
            if sid in EXCLUDED_STATION_IDS:
                continue
            try:
                item = fetch_station(session, args.token, sid, meta)
                if item:
                    stations[sid] = item
                else:
                    errors[sid] = "No AQI returned."
            except Exception as exc:
                errors[sid] = str(exc)

    payload = {
        "source": "waqi_geo_feed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "station_count": len(stations),
        "stations": stations,
        "errors": errors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(stations)} live station AQI values -> {args.out}")
    if errors:
        print(f"{len(errors)} station(s) had fetch issues; see errors in the JSON.")
    return 0 if stations else 1


if __name__ == "__main__":
    raise SystemExit(main())
