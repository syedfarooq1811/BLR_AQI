"""Fetch live CPCB AQI station data from data.gov.in and write app snapshot.

Usage:
    $env:CPCB_API_KEY="your_data_gov_key"
    python scripts/fetch_cpcb_live_station_aqi.py

Output:
    data/processed/live_station_aqi.json

The backend reads this file automatically and uses it to live-anchor 24h and
7-day station/street forecasts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
API_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
OUT_PATH = Path("data/processed/live_station_aqi.json")
DEFAULT_API_KEY = "579b464db66ec23bdd0000017571a3caaa2744b1676c553ab42050c7"
EXCLUDED_STATION_IDS = {"site_1558"}

INTERNAL_STATIONS = {
    "site_1553": {"name": "BWSSB Kadabesanahalli", "lat": 12.9279, "lon": 77.6271, "aliases": ["kadabesanahalli"]},
    "site_162": {"name": "Silk Board", "lat": 12.9174, "lon": 77.6235, "aliases": ["silk board"]},
    "site_165": {"name": "BTM Layout", "lat": 12.9166, "lon": 77.6101, "aliases": ["btm layout"]},
    "site_1554": {"name": "Hebbal", "lat": 13.0450, "lon": 77.5966, "aliases": ["hebbal"]},
    "site_1555": {"name": "Jayanagar", "lat": 12.9250, "lon": 77.5938, "aliases": ["jayanagar"]},
    "site_5729": {"name": "Peenya", "lat": 13.0289, "lon": 77.5199, "aliases": ["peenya"]},
    "site_5681": {"name": "Bapuji Nagar", "lat": 12.9634, "lon": 77.5559, "aliases": ["bapuji nagar"]},
    "site_163": {"name": "Hombegowda Nagar", "lat": 12.9609, "lon": 77.5996, "aliases": ["hombegowda"]},
    "site_5678": {"name": "City Railway Station", "lat": 12.9774, "lon": 77.5713, "aliases": ["city railway"]},
    "site_166": {"name": "Saneguruva Halli", "lat": 13.0068, "lon": 77.5090, "aliases": ["sanegurava", "saneguruva"]},
    "site_5686": {"name": "T Dasarahalli", "lat": 13.0450, "lon": 77.5116, "aliases": ["dasarahalli"]},
}

# Indian NAQI breakpoints: (concentration low, concentration high, AQI low, AQI high).
BREAKPOINTS = {
    "PM2.5": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500)],
    "PM10": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300), (351, 430, 301, 400), (431, 600, 401, 500)],
    "NO2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200), (181, 280, 201, 300), (281, 400, 301, 400), (401, 1000, 401, 500)],
    "SO2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200), (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2000, 401, 500)],
    "CO": [(0, 1, 0, 50), (1.1, 2, 51, 100), (2.1, 10, 101, 200), (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 50, 401, 500)],
    "OZONE": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200), (169, 208, 201, 300), (209, 748, 301, 400), (749, 1000, 401, 500)],
    "O3": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200), (169, 208, 201, 300), (209, 748, 301, 400), (749, 1000, 401, 500)],
    "NH3": [(0, 200, 0, 50), (201, 400, 51, 100), (401, 800, 101, 200), (801, 1200, 201, 300), (1201, 1800, 301, 400), (1801, 2400, 401, 500)],
}

PRIMARY_POLLUTANTS = ("PM2.5", "PM10", "NO2")
POLLUTANT_WEIGHTS = {
    "PM2.5": 1.0,
    "PM10": 0.95,
    "NO2": 0.9,
    "SO2": 0.7,
    "OZONE": 0.75,
    "O3": 0.75,
    "NH3": 0.6,
    "CO": 0.35,  # Down-weight CO because provider unit metadata is ambiguous.
}


def parse_number(value) -> float | None:
    try:
        if value in (None, "", "NA", "None"):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def subindex(pollutant: str, value: float) -> float | None:
    pollutant = pollutant.upper().replace("PM25", "PM2.5")
    ranges = BREAKPOINTS.get(pollutant)
    if not ranges:
        return None
    for c_low, c_high, i_low, i_high in ranges:
        if c_low <= value <= c_high:
            return ((i_high - i_low) / (c_high - c_low)) * (value - c_low) + i_low
    return 500.0 if value > ranges[-1][1] else None


def station_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    ref_lat = math.radians((lat1 + lat2) / 2.0)
    dy = (lat2 - lat1) * 111.32
    dx = (lon2 - lon1) * 111.32 * math.cos(ref_lat)
    return math.hypot(dx, dy)


def match_internal_station(cpcb_station: str, lat: float | None, lon: float | None) -> tuple[str | None, float | None, str]:
    text = cpcb_station.lower()
    for sid, meta in INTERNAL_STATIONS.items():
        if sid in EXCLUDED_STATION_IDS:
            continue
        if any(alias in text for alias in meta["aliases"]):
            distance = None
            if lat is not None and lon is not None:
                distance = station_distance_km(lat, lon, meta["lat"], meta["lon"])
            return sid, distance, "alias"

    if lat is None or lon is None:
        return None, None, "unmatched"

    nearest_sid = None
    nearest_distance = None
    for sid, meta in INTERNAL_STATIONS.items():
        if sid in EXCLUDED_STATION_IDS:
            continue
        distance = station_distance_km(lat, lon, meta["lat"], meta["lon"])
        if nearest_distance is None or distance < nearest_distance:
            nearest_sid = sid
            nearest_distance = distance
    if nearest_distance is not None and nearest_distance <= 3.0:
        return nearest_sid, nearest_distance, "nearest_coordinate"
    return None, nearest_distance, "too_far"


def fetch_records(api_key: str, state: str, city: str, page_size: int, max_pages: int) -> list[dict]:
    records: list[dict] = []
    total = None
    for page in range(max_pages):
        offset = page * page_size
        params = {
            "api-key": api_key,
            "format": "json",
            "offset": offset,
            "limit": page_size,
            "filters[state]": state,
            "filters[city]": city,
        }
        url = f"{API_URL}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 BluruAQI/1.0",
                "Accept": "application/json",
                "Connection": "close",
            },
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("status") != "ok":
            raise RuntimeError(f"CPCB API returned {payload.get('status')}: {payload.get('message')}")
        total = int(payload.get("total") or 0)
        page_records = payload.get("records", [])
        records.extend(page_records)
        if not page_records or len(records) >= total:
            break
    return records


def build_snapshot(records: list[dict]) -> dict:
    grouped: dict[str, dict] = defaultdict(lambda: {"pollutants": {}, "raw_rows": []})
    unmatched = []

    for row in records:
        station_name = row.get("station") or ""
        lat = parse_number(row.get("latitude"))
        lon = parse_number(row.get("longitude"))
        sid, distance_km, match_method = match_internal_station(station_name, lat, lon)
        if not sid:
            unmatched.append(station_name)
            continue

        value = parse_number(row.get("avg_value") or row.get("pollutant_avg"))
        pollutant = str(row.get("pollutant_id") or "").upper()
        index = subindex(pollutant, value) if value is not None else None
        if index is None:
            continue

        item = grouped[sid]
        item["station_id"] = sid
        item["station_name"] = INTERNAL_STATIONS[sid]["name"]
        item["provider_station"] = station_name
        item["lat"] = INTERNAL_STATIONS[sid]["lat"]
        item["lon"] = INTERNAL_STATIONS[sid]["lon"]
        item["provider_lat"] = lat
        item["provider_lon"] = lon
        item["observed_at"] = row.get("last_update")
        item["match_method"] = match_method
        item["match_distance_km"] = round(distance_km, 3) if distance_km is not None else None
        item["pollutants"][pollutant] = {
            "avg": value,
            "subindex": round(index, 2),
            "min": parse_number(row.get("min_value") or row.get("pollutant_min")),
            "max": parse_number(row.get("max_value") or row.get("pollutant_max")),
        }
        item["raw_rows"].append(row)

    stations = {}
    for sid, item in grouped.items():
        pollutants = item["pollutants"]
        primary_candidates = [
            (name, data)
            for name, data in pollutants.items()
            if name in PRIMARY_POLLUTANTS
        ]

        if primary_candidates:
            dominant_pollutant, dominant = max(
                primary_candidates,
                key=lambda pair: pair[1]["subindex"],
            )
            selection_mode = "priority_pollutants"
            weighted_dominant = dominant
            weighted_pollutant = dominant_pollutant
        else:
            weighted_candidates = []
            for name, data in pollutants.items():
                weight = POLLUTANT_WEIGHTS.get(name, 0.65)
                weighted_candidates.append((name, data, data["subindex"] * weight))
            weighted_pollutant, weighted_dominant, weighted_score = max(
                weighted_candidates,
                key=lambda triple: triple[2],
            )
            dominant_pollutant = weighted_pollutant
            dominant = dict(weighted_dominant)
            dominant["weighted_subindex"] = round(weighted_score, 2)
            selection_mode = "weighted_fallback"

        stations[sid] = {
            "station_id": sid,
            "station_name": item["station_name"],
            "lat": item["lat"],
            "lon": item["lon"],
            "aqi": round(float(dominant["subindex"]), 2),
            "dominant_pollutant": dominant_pollutant,
            "aqi_selection_mode": selection_mode,
            "weighted_dominant_pollutant": weighted_pollutant,
            "weighted_dominant_subindex": round(float(weighted_dominant["subindex"]), 2),
            "pollutants": item["pollutants"],
            "source": "cpcb_data_gov_in",
            "observed_at": item["observed_at"],
            "provider_station": item["provider_station"],
            "provider_lat": item["provider_lat"],
            "provider_lon": item["provider_lon"],
            "match_method": item["match_method"],
            "match_distance_km": item["match_distance_km"],
        }

    return {
        "source": "cpcb_data_gov_in",
        "resource_id": RESOURCE_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "station_count": len(stations),
        "stations": stations,
        "unmatched_provider_stations": sorted(set(unmatched)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch live CPCB Bengaluru AQI from data.gov.in.")
    parser.add_argument("--state", default="Karnataka")
    parser.add_argument("--city", default="Bengaluru")
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("CPCB_API_KEY") or os.environ.get("DATA_GOV_API_KEY") or DEFAULT_API_KEY,
    )
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Set CPCB_API_KEY, pass --api-key, or define DEFAULT_API_KEY in this script.")

    records = fetch_records(args.api_key, args.state, args.city, args.page_size, args.max_pages)
    snapshot = build_snapshot(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"Wrote {snapshot['station_count']} CPCB live station AQI values -> {args.out}")
    if snapshot["unmatched_provider_stations"]:
        print(f"Unmatched provider stations: {len(snapshot['unmatched_provider_stations'])}")
    return 0 if snapshot["station_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
