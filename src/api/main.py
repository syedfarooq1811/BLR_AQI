from __future__ import annotations

import json
import math
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import osmnx as ox
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal

from src.api.bias_store import BiasStore, bias_cell

from src.routing.astar import generate_pareto_fronts
from src.routing.cost_graph import build_cost_graph
from src.routing.exposure import (
    calculate_route_exposure,
    FORECAST_GRID_PATH,
    load_forecast_grid_bundle,
    sample_aqi_window_for_coordinates,
)
from src.routing.road_context import context_aqi_adjustment, road_context_for_coordinates

app = FastAPI(title="Bengaluru AQI API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path("frontend/public")
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

FORECAST_PATH = Path("data/processed/forecast_station_7day.json")
LIVE_STATION_PATH = Path("data/processed/live_station_aqi.json")
LIVE_STATION_MAX_AGE_MINUTES = 180
LIVE_FETCH_INTERVAL_SECONDS = 3600
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

G = None
AQI_DATA = None
GRAPH_LOADING = False
GRID_SURFACE_CACHE: dict[tuple, dict] = {}
LIVE_FETCHER_STARTED = False
LIVE_FETCH_STATUS = {
    "last_attempt": None,
    "last_success": None,
    "station_count": 0,
    "status": "not_started",
    "error": None,
}
LOCATION_BIAS_CACHE: dict[str, dict] = {}

# Canonical output scale for street AQI responses.
CANONICAL_AQI_SCALE = "IN_CPCB"

BIAS_DB_PATH = Path("data/processed/bias_store.sqlite")
BIAS_STORE: BiasStore | None = None


class RouteRequest(BaseModel):
    orig_lat: float
    orig_lon: float
    dest_lat: float
    dest_lon: float
    profile: str = "healthy"
    hour: int = 1
    transport_mode: str = "driving"
    hours_to_scan: int = 1


class Waypoint(BaseModel):
    lat: float
    lon: float


class TSPRequest(BaseModel):
    waypoints: List[Waypoint]
    profile: str = "healthy"
    hour: int = 1
    transport_mode: str = "driving"


class ObservationRequest(BaseModel):
    source: Literal["accuweather", "user", "sensor", "cpcb_data_gov_in", "other"] = "user"
    lat: float
    lon: float
    aqi: float
    aqi_scale: Literal["US_EPA", "IN_CPCB"] = "IN_CPCB"
    confidence: float = 0.85
    observed_at: str | None = None


@app.on_event("startup")
def load_runtime_assets():
    global AQI_DATA, GRAPH_LOADING, BIAS_STORE
    AQI_DATA = load_forecast_grid_bundle()
    if AQI_DATA is None:
        print("Forecast grid not loaded. Route AQI will use fallback values.")
    else:
        refresh_aqi_data_live_anchor()
    start_live_station_fetcher()
    BIAS_STORE = BiasStore(BIAS_DB_PATH)

    def load_graph_background():
        global G, GRAPH_LOADING
        GRAPH_LOADING = True
        try:
            G = build_cost_graph()
            print("Street graph loaded.")
        except Exception as exc:
            print(f"Graph not loaded: {exc}")
        finally:
            GRAPH_LOADING = False

    threading.Thread(target=load_graph_background, daemon=True).start()


def ensure_graph_started():
    global GRAPH_LOADING
    if G is not None or GRAPH_LOADING:
        return

    def load_graph_background():
        global G, GRAPH_LOADING
        GRAPH_LOADING = True
        try:
            G = build_cost_graph()
            print("Street graph loaded.")
        except Exception as exc:
            print(f"Graph not loaded: {exc}")
        finally:
            GRAPH_LOADING = False

    threading.Thread(target=load_graph_background, daemon=True).start()


def load_runtime_assets_sync():
    global AQI_DATA, G, BIAS_STORE
    try:
        G = build_cost_graph()
        print("Street graph loaded.")
    except Exception as exc:
        print(f"Graph not loaded: {exc}")

    AQI_DATA = load_forecast_grid_bundle()
    if AQI_DATA is None:
        print("Forecast grid not loaded. Route AQI will use fallback values.")
    else:
        refresh_aqi_data_live_anchor()
    BIAS_STORE = BiasStore(BIAS_DB_PATH)


def parse_live_generated_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def live_decay_weight(hour_index: int) -> float:
    if hour_index <= 0:
        return 1.0
    # Keep the live correction strong through the same day, then taper it so
    # the 7-day forecast gradually returns to the learned model trajectory.
    return float(math.exp(-hour_index / 24.0))


def load_live_station_snapshot(max_age_minutes: int = LIVE_STATION_MAX_AGE_MINUTES) -> dict:
    if not LIVE_STATION_PATH.exists():
        return {"available": False, "stations": {}, "reason": "No live station snapshot found."}

    try:
        with open(LIVE_STATION_PATH, encoding="utf-8") as file:
            payload = json.load(file)
    except Exception as exc:
        return {"available": False, "stations": {}, "reason": f"Could not read live station snapshot: {exc}"}

    generated_at = parse_live_generated_at(payload.get("generated_at"))
    age_minutes = None
    if generated_at is not None:
        now = datetime.now(generated_at.tzinfo or timezone.utc)
        age_minutes = max(0.0, (now - generated_at).total_seconds() / 60.0)
        if age_minutes > max_age_minutes:
            return {
                "available": False,
                "stations": {},
                "generated_at": payload.get("generated_at"),
                "age_minutes": round(age_minutes, 1),
                "reason": "Live station snapshot is stale.",
            }

    raw_stations = payload.get("stations", {})
    if isinstance(raw_stations, list):
        raw_stations = {item.get("station_id"): item for item in raw_stations if item.get("station_id")}

    stations = {}
    for sid, item in raw_stations.items():
        if sid not in STATION_META:
            continue
        try:
            aqi = float(item.get("aqi"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(aqi):
            continue
        stations[sid] = {
            "aqi": max(0.0, min(500.0, aqi)),
            "source": item.get("source", payload.get("source", "live_snapshot")),
            "observed_at": item.get("observed_at") or payload.get("generated_at"),
        }

    return {
        "available": bool(stations),
        "stations": stations,
        "generated_at": payload.get("generated_at"),
        "age_minutes": round(age_minutes, 1) if age_minutes is not None else None,
        "source": payload.get("source", "live_snapshot"),
        "reason": "Live station AQI loaded." if stations else "No matching Bengaluru stations in live snapshot.",
    }


def apply_live_station_anchor(station_data: dict, live_snapshot: dict) -> dict:
    if not live_snapshot.get("available"):
        return station_data

    anchored = deepcopy(station_data)
    for sid, live in live_snapshot.get("stations", {}).items():
        station = anchored.get(sid)
        if not station:
            continue
        hourly = station.get("hourly") or []
        if not hourly:
            continue

        live_aqi = float(live["aqi"])
        first_forecast = float(hourly[0].get("aqi", live_aqi))
        delta = live_aqi - first_forecast
        for hour_index, point in enumerate(hourly):
            corrected = max(0.0, min(500.0, float(point.get("aqi", first_forecast)) + delta * live_decay_weight(hour_index)))
            category, color = classify_aqi(corrected), color_for_aqi(corrected)
            point["aqi"] = round(corrected, 2)
            point["category"] = category
            point["color"] = color
            point["live_corrected"] = True
            point["live_weight"] = round(live_decay_weight(hour_index), 4)
            if hour_index == 0:
                point["source"] = live.get("source", "live_snapshot")
                point["observed_at"] = live.get("observed_at")
        station["live_anchor"] = {
            "aqi": round(live_aqi, 2),
            "delta_from_model_now": round(delta, 2),
            "source": live.get("source", "live_snapshot"),
            "observed_at": live.get("observed_at"),
        }

    anchored["_live_snapshot"] = {
        "available": True,
        "generated_at": live_snapshot.get("generated_at"),
        "age_minutes": live_snapshot.get("age_minutes"),
        "source": live_snapshot.get("source"),
    }
    return anchored


def load_station_forecast(use_live: bool = True):
    if not FORECAST_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Run `python -m src.models.forecast` first.",
        )
    with open(FORECAST_PATH, encoding="utf-8") as file:
        data = json.load(file)
    data = {sid: value for sid, value in data.items() if sid not in EXCLUDED_STATION_IDS}
    if not use_live:
        return data
    return apply_live_station_anchor(data, load_live_station_snapshot())


def refresh_aqi_data_live_anchor() -> None:
    if AQI_DATA is None:
        return
    try:
        AQI_DATA["station_anchor_forecast"] = load_station_forecast(use_live=True)
        snapshot = AQI_DATA["station_anchor_forecast"].get("_live_snapshot", {})
        AQI_DATA["live_snapshot"] = snapshot
    except Exception as exc:
        AQI_DATA["station_anchor_forecast"] = None
        AQI_DATA["live_snapshot"] = {"available": False, "reason": str(exc)}


def fetch_cpcb_live_snapshot_once() -> dict:
    global LIVE_FETCH_STATUS
    LIVE_FETCH_STATUS["last_attempt"] = datetime.now(timezone.utc).isoformat()
    LIVE_FETCH_STATUS["status"] = "fetching"
    LIVE_FETCH_STATUS["error"] = None
    try:
        from scripts.fetch_cpcb_live_station_aqi import (
            DEFAULT_API_KEY,
            OUT_PATH,
            build_snapshot,
            fetch_records,
        )

        records = fetch_records(
            DEFAULT_API_KEY,
            state="Karnataka",
            city="Bengaluru",
            page_size=20,
            max_pages=5,
        )
        snapshot = build_snapshot(records)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        GRID_SURFACE_CACHE.clear()
        refresh_aqi_data_live_anchor()
        LIVE_FETCH_STATUS.update(
            {
                "last_success": datetime.now(timezone.utc).isoformat(),
                "station_count": int(snapshot.get("station_count", 0)),
                "status": "ok",
                "error": None,
            }
        )
        return snapshot
    except Exception as exc:
        LIVE_FETCH_STATUS.update(
            {
                "status": "error",
                "error": str(exc),
            }
        )
        return {"source": "cpcb_data_gov_in", "station_count": 0, "error": str(exc)}


def live_station_fetch_loop() -> None:
    while True:
        fetch_cpcb_live_snapshot_once()
        threading.Event().wait(LIVE_FETCH_INTERVAL_SECONDS)


def start_live_station_fetcher() -> None:
    global LIVE_FETCHER_STARTED
    if LIVE_FETCHER_STARTED:
        return
    LIVE_FETCHER_STARTED = True
    threading.Thread(target=live_station_fetch_loop, daemon=True).start()


def interpolate_station_residual(lat: float, lon: float, hour_index: int, station_data: dict) -> float:
    if AQI_DATA is None:
        return 0.0

    forecast_grid = AQI_DATA.get("forecast_grid")
    lat_axis = AQI_DATA.get("lat_axis")
    lon_axis = AQI_DATA.get("lon_axis")
    if forecast_grid is None or lat_axis is None or lon_axis is None:
        return 0.0

    weighted_residual = 0.0
    total_weight = 0.0
    for sid, meta in STATION_META.items():
        station_entry = station_data.get(sid)
        if not station_entry:
            continue
        hourly = station_entry.get("hourly") or []
        if hour_index >= len(hourly):
            continue

        station_aqi = float(hourly[hour_index]["aqi"])
        station_lat = float(meta["lat"])
        station_lon = float(meta["lon"])
        lat_idx = int(np.abs(lat_axis - station_lat).argmin())
        lon_idx = int(np.abs(lon_axis - station_lon).argmin())
        grid_aqi = float(forecast_grid[hour_index, lat_idx, lon_idx])
        if not np.isfinite(grid_aqi):
            continue

        dy = (station_lat - lat) * 111.32
        dx = (station_lon - lon) * 111.32 * math.cos(math.radians(lat))
        distance_km = math.sqrt((dx * dx) + (dy * dy))
        if distance_km < 0.05:
            return station_aqi - grid_aqi

        weight = 1.0 / max(distance_km * distance_km, 0.01)
        weighted_residual += weight * (station_aqi - grid_aqi)
        total_weight += weight

    if total_weight <= 0:
        return 0.0
    return weighted_residual / total_weight


def geo_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    ref_lat = math.radians((lat1 + lat2) / 2.0)
    dy = (lat2 - lat1) * 111.32
    dx = (lon2 - lon1) * 111.32 * math.cos(ref_lat)
    return math.hypot(dx, dy)


def nearest_live_station_observation(lat: float, lon: float, station_data: dict) -> dict | None:
    nearest = None
    for sid, meta in STATION_META.items():
        station_entry = station_data.get(sid)
        hourly = station_entry.get("hourly") if station_entry else None
        if not hourly:
            continue
        current_aqi = parse_float_or_none(hourly[0].get("aqi"))
        if current_aqi is None:
            continue
        distance_km = geo_distance_km(lat, lon, float(meta["lat"]), float(meta["lon"]))
        candidate = {
            "station_id": sid,
            "station_name": station_entry.get("station_name", sid),
            "aqi": current_aqi,
            "distance_km": distance_km,
        }
        if nearest is None or distance_km < nearest["distance_km"]:
            nearest = candidate
    return nearest


def live_station_nowcast(lat: float, lon: float, max_distance_km: float = 10.0) -> dict | None:
    snapshot = load_live_station_snapshot()
    if not snapshot.get("available"):
        return None

    candidates = []
    for sid, station in snapshot.get("stations", {}).items():
        meta = STATION_META.get(sid)
        aqi = parse_float_or_none(station.get("aqi"))
        if not meta or aqi is None:
            continue
        distance_km = geo_distance_km(lat, lon, float(meta["lat"]), float(meta["lon"]))
        if distance_km > max_distance_km:
            continue
        candidates.append(
            {
                "station_id": sid,
                "station_name": meta["name"],
                "aqi": aqi,
                "distance_km": distance_km,
            }
        )

    if not candidates:
        return None

    candidates.sort(key=lambda item: item["distance_km"])
    nearest = candidates[0]
    top = candidates[:3]

    if nearest["distance_km"] < 0.05:
        blended_aqi = nearest["aqi"]
    else:
        weighted_sum = 0.0
        total_weight = 0.0
        for item in top:
            weight = 1.0 / max(item["distance_km"] ** 2, 0.01)
            weighted_sum += weight * item["aqi"]
            total_weight += weight
        blended_aqi = weighted_sum / max(total_weight, 1e-9)

    return {
        "aqi": round(float(blended_aqi), 2),
        "nearest_distance_km": round(float(nearest["distance_km"]), 3),
        "nearest_station_id": nearest["station_id"],
        "nearest_station_name": nearest["station_name"],
        "contributors": [
            {
                "station_id": item["station_id"],
                "station_name": item["station_name"],
                "aqi": round(float(item["aqi"]), 2),
                "distance_km": round(float(item["distance_km"]), 3),
            }
            for item in top
        ],
    }


def parse_float_or_none(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def station_residuals_for_hour(hour_index: int, station_data: dict) -> list[dict]:
    if AQI_DATA is None:
        return []

    forecast_grid = AQI_DATA.get("forecast_grid")
    lat_axis = AQI_DATA.get("lat_axis")
    lon_axis = AQI_DATA.get("lon_axis")
    if forecast_grid is None or lat_axis is None or lon_axis is None:
        return []

    residuals = []
    for sid, meta in STATION_META.items():
        station_entry = station_data.get(sid)
        hourly = station_entry.get("hourly") if station_entry else None
        if not hourly or hour_index >= len(hourly):
            continue

        station_lat = float(meta["lat"])
        station_lon = float(meta["lon"])
        lat_idx = int(np.abs(lat_axis - station_lat).argmin())
        lon_idx = int(np.abs(lon_axis - station_lon).argmin())
        grid_aqi = float(forecast_grid[hour_index, lat_idx, lon_idx])
        station_aqi = float(hourly[hour_index]["aqi"])
        if not np.isfinite(grid_aqi):
            continue

        residuals.append(
            {
                "station_id": sid,
                "station_name": station_entry.get("station_name", sid),
                "lat": station_lat,
                "lon": station_lon,
                "aqi": station_aqi,
                "residual": station_aqi - grid_aqi,
            }
        )
    return residuals


def corrected_surface_for_hour(hour: int, stride: int = 3) -> dict:
    if AQI_DATA is None:
        raise HTTPException(status_code=503, detail="Forecast grid not loaded yet.")

    forecast_grid = AQI_DATA.get("forecast_grid")
    grid_lat = AQI_DATA.get("grid_lat")
    grid_lon = AQI_DATA.get("grid_lon")
    lat_axis = AQI_DATA.get("lat_axis")
    lon_axis = AQI_DATA.get("lon_axis")
    if forecast_grid is None or grid_lat is None or grid_lon is None or lat_axis is None or lon_axis is None:
        raise HTTPException(status_code=503, detail="Forecast grid data missing.")

    hour_index = hour - 1
    stride = max(1, min(int(stride), 10))
    live_snapshot = load_live_station_snapshot()
    cache_key = (hour_index, stride, live_snapshot.get("generated_at"))
    if cache_key in GRID_SURFACE_CACHE:
        return GRID_SURFACE_CACHE[cache_key]

    station_data = apply_live_station_anchor(load_station_forecast(use_live=False), live_snapshot)
    residuals = station_residuals_for_hour(hour_index, station_data)
    lat_sample = np.asarray(grid_lat[::stride, ::stride], dtype=float)
    lon_sample = np.asarray(grid_lon[::stride, ::stride], dtype=float)
    base = np.asarray(forecast_grid[hour_index, ::stride, ::stride], dtype=float)
    corrected = np.nan_to_num(base, nan=75.0)

    if residuals:
        weighted_residual = np.zeros_like(corrected, dtype=float)
        total_weight = np.zeros_like(corrected, dtype=float)
        nearest_station = np.full_like(corrected, np.inf, dtype=float)
        for item in residuals:
            station_lat = float(item["lat"])
            station_lon = float(item["lon"])
            dy = (station_lat - lat_sample) * 111.32
            dx = (station_lon - lon_sample) * 111.32 * np.cos(np.deg2rad(lat_sample))
            distance_km = np.sqrt((dx * dx) + (dy * dy))
            nearest_station = np.minimum(nearest_station, distance_km)
            weights = 1.0 / np.maximum(distance_km * distance_km, 0.01)
            weighted_residual += weights * float(item["residual"])
            total_weight += weights

        corrected += np.divide(
            weighted_residual,
            np.maximum(total_weight, 1e-9),
            out=np.zeros_like(weighted_residual),
            where=total_weight > 0,
        )
    else:
        nearest_station = np.full_like(corrected, np.nan, dtype=float)

    # A light street-surface correction: areas with sharper local spatial
    # changes get slightly wider uncertainty and a small curbside uplift.
    local_gradient = np.zeros_like(corrected, dtype=float)
    if corrected.shape[0] > 1:
        local_gradient += np.abs(np.gradient(corrected, axis=0))
    if corrected.shape[1] > 1:
        local_gradient += np.abs(np.gradient(corrected, axis=1))
    context_adjustment = np.clip(local_gradient * 0.18, 0.0, 6.0)
    corrected = np.clip(corrected + context_adjustment, 0.0, 500.0)

    finite_nearest = np.where(np.isfinite(nearest_station), nearest_station, 5.0)
    uncertainty = np.clip(5.0 + finite_nearest * 2.5 + local_gradient * 0.22, 5.0, 35.0)
    trust = np.clip(100.0 - finite_nearest * 8.0 - uncertainty * 1.1, 5.0, 99.0)

    points = []
    flat_lat = lat_sample.ravel()
    flat_lon = lon_sample.ravel()
    flat_aqi = corrected.ravel()
    flat_uncertainty = uncertainty.ravel()
    flat_trust = trust.ravel()
    flat_nearest = finite_nearest.ravel()
    for lat, lon, aqi, unc, trust_score, nearest_km in zip(
        flat_lat,
        flat_lon,
        flat_aqi,
        flat_uncertainty,
        flat_trust,
        flat_nearest,
    ):
        points.append(
            {
                "lat": round(float(lat), 6),
                "lon": round(float(lon), 6),
                "aqi": round(float(aqi), 2),
                "category": classify_aqi(float(aqi)),
                "color": color_for_aqi(float(aqi)),
                "uncertainty": round(float(unc), 2),
                "trust_score": round(float(trust_score), 1),
                "nearest_station_km": round(float(nearest_km), 3),
            }
        )

    result = {
        "hour": hour,
        "stride": stride,
        "rows": int(corrected.shape[0]),
        "cols": int(corrected.shape[1]),
        "point_count": len(points),
        "correction": "station_residual_idw_with_uncertainty",
        "points": points,
        "stations": [
            {
                "station_id": item["station_id"],
                "station_name": item["station_name"],
                "lat": round(float(item["lat"]), 6),
                "lon": round(float(item["lon"]), 6),
                "aqi": round(float(item["aqi"]), 2),
                "category": classify_aqi(float(item["aqi"])),
                "color": color_for_aqi(float(item["aqi"])),
            }
            for item in residuals
        ],
    }
    GRID_SURFACE_CACHE[cache_key] = result
    return result


def get_preferred_edge_data(u, v) -> dict:
    edge_bundle = G.get_edge_data(u, v, default={}) or {}
    if not edge_bundle:
        return {}
    return min(
        edge_bundle.values(),
        key=lambda data: data.get("travel_time", data.get("length", float("inf"))),
    )


def parse_linestring_wkt(value: str) -> list[tuple[float, float]]:
    text = value.strip()
    if not text.upper().startswith("LINESTRING"):
        return []
    body = text[text.find("(") + 1:text.rfind(")")]
    coords = []
    for pair in body.split(","):
        parts = pair.strip().split()
        if len(parts) >= 2:
            coords.append((float(parts[0]), float(parts[1])))
    return coords


def edge_geometry_coordinates(u, v) -> list[list[float]]:
    edge_data = get_preferred_edge_data(u, v)
    geometry = edge_data.get("geometry")
    if hasattr(geometry, "coords"):
        lon_lat = [(float(lon), float(lat)) for lon, lat in geometry.coords]
    elif isinstance(geometry, str):
        lon_lat = parse_linestring_wkt(geometry)
    else:
        lon_lat = [
            (float(G.nodes[u]["x"]), float(G.nodes[u]["y"])),
            (float(G.nodes[v]["x"]), float(G.nodes[v]["y"])),
        ]

    if len(lon_lat) >= 2:
        u_lon, u_lat = float(G.nodes[u]["x"]), float(G.nodes[u]["y"])
        first_lon, first_lat = lon_lat[0]
        last_lon, last_lat = lon_lat[-1]
        first_dist = abs(first_lon - u_lon) + abs(first_lat - u_lat)
        last_dist = abs(last_lon - u_lon) + abs(last_lat - u_lat)
        if last_dist < first_dist:
            lon_lat.reverse()

    return [[round(lat, 6), round(lon, 6)] for lon, lat in lon_lat]


def route_to_coordinates(route):
    if not route:
        return []

    coordinates = []
    for u, v in zip(route, route[1:]):
        segment = edge_geometry_coordinates(u, v)
        if not segment:
            continue
        if coordinates and segment[0] == coordinates[-1]:
            coordinates.extend(segment[1:])
        else:
            coordinates.extend(segment)

    if not coordinates:
        return [
            [round(float(G.nodes[node]["y"]), 6), round(float(G.nodes[node]["x"]), 6)]
            for node in route
        ]
    return coordinates


def get_edge_display_name(u, v) -> str:
    edge_data = get_preferred_edge_data(u, v)
    if not edge_data:
        return "this road"
    name = edge_data.get("name") or edge_data.get("ref") or edge_data.get("highway") or "this road"
    if isinstance(name, (list, tuple)):
        name = name[0] if name else "this road"
    return str(name).replace("_", " ").title()


def bearing_degrees(a, b) -> float:
    lat1 = math.radians(float(a["y"]))
    lat2 = math.radians(float(b["y"]))
    dlon = math.radians(float(b["x"]) - float(a["x"]))
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def turn_label(delta: float) -> str:
    if abs(delta) < 25:
        return "Continue straight"
    if delta > 55:
        return "Turn right"
    if delta > 25:
        return "Bear right"
    if delta < -55:
        return "Turn left"
    return "Bear left"


def traffic_level_for_hour(hour: int) -> dict:
    clock_hour = max(0, min(23, (hour - 1) % 24))
    if 8 <= clock_hour <= 10 or 17 <= clock_hour <= 20:
        return {"level": "High", "summary": "Peak-hour traffic likely ahead"}
    if 7 <= clock_hour <= 11 or 16 <= clock_hour <= 21:
        return {"level": "Moderate", "summary": "Traffic is likely building on main roads"}
    return {"level": "Light", "summary": "Traffic pressure is likely lower now"}


def build_navigation_guidance(route, timeline, hour: int) -> dict:
    cumulative = [0.0]
    for item in timeline:
        cumulative.append(cumulative[-1] + float(item.get("segment_distance_m", 0.0)))

    steps = []
    signal_steps = []
    for index in range(1, len(route) - 1):
        previous_node = G.nodes[route[index - 1]]
        current_node = G.nodes[route[index]]
        next_node = G.nodes[route[index + 1]]
        incoming = bearing_degrees(previous_node, current_node)
        outgoing = bearing_degrees(current_node, next_node)
        delta = ((outgoing - incoming + 540.0) % 360.0) - 180.0
        label = turn_label(delta)
        if label == "Continue straight":
            continue

        node_tags = G.nodes[route[index]]
        distance_m = round(cumulative[index], 0)
        step = {
            "distance_m": distance_m,
            "instruction": f"{label} onto {get_edge_display_name(route[index], route[index + 1])}",
            "lat": round(float(current_node["y"]), 6),
            "lon": round(float(current_node["x"]), 6),
            "turn_angle": round(delta, 1),
        }
        steps.append(step)
        if node_tags.get("highway") == "traffic_signals":
            signal_steps.append({**step, "instruction": "Traffic signal ahead"})

    hotspot_segments = sorted(
        timeline,
        key=lambda item: float(item.get("aqi", 0.0)),
        reverse=True,
    )[:3]
    aqi_alerts = [
        {
            "distance_m": round(cumulative[min(i, len(cumulative) - 1)], 0),
            "aqi": item.get("aqi"),
            "lat": item.get("lat"),
            "lon": item.get("lon"),
            "message": f"High AQI ahead: {item.get('aqi')}",
        }
        for i, item in enumerate(hotspot_segments)
        if float(item.get("aqi", 0.0)) >= 100.0
    ]

    return {
        "traffic": traffic_level_for_hour(hour),
        "steps": steps[:12],
        "signals": signal_steps[:8],
        "aqi_alerts": aqi_alerts,
    }


def route_quality_summary(navigation: dict) -> dict:
    turn_count = len(navigation.get("steps") or [])
    signal_count = len(navigation.get("signals") or [])
    hotspot_count = len(navigation.get("aqi_alerts") or [])
    stress_score = round((turn_count * 1.4) + (signal_count * 2.4) + (hotspot_count * 2.8), 2)
    return {
        "turn_count": turn_count,
        "signal_count": signal_count,
        "hotspot_count": hotspot_count,
        "stress_score": stress_score,
    }


def summarize_route(route, metric, hour: int, transport_mode: str = "driving", profile: str = "healthy"):
    exposure, timeline = calculate_route_exposure(
        G,
        route,
        AQI_DATA,
        hour=hour,
        transport_mode=transport_mode,
        profile=profile,
    )
    total_time = sum(item["segment_time_seconds"] for item in timeline)
    total_distance = sum(item["segment_distance_m"] for item in timeline)
    concentration_exposure = sum(
        item["aqi"] * (item["segment_time_seconds"] / 3600.0)
        for item in timeline
    )
    average_aqi = 0.0 if total_time <= 0 else round((concentration_exposure * 3600.0) / total_time, 2)
    peak_aqi = max((item["aqi"] for item in timeline), default=0.0)
    navigation = build_navigation_guidance(route, timeline, hour)
    quality = route_quality_summary(navigation)

    return {
        "route": route,
        "coordinates": route_to_coordinates(route),
        "metric": round(float(metric), 2),
        "travel_time_seconds": round(total_time, 2),
        "travel_time_minutes": round(total_time / 60.0, 2),
        "distance_meters": round(total_distance, 2),
        "distance_km": round(total_distance / 1000.0, 2),
        "exposure_aqi_hours": round(float(exposure), 4),
        "dose_index": round(float(exposure), 4),
        "concentration_aqi_hours": round(float(concentration_exposure), 4),
        "average_aqi": average_aqi,
        "peak_aqi": round(float(peak_aqi), 2),
        "timeline": timeline,
        "navigation": navigation,
        "quality": quality,
    }


def route_tradeoff_summary(routes: dict) -> list[dict]:
    fastest = routes.get("fastest", {})
    fastest_time = max(float(fastest.get("travel_time_minutes", 0.0)), 0.01)
    fastest_dose = max(float(fastest.get("dose_index", 0.0)), 0.01)
    tradeoffs = []
    for name in ("fastest", "balanced", "cleanest", "least_harm"):
        route = routes.get(name)
        if not route:
            continue
        time_min = float(route["travel_time_minutes"])
        dose = float(route["dose_index"])
        tradeoffs.append(
            {
                "route": name,
                "travel_time_minutes": round(time_min, 2),
                "dose_index": round(dose, 4),
                "average_aqi": route["average_aqi"],
                "stress_score": route.get("quality", {}).get("stress_score", 0.0),
                "time_delta_percent_vs_fastest": round(((time_min - fastest_time) / fastest_time) * 100.0, 2),
                "dose_reduction_percent_vs_fastest": round(((fastest_dose - dose) / fastest_dose) * 100.0, 2),
            }
        )
    return tradeoffs


def route_uncertainty_summary(route: dict) -> dict:
    timeline = route.get("timeline", []) or []
    if not timeline:
        return {
            "mean_segment_uncertainty": 0.0,
            "peak_segment_uncertainty": 0.0,
            "confidence": "Unknown",
            "confidence_reason": "No segment-level uncertainty was available.",
        }

    uncertainties = [float(item.get("uncertainty", 0.0)) for item in timeline]
    mean_uncertainty = round(float(np.mean(uncertainties)), 2)
    peak_uncertainty = round(float(np.max(uncertainties)), 2)
    if mean_uncertainty <= 8:
        confidence = "High"
        reason = "Most route segments stay close to well-supported AQI estimates."
    elif mean_uncertainty <= 16:
        confidence = "Moderate"
        reason = "The route crosses a mix of well-supported and less-certain street segments."
    else:
        confidence = "Low"
        reason = "A meaningful share of the route passes through less-certain street estimates."
    return {
        "mean_segment_uncertainty": mean_uncertainty,
        "peak_segment_uncertainty": peak_uncertainty,
        "confidence": confidence,
        "confidence_reason": reason,
    }


def signal_exposure_forecast(route: dict) -> dict:
    timeline = route.get("timeline", []) or []
    signal_points = route.get("navigation", {}).get("signals", []) or []
    if not signal_points or not timeline:
        return {
            "count": 0,
            "total_expected_wait_seconds": 0,
            "expected_dose_at_signals": 0.0,
            "summary": "No mapped traffic-signal exposure hotspot on this route.",
            "top_signals": [],
        }

    top_signals = []
    total_wait = 0.0
    total_dose = 0.0
    for signal in signal_points[:6]:
        distance_m = float(signal.get("distance_m", 0.0))
        nearest_segment = min(
            timeline,
            key=lambda item: abs(float(item.get("segment_distance_m", 0.0)) - distance_m),
        )
        local_aqi = float(nearest_segment.get("aqi", 0.0))
        wait_seconds = max(20.0, min(95.0, 24.0 + (local_aqi * 0.22)))
        signal_dose = local_aqi * (wait_seconds / 3600.0)
        total_wait += wait_seconds
        total_dose += signal_dose
        top_signals.append(
            {
                "distance_m": round(distance_m, 0),
                "lat": signal.get("lat"),
                "lon": signal.get("lon"),
                "expected_wait_seconds": round(wait_seconds, 1),
                "aqi": round(local_aqi, 2),
                "extra_dose_index": round(signal_dose, 4),
            }
        )

    return {
        "count": len(signal_points),
        "total_expected_wait_seconds": round(total_wait, 1),
        "expected_dose_at_signals": round(total_dose, 4),
        "summary": f"Mapped signals may add about {round(total_wait / 60.0, 1)} min of waiting exposure on this route.",
        "top_signals": top_signals[:3],
    }


def personalized_dose_summary(route: dict, profile: str, transport_mode: str) -> dict:
    from src.routing.health_matrix import get_met, get_sensitivity

    dose = float(route.get("dose_index", 0.0))
    distance_km = float(route.get("distance_km", 0.0))
    met = float(get_met(transport_mode))
    sensitivity = float(get_sensitivity(profile))
    intensity = round(dose / max(distance_km, 0.25), 3)
    if intensity <= 0.12:
        band = "Low"
    elif intensity <= 0.24:
        band = "Moderate"
    else:
        band = "High"
    return {
        "profile": profile,
        "transport_mode": transport_mode,
        "met": round(met, 2),
        "sensitivity_multiplier": round(sensitivity, 2),
        "dose_per_km": intensity,
        "strain_band": band,
        "summary": (
            f"For a {profile} traveler using {transport_mode}, this trip lands in the "
            f"{band.lower()} inhaled-dose band at about {intensity} dose units per km."
        ),
    }


def counterfactual_route_interventions(route: dict, alternatives: dict, optimal_departure: dict | None) -> list[str]:
    suggestions: list[str] = []
    signals = route.get("signal_forecast", {}).get("top_signals", []) or []
    uncertainty = route.get("uncertainty", {}) or {}
    if signals:
        first_signal = signals[0]
        suggestions.append(
            f"Avoid the first signal-heavy segment around {int(first_signal['distance_m'])} m from the start to trim waiting exposure."
        )
    hotspots = route.get("navigation", {}).get("aqi_alerts", []) or []
    if hotspots:
        first_hotspot = hotspots[0]
        suggestions.append(
            f"Detour around the hotspot near {first_hotspot['lat']}, {first_hotspot['lon']} where route AQI rises to about {first_hotspot['aqi']}."
        )
    cleanest = alternatives.get("cleanest")
    fastest = alternatives.get("fastest")
    if cleanest and fastest:
        time_gap = float(cleanest.get("travel_time_minutes", 0.0)) - float(fastest.get("travel_time_minutes", 0.0))
        dose_gain = float(fastest.get("dose_index", 0.0)) - float(cleanest.get("dose_index", 0.0))
        if time_gap > 0.4 and dose_gain > 0.03:
            suggestions.append(
                f"Trading about {round(time_gap, 1)} extra minutes for the cleanest corridor cuts dose by roughly {round(dose_gain, 2)}."
            )
    if optimal_departure and optimal_departure.get("best_hour_offset", 0) > 0:
        best_avg = optimal_departure.get("best_avg_aqi", optimal_departure.get("best_avg_route_aqi", "N/A"))
        suggestions.append(
            f"Leaving about {int(optimal_departure['best_hour_offset'])} hour(s) later should lower average route AQI to about {best_avg}."
        )
    if float(uncertainty.get("mean_segment_uncertainty", 0.0)) > 14:
        suggestions.append("Favor the balanced route when you want a cleaner corridor without leaning too hard on uncertain street estimates.")
    return suggestions[:4]


def classify_aqi(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 200:
        return "Unhealthy for sensitive groups"
    if aqi <= 300:
        return "Poor"
    return "Very poor"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def piecewise_linear(x: float, ranges: list[tuple[float, float, float, float]]) -> float:
    for c_low, c_high, i_low, i_high in ranges:
        if c_low <= x <= c_high:
            return ((i_high - i_low) / (c_high - c_low)) * (x - c_low) + i_low
    if x > ranges[-1][1]:
        return ranges[-1][3]
    return ranges[0][2]


def invert_piecewise_linear(i: float, ranges: list[tuple[float, float, float, float]]) -> float:
    for c_low, c_high, i_low, i_high in ranges:
        if i_low <= i <= i_high:
            return ((c_high - c_low) / (i_high - i_low)) * (i - i_low) + c_low
    if i > ranges[-1][3]:
        return ranges[-1][1]
    return ranges[0][0]


def india_aqi_to_us_epa_aqi(india_aqi: float) -> float:
    # Convert through PM2.5 concentration so model/station values can share one
    # consistent output scale for user-facing comparisons.
    india_pm25 = [
        (0.0, 30.0, 0.0, 50.0),
        (31.0, 60.0, 51.0, 100.0),
        (61.0, 90.0, 101.0, 200.0),
        (91.0, 120.0, 201.0, 300.0),
        (121.0, 250.0, 301.0, 400.0),
        (251.0, 500.0, 401.0, 500.0),
    ]
    us_pm25 = [
        (0.0, 12.0, 0.0, 50.0),
        (12.1, 35.4, 51.0, 100.0),
        (35.5, 55.4, 101.0, 150.0),
        (55.5, 150.4, 151.0, 200.0),
        (150.5, 250.4, 201.0, 300.0),
        (250.5, 350.4, 301.0, 400.0),
        (350.5, 500.4, 401.0, 500.0),
    ]
    pm25_est = invert_piecewise_linear(clamp(float(india_aqi), 0.0, 500.0), india_pm25)
    return clamp(piecewise_linear(pm25_est, us_pm25), 0.0, 500.0)


def us_epa_to_india_aqi(us_aqi: float) -> float:
    india_pm25 = [
        (0.0, 30.0, 0.0, 50.0),
        (31.0, 60.0, 51.0, 100.0),
        (61.0, 90.0, 101.0, 200.0),
        (91.0, 120.0, 201.0, 300.0),
        (121.0, 250.0, 301.0, 400.0),
        (251.0, 500.0, 401.0, 500.0),
    ]
    us_pm25 = [
        (0.0, 12.0, 0.0, 50.0),
        (12.1, 35.4, 51.0, 100.0),
        (35.5, 55.4, 101.0, 150.0),
        (55.5, 150.4, 151.0, 200.0),
        (150.5, 250.4, 201.0, 300.0),
        (250.5, 350.4, 301.0, 400.0),
        (350.5, 500.4, 401.0, 500.0),
    ]
    pm25_est = invert_piecewise_linear(clamp(float(us_aqi), 0.0, 500.0), us_pm25)
    return clamp(piecewise_linear(pm25_est, india_pm25), 0.0, 500.0)


def us_aqi_category(aqi: float) -> str:
    value = float(aqi)
    if value <= 50:
        return "Good"
    if value <= 100:
        return "Moderate"
    if value <= 150:
        return "Unhealthy for Sensitive Groups"
    if value <= 200:
        return "Unhealthy"
    if value <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def bias_cell_key(lat: float, lon: float) -> str:
    return f"{round(lat, 3):.3f},{round(lon, 3):.3f}"


def get_location_bias(lat: float, lon: float) -> float:
    cell = LOCATION_BIAS_CACHE.get(bias_cell_key(lat, lon))
    if not cell:
        return 0.0
    return float(cell.get("ema_bias", 0.0))


def update_location_bias(lat: float, lon: float, observed_now: float, predicted_now: float) -> dict:
    key = bias_cell_key(lat, lon)
    current = LOCATION_BIAS_CACHE.get(key, {"ema_bias": 0.0, "samples": 0})
    previous_ema = float(current.get("ema_bias", 0.0))
    # Target is residual on canonical output scale.
    observed_residual = float(observed_now) - float(predicted_now)
    alpha = 0.35
    ema = (1.0 - alpha) * previous_ema + alpha * observed_residual
    samples = int(current.get("samples", 0)) + 1
    LOCATION_BIAS_CACHE[key] = {
        "ema_bias": round(ema, 3),
        "samples": samples,
        "last_observed_residual": round(observed_residual, 3),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    return LOCATION_BIAS_CACHE[key]


def ensure_bias_store() -> BiasStore:
    global BIAS_STORE
    if BIAS_STORE is None:
        BIAS_STORE = BiasStore(BIAS_DB_PATH)
    return BIAS_STORE


def color_for_aqi(aqi: float) -> str:
    if aqi <= 50:
        return "#00e400"
    if aqi <= 100:
        return "#ffff00"
    if aqi <= 200:
        return "#ff7e00"
    if aqi <= 300:
        return "#ff0000"
    return "#7e0023"


def build_street_explanation(current_aqi: float, uncertainty: float, road_context: dict, nearest_station_km: float | None):
    reasons = []
    tips = []

    major_share = float(road_context.get("major_road_share") or 0.0)
    nearest_major = road_context.get("nearest_major_road_m")
    density = float(road_context.get("road_density_km_per_sqkm") or 0.0)

    if nearest_major is not None and nearest_major < 120:
        reasons.append("This point sits very close to a major road, so traffic emissions are likely elevating the street reading.")
        tips.append("Prefer a parallel inner street for walking or cycling if one exists.")
    elif major_share > 0.35:
        reasons.append("A large share of nearby road length is made up of major corridors, which usually raises curbside pollution.")

    if density > 18:
        reasons.append("The surrounding road network is dense, which often traps more vehicle activity and stop-go exposure.")
        tips.append("Shorter waits at junctions help here, especially during peak traffic hours.")

    if current_aqi <= 50:
        reasons.append("The broader forecast field is relatively clean at this hour, so local road effects are staying modest.")
        tips.append("This is a good time for outdoor travel if you have flexibility.")
    elif current_aqi <= 100:
        reasons.append("Background AQI is moderate, so location within the street network matters more than citywide averages suggest.")
        tips.append("Choose cleaner side streets over arterial roads when possible.")
    else:
        reasons.append("The background AQI is already elevated, so nearby traffic and roadway geometry amplify the local burden.")
        tips.append("Reduce strenuous outdoor activity and keep trips shorter if possible.")

    if nearest_station_km is not None and nearest_station_km > 2.0:
        reasons.append("This estimate is farther from a reference station, so uncertainty is intentionally wider here.")
        tips.append("Treat this location as a directional risk signal rather than a precise curbside measurement.")

    if uncertainty >= 20:
        tips.append("Use the lower-exposure route and consider delaying departure until the next cleaner hour.")

    if not tips:
        tips.append("Stick to streets with fewer major intersections when you can.")

    return {
        "summary": f"Street AQI is {round(current_aqi, 1)} ({classify_aqi(current_aqi)}).",
        "reasons": reasons[:3],
        "tips": tips[:3],
    }


def street_trust_score(current_aqi: float, uncertainty: float, road_context: dict, nearest_station_km: float | None) -> dict:
    score = 100.0
    evidence = []

    if nearest_station_km is not None:
        score -= min(35.0, nearest_station_km * 8.0)
        if nearest_station_km <= 0.25:
            evidence.append("Very close to a reference station.")
        elif nearest_station_km <= 1.0:
            evidence.append("Reasonably close to a reference station.")
        else:
            evidence.append("Farther from the nearest reference station.")

    score -= min(35.0, uncertainty * 1.2)
    if uncertainty <= 8:
        evidence.append("Low modeled uncertainty.")
    elif uncertainty <= 16:
        evidence.append("Moderate modeled uncertainty.")
    else:
        evidence.append("Wide modeled uncertainty band.")

    major_share = float(road_context.get("major_road_share") or 0.0)
    if major_share > 0.45:
        score -= 8.0
        evidence.append("Major-road exposure can shift sharply over short distances here.")

    trust_score = round(max(5.0, min(99.0, score)), 1)
    if trust_score >= 75:
        band = "High"
    elif trust_score >= 50:
        band = "Moderate"
    else:
        band = "Low"
    return {
        "score": trust_score,
        "band": band,
        "summary": f"{band} trust street estimate.",
        "evidence": evidence[:3],
    }


def build_route_explanation(fastest: dict, balanced: dict, cleanest: dict, profile: str, transport_mode: str):
    fastest_dose = float(fastest.get("dose_index", 0.0))
    cleanest_dose = float(cleanest.get("dose_index", 0.0))
    time_gap = float(cleanest.get("travel_time_minutes", 0.0)) - float(fastest.get("travel_time_minutes", 0.0))
    dose_gain = fastest_dose - cleanest_dose

    hotspot_segments = sorted(cleanest.get("timeline", []), key=lambda item: item.get("aqi", 0.0), reverse=True)[:3]
    hotspot_text = []
    for seg in hotspot_segments:
        hotspot_text.append(
            f"A hotspot appears near {seg['lat']}, {seg['lon']} where the route touches AQI {seg['aqi']}."
        )

    if abs(time_gap) < 0.2 and abs(dose_gain) < 0.05:
        summary = (
            "The available road network offers very similar time and exposure trade-offs for this trip right now, "
            "so all three route labels collapse toward the same corridor."
        )
    else:
        summary = (
            f"For a {profile} traveler using {transport_mode}, the cleanest route cuts dose by about "
            f"{round(max(dose_gain, 0.0), 2)} while changing travel time by {round(time_gap, 2)} minutes."
        )

    tips = []
    if cleanest.get("average_aqi", 0.0) > 80:
        tips.append("If you can delay the trip, the departure-time panel is worth paying attention to.")
    if transport_mode in {"walking", "cycling"}:
        tips.append("Use the cleanest route for active travel, because time spent breathing curbside air matters more there.")
    if balanced.get("dose_index", 0.0) < fastest.get("dose_index", 0.0):
        tips.append("Balanced is a good compromise when you want some dose reduction without fully optimizing for air quality.")

    return {
        "summary": summary,
        "hotspots": hotspot_text,
        "tips": tips[:3],
    }


def route_signature(route_summary: dict) -> tuple[int, ...]:
    return tuple(route_summary.get("route", []))


def select_route_variants(route_summaries: list[dict]) -> tuple[dict, dict, dict]:
    unique = []
    seen = set()
    for item in route_summaries:
        signature = route_signature(item)
        if signature and signature not in seen:
            unique.append(item)
            seen.add(signature)

    if not unique:
        raise HTTPException(status_code=404, detail="No route candidates were available.")

    fastest = min(unique, key=lambda item: (item["travel_time_minutes"], item["dose_index"]))
    distinct_pool = [item for item in unique if route_signature(item) != route_signature(fastest)]

    # "Cleanest" is an air-quality route, not merely the route with the lowest
    # total dose. A short polluted corridor can have lower total dose while still
    # being worse air; rank concentration and peak AQI before time/dose.
    cleanest_pool = distinct_pool or unique
    reasonable_cleanest = [
        item for item in cleanest_pool
        if item["travel_time_minutes"] <= fastest["travel_time_minutes"] * 1.8 + 5.0
    ] or cleanest_pool
    cleanest = min(
        reasonable_cleanest,
        key=lambda item: (
            item["average_aqi"],
            item["peak_aqi"],
            item["dose_index"],
            item["travel_time_minutes"],
        ),
    )

    def balanced_score(item):
        time_penalty = abs(item["travel_time_minutes"] - fastest["travel_time_minutes"])
        air_penalty = abs(item["average_aqi"] - cleanest["average_aqi"])
        return time_penalty + 0.35 * air_penalty

    balanced_pool = [item for item in unique if route_signature(item) not in {route_signature(fastest), route_signature(cleanest)}]
    balanced = min(balanced_pool, key=balanced_score) if balanced_pool else fastest
    return fastest, balanced, cleanest


def select_least_harm_route(route_summaries: list[dict]) -> dict:
    if not route_summaries:
        raise HTTPException(status_code=404, detail="No route candidates were available.")

    fastest_time = max(min(item["travel_time_minutes"] for item in route_summaries), 0.01)
    lowest_aqi = min(item["average_aqi"] for item in route_summaries)
    lowest_dose = max(min(item["dose_index"] for item in route_summaries), 0.01)
    lowest_stress = min(item.get("quality", {}).get("stress_score", 0.0) for item in route_summaries)

    def score(item: dict) -> float:
        time_ratio = item["travel_time_minutes"] / fastest_time
        aqi_delta = max(0.0, item["average_aqi"] - lowest_aqi) / 10.0
        dose_ratio = item["dose_index"] / lowest_dose
        stress_delta = max(0.0, item.get("quality", {}).get("stress_score", 0.0) - lowest_stress) / 10.0
        peak_delta = max(0.0, item["peak_aqi"] - lowest_aqi) / 40.0
        return (0.16 * time_ratio) + (0.36 * aqi_delta) + (0.18 * dose_ratio) + (0.24 * stress_delta) + (0.06 * peak_delta)

    selected = min(route_summaries, key=score)
    fastest = min(route_summaries, key=lambda item: item["travel_time_minutes"])
    if route_signature(selected) == route_signature(fastest):
        cleaner_or_calmer = [
            item for item in route_summaries
            if route_signature(item) != route_signature(fastest)
            and item["travel_time_minutes"] <= fastest["travel_time_minutes"] * 1.7 + 5.0
            and (
                item["average_aqi"] <= fastest["average_aqi"] - 0.3
                or item.get("quality", {}).get("stress_score", 0.0)
                <= fastest.get("quality", {}).get("stress_score", 0.0) - 2.0
            )
        ]
        if cleaner_or_calmer:
            selected = min(cleaner_or_calmer, key=score)

    selected = dict(selected)
    selected["least_harm_score"] = round(score(selected), 4)
    return selected


@app.get("/")
def serve_dashboard():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"msg": "Frontend not found. Place index.html in frontend/public/"}


@app.get("/api/health")
def health():
    ensure_graph_started()
    station_coords = (AQI_DATA or {}).get("station_coords")
    return {
        "status": "ok",
        "graph_loaded": G is not None,
        "graph_loading": GRAPH_LOADING,
        "forecast_loaded": FORECAST_PATH.exists(),
        "forecast_grid_loaded": AQI_DATA is not None,
        "forecast_grid_path": str(FORECAST_GRID_PATH),
        "station_support_count": len(station_coords) if station_coords is not None else 0,
        "live_fetch": LIVE_FETCH_STATUS,
        "live_snapshot": (AQI_DATA or {}).get("live_snapshot") if AQI_DATA else None,
    }


@app.get("/api/research/novelty")
def research_novelty():
    return {
        "implemented_contributions": [
            "Hyperlocal street-level AQI forecasting from gridded downscaling.",
            "Present-location street AQI prediction using browser geolocation.",
            "Health-profile-aware and transport-mode-aware exposure dose routing.",
            "Fastest, balanced, and cleanest Pareto-style route alternatives.",
            "Optimal same-day departure-time recommendation for lower exposure.",
            "Multi-stop pollution-aware errand route optimization.",
            "Uncertainty bounds for street-level AQI forecasts.",
            "Explainable spatial attention overlay for graph-model interpretation.",
            "Longitudinal exposure-savings wallet for behavior feedback.",
        ],
        "paper_claim": (
            "An explainable, health-aware, hyperlocal AQI forecasting and routing "
            "system that converts sparse urban air-quality forecasts into "
            "street-level exposure-minimizing mobility decisions."
        ),
        "recommended_validation": [
            "Leave-one-station-out spatial validation.",
            "Temporal holdout validation by day/week.",
            "MAE, RMSE, R2, PICP, and CRPS for deterministic and uncertainty forecasts.",
            "Ablation of weather, temporal lag, spatial lag, traffic, and graph features.",
        ],
    }


@app.get("/api/forecast/today")
def get_today_forecast():
    data = load_station_forecast()
    today = []
    for sid, meta in STATION_META.items():
        if sid not in data:
            continue
        station = data[sid]
        today.append(
            {
                "station_id": sid,
                "station_name": station["station_name"],
                "lat": meta["lat"],
                "lon": meta["lon"],
                "hourly": station["hourly"][:24],
            }
        )
    return today


@app.get("/api/forecast/now")
def get_forecast_now():
    data = load_station_forecast()
    result = []
    for sid, meta in STATION_META.items():
        if sid in data:
            current = data[sid]["hourly"][0]
            result.append(
                {
                    "station_id": sid,
                    "station_name": data[sid]["station_name"],
                    "lat": meta["lat"],
                    "lon": meta["lon"],
                    "aqi": current["aqi"],
                    "category": current["category"],
                    "color": current["color"],
                    "timestamp": current["timestamp"],
                }
            )
    return result


@app.get("/api/live/stations")
def get_live_station_status():
    snapshot = load_live_station_snapshot(max_age_minutes=10_000)
    return {
        "snapshot_path": str(LIVE_STATION_PATH),
        "auto_fetch_interval_seconds": LIVE_FETCH_INTERVAL_SECONDS,
        "fetcher_started": LIVE_FETCHER_STARTED,
        "fetch_status": LIVE_FETCH_STATUS,
        "available": snapshot.get("available", False),
        "source": snapshot.get("source"),
        "generated_at": snapshot.get("generated_at"),
        "age_minutes": snapshot.get("age_minutes"),
        "station_count": len(snapshot.get("stations", {})),
        "max_age_minutes_for_forecast": LIVE_STATION_MAX_AGE_MINUTES,
        "reason": snapshot.get("reason"),
        "stations": [
            {
                "station_id": sid,
                "station_name": STATION_META[sid]["name"],
                "lat": STATION_META[sid]["lat"],
                "lon": STATION_META[sid]["lon"],
                "aqi": round(float(item["aqi"]), 2),
                "source": item.get("source"),
                "observed_at": item.get("observed_at"),
            }
            for sid, item in snapshot.get("stations", {}).items()
        ],
    }


@app.post("/api/observations")
def post_observation(req: ObservationRequest):
    """Ingest an external truth observation (e.g., AccuWeather or a sensor).

    This is used for per-location and time-of-day bias correction and source fusion.
    """
    store = ensure_bias_store()
    created = store.insert_observation(
        source=req.source,
        lat=req.lat,
        lon=req.lon,
        aqi=req.aqi,
        aqi_scale=req.aqi_scale,
        confidence=req.confidence,
        created_at=req.observed_at,
    )
    cell = bias_cell(req.lat, req.lon).key
    # Return last few obs for quick debugging.
    recent = store.latest_observations(cell, limit=5)
    return {"status": "ok", "stored": created, "recent": recent}


@app.post("/api/live/refresh")
def refresh_live_station_data():
    snapshot = fetch_cpcb_live_snapshot_once()
    if snapshot.get("error"):
        raise HTTPException(status_code=502, detail=snapshot["error"])
    return {
        "status": "ok",
        "station_count": snapshot.get("station_count", 0),
        "generated_at": snapshot.get("generated_at"),
        "source": snapshot.get("source"),
        "fetch_status": LIVE_FETCH_STATUS,
    }


@app.get("/api/forecast/hour/{hour}")
def get_forecast_at_hour(hour: int):
    if not (1 <= hour <= 168):
        raise HTTPException(status_code=400, detail="Hour must be 1-168.")

    data = load_station_forecast()
    result = []
    for sid, meta in STATION_META.items():
        if sid in data:
            point = data[sid]["hourly"][hour - 1]
            result.append(
                {
                    "station_id": sid,
                    "station_name": data[sid]["station_name"],
                    "lat": meta["lat"],
                    "lon": meta["lon"],
                    "aqi": point["aqi"],
                    "category": point["category"],
                    "color": point["color"],
                    "timestamp": point["timestamp"],
                }
            )
    return result


@app.get("/api/forecast/surface/{hour}")
def get_forecast_surface(hour: int, stride: int = 3):
    if not (1 <= hour <= 168):
        raise HTTPException(status_code=400, detail="Hour must be 1-168.")
    return corrected_surface_for_hour(hour, stride=stride)


@app.get("/api/forecast/station/{station_id}")
def get_station_forecast(station_id: str):
    if station_id in EXCLUDED_STATION_IDS:
        raise HTTPException(status_code=404, detail=f"Station {station_id} not found.")

    data = load_station_forecast()
    if station_id not in data:
        raise HTTPException(status_code=404, detail=f"Station {station_id} not found.")

    station_data = dict(data[station_id])
    if station_id in STATION_META:
        station_data["lat"] = STATION_META[station_id]["lat"]
        station_data["lon"] = STATION_META[station_id]["lon"]
    return station_data


@app.get("/api/forecast/street")
def get_street_forecast(lat: float, lon: float):
    if AQI_DATA is None:
        raise HTTPException(status_code=503, detail="Forecast grid not loaded yet.")

    forecast_grid = AQI_DATA.get("forecast_grid")
    lat_axis = AQI_DATA.get("lat_axis")
    lon_axis = AQI_DATA.get("lon_axis")

    if forecast_grid is None or lat_axis is None or lon_axis is None:
        raise HTTPException(status_code=503, detail="Forecast grid data missing.")

    lat_idx = int(np.abs(lat_axis - lat).argmin())
    lon_idx = int(np.abs(lon_axis - lon).argmin())

    station_data = load_station_forecast()
    hourly_aqi = forecast_grid[:, lat_idx, lon_idx]
    road_context = road_context_for_coordinates(G, lat, lon)
    local_adjustment = context_aqi_adjustment(road_context)
    corrected_hourly = []
    for i, base_value in enumerate(hourly_aqi):
        anchored = float(np.nan_to_num(base_value, nan=75.0))
        anchored += interpolate_station_residual(lat, lon, i, station_data)
        corrected_hourly.append(anchored)
    hourly_aqi_clean = np.clip(np.asarray(corrected_hourly) + local_adjustment, 0.0, 500.0).tolist()

    nowcast = live_station_nowcast(lat, lon, max_distance_km=10.0)
    nearest_live = nearest_live_station_observation(lat, lon, station_data)
    if nowcast and nowcast["nearest_distance_km"] <= 5.0:
        # Anchor near-term street forecast toward blended live-station nowcast.
        # Close points trust live data heavily; farther points taper smoothly.
        base_anchor = max(0.0, min(0.95, 0.92 * (1.0 - (nowcast["nearest_distance_km"] / 5.0))))
        for i in range(min(6, len(hourly_aqi_clean))):
            decay = math.exp(-i / 2.0)
            weight = base_anchor * decay
            hourly_aqi_clean[i] = (
                (1.0 - weight) * float(hourly_aqi_clean[i]) + weight * float(nowcast["aqi"])
            )

    # Canonical output scale is IN_CPCB.
    hourly_out = [float(value) for value in hourly_aqi_clean]
    nowcast_out = float(nowcast["aqi"]) if nowcast else None

    # Multi-source fusion: live stations + recent external observations
    store = ensure_bias_store()
    cell = bias_cell(lat, lon).key
    hour_of_day = datetime.now().hour
    external_obs = store.latest_observations(cell, limit=6)

    # Convert external observations to canonical scale (IN_CPCB).
    external_now_values = []
    for obs in external_obs[:3]:
        obs_aqi = float(obs.get("aqi", 0.0))
        scale = obs.get("aqi_scale")
        conf = float(obs.get("confidence", 0.7))
        if scale == "US_EPA":
            obs_aqi = us_epa_to_india_aqi(obs_aqi)
        external_now_values.append((obs_aqi, conf, obs.get("source")))

    fused_observed_now = None
    fused_confidence = 0.0
    fusion_sources = []

    station_weight = 0.0
    station_value = None
    if nowcast_out is not None:
        station_value = float(nowcast_out)
        # Station nowcast is useful but can be far from a user's micro-location.
        station_weight = 0.55

    ext_weight_sum = 0.0
    ext_value = None
    ext_has_high_conf_override = False
    if external_now_values:
        # Weighted blend of external observations (often closer to user microclimate).
        v_sum = 0.0
        for val, conf, source in external_now_values:
            w = clamp(conf, 0.2, 0.99)
            ext_weight_sum += w
            v_sum += w * float(val)
            fusion_sources.append({"source": source, "aqi": round(float(val), 2), "weight": round(w, 2)})
            if source in {"accuweather", "sensor"} and w >= 0.9:
                ext_has_high_conf_override = True
        ext_value = v_sum / max(ext_weight_sum, 1e-9)

    # If we have a high-confidence external "truth" (AccuWeather/sensor), let it dominate.
    if ext_has_high_conf_override:
        station_weight = min(station_weight, 0.15)

    if station_value is not None:
        fusion_sources.insert(
            0,
            {"source": "cpcb_data_gov_in", "aqi": round(float(nowcast_out), 2), "weight": round(station_weight, 2)},
        )

    if station_value is not None or ext_value is not None:
        total = station_weight + ext_weight_sum
        fused_observed_now = (
            ((station_weight * station_value) if station_value is not None else 0.0)
            + ((ext_weight_sum * ext_value) if ext_value is not None else 0.0)
        ) / max(total, 1e-9)
        fused_confidence = clamp((0.55 * (1.0 if station_value is not None else 0.0)) + (0.35 * min(1.0, ext_weight_sum / 1.2)), 0.25, 0.95)

    # Update hour-of-day bias model from fused truth.
    updated_bias = None
    if fused_observed_now is not None:
        updated_bias = store.update_bias_ema(
            cell_key=cell,
            hour_of_day=hour_of_day,
            observed_now=float(fused_observed_now),
            predicted_now=float(hourly_out[0]),
            confidence=float(fused_confidence or 0.7),
        )

    # Read bias for this cell + hour (fallback to 0).
    db_bias_row = store.get_bias(cell, hour_of_day) or {}
    ema_bias = float(db_bias_row.get("ema_bias", 0.0))
    for i in range(len(hourly_out)):
        # Apply strongest correction for "now", then decay for forward hours.
        decay = math.exp(-i / 2.8)
        hourly_out[i] = clamp(float(hourly_out[i]) + (ema_bias * decay), 0.0, 500.0)

    # High-confidence external truth should immediately reflect in the "now" readout.
    if ext_has_high_conf_override and ext_value is not None:
        hourly_out[0] = clamp(float(ext_value), 0.0, 500.0)
        if len(hourly_out) > 1:
            hourly_out[1] = clamp(0.7 * float(hourly_out[1]) + 0.3 * float(ext_value), 0.0, 500.0)

    uncertainty_points = [
        sample_aqi_window_for_coordinates(lat, lon, AQI_DATA, hour=i + 1)
        for i in range(len(hourly_aqi_clean))
    ]
    
    # Also return the actual snapped coordinates
    actual_lat = float(lat_axis[lat_idx])
    actual_lon = float(lon_axis[lon_idx])

    current_uncertainty = float(uncertainty_points[0]["uncertainty"])
    nearest_station_km = uncertainty_points[0].get("nearest_station_km")
    explanation = build_street_explanation(
        float(hourly_out[0]),
        current_uncertainty,
        road_context,
        nearest_station_km,
    )
    trust = street_trust_score(
        float(hourly_out[0]),
        current_uncertainty,
        road_context,
        nearest_station_km,
    )

    return {
        "requested_lat": lat,
        "requested_lon": lon,
        "actual_lat": actual_lat,
        "actual_lon": actual_lon,
        "road_context": road_context,
        "local_adjustment_aqi": round(local_adjustment, 2),
        "explanation": explanation,
        "trust": trust,
        "aqi_scale": CANONICAL_AQI_SCALE,
        "hourly_aqi": [round(float(val), 2) for val in hourly_out],
        "hourly_lower": [round(float(max(0.0, hourly_out[i] - float(point["uncertainty"]))), 2) for i, point in enumerate(uncertainty_points)],
        "hourly_upper": [round(float(hourly_out[i] + float(point["uncertainty"])), 2) for i, point in enumerate(uncertainty_points)],
        "hourly_uncertainty": [round(float(point["uncertainty"]), 2) for point in uncertainty_points],
        "current": {
            "aqi": round(float(hourly_out[0]), 2),
            "category": classify_aqi(float(hourly_out[0])),
            "lower": round(float(max(0.0, hourly_out[0] - current_uncertainty)), 2),
            "upper": round(float(hourly_out[0] + current_uncertainty), 2),
            "uncertainty": round(current_uncertainty, 2),
            "nearest_station_km": (
                round(float(nearest_station_km), 3)
                if nearest_station_km is not None
                else None
            ),
        },
        "live_anchor": (
            {
                "station_id": nearest_live["station_id"],
                "station_name": nearest_live["station_name"],
                "aqi": round(float(nearest_live["aqi"]), 2),
                "distance_km": round(float(nearest_live["distance_km"]), 3),
            }
            if nearest_live
            else None
        ),
        "live_nowcast": (
            {
                **nowcast,
                "aqi": round(float(nowcast_out), 2),
            }
            if nowcast and nowcast_out is not None
            else nowcast
        ),
        "bias_correction": {
            "cell": cell,
            "hour_of_day": hour_of_day,
            "ema_bias_applied": round(float(ema_bias), 2),
            "samples": int((db_bias_row or {}).get("samples", 0)),
            "last_residual": (db_bias_row or {}).get("last_residual"),
            "updated": updated_bias,
            "fusion": {
                "observed_now": round(float(fused_observed_now), 2) if fused_observed_now is not None else None,
                "sources": fusion_sources,
            },
        },
    }


@app.post("/api/route")
def get_route(req: RouteRequest):
    global G

    ensure_graph_started()
    refresh_aqi_data_live_anchor()
    if G is None:
        raise HTTPException(status_code=503, detail="Street graph not loaded yet.")
    if not (1 <= req.hour <= 168):
        raise HTTPException(status_code=400, detail="Hour must be 1-168.")

    try:
        orig_node = ox.distance.nearest_nodes(G, req.orig_lon, req.orig_lat)
        dest_node = ox.distance.nearest_nodes(G, req.dest_lon, req.dest_lat)
        pareto = generate_pareto_fronts(
            G,
            orig_node,
            dest_node,
            profile=req.profile,
            aqi_data=AQI_DATA,
            hour=req.hour,
            transport_mode=req.transport_mode,
        )

        fastest_route = pareto["fastest"]["route"]
        balanced_route = pareto["balanced"]["route"]
        cleanest_route = pareto["cleanest"]["route"]
        if not fastest_route or not balanced_route or not cleanest_route:
            raise HTTPException(status_code=404, detail="No route found for the selected points.")

        fastest_summary = summarize_route(
            fastest_route,
            pareto["fastest"]["metric"],
            req.hour,
            req.transport_mode,
            req.profile,
        )
        balanced_summary = summarize_route(
            balanced_route,
            pareto["balanced"]["metric"],
            req.hour,
            req.transport_mode,
            req.profile,
        )
        cleanest_summary = summarize_route(
            cleanest_route,
            pareto["cleanest"]["metric"],
            req.hour,
            req.transport_mode,
            req.profile,
        )

        fastest_summary, balanced_summary, cleanest_summary = select_route_variants(
            [fastest_summary, balanced_summary, cleanest_summary]
        )
        least_harm_summary = select_least_harm_route(
            [fastest_summary, balanced_summary, cleanest_summary]
        )

        result = {
            "status": "success",
            "data": {
                "requested_hour": req.hour,
                "profile": req.profile,
                "origin": {"lat": req.orig_lat, "lon": req.orig_lon, "node": orig_node},
                "destination": {"lat": req.dest_lat, "lon": req.dest_lon, "node": dest_node},
                "transport_mode": req.transport_mode,
                "fastest": fastest_summary,
                "balanced": balanced_summary,
                "cleanest": cleanest_summary,
                "least_harm": least_harm_summary,
            },
        }
        result["data"]["tradeoff_curve"] = route_tradeoff_summary(
            {
                "fastest": fastest_summary,
                "balanced": balanced_summary,
                "cleanest": cleanest_summary,
                "least_harm": least_harm_summary,
            }
        )

        # Keep route responses snappy by scanning only a small departure window.
        scan_hours = max(1, min(req.hours_to_scan, 3))

        # Calculate optimal departure by evaluating cleanest route over hours_to_scan
        best_hour_offset = 0
        best_aqi = float('inf')
        hourly_route_aqi = []

        for i in range(scan_hours):
            h = req.hour + i
            if h > 168:
                break
            exposure, timeline = calculate_route_exposure(
                G,
                cleanest_route,
                AQI_DATA,
                hour=h,
                transport_mode=req.transport_mode,
                profile=req.profile,
            )
            total_time = sum(item["segment_time_seconds"] for item in timeline)
            concentration = sum(item["aqi"] * (item["segment_time_seconds"] / 3600.0) for item in timeline)
            avg_aqi = 0.0 if total_time <= 0 else (concentration * 3600.0) / total_time
            
            hourly_route_aqi.append(round(avg_aqi, 2))
            if avg_aqi < best_aqi:
                best_aqi = avg_aqi
                best_hour_offset = i

        result["data"]["optimal_departure"] = {
            "best_hour_offset": best_hour_offset,
            "best_avg_aqi": round(best_aqi, 2),
            "hourly_route_aqi": hourly_route_aqi
        }

        route_bundle = {
            "fastest": fastest_summary,
            "balanced": balanced_summary,
            "cleanest": cleanest_summary,
            "least_harm": least_harm_summary,
        }
        for summary in route_bundle.values():
            summary["uncertainty"] = route_uncertainty_summary(summary)
            summary["signal_forecast"] = signal_exposure_forecast(summary)
            summary["personalized_dose"] = personalized_dose_summary(
                summary,
                req.profile,
                req.transport_mode,
            )
        for summary in route_bundle.values():
            summary["counterfactuals"] = counterfactual_route_interventions(
                summary,
                route_bundle,
                result["data"]["optimal_departure"],
            )

        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


import itertools

@app.post("/api/route/tsp")
def get_tsp_route(req: TSPRequest):
    global G
    ensure_graph_started()
    refresh_aqi_data_live_anchor()
    if G is None:
        raise HTTPException(status_code=503, detail="Street graph not loaded yet.")
    
    if len(req.waypoints) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 waypoints.")
    if len(req.waypoints) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 waypoints allowed.")

    from src.routing.astar import find_optimal_route
    from src.routing.cost_graph import build_exposure_weight

    nodes = [ox.distance.nearest_nodes(G, wp.lon, wp.lat) for wp in req.waypoints]
    
    # Precompute pairwise cleanest routes
    cleanest_weight = build_exposure_weight(G, aqi_data=AQI_DATA, profile=req.profile, hour=req.hour, transport_mode=req.transport_mode)
    
    pairwise_costs = {}
    pairwise_routes = {}
    
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                route, cost = find_optimal_route(G, nodes[i], nodes[j], weight=cleanest_weight)
                if route is None:
                    raise HTTPException(status_code=404, detail="Path disconnected between waypoints.")
                pairwise_costs[(i, j)] = cost
                pairwise_routes[(i, j)] = route

    # The start node is always index 0 (Origin), the rest are permutations
    best_cost = float('inf')
    best_seq = None
    
    # We must start at 0. If it's a closed loop, we end at 0. But for "Errand Runner", 
    # the user starts at Origin, goes to stops in any order, and finishes at the last stop.
    # Wait, the prompt says: "grocery store, pharmacy, gym... exact order to visit them to minimize total exposure".
    # Typically origin is fixed (0) and destination is fixed (last).
    # Let's assume nodes[0] is origin and nodes[-1] is final destination. 
    # Stops in between are permuted.
    middle_indices = list(range(1, len(nodes) - 1))
    for perm in itertools.permutations(middle_indices):
        seq = [0] + list(perm) + [len(nodes) - 1]
        cost = sum(pairwise_costs[(seq[k], seq[k+1])] for k in range(len(seq)-1))
        if cost < best_cost:
            best_cost = cost
            best_seq = seq

    if best_seq is None:
        best_seq = list(range(len(nodes))) # Fallback

    # Reconstruct unified route
    unified_route = []
    for k in range(len(best_seq)-1):
        segment = pairwise_routes[(best_seq[k], best_seq[k+1])]
        if k == 0:
            unified_route.extend(segment)
        else:
            unified_route.extend(segment[1:]) # Avoid duplicating the overlapping node
            
    summary = summarize_route(unified_route, best_cost, req.hour, req.transport_mode, req.profile)
    
    return {
        "status": "success",
        "data": {
            "requested_hour": req.hour,
            "profile": req.profile,
            "transport_mode": req.transport_mode,
            "cleanest": summary,
            "optimal_sequence": best_seq,
        }
    }

ATTENTION_PATH = Path("data/processed/attention_weights.json")

@app.get("/api/xai/attention")
def get_attention_weights():
    if not ATTENTION_PATH.exists():
        raise HTTPException(status_code=404, detail="Attention weights not found.")
    with open(ATTENTION_PATH, "r", encoding="utf-8") as f:
        weights = json.load(f)
    return [
        item
        for item in weights
        if item.get("from_node") not in EXCLUDED_STATION_IDS
        and item.get("to_node") not in EXCLUDED_STATION_IDS
    ]

@app.post("/api/xai/route_insight")
def get_route_insight(req: RouteRequest):
    try:
        weights = []
        if ATTENTION_PATH.exists():
            with open(ATTENTION_PATH, "r", encoding="utf-8") as f:
                weights = json.load(f)
            weights = [
                item
                for item in weights
                if item.get("from_node") not in EXCLUDED_STATION_IDS
                and item.get("to_node") not in EXCLUDED_STATION_IDS
            ]
            weights.sort(key=lambda x: x["weight"], reverse=True)

        ensure_graph_started()
        refresh_aqi_data_live_anchor()
        if G is None:
            return {"insight": "Street graph not loaded yet.", "highlights": []}

        orig_node = ox.distance.nearest_nodes(G, req.orig_lon, req.orig_lat)
        dest_node = ox.distance.nearest_nodes(G, req.dest_lon, req.dest_lat)
        pareto = generate_pareto_fronts(
            G,
            orig_node,
            dest_node,
            profile=req.profile,
            aqi_data=AQI_DATA,
            hour=req.hour,
            transport_mode=req.transport_mode,
        )
        fastest_summary = summarize_route(pareto["fastest"]["route"], pareto["fastest"]["metric"], req.hour, req.transport_mode, req.profile)
        balanced_summary = summarize_route(pareto["balanced"]["route"], pareto["balanced"]["metric"], req.hour, req.transport_mode, req.profile)
        cleanest_summary = summarize_route(pareto["cleanest"]["route"], pareto["cleanest"]["metric"], req.hour, req.transport_mode, req.profile)
        explanation = build_route_explanation(fastest_summary, balanced_summary, cleanest_summary, req.profile, req.transport_mode)

        highlights_with_coords = []
        for w in weights[:10]:
            from_node = w["from_node"]
            to_node = w["to_node"]
            if from_node in STATION_META and to_node in STATION_META:
                highlights_with_coords.append({
                    "from_lat": STATION_META[from_node]["lat"],
                    "from_lon": STATION_META[from_node]["lon"],
                    "to_lat": STATION_META[to_node]["lat"],
                    "to_lon": STATION_META[to_node]["lon"],
                    "weight": w["weight"]
                })

        insight_parts = [explanation["summary"]]
        if explanation["hotspots"]:
            insight_parts.append(" ".join(explanation["hotspots"][:2]))
        if explanation["tips"]:
            insight_parts.append("Advice: " + " ".join(explanation["tips"]))

        return {
            "insight": " ".join(insight_parts),
            "summary": explanation["summary"],
            "hotspots": explanation["hotspots"],
            "tips": explanation["tips"],
            "highlights": highlights_with_coords
        }
    except Exception as e:
        return {"insight": str(e), "highlights": []}
