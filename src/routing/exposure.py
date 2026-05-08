from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_AQI = 75.0
FORECAST_GRID_PATH = Path("data/processed/forecast_grid_7day_filtered.npy")
LEGACY_FORECAST_GRID_PATH = Path("data/processed/forecast_grid_7day.npy")
GRID_LAT_PATH = Path("data/processed/grid_lat.npy")
GRID_LON_PATH = Path("data/processed/grid_lon.npy")
FEATURES_PATH = Path("data/processed/features.parquet")
EXCLUDED_STATION_IDS = {"site_1558"}
KNOWN_STATION_COORDS = {
    "site_1553": (12.9279, 77.6271),
    "site_162": (12.9174, 77.6235),
    "site_165": (12.9166, 77.6101),
    "site_1554": (13.0450, 77.5966),
    "site_1555": (12.9250, 77.5938),
    "site_5729": (13.0289, 77.5199),
    "site_5681": (12.9634, 77.5559),
    "site_163": (12.9609, 77.5996),
    "site_5678": (12.9774, 77.5713),
    "site_166": (13.0068, 77.5090),
    "site_5686": (13.0450, 77.5116),
}


def normalize_hour_index(hour: int) -> int:
    return max(0, min(int(hour) - 1, 167))


def load_forecast_grid_bundle():
    if not (
        (FORECAST_GRID_PATH.exists() or LEGACY_FORECAST_GRID_PATH.exists())
        and GRID_LAT_PATH.exists()
        and GRID_LON_PATH.exists()
    ):
        return None

    grid_lat = np.load(GRID_LAT_PATH)
    grid_lon = np.load(GRID_LON_PATH)
    station_coords = load_station_coordinates()
    forecast_grid_path = FORECAST_GRID_PATH if FORECAST_GRID_PATH.exists() else LEGACY_FORECAST_GRID_PATH
    return {
        "forecast_grid": np.load(forecast_grid_path, mmap_mode="r"),
        "grid_lat": grid_lat,
        "grid_lon": grid_lon,
        "lat_axis": grid_lat[:, 0],
        "lon_axis": grid_lon[0, :],
        "station_coords": station_coords,
    }


def load_station_coordinates() -> np.ndarray | None:
    coords_map: dict[str, tuple[float, float]] = dict(KNOWN_STATION_COORDS)
    try:
        if FEATURES_PATH.exists():
            df = pd.read_parquet(FEATURES_PATH, columns=["station_id", "lat", "lon"])
            coords = df.dropna().groupby("station_id")[["lat", "lon"]].first()
            for station_id, row in coords.iterrows():
                if station_id in EXCLUDED_STATION_IDS:
                    continue
                coords_map.setdefault(station_id, (float(row["lat"]), float(row["lon"])))
    except Exception:
        pass
    if not coords_map:
        return None
    return np.asarray(list(coords_map.values()), dtype=float)


def nearest_station_distance_km(lat: float, lon: float, station_coords: np.ndarray | None) -> float | None:
    if station_coords is None or len(station_coords) == 0:
        return None
    ref_lat = float(np.mean(station_coords[:, 0]))
    dy = (station_coords[:, 0] - lat) * 111.32
    dx = (station_coords[:, 1] - lon) * 111.32 * np.cos(np.deg2rad(ref_lat))
    return float(np.sqrt(dx * dx + dy * dy).min())


def live_decay_weight(hour_index: int) -> float:
    if hour_index <= 0:
        return 1.0
    return float(np.exp(-hour_index / 24.0))


def station_anchor_residual(lat: float, lon: float, aqi_data=None, hour: int = 1) -> float:
    if not aqi_data:
        return 0.0

    station_forecast = aqi_data.get("station_anchor_forecast")
    forecast_grid = aqi_data.get("forecast_grid")
    lat_axis = aqi_data.get("lat_axis")
    lon_axis = aqi_data.get("lon_axis")
    if not station_forecast or forecast_grid is None or lat_axis is None or lon_axis is None:
        return 0.0

    h = normalize_hour_index(hour)
    weighted_residual = 0.0
    total_weight = 0.0
    for sid, coords in KNOWN_STATION_COORDS.items():
        station = station_forecast.get(sid)
        hourly = station.get("hourly") if station else None
        if not hourly or h >= len(hourly):
            continue

        station_lat, station_lon = coords
        station_aqi = float(hourly[h].get("aqi", DEFAULT_AQI))
        station_lat_idx = int(np.abs(lat_axis - station_lat).argmin())
        station_lon_idx = int(np.abs(lon_axis - station_lon).argmin())
        grid_aqi = float(forecast_grid[h, station_lat_idx, station_lon_idx])
        if not np.isfinite(grid_aqi):
            continue

        dy = (station_lat - lat) * 111.32
        dx = (station_lon - lon) * 111.32 * np.cos(np.deg2rad(lat))
        distance_km = float(np.sqrt((dx * dx) + (dy * dy)))
        residual = station_aqi - grid_aqi
        if distance_km < 0.05:
            return residual

        weight = 1.0 / max(distance_km * distance_km, 0.01)
        weighted_residual += weight * residual
        total_weight += weight

    if total_weight <= 0:
        return 0.0
    return weighted_residual / total_weight


def sample_aqi_for_coordinates(lat: float, lon: float, aqi_data=None, hour: int = 1) -> float:
    if not aqi_data:
        return DEFAULT_AQI

    forecast_grid = aqi_data.get("forecast_grid")
    lat_axis = aqi_data.get("lat_axis")
    lon_axis = aqi_data.get("lon_axis")
    station_coords = aqi_data.get("station_coords")
    if forecast_grid is None or lat_axis is None or lon_axis is None:
        return DEFAULT_AQI

    lat_idx = int(np.abs(lat_axis - lat).argmin())
    lon_idx = int(np.abs(lon_axis - lon).argmin())
    aqi_value = float(forecast_grid[normalize_hour_index(hour), lat_idx, lon_idx])
    if not np.isfinite(aqi_value):
        return DEFAULT_AQI
    aqi_value += station_anchor_residual(lat, lon, aqi_data=aqi_data, hour=hour)
    return float(np.clip(aqi_value, 0.0, 500.0))


def sample_aqi_window_for_coordinates(
    lat: float,
    lon: float,
    aqi_data=None,
    hour: int = 1,
    spatial_radius: int = 1,
) -> dict:
    if not aqi_data:
        return {
            "aqi": DEFAULT_AQI,
            "uncertainty": 25.0,
            "lower": max(0.0, DEFAULT_AQI - 25.0),
            "upper": DEFAULT_AQI + 25.0,
        }

    forecast_grid = aqi_data.get("forecast_grid")
    lat_axis = aqi_data.get("lat_axis")
    lon_axis = aqi_data.get("lon_axis")
    station_coords = aqi_data.get("station_coords")
    if forecast_grid is None or lat_axis is None or lon_axis is None:
        return {
            "aqi": DEFAULT_AQI,
            "uncertainty": 25.0,
            "lower": max(0.0, DEFAULT_AQI - 25.0),
            "upper": DEFAULT_AQI + 25.0,
        }

    h = normalize_hour_index(hour)
    lat_idx = int(np.abs(lat_axis - lat).argmin())
    lon_idx = int(np.abs(lon_axis - lon).argmin())
    lat_start = max(0, lat_idx - spatial_radius)
    lat_end = min(len(lat_axis), lat_idx + spatial_radius + 1)
    lon_start = max(0, lon_idx - spatial_radius)
    lon_end = min(len(lon_axis), lon_idx + spatial_radius + 1)

    spatial_values = np.asarray(forecast_grid[h, lat_start:lat_end, lon_start:lon_end], dtype=float)
    local_value = float(forecast_grid[h, lat_idx, lon_idx])
    if not np.isfinite(local_value):
        local_value = DEFAULT_AQI
    residual = station_anchor_residual(lat, lon, aqi_data=aqi_data, hour=hour)
    local_value = float(np.clip(local_value + residual, 0.0, 500.0))

    finite_spatial = spatial_values[np.isfinite(spatial_values)]
    spatial_std = float(np.std(finite_spatial)) if finite_spatial.size else 20.0

    h_start = max(0, h - 2)
    h_end = min(forecast_grid.shape[0], h + 3)
    temporal_values = np.asarray(forecast_grid[h_start:h_end, lat_idx, lon_idx], dtype=float)
    finite_temporal = temporal_values[np.isfinite(temporal_values)]
    temporal_std = float(np.std(finite_temporal)) if finite_temporal.size else 15.0

    nearest_station_km = nearest_station_distance_km(lat, lon, station_coords)
    # Keep street-level uncertainty directional rather than overwhelming the AQI
    # estimate when the nearest reference station is a few kilometers away.
    distance_uncertainty = 0.0 if nearest_station_km is None else min(20.0, 2.5 * nearest_station_km)
    base_uncertainty = 0.30 * spatial_std + 0.15 * temporal_std
    uncertainty = max(4.0, min(35.0, base_uncertainty + distance_uncertainty))
    return {
        "aqi": local_value,
        "uncertainty": uncertainty,
        "lower": max(0.0, local_value - uncertainty),
        "upper": local_value + uncertainty,
        "lat_idx": lat_idx,
        "lon_idx": lon_idx,
        "nearest_station_km": nearest_station_km,
        "station_anchor_residual": residual,
        "live_corrected": bool(aqi_data.get("station_anchor_forecast")),
    }


def get_preferred_edge_data(G, u, v, weight: str = "travel_time"):
    edge_bundle = G.get_edge_data(u, v)
    if not edge_bundle:
        return {}
    return min(
        edge_bundle.values(),
        key=lambda data: data.get(weight, data.get("length", float("inf"))),
    )


def calculate_route_exposure(G, route, aqi_data=None, hour: int = 1, transport_mode: str = "driving", profile: str = "healthy"):
    """
    Integrate forecast AQI along a route path using edge midpoints, adjusted for inhaled dose (MET).
    """
    from src.routing.health_matrix import get_met, get_sensitivity
    from src.routing.cost_graph import get_traffic_multiplier
    met = get_met(transport_mode)
    sensitivity = get_sensitivity(profile)
    if not route or len(route) < 2:
        return 0.0, []

    total_exposure = 0.0
    exposure_timeline = []
    elapsed_seconds = 0.0
    traffic_mult = get_traffic_multiplier(hour)

    def edge_road_multiplier(edge_data):
        highway = edge_data.get("highway")
        if isinstance(highway, (list, tuple, set)):
            values = [str(item) for item in highway]
        elif highway is None:
            values = []
        else:
            values = [str(highway)]
        penalty = 1.0
        for value in values:
            if value == "motorway":
                penalty = max(penalty, 1.35)
            elif value == "trunk":
                penalty = max(penalty, 1.28)
            elif value == "primary":
                penalty = max(penalty, 1.20)
            elif value == "secondary":
                penalty = max(penalty, 1.12)
            elif value == "tertiary":
                penalty = max(penalty, 1.06)
        return penalty

    for u, v in zip(route, route[1:]):
        edge_data = get_preferred_edge_data(G, u, v)
        base_time = float(edge_data.get("travel_time", 60.0))
        time_seconds = base_time * traffic_mult
        distance_meters = float(edge_data.get("length", 0.0))

        start_node = G.nodes[u]
        end_node = G.nodes[v]
        lat = (float(start_node["y"]) + float(end_node["y"])) / 2.0
        lon = (float(start_node["x"]) + float(end_node["x"])) / 2.0
        aqi_window = sample_aqi_window_for_coordinates(lat, lon, aqi_data=aqi_data, hour=hour)
        aqi = float(aqi_window.get("aqi", sample_aqi_for_coordinates(lat, lon, aqi_data=aqi_data, hour=hour)))
        uncertainty = float(aqi_window.get("uncertainty", 0.0))
        road_multiplier = edge_road_multiplier(edge_data)
        adjusted_aqi = aqi * road_multiplier

        segment_exposure = adjusted_aqi * (time_seconds / 3600.0) * met * sensitivity
        total_exposure += segment_exposure
        elapsed_seconds += time_seconds
        exposure_timeline.append(
            {
                "from_node": u,
                "to_node": v,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "segment_time_seconds": round(time_seconds, 2),
                "segment_distance_m": round(distance_meters, 2),
                "time_elapsed_seconds": round(elapsed_seconds, 2),
                "aqi": round(float(adjusted_aqi), 2),
                "base_aqi": round(float(aqi), 2),
                "uncertainty": round(float(uncertainty), 2),
                "aqi_upper": round(float((aqi + uncertainty) * road_multiplier), 2),
                "aqi_lower": round(float(max(0.0, aqi - uncertainty) * road_multiplier), 2),
                "road_multiplier": round(float(road_multiplier), 3),
                "segment_dose_index": round(float(segment_exposure), 4),
                "segment_exposure": round(float(segment_exposure), 4),
            }
        )

    return total_exposure, exposure_timeline
