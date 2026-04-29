from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_AQI = 75.0
FORECAST_GRID_PATH = Path("data/processed/forecast_grid_7day.npy")
GRID_LAT_PATH = Path("data/processed/grid_lat.npy")
GRID_LON_PATH = Path("data/processed/grid_lon.npy")


def normalize_hour_index(hour: int) -> int:
    return max(0, min(int(hour) - 1, 167))


def load_forecast_grid_bundle():
    if not (
        FORECAST_GRID_PATH.exists()
        and GRID_LAT_PATH.exists()
        and GRID_LON_PATH.exists()
    ):
        return None

    grid_lat = np.load(GRID_LAT_PATH)
    grid_lon = np.load(GRID_LON_PATH)
    return {
        "forecast_grid": np.load(FORECAST_GRID_PATH, mmap_mode="r"),
        "grid_lat": grid_lat,
        "grid_lon": grid_lon,
        "lat_axis": grid_lat[:, 0],
        "lon_axis": grid_lon[0, :],
    }


def sample_aqi_for_coordinates(lat: float, lon: float, aqi_data=None, hour: int = 1) -> float:
    if not aqi_data:
        return DEFAULT_AQI

    forecast_grid = aqi_data.get("forecast_grid")
    lat_axis = aqi_data.get("lat_axis")
    lon_axis = aqi_data.get("lon_axis")
    if forecast_grid is None or lat_axis is None or lon_axis is None:
        return DEFAULT_AQI

    lat_idx = int(np.abs(lat_axis - lat).argmin())
    lon_idx = int(np.abs(lon_axis - lon).argmin())
    aqi_value = float(forecast_grid[normalize_hour_index(hour), lat_idx, lon_idx])
    if not np.isfinite(aqi_value):
        return DEFAULT_AQI
    return aqi_value


def get_preferred_edge_data(G, u, v, weight: str = "travel_time"):
    edge_bundle = G.get_edge_data(u, v)
    if not edge_bundle:
        return {}
    return min(
        edge_bundle.values(),
        key=lambda data: data.get(weight, data.get("length", float("inf"))),
    )


def calculate_route_exposure(G, route, aqi_data=None, hour: int = 1, transport_mode: str = "driving"):
    """
    Integrate forecast AQI along a route path using edge midpoints, adjusted for inhaled dose (MET).
    """
    from src.routing.health_matrix import get_met
    from src.routing.cost_graph import get_traffic_multiplier
    met = get_met(transport_mode)
    if not route or len(route) < 2:
        return 0.0, []

    total_exposure = 0.0
    exposure_timeline = []
    elapsed_seconds = 0.0
    traffic_mult = get_traffic_multiplier(hour)

    for u, v in zip(route, route[1:]):
        edge_data = get_preferred_edge_data(G, u, v)
        base_time = float(edge_data.get("travel_time", 60.0))
        time_seconds = base_time * traffic_mult
        distance_meters = float(edge_data.get("length", 0.0))

        start_node = G.nodes[u]
        end_node = G.nodes[v]
        lat = (float(start_node["y"]) + float(end_node["y"])) / 2.0
        lon = (float(start_node["x"]) + float(end_node["x"])) / 2.0
        aqi = sample_aqi_for_coordinates(lat, lon, aqi_data=aqi_data, hour=hour)

        segment_exposure = aqi * (time_seconds / 3600.0) * met
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
                "aqi": round(float(aqi), 2),
                "segment_exposure": round(float(segment_exposure), 4),
            }
        )

    return total_exposure, exposure_timeline
