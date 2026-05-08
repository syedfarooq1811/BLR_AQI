from __future__ import annotations

import math
from functools import lru_cache

import networkx as nx


MAJOR_HIGHWAYS = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
}


def _as_set(value) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(v) for v in value}
    return {str(value)}


def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    ref_lat = math.radians((lat1 + lat2) / 2.0)
    dy = (lat2 - lat1) * 111_320.0
    dx = (lon2 - lon1) * 111_320.0 * math.cos(ref_lat)
    return math.hypot(dx, dy)


@lru_cache(maxsize=16_384)
def _rounded_key(lat: float, lon: float, radius_m: float) -> tuple[float, float, float]:
    return (round(lat, 4), round(lon, 4), round(radius_m, 1))


def road_context_for_coordinates(
    G: nx.MultiDiGraph | None,
    lat: float,
    lon: float,
    radius_m: float = 500.0,
) -> dict[str, float]:
    if G is None:
        return {
            "road_context_available": False,
            "road_density_km_per_sqkm": 0.0,
            "major_road_share": 0.0,
            "nearest_major_road_m": None,
        }

    total_length = 0.0
    major_length = 0.0
    nearest_major = None
    radius_sqkm = math.pi * (radius_m / 1000.0) ** 2

    for u, v, data in G.edges(data=True):
        u_node = G.nodes[u]
        v_node = G.nodes[v]
        mid_lat = (float(u_node["y"]) + float(v_node["y"])) / 2.0
        mid_lon = (float(u_node["x"]) + float(v_node["x"])) / 2.0
        dist = _distance_m(lat, lon, mid_lat, mid_lon)
        if dist > radius_m:
            continue

        length = float(data.get("length", 0.0))
        total_length += length
        is_major = bool(_as_set(data.get("highway")) & MAJOR_HIGHWAYS)
        if is_major:
            major_length += length
            nearest_major = dist if nearest_major is None else min(nearest_major, dist)

    return {
        "road_context_available": True,
        "road_density_km_per_sqkm": round((total_length / 1000.0) / max(radius_sqkm, 1e-6), 4),
        "major_road_share": round(major_length / total_length, 4) if total_length > 0 else 0.0,
        "nearest_major_road_m": round(nearest_major, 2) if nearest_major is not None else None,
    }


def context_aqi_adjustment(road_context: dict | None) -> float:
    if not road_context or not road_context.get("road_context_available"):
        return 0.0

    density = float(road_context.get("road_density_km_per_sqkm") or 0.0)
    major_share = float(road_context.get("major_road_share") or 0.0)
    nearest_major = road_context.get("nearest_major_road_m")
    proximity = 0.0
    if nearest_major is not None:
        proximity = max(0.0, 1.0 - min(float(nearest_major), 400.0) / 400.0)

    return round(min(14.0, 0.3 * density + 6.0 * major_share + 5.0 * proximity), 2)
