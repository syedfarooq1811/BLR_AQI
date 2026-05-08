from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import osmnx as ox
import yaml


def load_routing_config():
    with open("configs/routing.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["routing"]


def get_traffic_multiplier(hour: int) -> float:
    """
    Returns a traffic multiplier based on the hour of the day.
    Peak hours (8-10, 17-19) increase travel time by 60%.
    """
    h = hour % 24
    if (8 <= h <= 10) or (17 <= h <= 19):
        return 1.6
    return 1.0


def normalized_edge(u, v):
    return (u, v) if u <= v else (v, u)


def build_cost_graph(city_name="Bangalore, India", save_path="data/processed/blr_graph.graphml"):
    """
    Download the OSM street network once, then reuse the cached GraphML artifact.
    """
    if Path(save_path).exists():
        print(f"Loading cached graph from {save_path}...")
        return ox.load_graphml(save_path)

    print(
        f"Downloading street network for {city_name} from OpenStreetMap. "
        "This will take a few minutes..."
    )
    graph = ox.graph_from_place(city_name, network_type="drive")
    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)
    ox.save_graphml(graph, save_path)
    return graph


def build_exposure_weight(
    G,
    aqi_data=None,
    profile="healthy",
    hour: int = 1,
    transport_mode: str = "driving",
    exposure_scale: float = 1.0,
    avoided_edges: set[tuple[int, int]] | None = None,
    route_penalty_seconds: float = 0.0,
    edge_air_quality_cache: dict | None = None,
):
    """
    Build a lazy weight function that mixes travel time with AQI burden.
    """
    from src.routing.exposure import sample_aqi_window_for_coordinates
    from src.routing.health_matrix import get_beta_aqi, get_sensitivity, get_met

    config = load_routing_config()
    alpha = config["alpha_time"]
    gamma = config["gamma_turn_penalty"]
    delta = config["delta_uncertainty_risk"]
    beta = get_beta_aqi(profile)
    sensitivity = get_sensitivity(profile)
    met = get_met(transport_mode)
    major_penalty = {
        "motorway": 1.35,
        "trunk": 1.28,
        "primary": 1.2,
        "secondary": 1.12,
        "tertiary": 1.06,
    }

    avoided_edges = avoided_edges or set()
    edge_air_quality_cache = edge_air_quality_cache if edge_air_quality_cache is not None else {}

    @lru_cache(maxsize=None)
    def edge_air_quality(u, v):
        cache_key = (int(hour), normalized_edge(u, v))
        if cache_key in edge_air_quality_cache:
            return edge_air_quality_cache[cache_key]
        start = G.nodes[u]
        end = G.nodes[v]
        lat = (float(start["y"]) + float(end["y"])) / 2.0
        lon = (float(start["x"]) + float(end["x"])) / 2.0
        value = sample_aqi_window_for_coordinates(lat, lon, aqi_data=aqi_data, hour=hour)
        edge_air_quality_cache[cache_key] = value
        return value

    def edge_highway_penalty(edge_data):
        highway = edge_data.get("highway")
        if isinstance(highway, (list, tuple, set)):
            values = [str(item) for item in highway]
        elif highway is None:
            values = []
        else:
            values = [str(highway)]
        penalty = 1.0
        for value in values:
            penalty = max(penalty, major_penalty.get(value, 1.0))
        return penalty

    def weight(u, v, edge_bundle):
        edge_data = min(
            edge_bundle.values(),
            key=lambda data: data.get("travel_time", data.get("length", float("inf"))),
        )
        base_travel_time = float(edge_data.get("travel_time", 60.0))
        travel_time = base_travel_time * get_traffic_multiplier(hour)
        aqi_window = edge_air_quality(u, v)
        edge_aqi = float(aqi_window["aqi"])
        uncertainty = float(aqi_window["uncertainty"])
        road_multiplier = edge_highway_penalty(edge_data)
        edge_dose = edge_aqi * road_multiplier * travel_time * sensitivity * met
        uncertainty_risk = uncertainty * delta * (travel_time / 60.0)
        route_penalty = route_penalty_seconds if normalized_edge(u, v) in avoided_edges else 0.0
        turn_penalty = gamma * float(edge_data.get("length", 0.0)) / 100.0
        return alpha * travel_time + (beta * exposure_scale) * edge_dose + uncertainty_risk + route_penalty + turn_penalty

    return weight
