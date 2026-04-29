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


def build_exposure_weight(G, aqi_data=None, profile="healthy", hour: int = 1, transport_mode: str = "driving"):
    """
    Build a lazy weight function that mixes travel time with AQI burden.
    """
    from src.routing.exposure import sample_aqi_for_coordinates
    from src.routing.health_matrix import get_beta_aqi, get_sensitivity, get_met

    config = load_routing_config()
    alpha = config["alpha_time"]
    beta = get_beta_aqi(profile)
    sensitivity = get_sensitivity(profile)
    met = get_met(transport_mode)

    @lru_cache(maxsize=None)
    def node_aqi(node_id):
        node = G.nodes[node_id]
        return sample_aqi_for_coordinates(
            float(node["y"]),
            float(node["x"]),
            aqi_data=aqi_data,
            hour=hour,
        )

    def weight(u, v, edge_bundle):
        edge_data = min(
            edge_bundle.values(),
            key=lambda data: data.get("travel_time", data.get("length", float("inf"))),
        )
        base_travel_time = float(edge_data.get("travel_time", 60.0))
        travel_time = base_travel_time * get_traffic_multiplier(hour)
        
        edge_aqi = (node_aqi(u) + node_aqi(v)) / 2.0
        # Integrate AQI over time (exposure) and MET rather than just adding concentration
        return alpha * travel_time + beta * (edge_aqi * travel_time * sensitivity * met)

    return weight
