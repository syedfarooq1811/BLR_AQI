import networkx as nx


def find_optimal_route(G, orig_node, dest_node, weight="cost"):
    """
    Use NetworkX shortest-path search with either an edge attribute or weight function.
    """
    try:
        route = nx.shortest_path(G, orig_node, dest_node, weight=weight)
        cost = nx.shortest_path_length(G, orig_node, dest_node, weight=weight)
        return route, cost
    except nx.NetworkXNoPath:
        return None, float("inf")


def generate_pareto_fronts(G, orig_node, dest_node, profile="healthy", aqi_data=None, hour: int = 1, transport_mode: str = "driving"):
    """
    Return the fastest route and the lowest-exposure route for the requested profile.
    """
    from src.routing.cost_graph import build_exposure_weight

    fastest_route, fastest_time = find_optimal_route(
        G,
        orig_node,
        dest_node,
        weight="travel_time",
    )
    cleanest_weight = build_exposure_weight(G, aqi_data=aqi_data, profile=profile, hour=hour, transport_mode=transport_mode)
    cleanest_route, cleanest_cost = find_optimal_route(
        G,
        orig_node,
        dest_node,
        weight=cleanest_weight,
    )

    return {
        "fastest": {"route": fastest_route, "metric": fastest_time},
        "cleanest": {"route": cleanest_route, "metric": cleanest_cost},
    }
