import networkx as nx


def route_cost(G, route, weight="cost"):
    if not route or len(route) < 2:
        return 0.0

    total = 0.0
    for u, v in zip(route, route[1:]):
        edge_bundle = G.get_edge_data(u, v)
        if edge_bundle is None:
            return float("inf")
        if callable(weight):
            value = weight(u, v, edge_bundle)
        else:
            value = min(
                (data.get(weight, data.get("length", float("inf"))) for data in edge_bundle.values()),
                default=float("inf"),
            )
        if value is None:
            return float("inf")
        total += float(value)
    return total


def find_optimal_route(G, orig_node, dest_node, weight="cost"):
    """
    Use NetworkX shortest-path search with either an edge attribute or weight function.
    """
    try:
        route = nx.shortest_path(G, orig_node, dest_node, weight=weight)
        cost = route_cost(G, route, weight=weight)
        return route, cost
    except nx.NetworkXNoPath:
        return None, float("inf")


def normalized_edge(u, v):
    return (u, v) if u <= v else (v, u)


def route_edges(route):
    return {normalized_edge(u, v) for u, v in zip(route, route[1:])} if route else set()


def route_overlap_ratio(route_a, route_b):
    edges_a = route_edges(route_a)
    edges_b = route_edges(route_b)
    if not edges_a or not edges_b:
        return 0.0
    return len(edges_a & edges_b) / max(1, min(len(edges_a), len(edges_b)))


def pick_distinct_route(base_route, candidates, min_difference: float = 0.02):
    if not base_route:
        return candidates[0] if candidates else (None, float("inf"))
    max_overlap = 1.0 - min_difference
    for route, cost in candidates:
        if route and route != base_route and route_overlap_ratio(base_route, route) < max_overlap:
            return route, cost
    for route, cost in candidates:
        if route and route != base_route:
            return route, cost
    return candidates[0] if candidates else (None, float("inf"))


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
    avoided_fastest = route_edges(fastest_route)
    edge_air_quality_cache = {}

    balanced_candidates = []
    for scale, penalty, avoided in (
        (0.8, 0.0, set()),
        (1.6, 480.0, avoided_fastest),
    ):
        balanced_weight = build_exposure_weight(
            G,
            aqi_data=aqi_data,
            profile=profile,
            hour=hour,
            transport_mode=transport_mode,
            exposure_scale=scale,
            avoided_edges=avoided,
            route_penalty_seconds=penalty,
            edge_air_quality_cache=edge_air_quality_cache,
        )
        balanced_candidates.append(find_optimal_route(G, orig_node, dest_node, weight=balanced_weight))
    balanced_route, balanced_cost = pick_distinct_route(fastest_route, balanced_candidates, min_difference=0.15)

    avoided_cleanest = avoided_fastest | route_edges(balanced_route)
    cleanest_candidates = []
    for scale, penalty, avoided in (
        (2.4, 0.0, set()),
        (5.5, 1800.0, avoided_cleanest),
    ):
        cleanest_weight = build_exposure_weight(
            G,
            aqi_data=aqi_data,
            profile=profile,
            hour=hour,
            transport_mode=transport_mode,
            exposure_scale=scale,
            avoided_edges=avoided,
            route_penalty_seconds=penalty,
            edge_air_quality_cache=edge_air_quality_cache,
        )
        cleanest_candidates.append(find_optimal_route(G, orig_node, dest_node, weight=cleanest_weight))
    cleanest_route, cleanest_cost = pick_distinct_route(fastest_route, cleanest_candidates, min_difference=0.25)

    return {
        "fastest": {"route": fastest_route, "metric": fastest_time},
        "balanced": {"route": balanced_route, "metric": balanced_cost},
        "cleanest": {"route": cleanest_route, "metric": cleanest_cost},
    }
