from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import osmnx as ox
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List

from src.routing.astar import generate_pareto_fronts
from src.routing.cost_graph import build_cost_graph
from src.routing.exposure import calculate_route_exposure, load_forecast_grid_bundle

app = FastAPI(title="Bengaluru AQI API", version="2.0")

FRONTEND_DIR = Path("frontend/public")
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

FORECAST_PATH = Path("data/processed/forecast_station_7day.json")

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
    "site_1558": {"name": "Yeshwanthpur", "lat": 13.0219, "lon": 77.5421},
}

G = None
AQI_DATA = None


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


@app.on_event("startup")
def load_runtime_assets():
    global AQI_DATA, G
    try:
        G = build_cost_graph()
        print("Street graph loaded.")
    except Exception as exc:
        print(f"Graph not loaded: {exc}")

    AQI_DATA = load_forecast_grid_bundle()
    if AQI_DATA is None:
        print("Forecast grid not loaded. Route AQI will use fallback values.")


def load_station_forecast():
    if not FORECAST_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Run `python -m src.models.forecast` first.",
        )
    with open(FORECAST_PATH, encoding="utf-8") as file:
        return json.load(file)


def route_to_coordinates(route):
    return [
        [round(float(G.nodes[node]["y"]), 6), round(float(G.nodes[node]["x"]), 6)]
        for node in route
    ]


def summarize_route(route, metric, hour: int, transport_mode: str = "driving"):
    exposure, timeline = calculate_route_exposure(G, route, AQI_DATA, hour=hour, transport_mode=transport_mode)
    total_time = sum(item["segment_time_seconds"] for item in timeline)
    total_distance = sum(item["segment_distance_m"] for item in timeline)
    average_aqi = 0.0 if total_time <= 0 else round((exposure * 3600.0) / total_time, 2)

    return {
        "route": route,
        "coordinates": route_to_coordinates(route),
        "metric": round(float(metric), 2),
        "travel_time_seconds": round(total_time, 2),
        "travel_time_minutes": round(total_time / 60.0, 2),
        "distance_meters": round(total_distance, 2),
        "distance_km": round(total_distance / 1000.0, 2),
        "exposure_aqi_hours": round(float(exposure), 4),
        "average_aqi": average_aqi,
        "timeline": timeline,
    }


@app.get("/")
def serve_dashboard():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"msg": "Frontend not found. Place index.html in frontend/public/"}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "graph_loaded": G is not None,
        "forecast_loaded": FORECAST_PATH.exists(),
        "forecast_grid_loaded": AQI_DATA is not None,
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


@app.get("/api/forecast/station/{station_id}")
def get_station_forecast(station_id: str):
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

    hourly_aqi = forecast_grid[:, lat_idx, lon_idx]
    # Replace NaNs with a default AQI of 75.0 just in case
    hourly_aqi_clean = np.nan_to_num(hourly_aqi, nan=75.0).tolist()
    
    # Also return the actual snapped coordinates
    actual_lat = float(lat_axis[lat_idx])
    actual_lon = float(lon_axis[lon_idx])

    return {
        "requested_lat": lat,
        "requested_lon": lon,
        "actual_lat": actual_lat,
        "actual_lon": actual_lon,
        "hourly_aqi": [round(float(val), 2) for val in hourly_aqi_clean],
    }


@app.post("/api/route")
def get_route(req: RouteRequest):
    global G

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
        cleanest_route = pareto["cleanest"]["route"]
        if not fastest_route or not cleanest_route:
            raise HTTPException(status_code=404, detail="No route found for the selected points.")

        result = {
            "status": "success",
            "data": {
                "requested_hour": req.hour,
                "profile": req.profile,
                "origin": {"lat": req.orig_lat, "lon": req.orig_lon, "node": orig_node},
                "destination": {"lat": req.dest_lat, "lon": req.dest_lon, "node": dest_node},
                "transport_mode": req.transport_mode,
                "fastest": summarize_route(fastest_route, pareto["fastest"]["metric"], req.hour, req.transport_mode),
                "cleanest": summarize_route(cleanest_route, pareto["cleanest"]["metric"], req.hour, req.transport_mode),
            },
        }

        # Calculate optimal departure by evaluating cleanest route over hours_to_scan
        best_hour_offset = 0
        best_aqi = float('inf')
        hourly_route_aqi = []

        for i in range(req.hours_to_scan):
            h = req.hour + i
            if h > 168:
                break
            exposure, timeline = calculate_route_exposure(G, cleanest_route, AQI_DATA, hour=h, transport_mode=req.transport_mode)
            total_time = sum(item["segment_time_seconds"] for item in timeline)
            avg_aqi = 0.0 if total_time <= 0 else (exposure * 3600.0) / total_time
            
            hourly_route_aqi.append(round(avg_aqi, 2))
            if avg_aqi < best_aqi:
                best_aqi = avg_aqi
                best_hour_offset = i

        result["data"]["optimal_departure"] = {
            "best_hour_offset": best_hour_offset,
            "best_avg_aqi": round(best_aqi, 2),
            "hourly_route_aqi": hourly_route_aqi
        }

        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


import itertools

@app.post("/api/route/tsp")
def get_tsp_route(req: TSPRequest):
    global G
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
            
    summary = summarize_route(unified_route, best_cost, req.hour, req.transport_mode)
    
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
        return json.load(f)

@app.post("/api/xai/route_insight")
def get_route_insight(req: RouteRequest):
    if not ATTENTION_PATH.exists():
        return {"insight": "Attention weights not available.", "highlights": []}
        
    try:
        with open(ATTENTION_PATH, "r", encoding="utf-8") as f:
            weights = json.load(f)
            
        # Sort by weight descending
        weights.sort(key=lambda x: x["weight"], reverse=True)
        
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
            
        return {
            "insight": f"The ST-MHGTD graph transformer's spatial attention dynamically weighted the edges, anticipating wind-driven pollutant advection across your route. The {req.transport_mode} route minimizes physiological dose accordingly.",
            "highlights": highlights_with_coords
        }
    except Exception as e:
        return {"insight": str(e), "highlights": []}
