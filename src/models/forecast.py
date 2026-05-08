"""
7-Day (168-Hour) AQI Forecasting Engine for Bengaluru.

Architecture:
  1. Load latest `seq_len` hours of station feature data from features.parquet.
  2. Infer with ST_MHGTD -> head_168h gives (1, 12, 168) station-level AQI predictions.
  3. For each of the 168 hours, project station values onto the IDW spatial grid using SpatialUNet.
  4. Save outputs:
     - data/processed/forecast_station_7day.json  -> per-station, per-hour AQI (human-readable)
     - data/processed/forecast_grid_7day.npy       -> spatial tensor (168, H, W)
"""

import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

from src.models.st_mhgtd import ST_MHGTD


# -----------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------
SEQ_LEN    = 168          # 7 days of hourly context fed to the model
HORIZON    = 168          # 7 days of forecast
NUM_NODES  = 12
HIDDEN_DIM = 256          # Best Optuna trial hidden dim
MODEL_PATH    = "data/models/best_st_mhgtd.pt"
FEATURES_PATH = "data/processed/features.parquet"
GRID_PATH     = "data/processed/aqi_grid.npy"
OUT_DIR       = Path("data/processed")
FORECAST_GRID_OUT_PATH = OUT_DIR / "forecast_grid_7day_filtered.npy"
INTERPOLATION_METADATA_PATH = Path("reports/street_level/interpolation_loso_metadata.json")
EXCLUDED_STATION_IDS = {"site_1558"}
DEFAULT_MODEL_INPUT_DIM = 15

# Bengaluru major CPCB stations
STATION_NAMES = {
    "site_1553": "BWSSB Kadabesanahalli",
    "site_162":  "Silk Board",
    "site_165":  "BTM Layout",
    "site_1554": "Hebbal",
    "site_1555": "Jayanagar",
    "site_5729": "Peenya",
    "site_5681": "Bapuji Nagar",
    "site_163":  "Hombegowda Nagar",
    "site_5678": "City Railway Station",
    "site_166":  "Saneguruva Halli",
    "site_5686": "T Dasarahalli",
}

AQI_CATEGORIES = [
    (0,   50,  "Good",         "#00e400"),
    (51,  100, "Satisfactory", "#92d050"),
    (101, 200, "Moderate",     "#ffff00"),
    (201, 300, "Poor",         "#ff7e00"),
    (301, 400, "Very Poor",    "#ff0000"),
    (401, 500, "Severe",       "#7e0023"),
]

def categorize_aqi(val):
    for lo, hi, label, color in AQI_CATEGORIES:
        if lo <= val <= hi:
            return label, color
    return "Severe", "#7e0023"


def model_input_dim() -> int:
    if not Path(MODEL_PATH).exists():
        return DEFAULT_MODEL_INPUT_DIM
    try:
        state = torch.load(MODEL_PATH, map_location="cpu")
        weight = state.get("tcn.0.conv.weight")
        if weight is not None and len(weight.shape) >= 2:
            return int(weight.shape[1])
    except Exception:
        pass
    return DEFAULT_MODEL_INPUT_DIM


# -----------------------------------------------------------------
# STEP 1: Load latest SEQ_LEN hours of features
# -----------------------------------------------------------------
def load_input_sequence():
    print("[1/4] Loading latest input features...")
    df = pd.read_parquet(FEATURES_PATH)
    df = df.sort_values('timestamp')

    preferred_cols = [
        "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI",
        "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
        "is_weekend", "AQI_lag_1h", "AQI_lag_3h", "AQI_lag_6h", "AQI_lag_24h",
        "AQI_rolling_mean_6h", "AQI_rolling_std_6h", "AQI_rolling_mean_12h",
        "AQI_rolling_std_12h", "AQI_rolling_mean_24h", "AQI_rolling_std_24h",
    ]
    feature_cols = [col for col in preferred_cols if col in df.columns][:model_input_dim()]

    all_stations = [
        sid for sid in sorted(df['station_id'].unique())
        if sid not in EXCLUDED_STATION_IDS
    ]
    station_sequences = {}
    for sid in all_stations:
        sdf = df[df['station_id'] == sid].tail(SEQ_LEN)
        if len(sdf) < SEQ_LEN:
            pad = pd.DataFrame(np.zeros((SEQ_LEN - len(sdf), len(feature_cols))),
                               columns=feature_cols)
            sdf_vals = pd.concat([pad, sdf[feature_cols]], ignore_index=True)
        else:
            sdf_vals = sdf[feature_cols].reset_index(drop=True)
        station_sequences[sid] = sdf_vals.values.astype(np.float32)

    in_dim = len(feature_cols)
    x = np.zeros((1, NUM_NODES, SEQ_LEN, in_dim), dtype=np.float32)
    for i, sid in enumerate(all_stations[:NUM_NODES]):
        x[0, i] = station_sequences[sid]

    print(f"  Input tensor shape: {x.shape}")
    # Use the current time (India Standard Time) as the reference point for timestamps
    last_ts = pd.Timestamp.now(tz='Asia/Kolkata')
    # Align to the next full hour to keep 1‑hour steps consistent
    if last_ts.minute != 0 or last_ts.second != 0 or last_ts.microsecond != 0:
        last_ts = (last_ts + pd.Timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    return torch.tensor(x), all_stations[:NUM_NODES], last_ts


# -----------------------------------------------------------------
# STEP 2: Run ST_MHGTD -> 168h station-level forecast
# -----------------------------------------------------------------
def forecast_stations(x_tensor, in_dim):
    print("[2/4] Running ST-MHGTD for 7-day station-level forecast...")
    model = ST_MHGTD(num_nodes=NUM_NODES, in_dim=in_dim, hidden_dim=HIDDEN_DIM)
    if Path(MODEL_PATH).exists():
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'), strict=False)
            print(f"  Weights loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"  Warning: Could not load weights ({e}). Using random init.")
    model.eval()

    with torch.no_grad():
        outputs = model(x_tensor)

    # head_168h shape: (1, num_nodes, 168)
    station_forecast = outputs['head_168h'].squeeze(0).numpy()  # (12, 168)
    # De-normalize: shift from normalized space to AQI range
    station_forecast = np.clip(station_forecast * 50 + 100, 0, 500)
    adj_matrix = outputs['adj'].squeeze(0).numpy()
    print(f"  Station forecast shape: {station_forecast.shape}")
    return station_forecast, adj_matrix


from scipy.interpolate import griddata
from src.models.spatial_interpolation import grid_idw_weights

STATION_COORDS = {
    "site_1553": [12.9279, 77.6271],
    "site_162": [12.9174, 77.6235],
    "site_165": [12.9166, 77.6101],
    "site_1554": [13.0450, 77.5966],
    "site_1555": [12.9250, 77.5938],
    "site_5729": [13.0289, 77.5199],
    "site_5681": [12.9634, 77.5559],
    "site_163": [12.9609, 77.5996],
    "site_5678": [12.9774, 77.5713],
    "site_166": [13.0068, 77.5090],
    "site_5686": [13.0450, 77.5116],
}

def project_to_grid(station_forecast, grid_shape, station_ids):
    print("[3/4] Projecting station forecasts to spatial grid with station-anchored interpolation...")
    H, W = grid_shape
    idw_power, idw_blend = load_interpolation_settings()
    
    try:
        grid_lat = np.load(OUT_DIR / "grid_lat.npy")
        grid_lon = np.load(OUT_DIR / "grid_lon.npy")
    except Exception as e:
        print(f"Warning: Could not load lat/lon grids ({e}). Falling back to flat mean.")
        grid_lat, grid_lon = None, None

    spatial_grids = []
    
    if grid_lat is not None and grid_lon is not None:
        points = np.array([STATION_COORDS[sid] for sid in station_ids])
        idw_weights = grid_idw_weights(points, grid_lat, grid_lon, power=idw_power)
        
    for h in range(HORIZON):
        values = station_forecast[:len(station_ids), h]
        avg_aqi = values.mean()
        
        if grid_lat is not None and grid_lon is not None:
            grid_z = griddata(points, values, (grid_lat, grid_lon), method='linear')
            grid_z_nearest = griddata(points, values, (grid_lat, grid_lon), method='nearest')
            linear_grid = np.where(np.isnan(grid_z), grid_z_nearest, grid_z)
            linear_grid = np.nan_to_num(linear_grid, nan=avg_aqi)
            idw_grid = np.tensordot(idw_weights, values, axes=([2], [0]))
            # Blend the smoother linear surface with station-anchored IDW. This is
            # deterministic and avoids adding structure from an untrained U-Net.
            base = (1.0 - idw_blend) * linear_grid + idw_blend * idw_grid
        else:
            base = np.full((H, W), avg_aqi, dtype=np.float32)

        grid_aqi = np.clip(base, 0, 500).astype(np.float32)
        spatial_grids.append(grid_aqi)

    stacked = np.stack(spatial_grids, axis=0)  # (168, H, W)
    print(f"  Spatial grid shape: {stacked.shape}")
    return stacked, idw_power, idw_blend


def load_interpolation_settings():
    default_power = 1.0
    default_blend = 1.0
    if not INTERPOLATION_METADATA_PATH.exists():
        return default_power, default_blend
    try:
        metadata = json.loads(INTERPOLATION_METADATA_PATH.read_text(encoding="utf-8"))
        best = metadata.get("best", {})
        return float(best.get("idw_power", default_power)), float(best.get("idw_blend", default_blend))
    except Exception as exc:
        print(f"  Warning: could not load interpolation settings ({exc}); using defaults.")
        return default_power, default_blend


# -----------------------------------------------------------------
# STEP 4: Save all outputs
# -----------------------------------------------------------------
def save_outputs(station_forecast, spatial_grids, station_ids, last_ts, adj_matrix, idw_power, idw_blend):
    print("[4/4] Saving outputs...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    forecast_json = {}
    for i, sid in enumerate(station_ids):
        name = STATION_NAMES.get(sid, sid)
        hourly = []
        for h in range(HORIZON):
            ts = last_ts + timedelta(hours=h + 1)
            aqi_val = float(station_forecast[i, h])
            category, color = categorize_aqi(aqi_val)
            hourly.append({
                "hour": h + 1,
                "timestamp": str(ts),
                "aqi": round(aqi_val, 2),
                "category": category,
                "color": color
            })
        forecast_json[sid] = {
            "station_name": name,
            "forecast_horizon_hours": HORIZON,
            "generated_at": str(last_ts),
            "hourly": hourly,
            "daily_summary": [
                {
                    "day": d + 1,
                    "date": str((last_ts + timedelta(days=d + 1)).date()),
                    "aqi_mean": round(float(station_forecast[i, d*24:(d+1)*24].mean()), 2),
                    "aqi_max":  round(float(station_forecast[i, d*24:(d+1)*24].max()), 2),
                    "category": categorize_aqi(station_forecast[i, d*24:(d+1)*24].mean())[0]
                }
                for d in range(7)
            ]
        }

    json_path = OUT_DIR / "forecast_station_7day.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(forecast_json, f, indent=2)
    print(f"  Station forecast saved -> {json_path}")

    adj_path = OUT_DIR / "attention_weights.json"
    adj_list = []
    n = len(station_ids)
    # Collect all cross-station (non-self) edges
    cross_weights = []
    for i in range(n):
        for j in range(n):
            if i != j:  # exclude self-loops
                cross_weights.append((i, j, float(adj_matrix[i, j])))

    # Normalize relative to max cross-station weight so we always get values
    max_w = max((w for _, _, w in cross_weights), default=1.0)
    if max_w == 0:
        max_w = 1.0
    for i, j, weight in cross_weights:
        norm_weight = weight / max_w
        if norm_weight > 0.001:  # very low threshold — keep meaningful edges
            adj_list.append({
                "from_node": station_ids[i],
                "to_node": station_ids[j],
                "weight": round(norm_weight, 4)
            })

    # Sort by weight descending, keep top 30
    adj_list.sort(key=lambda x: x["weight"], reverse=True)
    adj_list = adj_list[:30]

    with open(adj_path, "w", encoding="utf-8") as f:
        json.dump(adj_list, f, indent=2)
    print(f"  Attention weights saved -> {adj_path} ({len(adj_list)} cross-station edges)")

    np.save(FORECAST_GRID_OUT_PATH, spatial_grids)
    print(f"  Spatial grid saved     -> {FORECAST_GRID_OUT_PATH}")

    # Summary table
    sep = "=" * 75
    print(f"\n{sep}")
    print(f"  7-DAY AQI FORECAST SUMMARY  --  Bengaluru Major Stations")
    print(sep)
    print(f"  {'Station':<35} {'Day1':>7} {'Day2':>7} {'Day3':>7} {'Day4':>7} {'Day7':>7}")
    print("-" * 75)
    for i, sid in enumerate(station_ids):
        name = STATION_NAMES.get(sid, sid)[:33]
        days = [round(float(station_forecast[i, d*24:(d+1)*24].mean()), 1) for d in range(7)]
        print(f"  {name:<35} {days[0]:>7} {days[1]:>7} {days[2]:>7} {days[3]:>7} {days[6]:>7}")
    print(sep)
    print(f"\n  Grid: {spatial_grids.shape}  (168 hours x H x W)")
    print(f"  Street interpolation: IDW power={idw_power:.2f}, IDW blend={idw_blend:.2f}\n")


# -----------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 75)
    print("  ST-MHGTD 7-Day Forecast Engine  --  Bengaluru AQI")
    print("=" * 75 + "\n")

    x_tensor, station_ids, last_ts = load_input_sequence()
    in_dim = x_tensor.shape[-1]

    station_forecast, adj_matrix = forecast_stations(x_tensor, in_dim)

    if Path(GRID_PATH).exists():
        base_grid = np.load(GRID_PATH)
        grid_shape = base_grid.shape
    else:
        grid_shape = (241, 248)  # Grid shape from grid_builder output

    spatial_grids, idw_power, idw_blend = project_to_grid(station_forecast, grid_shape, station_ids)
    save_outputs(station_forecast, spatial_grids, station_ids, last_ts, adj_matrix, idw_power, idw_blend)
