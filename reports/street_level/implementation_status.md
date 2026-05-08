# Street-Level Accuracy Status

## What Is Implemented

- Leave-one-station-out interpolation validation using the `AQI` target only.
- Automatic interpolation tuning report in:
  - `reports/street_level/interpolation_loso_report.md`
  - `reports/street_level/interpolation_loso_tuning.csv`
  - `reports/street_level/interpolation_loso_by_station.csv`
- Forecast grid generation now uses deterministic station-anchored interpolation instead of an untrained `SpatialUNet`.
- Street forecast uncertainty now widens with distance from the nearest station.
- Street forecast API now includes road-context descriptors:
  - `road_density_km_per_sqkm`
  - `major_road_share`
  - `nearest_major_road_m`
- Learned street-level downscaler training is blocked unless real street labels exist.

## Current Validation Result

- Best interpolation setting from LOSO station validation:
  - `idw_power = 1.0`
  - `idw_blend = 1.0`
- Mean hidden-station interpolation error:
  - `MAE = 0.6574`
  - `RMSE = 0.8572`
  - `R2 = 0.2098`

These metrics are in normalized AQI units from `data/processed/features.parquet`.

## What This Means

- The street map is now more honest and more stable.
- The current station network is not dense enough to claim highly accurate street-level AQI everywhere in the city.
- Some hidden stations interpolate reasonably well, while others remain difficult, especially where local micro-environment effects are strong.

## What Is Still Needed

- Real street-level AQI labels from low-cost sensors, mobile transects, or repeated route measurements.
- Optional context-model training that uses road density, major-road proximity, land use, vegetation, and elevation.
- Re-generation of `forecast_grid_7day.npy` after the Torch paging-file issue is resolved on this machine.
