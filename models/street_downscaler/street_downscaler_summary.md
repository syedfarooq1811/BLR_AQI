# Street Downscaler Training Summary

- Labels: `data\raw\street_labels.parquet`
- Features: `data\processed\features.parquet`
- Target: `AQI`

## Paper Novelty Model: STARLING-AQI
- Full name: Spatial-Temporal Adaptive Residual Learning with Interpolation-Guided Stacking
- Novelty: interpolation-guided heterogeneous stacking with horizon-specific validation.
- Caution: final street-level claims require real street labels, not station-proxy labels alone.

## Horizon 24h
- Rows: 3000
- Split train/val/test: 2100/450/450
- Test R2: 0.5385
- Test RMSE: 0.5748
- Test MAE: 0.4088
- Target gate pass: False
- Model path: `models\street_downscaler\street_downscaler_h24.joblib`

## Horizon 168h
- Rows: 3000
- Split train/val/test: 2100/450/450
- Test R2: 0.3221
- Test RMSE: 0.7179
- Test MAE: 0.5026
- Target gate pass: False
- Model path: `models\street_downscaler\street_downscaler_h168.joblib`
