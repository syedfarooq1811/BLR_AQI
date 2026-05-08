# Journal Validation

Run:

```bash
python scripts/run_journal_validation.py
```

Outputs:

- `temporal_holdout_metrics.csv`
- `leave_one_station_out_metrics.csv`
- `ablation_metrics.csv`
- `journal_validation_tables.md`
- `run_metadata.json`

The processed target in `features.parquet` is normalized, so the generated MAE,
RMSE, PICP, and CRPS tables are in normalized AQI units.
