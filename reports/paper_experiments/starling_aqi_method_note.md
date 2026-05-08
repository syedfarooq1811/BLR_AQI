# STARLING-AQI Method Note

## Proposed Novelty

**STARLING-AQI** stands for **Spatial-Temporal Adaptive Residual Learning with Interpolation-Guided Stacking**.

The novelty is the combined modelling design used for Bengaluru street-level AQI forecasting:

- A spatial interpolation prior built from blended IDW and Delaunay-linear station interpolation.
- Horizon-specific station distribution context using mean, spread, minimum, and maximum station AQI at the forecast horizon.
- Cyclic temporal context through hour-of-day and day-of-week encodings.
- Site-aware residual learning through station/site identity encoding when station-proxy labels are used.
- A heterogeneous stacking ensemble combining tree boosting, random forests, extremely randomized trees, local analogue KNN, and neural nonlinear correction.
- RidgeCV meta-learning with original feature passthrough so the final estimator can blend raw spatial priors with learned residual corrections.

## Suggested Paper Framing

Use wording like:

> We propose STARLING-AQI, an interpolation-guided heterogeneous residual stacking framework for horizon-specific urban AQI downscaling. The framework combines physically interpretable spatial priors with nonlinear residual learners and a regularized meta-learner, enabling both predictive performance and interpretable ablation of spatial, temporal, and site-specific contributions.

## Performance Claim Rule

Only claim **R2 >= 0.94** and **RMSE <= 0.15** if `reports/paper_experiments/target_checks.json` reports a pass on the intended validation protocol.

If training uses `--use-station-proxy-labels`, report it as a proxy-data development experiment. Final street-level claims require real street labels in `data/raw/street_labels.parquet` or another clearly documented external validation set.
