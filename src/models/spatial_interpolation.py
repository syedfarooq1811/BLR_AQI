from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay


@dataclass(frozen=True)
class InterpolationWeights:
    station_ids: list[str]
    linear_weights: np.ndarray
    idw_weights: np.ndarray


def km_projected_delta(
    lat: np.ndarray,
    lon: np.ndarray,
    ref_lat: float,
    ref_lon: float,
) -> tuple[np.ndarray, np.ndarray]:
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(np.deg2rad(ref_lat))
    dy = (lat - ref_lat) * km_per_deg_lat
    dx = (lon - ref_lon) * km_per_deg_lon
    return dy, dx


def idw_weights_for_targets(
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    power: float = 2.0,
) -> np.ndarray:
    ref_lat = float(np.mean(source_coords[:, 0]))
    target_lat = target_coords[:, 0][:, None]
    target_lon = target_coords[:, 1][:, None]
    source_lat = source_coords[:, 0][None, :]
    source_lon = source_coords[:, 1][None, :]
    dy = (target_lat - source_lat) * 111.32
    dx = (target_lon - source_lon) * 111.32 * np.cos(np.deg2rad(ref_lat))
    dist = np.sqrt(dx * dx + dy * dy)

    exact = dist < 1e-9
    weights = 1.0 / np.maximum(dist, 1e-6) ** power
    weights = weights / weights.sum(axis=1, keepdims=True)
    if np.any(exact):
        exact_any = exact.any(axis=1, keepdims=True)
        weights = np.where(exact_any, exact.astype(float), weights)
        weights = weights / weights.sum(axis=1, keepdims=True)
    return weights.astype(np.float32)


def linear_weights_for_targets(source_coords: np.ndarray, target_coords: np.ndarray) -> np.ndarray:
    weights = np.zeros((len(target_coords), len(source_coords)), dtype=np.float32)
    if len(source_coords) < 3:
        return idw_weights_for_targets(source_coords, target_coords, power=2.0)

    tri = Delaunay(source_coords)
    simplex = tri.find_simplex(target_coords)
    nearest = idw_weights_for_targets(source_coords, target_coords, power=8.0)

    for target_idx, simplex_idx in enumerate(simplex):
        if simplex_idx < 0:
            weights[target_idx] = nearest[target_idx]
            continue
        transform = tri.transform[simplex_idx]
        delta = target_coords[target_idx] - transform[2]
        bary = np.dot(transform[:2], delta)
        bary = np.append(bary, 1.0 - bary.sum())
        vertices = tri.simplices[simplex_idx]
        weights[target_idx, vertices] = bary

    return weights


def blend_predictions(
    source_values: np.ndarray,
    linear_weights: np.ndarray,
    idw_weights: np.ndarray,
    idw_blend: float,
) -> np.ndarray:
    linear_pred = source_values @ linear_weights.T
    idw_pred = source_values @ idw_weights.T
    return (1.0 - idw_blend) * linear_pred + idw_blend * idw_pred


def grid_idw_weights(points: np.ndarray, grid_lat: np.ndarray, grid_lon: np.ndarray, power: float = 2.0):
    flat_targets = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])
    return idw_weights_for_targets(points, flat_targets, power=power).reshape(
        grid_lat.shape[0],
        grid_lat.shape[1],
        len(points),
    )
