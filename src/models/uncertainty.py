import torch
import numpy as np

def compute_picp(preds_lower, preds_upper, target):
    """
    Prediction Interval Coverage Probability (PICP)
    """
    in_bound = (target >= preds_lower) & (target <= preds_upper)
    return in_bound.float().mean().item()

def compute_crps(preds_quantiles, target, quantiles=[0.05, 0.5, 0.95]):
    """
    Continuous Ranked Probability Score (CRPS) approximation using quantiles
    """
    # Simplified approximation
    loss = 0
    for i, q in enumerate(quantiles):
        errors = target - preds_quantiles[..., i]
        loss += torch.max((q - 1) * errors, q * errors).mean().item()
    return loss / len(quantiles)
