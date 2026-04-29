import torch
import torch.nn as nn

class AdvectionDiffusionLoss(nn.Module):
    """
    Physics-Informed Regularization enforcing advection-diffusion-decay mechanics
    dc/dt + u*grad(c) = D*laplacian(c) - k*c
    """
    def __init__(self, lambda_phys=0.1):
        super().__init__()
        self.lambda_phys = lambda_phys
        self.D = nn.Parameter(torch.tensor([0.1])) # Diffusion
        self.k = nn.Parameter(torch.tensor([0.01])) # Decay
        
    def forward(self, preds, preds_prev=None):
        if preds.dim() == 4: # Grid outputs from SpatialUNet: (B, C, H, W)
            # Spatial gradients (dx, dy)
            dy = preds[:, :, 1:, :] - preds[:, :, :-1, :]
            dx = preds[:, :, :, 1:] - preds[:, :, :, :-1]
            
            # Laplacian proxy (2nd derivative)
            d2y = dy[:, :, 1:, :] - dy[:, :, :-1, :]
            d2x = dx[:, :, :, 1:] - dx[:, :, :, :-1]
            
            # Temporal derivative (if previous frame is provided)
            time_penalty = torch.mean((preds - preds_prev)**2) if preds_prev is not None else 0.0
            laplacian_penalty = torch.mean(d2y**2) + torch.mean(d2x**2)
            decay_penalty = preds.mean()
            
            phys_loss = time_penalty + self.D * laplacian_penalty + self.k * decay_penalty
        else:
            # Fallback for 1D station signals: (B, N, horizon)
            dc_dt = preds[:, :, 1:] - preds[:, :, :-1]
            spatial_var = preds.var(dim=1).mean()
            phys_loss = torch.mean(dc_dt**2) + self.D * spatial_var + self.k * preds.mean()
            
        return self.lambda_phys * phys_loss

class QuantileLoss(nn.Module):
    def __init__(self, quantiles=[0.05, 0.5, 0.95]):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        # preds: (B, N, horizon, len(quantiles))
        loss = 0
        for i, q in enumerate(self.quantiles):
            errors = target - preds[..., i]
            loss += torch.max((q - 1) * errors, q * errors).mean()
        return loss
