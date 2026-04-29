import torch
import torch.nn as nn
from src.models.dynamic_graph import DynamicGraphLearner

class DilatedTCN(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (B, C, L)
        return self.relu(self.conv(x))

class PatchTSTEncoder(nn.Module):
    def __init__(self, in_dim, patch_size, stride, hidden_dim):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.proj = nn.Linear(patch_size * in_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        # Create patches
        # Pad sequence if necessary
        # Simplified patching logic
        seq_len = L // self.patch_size
        x_patched = x[:, :seq_len * self.patch_size, :].reshape(B, seq_len, self.patch_size * C)
        emb = self.proj(x_patched)
        out = self.transformer(emb)
        # Shape: (B, seq_len, hidden_dim)
        return out

class ST_MHGTD(nn.Module):
    """
    Spatio-Temporal Multi-Horizon Graph Transformer with Adaptive Diffusion.
    """
    def __init__(self, num_nodes, in_dim, hidden_dim, horizon_steps=[24, 168]):
        super().__init__()
        self.graph_learner = DynamicGraphLearner(hidden_dim, num_nodes)
        
        self.tcn = nn.Sequential(
            DilatedTCN(in_dim, hidden_dim, dilation=1),
            DilatedTCN(hidden_dim, hidden_dim, dilation=2),
            DilatedTCN(hidden_dim, hidden_dim, dilation=4)
        )
        
        self.patch_tst = PatchTSTEncoder(hidden_dim, patch_size=6, stride=3, hidden_dim=hidden_dim)
        
        # Multi-Horizon Heads
        self.heads = nn.ModuleDict({
            f"head_{h}h": nn.Linear(hidden_dim, h) for h in horizon_steps
        })
        
        # Residual projection to map original input features directly to horizon outputs
        self.residual_proj = nn.ModuleDict({
            f"head_{h}h": nn.Linear(in_dim, h) for h in horizon_steps
        })
        
    def forward(self, x):
        # x shape: (B, num_nodes, seq_len, in_dim)
        B, N, L, C = x.shape
        
        # 1. Temporal Encoding (TCN over seq_len)
        x_flat = x.view(B * N, L, C).transpose(1, 2)  # (B*N, C, L)
        tcn_out = self.tcn(x_flat).transpose(1, 2)    # (B*N, L, hidden)
        
        # 2. PatchTST
        tst_out = self.patch_tst(tcn_out)             # (B*N, patches, hidden)
        
        # Aggregate temporal features (e.g., mean)
        node_feats = tst_out.mean(dim=1).view(B, N, -1) # (B, N, hidden)
        
        # 3. Spatial Graph
        adj = self.graph_learner(node_feats)          # (B, N, N)
        spatial_out = torch.bmm(adj, node_feats)      # (B, N, hidden)
        
        # 4. Multi-Horizon Outputs with Residual Connections
        # Take the features from the very last time step for the residual
        last_step_feats = x[:, :, -1, :] # (B, N, C)
        
        outputs = {'adj': adj}
        for k, head in self.heads.items():
            outputs[k] = head(spatial_out) + self.residual_proj[k](last_step_feats)
            
        return outputs
