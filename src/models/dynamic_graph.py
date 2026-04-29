import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicGraphLearner(nn.Module):
    """
    Learns dynamic adjacency matrix based on spatial embeddings and temporal node states.
    Incorporates directional wind-like bias for asymmetric edge weights.
    """
    def __init__(self, node_dim, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_emb = nn.Parameter(torch.randn(num_nodes, node_dim))
        self.w1 = nn.Linear(node_dim, node_dim)
        self.w2 = nn.Linear(node_dim, node_dim)
        self.wind_proj = nn.Linear(node_dim, 1) # Projects temporal state to wind potential
        
    def forward(self, x_temp=None):
        # Base graph from node embeddings
        node_1 = self.w1(self.node_emb)
        node_2 = self.w2(self.node_emb)
        
        adj_base = torch.relu(torch.matmul(node_1, node_2.T))
        
        if x_temp is not None:
            # x_temp shape: (B, num_nodes, feat_dim)
            # dynamic undirected modifier
            dyn = torch.bmm(x_temp, x_temp.transpose(1, 2))
            
            # Dynamic directional bias (asymmetric) to represent wind advection
            v = self.wind_proj(x_temp) # (B, num_nodes, 1)
            directional_bias = v - v.transpose(1, 2) # (B, num_nodes, num_nodes)
            
            adj = adj_base.unsqueeze(0) + dyn + directional_bias
        else:
            adj = adj_base.unsqueeze(0)
            
        # Normalize
        adj = F.softmax(adj, dim=-1)
        # Shape: (B, num_nodes, num_nodes)
        return adj
