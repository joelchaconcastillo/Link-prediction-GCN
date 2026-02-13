"""
Graph Attention Network (GAT) implementation for link prediction.

Reference: Veličković et al. (2018), "Graph Attention Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    """
    Graph Attention Network for node embeddings in link prediction.
    
    Args:
        in_channels (int): Number of input features per node
        hidden_channels (int): Number of hidden units
        out_channels (int): Dimension of output embeddings
        num_layers (int): Number of GAT layers (default: 2)
        heads (int): Number of attention heads (default: 8)
        dropout (float): Dropout rate (default: 0.6)
        use_batch_norm (bool): Whether to use batch normalization (default: False)
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_layers=2, heads=8, dropout=0.6, use_batch_norm=False):
        super(GAT, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        if use_batch_norm:
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, 
                                     heads=heads, dropout=dropout))
            if use_batch_norm:
                self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # Output layer (single head for final embedding)
        self.convs.append(GATConv(hidden_channels * heads, out_channels, 
                                 heads=1, concat=False, dropout=dropout))
        
    def forward(self, x, edge_index):
        """
        Forward pass of GAT.
        
        Args:
            x (Tensor): Node feature matrix [num_nodes, in_channels]
            edge_index (LongTensor): Graph connectivity [2, num_edges]
            
        Returns:
            Tensor: Node embeddings [num_nodes, out_channels]
        """
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            if self.use_batch_norm:
                x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        return x
    
    def encode(self, x, edge_index):
        """Alias for forward pass to generate node embeddings."""
        return self.forward(x, edge_index)
    
    def decode(self, z, edge_label_index):
        """
        Decode edge predictions from node embeddings using dot product.
        
        Args:
            z (Tensor): Node embeddings [num_nodes, out_channels]
            edge_label_index (LongTensor): Edge indices to predict [2, num_edges]
            
        Returns:
            Tensor: Edge predictions (logits) [num_edges]
        """
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)
    
    def decode_all(self, z):
        """
        Decode all possible edges (full adjacency matrix).
        
        Args:
            z (Tensor): Node embeddings [num_nodes, out_channels]
            
        Returns:
            Tensor: Full adjacency matrix [num_nodes, num_nodes]
        """
        return torch.matmul(z, z.t())
