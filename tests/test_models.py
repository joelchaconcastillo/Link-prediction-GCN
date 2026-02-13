"""
Basic tests for the link prediction framework.
"""

import torch
import pytest
from src.models import GCN, GAT, GraphSAGE, SEAL


def test_gcn_forward():
    """Test GCN forward pass."""
    model = GCN(in_channels=16, hidden_channels=32, out_channels=16, num_layers=2)
    
    # Create dummy data
    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    
    # Forward pass
    out = model(x, edge_index)
    
    assert out.shape == (10, 16)


def test_gcn_encode_decode():
    """Test GCN encode and decode."""
    model = GCN(in_channels=16, hidden_channels=32, out_channels=16, num_layers=2)
    
    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    
    # Encode
    z = model.encode(x, edge_index)
    assert z.shape == (10, 16)
    
    # Decode
    edge_label_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    scores = model.decode(z, edge_label_index)
    assert scores.shape == (2,)


def test_gat_forward():
    """Test GAT forward pass."""
    model = GAT(in_channels=16, hidden_channels=32, out_channels=16, 
                num_layers=2, heads=4)
    
    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    
    out = model(x, edge_index)
    assert out.shape == (10, 16)


def test_graphsage_forward():
    """Test GraphSAGE forward pass."""
    model = GraphSAGE(in_channels=16, hidden_channels=32, out_channels=16, num_layers=2)
    
    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    
    out = model(x, edge_index)
    assert out.shape == (10, 16)


def test_seal_forward():
    """Test SEAL forward pass."""
    model = SEAL(in_channels=16, hidden_channels=32, out_channels=16, 
                 num_layers=2, use_attention=True)
    
    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    
    out = model(x, edge_index)
    assert out.shape == (10, 16)


def test_seal_attention_decode():
    """Test SEAL with attention-based decoding."""
    model = SEAL(in_channels=16, hidden_channels=32, out_channels=16, 
                 num_layers=2, use_attention=True)
    
    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    
    z = model.encode(x, edge_index)
    edge_label_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    
    scores = model.decode(z, edge_label_index)
    assert scores.shape == (2,)


def test_batch_norm():
    """Test models with batch normalization."""
    model = GCN(in_channels=16, hidden_channels=32, out_channels=16, 
                num_layers=2, use_batch_norm=True)
    
    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    
    out = model(x, edge_index)
    assert out.shape == (10, 16)


def test_model_training_mode():
    """Test model training/eval modes."""
    model = GCN(in_channels=16, hidden_channels=32, out_channels=16, num_layers=2, dropout=0.5)
    
    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    
    # Training mode
    model.train()
    out_train = model(x, edge_index)
    
    # Eval mode
    model.eval()
    out_eval = model(x, edge_index)
    
    # Shapes should be same
    assert out_train.shape == out_eval.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
