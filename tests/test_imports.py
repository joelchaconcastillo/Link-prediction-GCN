"""Test configuration and initialization."""

import pytest


def test_imports():
    """Test that all modules can be imported."""
    from src.models import GCN, GAT, GraphSAGE, SEAL
    from src.data import load_dataset, split_edges, negative_sampling
    from src.utils import evaluate_auc_ap, evaluate_hits_at_k
    
    # If we get here, imports work
    assert True


def test_pytorch_geometric_available():
    """Test PyTorch Geometric availability."""
    try:
        import torch_geometric
        assert True
    except ImportError:
        pytest.skip("PyTorch Geometric not installed")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
