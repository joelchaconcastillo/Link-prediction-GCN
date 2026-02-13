"""
Tests for data loading and preprocessing utilities.
"""

import torch
import pytest
from src.data import load_dataset, split_edges, negative_sampling, create_edge_labels


def test_load_cora():
    """Test loading Cora dataset."""
    data = load_dataset('Cora')
    
    assert data.num_nodes == 2708
    assert data.num_features == 1433
    assert data.edge_index.size(0) == 2  # 2 rows (source, target)


def test_split_edges():
    """Test edge splitting."""
    data = load_dataset('Cora')
    split_data = split_edges(data, val_ratio=0.1, test_ratio=0.2)
    
    assert 'train_edge_index' in split_data
    assert 'val_edge_index' in split_data
    assert 'test_edge_index' in split_data
    
    # Check that train edges are bidirectional
    assert split_data['train_edge_index'].size(0) == 2
    
    # Check ratios approximately correct
    total_edges = data.edge_index.size(1) // 2  # Undirected
    val_edges = split_data['val_edge_index'].size(1)
    test_edges = split_data['test_edge_index'].size(1)
    
    assert abs(val_edges / total_edges - 0.1) < 0.02  # Within 2%
    assert abs(test_edges / total_edges - 0.2) < 0.02


def test_negative_sampling():
    """Test negative sampling."""
    # Create simple edge index
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    num_nodes = 10
    
    neg_edges = negative_sampling(edge_index, num_nodes, num_neg_samples=5)
    
    assert neg_edges.size() == (2, 5)
    
    # Check that negative edges don't exist in positive edges
    pos_set = set()
    for i in range(edge_index.size(1)):
        pos_set.add((edge_index[0, i].item(), edge_index[1, i].item()))
        pos_set.add((edge_index[1, i].item(), edge_index[0, i].item()))
    
    for i in range(neg_edges.size(1)):
        src, dst = neg_edges[0, i].item(), neg_edges[1, i].item()
        assert (src, dst) not in pos_set


def test_create_edge_labels():
    """Test edge label creation."""
    pos_edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    neg_edges = torch.tensor([[0, 3], [2, 3]], dtype=torch.long)
    
    edge_index, labels = create_edge_labels(pos_edges, neg_edges)
    
    assert edge_index.size() == (2, 4)
    assert labels.size() == (4,)
    assert labels[:2].sum() == 2  # First two are positive
    assert labels[2:].sum() == 0  # Last two are negative


def test_dataset_info():
    """Test dataset info retrieval."""
    from src.data import get_dataset_info
    
    # Get all info
    all_info = get_dataset_info()
    assert 'Cora' in all_info
    assert 'CiteSeer' in all_info
    
    # Get specific info
    cora_info = get_dataset_info('Cora')
    assert cora_info['num_nodes'] == 2708


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
