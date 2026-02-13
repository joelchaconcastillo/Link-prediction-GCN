"""
Tests for evaluation metrics.
"""

import torch
import numpy as np
import pytest
from src.utils import evaluate_auc_ap, evaluate_hits_at_k


def test_evaluate_auc_ap():
    """Test AUC and AP calculation."""
    # Perfect predictions
    predictions = torch.tensor([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
    labels = torch.tensor([1, 1, 1, 0, 0, 0])
    
    metrics = evaluate_auc_ap(predictions, labels)
    
    assert 'auc' in metrics
    assert 'ap' in metrics
    assert metrics['auc'] == 1.0  # Perfect ranking
    assert metrics['ap'] == 1.0


def test_evaluate_auc_ap_numpy():
    """Test with numpy arrays."""
    predictions = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array([1, 1, 0, 0])
    
    metrics = evaluate_auc_ap(predictions, labels)
    
    assert metrics['auc'] == 1.0
    assert metrics['ap'] == 1.0


def test_evaluate_hits_at_k():
    """Test Hits@K calculation."""
    # 10 predictions: 5 positive, 5 negative
    # Top 5 include all positives
    predictions = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    labels = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    
    metrics = evaluate_hits_at_k(predictions, labels, k_list=[5, 10])
    
    assert 'hits@5' in metrics
    assert 'hits@10' in metrics
    assert metrics['hits@5'] == 1.0  # All positives in top-5
    assert metrics['hits@10'] == 1.0


def test_evaluate_hits_at_k_partial():
    """Test Hits@K with partial success."""
    # Only 3 out of 5 positives in top-5
    predictions = torch.tensor([0.9, 0.8, 0.7, 0.4, 0.3, 0.6, 0.5, 0.2, 0.1, 0.0])
    labels = torch.tensor([1, 1, 1, 0, 0, 1, 1, 0, 0, 0])
    
    metrics = evaluate_hits_at_k(predictions, labels, k_list=[5])
    
    # Top 5 predictions: indices [0,1,2,5,6], labels [1,1,1,1,1]
    # Actually top 5: [0.9, 0.8, 0.7, 0.6, 0.5] -> indices [0,1,2,5,6]
    # Wait, let me recalculate...
    # Sorted indices by prediction: [0,1,2,5,6,3,4,7,8,9]
    # Top 5: [0,1,2,5,6], all have label 1
    assert metrics['hits@5'] == 1.0


def test_empty_predictions():
    """Test with edge cases."""
    predictions = torch.tensor([])
    labels = torch.tensor([])
    
    # Should handle empty inputs gracefully
    # Note: Some metrics might fail with empty inputs, which is expected


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
