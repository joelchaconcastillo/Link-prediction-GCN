"""
Evaluation metrics for link prediction.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_auc_ap(predictions, labels):
    """
    Evaluate AUC (Area Under ROC Curve) and AP (Average Precision).
    
    Args:
        predictions (Tensor or ndarray): Predicted scores for edges
        labels (Tensor or ndarray): True labels (1 for positive, 0 for negative)
        
    Returns:
        dict: Dictionary containing 'auc' and 'ap' scores
    """
    # Convert to numpy arrays
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    
    # Calculate metrics
    auc = roc_auc_score(labels, predictions)
    ap = average_precision_score(labels, predictions)
    
    return {
        'auc': auc,
        'ap': ap
    }


def evaluate_hits_at_k(predictions, labels, k_list=[10, 20, 50, 100]):
    """
    Evaluate Hits@K metric for link prediction.
    
    Hits@K measures the proportion of true positive edges that appear
    in the top-K predicted edges.
    
    Args:
        predictions (Tensor or ndarray): Predicted scores for edges
        labels (Tensor or ndarray): True labels (1 for positive, 0 for negative)
        k_list (list): List of K values to evaluate
        
    Returns:
        dict: Dictionary containing hits@k for each k
    """
    # Convert to numpy arrays
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    
    # Get indices of positive edges
    pos_indices = np.where(labels == 1)[0]
    num_pos = len(pos_indices)
    
    # Sort predictions in descending order
    sorted_indices = np.argsort(-predictions)
    
    results = {}
    for k in k_list:
        if k > len(predictions):
            continue
        
        # Get top-k predictions
        top_k_indices = sorted_indices[:k]
        
        # Count how many positive edges are in top-k
        hits = len(set(top_k_indices) & set(pos_indices))
        
        # Calculate hits@k (normalize by number of positive edges or k)
        hits_at_k = hits / min(num_pos, k)
        
        results[f'hits@{k}'] = hits_at_k
    
    return results


def evaluate_mrr(predictions, labels):
    """
    Evaluate Mean Reciprocal Rank (MRR).
    
    MRR measures how well the model ranks positive edges.
    
    Args:
        predictions (Tensor or ndarray): Predicted scores for edges
        labels (Tensor or ndarray): True labels (1 for positive, 0 for negative)
        
    Returns:
        float: MRR score
    """
    # Convert to numpy arrays
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    
    # Get indices of positive edges
    pos_indices = np.where(labels == 1)[0]
    
    # Sort predictions in descending order
    sorted_indices = np.argsort(-predictions)
    
    # Find ranks of positive edges
    ranks = []
    for pos_idx in pos_indices:
        rank = np.where(sorted_indices == pos_idx)[0][0] + 1  # 1-indexed
        ranks.append(1.0 / rank)
    
    # Calculate MRR
    mrr = np.mean(ranks)
    
    return mrr


def evaluate_all_metrics(predictions, labels, k_list=[10, 20, 50, 100]):
    """
    Evaluate all metrics at once.
    
    Args:
        predictions (Tensor or ndarray): Predicted scores for edges
        labels (Tensor or ndarray): True labels (1 for positive, 0 for negative)
        k_list (list): List of K values for Hits@K
        
    Returns:
        dict: Dictionary containing all metrics
    """
    results = {}
    
    # AUC and AP
    auc_ap = evaluate_auc_ap(predictions, labels)
    results.update(auc_ap)
    
    # Hits@K
    hits = evaluate_hits_at_k(predictions, labels, k_list)
    results.update(hits)
    
    # MRR
    mrr = evaluate_mrr(predictions, labels)
    results['mrr'] = mrr
    
    return results


def print_metrics(metrics, prefix=''):
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics (dict): Dictionary of metrics
        prefix (str): Prefix for the output (e.g., 'Train', 'Val', 'Test')
    """
    if prefix:
        print(f"\n{prefix} Metrics:")
    else:
        print("\nMetrics:")
    
    # Print in a nice format
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
