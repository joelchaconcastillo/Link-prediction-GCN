"""
Run experiments across multiple datasets and models.
"""

import os
import sys
import json
import torch
from datetime import datetime
from train import train
from src.models import GCN, GAT, GraphSAGE, SEAL
from src.data import load_dataset, get_edge_split_data


# Experiment configurations
DATASETS = ['Cora', 'CiteSeer', 'PubMed']
MODELS = ['GCN', 'GAT', 'GraphSAGE', 'SEAL']

DEFAULT_CONFIG = {
    'hidden_channels': 128,
    'out_channels': 64,
    'num_layers': 2,
    'dropout': 0.5,
    'lr': 0.01,
    'weight_decay': 5e-4,
    'epochs': 200,
    'eval_every': 10,
    'patience': 20,
    'seed': 42,
    'save_model': True,
    'save_dir': 'checkpoints'
}


def run_single_experiment(dataset_name, model_name, device):
    """Run a single experiment."""
    print(f"\n{'='*60}")
    print(f"Running: {model_name} on {dataset_name}")
    print(f"{'='*60}")
    
    # Load dataset
    data = load_dataset(dataset_name)
    split_data = get_edge_split_data(data, val_ratio=0.1, test_ratio=0.2,
                                     random_state=DEFAULT_CONFIG['seed'])
    
    # Initialize model
    model_class = {
        'GCN': GCN,
        'GAT': GAT,
        'GraphSAGE': GraphSAGE,
        'SEAL': SEAL
    }[model_name]
    
    model = model_class(
        in_channels=data.num_features,
        hidden_channels=DEFAULT_CONFIG['hidden_channels'],
        out_channels=DEFAULT_CONFIG['out_channels'],
        num_layers=DEFAULT_CONFIG['num_layers'],
        dropout=DEFAULT_CONFIG['dropout']
    )
    
    model = model.to(device)
    
    # Update config
    config = DEFAULT_CONFIG.copy()
    config.update({
        'dataset': dataset_name,
        'model': model_name,
        'device': str(device)
    })
    
    # Train
    test_metrics = train(model, split_data, config, device)
    
    return test_metrics


def run_all_experiments():
    """Run experiments for all dataset-model combinations."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for dataset_name in DATASETS:
        results[dataset_name] = {}
        
        for model_name in MODELS:
            try:
                metrics = run_single_experiment(dataset_name, model_name, device)
                results[dataset_name][model_name] = metrics
            except Exception as e:
                print(f"Error running {model_name} on {dataset_name}: {e}")
                results[dataset_name][model_name] = {'error': str(e)}
    
    # Save results
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f'results_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print(f"{'Dataset':<15} {'Model':<12} {'AUC':<8} {'AP':<8} {'Hits@10':<10}")
    print("-"*80)
    
    for dataset_name in DATASETS:
        for model_name in MODELS:
            if 'error' not in results[dataset_name][model_name]:
                metrics = results[dataset_name][model_name]
                print(f"{dataset_name:<15} {model_name:<12} "
                      f"{metrics['auc']:<8.4f} {metrics['ap']:<8.4f} "
                      f"{metrics.get('hits@10', 0):<10.4f}")
    
    return results


if __name__ == '__main__':
    run_all_experiments()
