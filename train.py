"""
Training script for link prediction with GNN models.
"""

import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import argparse
import yaml

from src.models import GCN, GAT, GraphSAGE, SEAL
from src.data import load_dataset, get_edge_split_data
from src.utils import evaluate_all_metrics, print_metrics


def train_epoch(model, data, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: GNN model
        data: Training data dictionary
        optimizer: Optimizer
        device: Device to use
        
    Returns:
        float: Training loss
    """
    model.train()
    optimizer.zero_grad()
    
    # Move data to device
    x = data['x'].to(device)
    train_edge_index = data['train_edge_index'].to(device)
    
    # Generate training samples
    from src.data import negative_sampling, create_edge_labels
    
    # Positive edges (subsample for efficiency)
    num_pos = train_edge_index.size(1) // 2  # Divided by 2 because edges are bidirectional
    perm = torch.randperm(train_edge_index.size(1))[:num_pos]
    pos_edge_index = train_edge_index[:, perm]
    
    # Negative edges
    neg_edge_index = negative_sampling(pos_edge_index, data['num_nodes'],
                                       num_neg_samples=num_pos,
                                       existing_edges=train_edge_index)
    neg_edge_index = neg_edge_index.to(device)
    
    # Create labels
    edge_index, labels = create_edge_labels(pos_edge_index, neg_edge_index)
    labels = labels.to(device)
    
    # Forward pass
    z = model.encode(x, train_edge_index)
    predictions = model.decode(z, edge_index)
    
    # Compute loss
    loss = F.binary_cross_entropy_with_logits(predictions, labels)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(model, data, split, device):
    """
    Evaluate the model on validation or test set.
    
    Args:
        model: GNN model
        data: Data dictionary
        split: 'val' or 'test'
        device: Device to use
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    # Move data to device
    x = data['x'].to(device)
    train_edge_index = data['train_edge_index'].to(device)
    edge_index = data[f'{split}_edge_index'].to(device)
    labels = data[f'{split}_labels'].to(device)
    
    # Forward pass
    z = model.encode(x, train_edge_index)
    predictions = model.decode(z, edge_index)
    
    # Apply sigmoid to get probabilities
    predictions = torch.sigmoid(predictions)
    
    # Evaluate metrics
    metrics = evaluate_all_metrics(predictions, labels)
    
    return metrics


def train(model, data, config, device):
    """
    Complete training loop.
    
    Args:
        model: GNN model
        data: Data dictionary
        config: Configuration dictionary
        device: Device to use
        
    Returns:
        dict: Best validation metrics and test metrics
    """
    optimizer = Adam(model.parameters(), lr=config['lr'], 
                    weight_decay=config['weight_decay'])
    
    best_val_auc = 0
    best_test_metrics = None
    patience_counter = 0
    
    print("\nStarting training...")
    print(f"Device: {device}")
    print(f"Model: {config['model']}")
    print(f"Dataset: {config['dataset']}")
    
    for epoch in range(1, config['epochs'] + 1):
        # Training
        loss = train_epoch(model, data, optimizer, device)
        
        # Validation
        if epoch % config['eval_every'] == 0:
            val_metrics = evaluate(model, data, 'val', device)
            test_metrics = evaluate(model, data, 'test', device)
            
            print(f'\nEpoch {epoch:03d}, Loss: {loss:.4f}')
            print_metrics(val_metrics, 'Validation')
            print_metrics(test_metrics, 'Test')
            
            # Early stopping
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                best_test_metrics = test_metrics
                patience_counter = 0
                
                # Save best model
                if config.get('save_model', False):
                    save_path = os.path.join(config['save_dir'], 
                                           f"best_model_{config['model']}.pt")
                    torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1
                
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    return best_test_metrics


def main():
    parser = argparse.ArgumentParser(description='Link Prediction with GNNs')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name (overrides config)')
    parser.add_argument('--model', type=str, default=None,
                       help='Model name (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'dataset': 'Cora',
            'model': 'GCN',
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
    
    # Override with command-line arguments
    if args.dataset:
        config['dataset'] = args.dataset
    if args.model:
        config['model'] = args.model
    if args.device:
        config['device'] = args.device
    else:
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set random seed
    torch.manual_seed(config['seed'])
    
    # Create save directory
    if config.get('save_model', False):
        os.makedirs(config.get('save_dir', 'checkpoints'), exist_ok=True)
    
    # Load dataset
    print(f"Loading {config['dataset']} dataset...")
    data = load_dataset(config['dataset'])
    
    # Split edges
    print("Splitting edges...")
    split_data = get_edge_split_data(data, val_ratio=0.1, test_ratio=0.2,
                                     random_state=config['seed'])
    
    # Initialize model
    print(f"Initializing {config['model']} model...")
    model_class = {
        'GCN': GCN,
        'GAT': GAT,
        'GraphSAGE': GraphSAGE,
        'SEAL': SEAL
    }[config['model']]
    
    model = model_class(
        in_channels=data.num_features,
        hidden_channels=config['hidden_channels'],
        out_channels=config['out_channels'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    device = torch.device(config['device'])
    model = model.to(device)
    
    # Train model
    test_metrics = train(model, split_data, config, device)
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL TEST RESULTS")
    print("="*50)
    print_metrics(test_metrics, 'Test')
    

if __name__ == '__main__':
    main()
