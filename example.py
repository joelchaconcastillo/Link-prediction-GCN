"""
Simple example script demonstrating the link prediction framework.

This script shows how to:
1. Load a dataset
2. Initialize a model
3. Train and evaluate
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.models import GCN, SEAL
from src.data import load_dataset, get_edge_split_data
from src.utils import evaluate_all_metrics, print_metrics


def simple_example():
    """Run a simple link prediction example."""
    
    print("="*60)
    print("Link Prediction Framework - Simple Example")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load dataset
    print("\n1. Loading Cora dataset...")
    data = load_dataset('Cora')
    print(f"   Nodes: {data.num_nodes}")
    print(f"   Edges: {data.edge_index.size(1)}")
    print(f"   Features: {data.num_features}")
    
    # Split edges
    print("\n2. Splitting edges into train/val/test...")
    split_data = get_edge_split_data(data, val_ratio=0.1, test_ratio=0.2)
    print(f"   Train edges: {split_data['train_edge_index'].size(1)}")
    print(f"   Val edges: {split_data['val_edge_index'].size(1)}")
    print(f"   Test edges: {split_data['test_edge_index'].size(1)}")
    
    # Initialize models
    print("\n3. Initializing models...")
    
    # Standard GCN
    gcn_model = GCN(
        in_channels=data.num_features,
        hidden_channels=128,
        out_channels=64,
        num_layers=2,
        dropout=0.5
    ).to(device)
    print(f"   GCN parameters: {sum(p.numel() for p in gcn_model.parameters())}")
    
    # Enhanced SEAL
    seal_model = SEAL(
        in_channels=data.num_features,
        hidden_channels=128,
        out_channels=64,
        num_layers=3,
        dropout=0.5,
        use_attention=True
    ).to(device)
    print(f"   SEAL parameters: {sum(p.numel() for p in seal_model.parameters())}")
    
    # Quick training demo with GCN
    print("\n4. Training GCN (10 epochs demo)...")
    model = gcn_model
    optimizer = Adam(model.parameters(), lr=0.01)
    
    x = split_data['x'].to(device)
    train_edge_index = split_data['train_edge_index'].to(device)
    val_edge_index = split_data['val_edge_index'].to(device)
    val_labels = split_data['val_labels'].to(device)
    
    for epoch in range(1, 11):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # Simple training step
        from src.data import negative_sampling, create_edge_labels
        
        # Sample positive edges
        perm = torch.randperm(train_edge_index.size(1))[:1000]
        pos_edges = train_edge_index[:, perm]
        
        # Generate negative edges
        neg_edges = negative_sampling(pos_edges, data.num_nodes, 
                                      num_neg_samples=1000,
                                      existing_edges=train_edge_index).to(device)
        
        # Create labels
        edge_index, labels = create_edge_labels(pos_edges, neg_edges)
        labels = labels.to(device)
        
        # Forward pass
        z = model.encode(x, train_edge_index)
        pred = model.decode(z, edge_index)
        loss = F.binary_cross_entropy_with_logits(pred, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Validation
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                z = model.encode(x, train_edge_index)
                val_pred = torch.sigmoid(model.decode(z, val_edge_index))
                val_metrics = evaluate_all_metrics(val_pred, val_labels)
            
            print(f"   Epoch {epoch:02d} | Loss: {loss:.4f} | "
                  f"Val AUC: {val_metrics['auc']:.4f} | "
                  f"Val AP: {val_metrics['ap']:.4f}")
    
    # Final evaluation
    print("\n5. Final evaluation on test set...")
    model.eval()
    test_edge_index = split_data['test_edge_index'].to(device)
    test_labels = split_data['test_labels'].to(device)
    
    with torch.no_grad():
        z = model.encode(x, train_edge_index)
        test_pred = torch.sigmoid(model.decode(z, test_edge_index))
        test_metrics = evaluate_all_metrics(test_pred, test_labels)
    
    print_metrics(test_metrics, 'Test')
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("  - Train full models with: python train.py")
    print("  - Run experiments with: python experiments/run_experiments.py")
    print("  - Try different models: --model SEAL")
    print("  - Try different datasets: --dataset CiteSeer")
    print("="*60)


if __name__ == '__main__':
    simple_example()
