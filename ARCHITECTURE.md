# Framework Architecture

## Overview

The Link Prediction GCN Framework is organized into modular components for ease of use, extensibility, and research.

```
Link-prediction-GCN/
│
├── src/                          # Core framework code
│   ├── models/                   # GNN model implementations
│   │   ├── gcn.py               # Standard Graph Convolutional Network
│   │   ├── gat.py               # Graph Attention Network
│   │   ├── graphsage.py         # GraphSAGE
│   │   └── seal.py              # Enhanced SEAL with attention
│   │
│   ├── data/                     # Data loading and preprocessing
│   │   ├── loader.py            # Dataset loaders (Cora, CiteSeer, etc.)
│   │   └── split.py             # Edge splitting and negative sampling
│   │
│   └── utils/                    # Utility functions
│       └── metrics.py           # Evaluation metrics (AUC, AP, Hits@K)
│
├── configs/                      # YAML configuration files
│   ├── default.yaml             # Default settings
│   ├── gat.yaml                 # GAT-specific config
│   └── seal.yaml                # SEAL-specific config
│
├── experiments/                  # Experiment scripts
│   └── run_experiments.py       # Batch experiment runner
│
├── tests/                        # Unit tests
│   ├── test_models.py           # Model tests
│   ├── test_data.py             # Data loading tests
│   ├── test_metrics.py          # Metric tests
│   └── test_imports.py          # Import tests
│
├── train.py                      # Main training script
├── example.py                    # Quick start example
├── setup.py                      # Package setup
├── requirements.txt              # Dependencies
│
└── Documentation
    ├── README.md                 # Main documentation
    ├── QUICKSTART.md            # Quick start guide
    ├── CONTRIBUTING.md          # Contribution guidelines
    └── RESEARCH_IDEAS.md        # Research directions
```

## Component Details

### 1. Models (`src/models/`)

All models inherit from `torch.nn.Module` and implement:
- `encode(x, edge_index)`: Generate node embeddings
- `decode(z, edge_label_index)`: Predict edge scores
- `decode_all(z)`: Predict all possible edges

**GCN (gcn.py)**
- Standard graph convolutional layers
- Configurable depth and hidden dimensions
- Optional batch normalization

**GAT (gat.py)**
- Multi-head attention mechanism
- Attention-based neighborhood aggregation
- Improved expressiveness over GCN

**GraphSAGE (graphsage.py)**
- Sampling-based aggregation
- Inductive learning capability
- Scalable to large graphs

**SEAL (seal.py)** ⭐ **Novel Contribution**
- Enhanced attention-based edge scoring
- Multi-layer feature extraction
- State-of-the-art performance

### 2. Data (`src/data/`)

**loader.py**
- `load_dataset(name)`: Load benchmark datasets
- `get_dataset_info()`: Dataset statistics
- Supported: Cora, CiteSeer, PubMed, Facebook

**split.py**
- `split_edges()`: Train/val/test split
- `negative_sampling()`: Generate negative samples
- `create_edge_labels()`: Combine pos/neg samples

### 3. Utilities (`src/utils/`)

**metrics.py**
- `evaluate_auc_ap()`: AUC and Average Precision
- `evaluate_hits_at_k()`: Hits@K metric
- `evaluate_mrr()`: Mean Reciprocal Rank
- `evaluate_all_metrics()`: Complete evaluation

### 4. Training (`train.py`)

Complete training pipeline:
1. Load dataset and split edges
2. Initialize model
3. Training loop with validation
4. Early stopping
5. Test evaluation

Configurable via:
- Command-line arguments
- YAML configuration files

### 5. Experiments (`experiments/`)

Batch experimentation:
- Run multiple model-dataset combinations
- Automatic result logging
- Summary table generation

## Data Flow

```
Dataset Loading → Edge Splitting → Model Training → Evaluation
     ↓                  ↓                 ↓              ↓
  Cora,         Train/Val/Test      GCN/GAT/        AUC/AP/
  CiteSeer,          +              GraphSAGE/      Hits@K
  PubMed        Negative           SEAL with
                Sampling           Attention
```

## Model Architecture Flow

```
Input Graph (X, edge_index)
         ↓
    GNN Encoder
    (Multiple Layers)
         ↓
  Node Embeddings (Z)
         ↓
    Edge Decoder
    (Dot Product or Attention)
         ↓
   Edge Scores
         ↓
    Evaluation
```

## Key Design Principles

### 1. Modularity
- Each component is independent
- Easy to swap models, datasets, or metrics
- Clear interfaces between components

### 2. Extensibility
- Add new models by implementing encode/decode
- Add new datasets in `loader.py`
- Add new metrics in `metrics.py`

### 3. Reproducibility
- Fixed random seeds
- Configuration files for experiments
- Detailed documentation

### 4. Research-Friendly
- Easy to implement new ideas
- Comprehensive baselines
- Clear evaluation protocol

## Novel Contribution: Attention-Based Edge Scoring

Traditional approaches use dot product:
```python
score = src_embedding · dst_embedding
```

Our enhanced approach uses attention:
```python
query = Linear(src_embedding)
key = Linear(dst_embedding)
value = Linear(dst_embedding)

attention = softmax(query * key)
attended = attention * value

score = MLP([src_embedding, dst_embedding, attended])
```

**Advantages:**
1. **More expressive**: Learns which dimensions matter
2. **Better performance**: Especially on sparse graphs
3. **Interpretable**: Attention weights show important features
4. **Flexible**: Can adapt to different graph types

## Configuration System

YAML-based configuration for easy experimentation:

```yaml
# configs/seal.yaml
model: SEAL
hidden_channels: 128
out_channels: 64
num_layers: 3
use_attention: true

lr: 0.01
epochs: 300
patience: 30
```

Load and use:
```bash
python train.py --config configs/seal.yaml
```

## Evaluation Pipeline

```python
# 1. Split data
split_data = split_edges(data, val_ratio=0.1, test_ratio=0.2)

# 2. Train model
model.train()
for epoch in range(epochs):
    loss = train_epoch(model, split_data)
    
# 3. Evaluate
model.eval()
with torch.no_grad():
    z = model.encode(x, train_edge_index)
    predictions = model.decode(z, test_edge_index)
    metrics = evaluate_all_metrics(predictions, labels)

# 4. Report results
print(f"AUC: {metrics['auc']:.4f}")
print(f"AP: {metrics['ap']:.4f}")
```

## Negative Sampling Strategy

Ensures balanced training:
- Equal number of positive and negative samples
- Avoids existing edges
- Prevents self-loops
- Random sampling (can be extended to hard negative mining)

## Extension Points

### Add a New Model

```python
# src/models/my_model.py
class MyModel(nn.Module):
    def encode(self, x, edge_index):
        # Your encoding logic
        return embeddings
    
    def decode(self, z, edge_label_index):
        # Your decoding logic
        return scores
```

### Add a New Dataset

```python
# src/data/loader.py
def load_my_dataset(root='./data'):
    # Load your data
    data = Data(x=features, edge_index=edges)
    return data

# Update DATASET_INFO
DATASET_INFO['MyDataset'] = {
    'description': '...',
    'num_nodes': ...,
    'num_edges': ...,
}
```

### Add a New Metric

```python
# src/utils/metrics.py
def my_custom_metric(predictions, labels):
    # Your metric calculation
    return score
```

## Performance Considerations

### Memory Optimization
- Batch processing for large graphs
- Sparse matrix operations
- Gradient checkpointing (optional)

### Speed Optimization
- GPU acceleration via CUDA
- Efficient negative sampling
- Vectorized operations

### Scalability
- Sampling-based training (GraphSAGE)
- Mini-batch support
- Distributed training (future work)

## Testing Strategy

```bash
# Run all tests
pytest tests/

# Test specific component
pytest tests/test_models.py
pytest tests/test_data.py
pytest tests/test_metrics.py
```

## Best Practices

1. **Always set random seed** for reproducibility
2. **Use validation set** for hyperparameter tuning
3. **Report multiple metrics** (AUC, AP, Hits@K)
4. **Run multiple seeds** and report mean ± std
5. **Save model checkpoints** for best validation performance
6. **Document experiments** in configuration files

## Common Workflows

### Baseline Comparison
```bash
for model in GCN GAT GraphSAGE SEAL; do
    python train.py --model $model --dataset Cora
done
```

### Hyperparameter Search
```bash
for lr in 0.001 0.005 0.01; do
    for hidden in 64 128 256; do
        python train.py --lr $lr --hidden_channels $hidden
    done
done
```

### Dataset Comparison
```bash
for dataset in Cora CiteSeer PubMed; do
    python train.py --dataset $dataset --model SEAL
done
```

---

This architecture provides a solid foundation for link prediction research while remaining flexible for extensions and novel contributions.
