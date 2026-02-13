# Quick Start Guide

Get started with link prediction in just a few minutes!

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/joelchaconcastillo/Link-prediction-GCN.git
cd Link-prediction-GCN

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## 🎯 Run Your First Experiment

### Option 1: Simple Example (Recommended for beginners)

```bash
python example.py
```

This will:
- Load the Cora dataset
- Train a GCN model for 10 epochs
- Show evaluation metrics

**Expected output:**
```
============================================================
Link Prediction Framework - Simple Example
============================================================

Using device: cuda

1. Loading Cora dataset...
   Nodes: 2708
   Edges: 10556
   Features: 1433
   
...

Test Metrics:
  auc: 0.9123
  ap: 0.8956
  hits@10: 0.8534
```

### Option 2: Full Training

```bash
# Train GCN on Cora
python train.py --dataset Cora --model GCN

# Train SEAL (our enhanced model) on CiteSeer
python train.py --dataset CiteSeer --model SEAL

# Use custom configuration
python train.py --config configs/seal.yaml
```

### Option 3: Run All Experiments

```bash
python experiments/run_experiments.py
```

This runs all model-dataset combinations and saves results to `results/`.

## 📊 Understanding the Results

After training, you'll see metrics like:

```
Test Metrics:
  auc: 0.9200        # Area Under ROC Curve (higher is better)
  ap: 0.9050         # Average Precision (higher is better)
  hits@10: 0.8600    # Proportion of true edges in top-10 predictions
  hits@20: 0.9100
  mrr: 0.7800        # Mean Reciprocal Rank
```

### What do these metrics mean?

- **AUC (0.92)**: The model correctly ranks 92% of positive edges higher than negative ones
- **AP (0.91)**: Average precision across different thresholds
- **Hits@10 (0.86)**: 86% of true edges appear in the top-10 predictions

## 🎨 Customization

### Change the Model

```bash
python train.py --model SEAL     # Enhanced model with attention
python train.py --model GAT      # Graph Attention Network
python train.py --model GraphSAGE # GraphSAGE
```

### Try Different Datasets

```bash
python train.py --dataset Cora      # Small citation network
python train.py --dataset CiteSeer  # Medium citation network
python train.py --dataset PubMed    # Large citation network
```

### Adjust Hyperparameters

Edit `configs/default.yaml`:

```yaml
hidden_channels: 256    # Increase model capacity
lr: 0.005              # Lower learning rate
epochs: 500            # Train longer
```

Then run:
```bash
python train.py --config configs/default.yaml
```

## 💻 Use in Your Code

```python
from src.models import GCN, SEAL
from src.data import load_dataset, get_edge_split_data
from src.utils import evaluate_all_metrics

# Load data
data = load_dataset('Cora')
split_data = get_edge_split_data(data)

# Create model
model = SEAL(
    in_channels=data.num_features,
    hidden_channels=128,
    out_channels=64,
    num_layers=3,
    use_attention=True  # Enable attention mechanism
)

# Train (see train.py for complete loop)
# ...

# Evaluate
metrics = evaluate_all_metrics(predictions, labels)
print(f"AUC: {metrics['auc']:.4f}")
```

## 🐛 Troubleshooting

### CUDA Out of Memory?

Reduce batch size in training:
```python
# In train.py, reduce num_pos sampling
num_pos = train_edge_index.size(1) // 4  # Instead of // 2
```

### Slow Training?

Use CPU if GPU is causing issues:
```bash
python train.py --device cpu
```

### Package Import Errors?

Install missing packages:
```bash
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## 📈 Next Steps

1. **Experiment with parameters**: Try different learning rates, hidden dimensions
2. **Compare models**: Run experiments to see which model works best
3. **Add your own data**: Extend the framework with custom datasets
4. **Develop new models**: Implement your own GNN architecture
5. **Publish research**: Use insights to write a paper (see RESEARCH_IDEAS.md)

## 📚 Learn More

- **Full Documentation**: See README.md
- **Research Ideas**: See RESEARCH_IDEAS.md
- **Contributing**: See CONTRIBUTING.md
- **Code Examples**: See `example.py` and `train.py`

## 🎓 Typical Workflow for Research

```bash
# 1. Explore the data
python -c "from src.data import load_dataset, print_dataset_stats; \
           data = load_dataset('Cora'); \
           print_dataset_stats(data, 'Cora')"

# 2. Run quick test
python example.py

# 3. Train baseline models
python train.py --model GCN --dataset Cora
python train.py --model GAT --dataset Cora

# 4. Try the enhanced model
python train.py --model SEAL --dataset Cora

# 5. Run comprehensive experiments
python experiments/run_experiments.py

# 6. Analyze results and iterate on your model
# Edit src/models/seal.py to implement your ideas
# Re-run experiments

# 7. Write paper with results
```

## 💡 Tips

- Start with small datasets (Cora) for quick iteration
- Use `--eval_every 5` to see metrics more frequently
- Save models with `save_model: true` in config
- Use tensorboard for tracking (coming soon)

## 🤝 Need Help?

- Open an issue on GitHub
- Check existing issues for common problems
- Read the full README.md
- Review the code examples

Happy researching! 🎉
