# Installation Verification Guide

This guide helps you verify that the Link Prediction GCN framework is correctly installed and working.

## Step 1: Check Python Version

```bash
python --version
# Required: Python >= 3.7
```

Expected output: `Python 3.x.x` (where x >= 7)

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.3.0
- NumPy, SciPy, scikit-learn
- NetworkX, matplotlib
- Other utilities

**Note:** PyTorch Geometric installation may require additional steps depending on your CUDA version. See [PyTorch Geometric Installation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

## Step 3: Verify Imports

```bash
python -c "from src.models import GCN, GAT, GraphSAGE, SEAL; print('✅ Models imported successfully')"
python -c "from src.data import load_dataset; print('✅ Data utilities imported successfully')"
python -c "from src.utils import evaluate_auc_ap; print('✅ Metrics imported successfully')"
```

All three commands should print success messages.

## Step 4: Run Tests

```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest tests/ -v
```

Expected: All tests pass (or skip if PyTorch Geometric not installed)

## Step 5: Run Quick Example

```bash
python example.py
```

This should:
1. Load the Cora dataset
2. Split edges into train/val/test
3. Train a GCN model for 10 epochs
4. Show evaluation metrics

**Expected runtime:** 1-2 minutes on CPU, faster on GPU

**Expected output:**
```
============================================================
Link Prediction Framework - Simple Example
============================================================

Using device: cpu (or cuda)

1. Loading Cora dataset...
   Nodes: 2708
   Edges: 10556
   ...

Test Metrics:
  auc: 0.XXXX
  ap: 0.XXXX
  ...
```

## Step 6: Train a Model

```bash
python train.py --dataset Cora --model GCN --epochs 50
```

This performs a full training run with:
- Train/validation/test split
- Early stopping
- Metric evaluation

**Expected runtime:** 2-5 minutes

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
pip install torch>=2.0.0
```

**Problem:** `ModuleNotFoundError: No module named 'torch_geometric'`

**Solution:**
```bash
pip install torch-geometric
# Or follow: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
```

### CUDA Issues

**Problem:** `RuntimeError: CUDA out of memory`

**Solution:** Use CPU or reduce model size
```bash
python train.py --device cpu
# Or reduce hidden dimensions
python train.py --hidden_channels 64
```

**Problem:** CUDA version mismatch

**Solution:** Install PyTorch for your CUDA version
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch
# See: https://pytorch.org/get-started/locally/
```

### Dataset Download Issues

**Problem:** Network errors when downloading datasets

**Solution:** 
- Datasets are downloaded automatically to `./data/`
- If download fails, retry or download manually
- Check internet connection

### Performance Issues

**Problem:** Training is very slow

**Solutions:**
1. Use GPU: `python train.py --device cuda`
2. Reduce dataset size: Use Cora instead of PubMed
3. Reduce model size: `--hidden_channels 64`
4. Reduce epochs: `--epochs 50`

## Verification Checklist

- [ ] Python 3.7+ installed
- [ ] Dependencies installed successfully
- [ ] All imports work
- [ ] Tests pass (or skip gracefully)
- [ ] Example script runs without errors
- [ ] Training script works
- [ ] Can load at least one dataset
- [ ] GPU works (if available)

## Success!

If all steps complete successfully, your installation is verified! 🎉

Next steps:
- Try different models: `python train.py --model SEAL`
- Experiment with datasets: `python train.py --dataset CiteSeer`
- Run full experiments: `python experiments/run_experiments.py`
- Read documentation: `README.md`, `QUICKSTART.md`

## Getting Help

If you encounter issues:

1. Check the error message carefully
2. Review this troubleshooting guide
3. Check the [PyTorch Geometric Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
4. Open an issue on GitHub with:
   - Python version
   - PyTorch version
   - Error message
   - Steps to reproduce

## Alternative Installation (Conda)

If you prefer conda:

```bash
# Create environment
conda create -n linkpred python=3.9
conda activate linkpred

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install PyTorch Geometric
conda install pyg -c pyg

# Install other dependencies
pip install networkx scikit-learn pyyaml tqdm pandas matplotlib
```

## Docker Installation (Advanced)

For a containerized setup:

```bash
# Build image
docker build -t link-prediction-gcn .

# Run container
docker run -it --gpus all link-prediction-gcn python example.py
```

(Note: Dockerfile not included yet - can be added if needed)

---

**Happy researching!** 🔬
