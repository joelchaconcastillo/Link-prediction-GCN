# Contributing to Link-prediction-GCN

Thank you for considering contributing to this project! This document provides guidelines and suggestions for contributing.

## 🎯 Areas for Contribution

### 1. New GNN Architectures

Add new graph neural network models:
- Graph Isomorphism Network (GIN)
- Principal Neighbourhood Aggregation (PNA)
- Message Passing Neural Networks (MPNN)
- Custom architectures

**How to add a new model:**
1. Create a new file in `src/models/` (e.g., `gin.py`)
2. Implement the model class with `encode()` and `decode()` methods
3. Add import to `src/models/__init__.py`
4. Create a config file in `configs/`
5. Test with `train.py`

### 2. Additional Datasets

Extend dataset support:
- Open Graph Benchmark (OGB) datasets
- Custom domain-specific graphs
- Temporal/dynamic graphs
- Heterogeneous graphs

**How to add a dataset:**
1. Add loader function to `src/data/loader.py`
2. Update `DATASET_INFO` dictionary
3. Ensure compatibility with edge splitting

### 3. Advanced Features

Potential enhancements:
- Temporal link prediction
- Inductive link prediction
- Explainability tools (attention visualization)
- Multi-relational link prediction
- Hyperparameter optimization (Optuna integration)
- Distributed training support

### 4. Evaluation Metrics

Additional metrics to implement:
- Precision@K, Recall@K
- NDCG (Normalized Discounted Cumulative Gain)
- Time-aware metrics for temporal graphs
- Domain-specific metrics

### 5. Negative Sampling Strategies

More sophisticated sampling:
- Hard negative mining
- Adversarial negative sampling
- Structure-aware sampling
- Importance sampling

## 💻 Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Link-prediction-GCN.git
cd Link-prediction-GCN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

## 🧪 Testing

Before submitting a PR:

```bash
# Run tests (when available)
pytest tests/

# Check code style
black src/ train.py example.py
flake8 src/ train.py example.py

# Type checking
mypy src/
```

## 📝 Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and modular
- Use descriptive variable names

**Example:**

```python
def evaluate_model(
    model: nn.Module,
    data: Data,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on given edge set.
    
    Args:
        model: Trained GNN model
        data: Graph data object
        edge_index: Edges to evaluate
        labels: Ground truth labels
        device: Computing device
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Implementation
    pass
```

## 🔄 Pull Request Process

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** with clear, focused commits
3. **Add tests** if applicable
4. **Update documentation** (README, docstrings, etc.)
5. **Ensure code passes** linting and tests
6. **Submit PR** with clear description of changes

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How has this been tested?

## Checklist
- [ ] Code follows project style
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests added/updated
```

## 📚 Documentation

When adding features:
- Update README.md with usage examples
- Add docstrings to all public functions/classes
- Create config examples if needed
- Update experiment scripts if applicable

## 🐛 Bug Reports

Good bug reports include:
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- System info (OS, Python version, PyTorch version)
- Error messages/stack traces
- Minimal code example

## 💡 Feature Requests

For feature requests:
- Check existing issues first
- Describe the problem you're trying to solve
- Propose potential solutions
- Consider implementation complexity
- Discuss potential impact on existing code

## 📖 Research Ideas

See `RESEARCH_IDEAS.md` for potential research directions and paper ideas.

## 🤝 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions
- Help others learn and grow

## 📞 Questions?

- Open an issue for general questions
- Use discussions for broader topics
- Tag maintainers for urgent matters

## 🎓 Academic Contributions

If your contribution leads to a publication:
- Cite this repository appropriately
- Consider adding your paper to a "Papers Using This Framework" section
- Share your results with the community

---

Thank you for contributing! 🙏
