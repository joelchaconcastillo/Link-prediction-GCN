"""Graph Neural Network models for link prediction."""

from .gcn import GCN
from .gat import GAT
from .graphsage import GraphSAGE
from .seal import SEAL

__all__ = ['GCN', 'GAT', 'GraphSAGE', 'SEAL']
