"""
PyTorch Dataset for Stock Time-Series
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple


class StockTimeSeriesDataset(Dataset):
    """
    Dataset for LSTM training
    
    Pattern: PyTorch-compatible while maintaining project conventions
    """
    
    def __init__(
        self, 
        sequences: np.ndarray, 
        targets: np.ndarray
    ):
        """
        Initialize dataset
        
        Args:
            sequences: (N, sequence_length, features) array
            targets: (N,) target array
        """
        if len(sequences) != len(targets):
            raise ValueError(f"Sequences and targets must have same length: {len(sequences)} vs {len(targets)}")
        
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index
        
        Returns:
            (sequence, target) tuple
        """
        return self.sequences[idx], self.targets[idx]
    
    def get_feature_dim(self) -> int:
        """Get number of features"""
        return self.sequences.shape[2] if len(self.sequences.shape) == 3 else 1
