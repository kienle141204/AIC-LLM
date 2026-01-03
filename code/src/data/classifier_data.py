"""
Data loader for Domain Classifier training.
Loads data from all 4 PEMS datasets and creates labeled samples for classification.
Simple and fast loading - directly from .npz files.
"""

import torch
import torch.utils.data as data
import numpy as np
from typing import Dict, List, Tuple, Optional
import os


# Dataset configurations
DATASET_CONFIG = {
    'PEMS03': {'label': 0, 'file': 'PEMS03/PEMS03.npz', 'node_count': 358, 'feature_count': 1},
    'PEMS04': {'label': 1, 'file': 'PEMS04/PEMS04.npz', 'node_count': 307, 'feature_count': 3},
    'PEMS07': {'label': 2, 'file': 'PEMS07/PEMS07.npz', 'node_count': 883, 'feature_count': 1},
    'PEMS08': {'label': 3, 'file': 'PEMS08/PEMS08.npz', 'node_count': 170, 'feature_count': 3},
}


class ClassifierDataset(data.Dataset):
    """Dataset for domain classification task."""
    
    def __init__(
        self,
        samples: List[torch.Tensor],
        labels: List[int],
        metadata: List[Tuple[int, int]],
    ):
        self.samples = samples
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.metadata = metadata
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        return self.samples[idx], self.labels[idx], self.metadata[idx]


def collate_fn(batch):
    """Custom collate function to handle variable-sized inputs."""
    samples, labels, metadata = zip(*batch)
    
    # Find max nodes and max features in this batch
    max_nodes = max(s.shape[1] for s in samples)
    max_features = max(s.shape[2] for s in samples)
    T = samples[0].shape[0]
    
    # Pad samples
    padded_samples = []
    for s in samples:
        T_s, N, F = s.shape
        
        # Pad features if needed
        if F < max_features:
            feat_padding = torch.zeros(T_s, N, max_features - F, dtype=s.dtype)
            s = torch.cat([s, feat_padding], dim=2)
        
        # Pad nodes if needed
        if N < max_nodes:
            node_padding = torch.zeros(T_s, max_nodes - N, max_features, dtype=s.dtype)
            s = torch.cat([s, node_padding], dim=1)
            
        padded_samples.append(s)
    
    batch_samples = torch.stack(padded_samples, dim=0)  # (B, T, N, F)
    batch_labels = torch.stack(list(labels), dim=0)
    batch_metadata = torch.tensor(metadata, dtype=torch.float32)
    
    return batch_samples, batch_labels, batch_metadata


def load_pems_data(
    data_path: str,
    sample_len: int = 12,
    step: int = 12,
    max_samples_per_dataset: Optional[int] = None
) -> Tuple[List[torch.Tensor], List[int], List[Tuple[int, int]]]:
    """Load data from all PEMS datasets - simple and fast."""
    samples = []
    labels = []
    metadata_list = []
    
    for dataset_name, config in DATASET_CONFIG.items():
        file_path = os.path.join(data_path, config['file'])
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping")
            continue
        
        # Load data directly from npz
        data_npz = np.load(file_path)
        data_array = data_npz['data'].astype(np.float32)  # (T_total, N, F)
        
        T_total, N, F = data_array.shape
        
        # Normalize (min-max per feature)
        for f in range(F):
            feat = data_array[:, :, f]
            min_val, max_val = feat.min(), feat.max()
            if max_val > min_val:
                data_array[:, :, f] = (feat - min_val) / (max_val - min_val)
        
        # Create samples using sliding window
        count = 0
        for start in range(0, T_total - sample_len + 1, step):
            if max_samples_per_dataset and count >= max_samples_per_dataset:
                break
                
            sample = data_array[start:start + sample_len]  # (T, N, F)
            
            samples.append(torch.tensor(sample, dtype=torch.float32))
            labels.append(config['label'])
            metadata_list.append((config['node_count'], config['feature_count']))
            count += 1
        
        print(f"Loaded {count} samples from {dataset_name}")
    
    return samples, labels, metadata_list


def split_data(
    samples: List[torch.Tensor],
    labels: List[int],
    metadata: List[Tuple[int, int]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
):
    """Split data into train/val/test sets, stratified by label."""
    np.random.seed(seed)
    
    # Group by label
    label_to_indices = {}
    for i, label in enumerate(labels):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(i)
    
    train_indices, val_indices, test_indices = [], [], []
    
    for label, indices in label_to_indices.items():
        np.random.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])
    
    train_data = ([samples[i] for i in train_indices], 
                  [labels[i] for i in train_indices],
                  [metadata[i] for i in train_indices])
    val_data = ([samples[i] for i in val_indices],
                [labels[i] for i in val_indices], 
                [metadata[i] for i in val_indices])
    test_data = ([samples[i] for i in test_indices],
                 [labels[i] for i in test_indices],
                 [metadata[i] for i in test_indices])
    
    return train_data, val_data, test_data


def get_classifier_dataloaders(
    data_path: str,
    batch_size: int = 32,
    sample_len: int = 12,
    step: int = 12,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_samples_per_dataset: Optional[int] = 2000,
    num_workers: int = 0,
    seed: int = 42
) -> Dict[str, data.DataLoader]:
    """Get data loaders for classifier training."""
    
    # Load all data
    samples, labels, metadata = load_pems_data(
        data_path, sample_len, step, max_samples_per_dataset
    )
    
    # Split
    train_data, val_data, test_data = split_data(
        samples, labels, metadata, train_ratio, val_ratio, seed
    )
    
    # Create datasets
    train_dataset = ClassifierDataset(*train_data)
    val_dataset = ClassifierDataset(*val_data)
    test_dataset = ClassifierDataset(*test_data)
    
    print(f"\nDataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create dataloaders
    loaders = {
        'train': data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=num_workers, drop_last=True
        ),
        'val': data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers
        ),
        'test': data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers
        )
    }
    
    return loaders


if __name__ == "__main__":
    loaders = get_classifier_dataloaders(
        "../../data/traffic", batch_size=8, sample_len=12, max_samples_per_dataset=50
    )
    
    for split, loader in loaders.items():
        batch = next(iter(loader))
        samples, labels, metadata = batch
        print(f"{split}: samples={samples.shape}, labels={labels.tolist()}, meta={metadata[:3].tolist()}")
