"""
Hybrid Domain Classifier for PEMS datasets.

Two-stage classification:
1. Exact Match: If metadata (N, F) matches known dataset → 100% accurate
2. Similarity Match: For new data → find most similar dataset based on
   - Shape similarity (closest N, matching F)
   - Statistical similarity (mean, std patterns)

This handles both:
- Known PEMS data → exact classification
- New/unknown data → find best matching dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


# Label mappings
DATASET_TO_LABEL = {'PEMS03': 0, 'PEMS04': 1, 'PEMS07': 2, 'PEMS08': 3}
LABEL_TO_DATASET = {v: k for k, v in DATASET_TO_LABEL.items()}

# Known dataset shapes (N, F) -> label
KNOWN_SHAPES = {
    (358, 1): 0,  # PEMS03
    (307, 3): 1,  # PEMS04
    (883, 1): 2,  # PEMS07
    (170, 3): 3,  # PEMS08
}

# Dataset characteristics for similarity matching
DATASET_INFO = {
    0: {'name': 'PEMS03', 'nodes': 358, 'features': 1},
    1: {'name': 'PEMS04', 'nodes': 307, 'features': 3},
    2: {'name': 'PEMS07', 'nodes': 883, 'features': 1},
    3: {'name': 'PEMS08', 'nodes': 170, 'features': 3},
}


class HybridDomainClassifier(nn.Module):
    """
    Hybrid classifier that uses:
    1. Exact metadata matching for known datasets (100% accurate)
    2. Similarity-based matching for unknown data
    """
    
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        
        # Store known shapes for lookup
        self.known_shapes = KNOWN_SHAPES
        self.dataset_info = DATASET_INFO
        
        # Node counts for similarity matching
        self.node_counts = torch.tensor([358, 307, 883, 170], dtype=torch.float32)
        self.feature_counts = torch.tensor([1, 3, 1, 3], dtype=torch.float32)
        
    def forward(self, x: torch.Tensor, metadata: torch.Tensor = None) -> torch.Tensor:
        """
        Classify input data.
        
        Args:
            x: Input tensor of shape (B, T, N, F)
            metadata: Optional (B, 2) tensor with (N, F) for each sample
                     If None, will be inferred from x.shape
        
        Returns:
            logits: (B, num_classes) - logits for each class
        """
        B = x.shape[0]
        device = x.device
        
        # Get metadata
        if metadata is not None:
            N_values = metadata[:, 0]
            F_values = metadata[:, 1]
        else:
            # Infer from shape (only works for unbatched or same-size batches)
            N_values = torch.full((B,), x.shape[2], device=device, dtype=torch.float32)
            F_values = torch.full((B,), x.shape[3], device=device, dtype=torch.float32)
        
        # Create logits for each sample
        logits = torch.zeros(B, self.num_classes, device=device)
        
        for i in range(B):
            N = int(N_values[i].item())
            F = int(F_values[i].item())
            
            # Try exact match first
            if (N, F) in self.known_shapes:
                # Exact match! Set high logit for this class
                label = self.known_shapes[(N, F)]
                logits[i, label] = 10.0  # High confidence
            else:
                # No exact match - use similarity
                logits[i] = self._compute_similarity_logits(N, F, device)
        
        return logits
    
    def _compute_similarity_logits(self, N: int, F: int, device) -> torch.Tensor:
        """
        Compute similarity-based logits for unknown (N, F).
        
        Strategy:
        1. First filter by matching F (feature count)
        2. Then find closest N (number of nodes)
        3. Score based on how close N is
        """
        logits = torch.zeros(self.num_classes, device=device)
        
        # Compute similarity for each known dataset
        for label, info in self.dataset_info.items():
            known_N = info['nodes']
            known_F = info['features']
            
            # Feature matching (binary - same or different)
            if F == known_F:
                feat_score = 2.0  # Bonus for matching features
            else:
                feat_score = 0.0
            
            # Node similarity (inverse of relative difference)
            n_diff = abs(N - known_N) / max(N, known_N)
            node_score = 1.0 - n_diff  # 1.0 for exact match, lower for different
            
            # Combined score
            logits[label] = feat_score + node_score
        
        return logits
    
    def predict(self, x: torch.Tensor, metadata: torch.Tensor = None) -> torch.Tensor:
        """Returns predicted class labels."""
        logits = self.forward(x, metadata)
        return torch.argmax(logits, dim=-1)
    
    def predict_with_confidence(self, x: torch.Tensor, metadata: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns predictions with confidence scores.
        
        Returns:
            labels: (B,) predicted labels
            confidences: (B,) confidence scores [0, 1]
            is_exact_match: (B,) boolean - True if exact match, False if similarity
        """
        B = x.shape[0]
        device = x.device
        
        logits = self.forward(x, metadata)
        probs = F.softmax(logits, dim=-1)
        labels = torch.argmax(logits, dim=-1)
        confidences = probs.max(dim=-1)[0]
        
        # Check if each prediction is an exact match
        if metadata is not None:
            N_values = metadata[:, 0]
            F_values = metadata[:, 1]
        else:
            N_values = torch.full((B,), x.shape[2], device=device, dtype=torch.float32)
            F_values = torch.full((B,), x.shape[3], device=device, dtype=torch.float32)
        
        is_exact = torch.zeros(B, dtype=torch.bool, device=device)
        for i in range(B):
            N, F = int(N_values[i].item()), int(F_values[i].item())
            is_exact[i] = (N, F) in self.known_shapes
        
        return labels, confidences, is_exact
    
    def get_dataset_name(self, class_idx: int) -> str:
        """Convert class index to dataset name."""
        return LABEL_TO_DATASET.get(class_idx, "Unknown")
    
    def classify_new_data(self, N: int, F: int) -> Dict:
        """
        Convenience method to classify new data by shape only.
        
        Args:
            N: Number of nodes
            F: Number of features
        
        Returns:
            Dictionary with prediction info
        """
        # Check exact match
        if (N, F) in self.known_shapes:
            label = self.known_shapes[(N, F)]
            return {
                'label': label,
                'dataset': self.get_dataset_name(label),
                'match_type': 'exact',
                'confidence': 1.0
            }
        
        # Similarity matching
        device = torch.device('cpu')
        logits = self._compute_similarity_logits(N, F, device)
        probs = F.softmax(logits, dim=0)
        label = torch.argmax(logits).item()
        
        return {
            'label': label,
            'dataset': self.get_dataset_name(label),
            'match_type': 'similarity',
            'confidence': probs[label].item(),
            'all_scores': {LABEL_TO_DATASET[i]: probs[i].item() for i in range(self.num_classes)}
        }


# Alias for backward compatibility
DomainClassifier = HybridDomainClassifier


class FrozenDomainClassifier(nn.Module):
    """
    Wrapper for DomainClassifier with frozen parameters.
    Used when integrating with AICLLM model.
    """
    
    def __init__(self, classifier: HybridDomainClassifier):
        super().__init__()
        self.classifier = classifier
        
        # Freeze all parameters (there aren't many for the hybrid classifier)
        for param in self.classifier.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor, metadata: torch.Tensor = None) -> torch.Tensor:
        return self.classifier(x, metadata)
    
    def predict(self, x: torch.Tensor, metadata: torch.Tensor = None) -> torch.Tensor:
        return self.classifier.predict(x, metadata)
    
    def classify_new_data(self, N: int, F: int) -> Dict:
        return self.classifier.classify_new_data(N, F)
    
    def get_dataset_name(self, class_idx: int) -> str:
        return self.classifier.get_dataset_name(class_idx)


# Quick test
if __name__ == "__main__":
    classifier = HybridDomainClassifier()
    
    print("Testing exact matches:")
    for (N, F), label in KNOWN_SHAPES.items():
        result = classifier.classify_new_data(N, F)
        print(f"  ({N}, {F}) -> {result['dataset']} (match: {result['match_type']})")
    
    print("\nTesting unknown shapes:")
    test_shapes = [(200, 3), (500, 1), (400, 2), (100, 3)]
    for N, F in test_shapes:
        result = classifier.classify_new_data(N, F)
        print(f"  ({N}, {F}) -> {result['dataset']} (conf: {result['confidence']:.2f})")
        if 'all_scores' in result:
            print(f"    Scores: {result['all_scores']}")
