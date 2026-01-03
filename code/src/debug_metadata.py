"""Debug script to verify PEMS08 metadata is correct"""
import sys
sys.path.insert(0, '.')
from data.classifier_data import get_classifier_dataloaders, DATASET_CONFIG

print("Dataset Config:")
for name, cfg in DATASET_CONFIG.items():
    print(f"  {name}: label={cfg['label']}")

print("\nLoading data...")
loaders = get_classifier_dataloaders('../../data/traffic', batch_size=16, max_samples_per_dataset=50)

print("\nChecking test batches for PEMS08 samples...")
for i, (samples, labels, metadata) in enumerate(loaders['test']):
    if i >= 5:
        break
    
    # For each unique label in batch
    for label_val in labels.unique():
        mask = labels == label_val
        label_name = ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08'][label_val.item()]
        meta = metadata[mask]
        print(f"Batch {i}, {label_name} (label={label_val.item()}): metadata samples = {meta[:3].tolist()}")
        
print("\n--- Expected metadata values ---")
print("PEMS03: 358 nodes, 1 feature -> [358, 1]")
print("PEMS04: 307 nodes, 3 features -> [307, 3]")
print("PEMS07: 883 nodes, 1 feature -> [883, 1]")
print("PEMS08: 170 nodes, 3 features -> [170, 3]")
