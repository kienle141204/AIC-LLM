"""
Training script for Domain Classifier.
Trains the classifier to identify which PEMS dataset a sample belongs to.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
import numpy as np

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.domain_classifier import DomainClassifier, LABEL_TO_DATASET
from data.classifier_data import get_classifier_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='Train Domain Classifier')
    
    # Data
    parser.add_argument('--data_path', type=str, default='../../data/traffic',
                        help='Path to traffic data directory')
    parser.add_argument('--sample_len', type=int, default=12,
                        help='Length of each sample (timesteps)')
    parser.add_argument('--step', type=int, default=12,
                        help='Step size for sliding window')
    parser.add_argument('--max_samples', type=int, default=2000,
                        help='Max samples per dataset (for balance)')
    
    # Model
    parser.add_argument('--common_nodes', type=int, default=256,
                        help='Common node dimension after projection')
    parser.add_argument('--common_features', type=int, default=16,
                        help='Common feature dimension after projection')
    parser.add_argument('--temporal_hidden', type=int, default=64,
                        help='Hidden dimension for temporal encoder')
    parser.add_argument('--classifier_hidden', type=int, default=128,
                        help='Hidden dimension for classifier')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay (increased for regularization)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Output
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    use_amp = scaler is not None
    
    for batch_idx, (samples, labels, _metadata) in enumerate(loader):
        samples = samples.to(device)
        labels = labels.to(device)
        # metadata is no longer used - model learns from data content
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast(enabled=use_amp):
            outputs = model(samples)  # No metadata needed!
            loss = criterion(outputs, labels)
        
        # Mixed precision backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # Per-class metrics
    class_correct = [0] * 4
    class_total = [0] * 4
    
    with torch.no_grad():
        for samples, labels, _metadata in loader:
            samples = samples.to(device)
            labels = labels.to(device)
            # metadata is no longer used
            
            outputs = model(samples)  # No metadata needed!
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Per-class accuracy
            for i in range(4):
                mask = labels == i
                class_correct[i] += predicted[mask].eq(labels[mask]).sum().item()
                class_total[i] += mask.sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    
    # Per-class accuracy
    class_acc = {}
    for i in range(4):
        if class_total[i] > 0:
            class_acc[LABEL_TO_DATASET[i]] = 100.0 * class_correct[i] / class_total[i]
        else:
            class_acc[LABEL_TO_DATASET[i]] = 0.0
    
    return avg_loss, accuracy, class_acc


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Data
    print("\nLoading data...")
    loaders = get_classifier_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        sample_len=args.sample_len,
        max_samples_per_dataset=args.max_samples,
        seed=args.seed
    )
    
    # Model - Hybrid classifier (exact match + similarity fallback)
    print("\nCreating model...")
    model = DomainClassifier().to(device)  # No training params needed!
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("Note: HybridDomainClassifier uses rule-based matching, minimal training needed.")
    
    # Loss and optimizer (with label smoothing for regularization)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Training
    print("\nStarting training...")
    best_val_acc = 0
    patience_counter = 0
    
    # Mixed precision scaler for faster training
    scaler = GradScaler() if device.type == 'cuda' else None
    print(f"Mixed precision (FP16): {'Enabled' if scaler else 'Disabled'}")
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, loaders['train'], criterion, optimizer, device, scaler
        )
        
        # Validate
        val_loss, val_acc, _ = evaluate(
            model, loaders['val'], criterion, device
        )
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': vars(args)
            }
            save_path = os.path.join(args.save_dir, 'domain_classifier_best.pth')
            torch.save(checkpoint, save_path)
            print(f"  -> Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model for testing
    print("\nLoading best model for testing...")
    checkpoint = torch.load(os.path.join(args.save_dir, 'domain_classifier_best.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test
    test_loss, test_acc, class_acc = evaluate(
        model, loaders['test'], criterion, device
    )
    
    print(f"\n{'='*50}")
    print(f"Test Results:")
    print(f"  Overall Accuracy: {test_acc:.2f}%")
    print(f"  Per-class Accuracy:")
    for dataset, acc in class_acc.items():
        print(f"    {dataset}: {acc:.2f}%")
    print(f"{'='*50}")
    
    # Save final results
    result_path = os.path.join(args.save_dir, 'training_results.txt')
    with open(result_path, 'w') as f:
        f.write(f"Training completed at: {datetime.now()}\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Per-class Accuracy:\n")
        for dataset, acc in class_acc.items():
            f.write(f"  {dataset}: {acc:.2f}%\n")
        f.write(f"\nArgs: {vars(args)}\n")
    
    print(f"\nResults saved to {result_path}")
    print(f"Model saved to {os.path.join(args.save_dir, 'domain_classifier_best.pth')}")


if __name__ == "__main__":
    main()
