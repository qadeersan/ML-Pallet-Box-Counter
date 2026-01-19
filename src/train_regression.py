"""
CNN Regression Training Script for Box Counting

This script trains a ResNet18-based regression model to predict box counts.
This serves as a baseline comparison to YOLOv8 object detection.

Usage:
    python src/train_regression.py --epochs 30 --batch 32
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
DOCS_DIR = PROJECT_ROOT / 'docs'


class BoxCountDataset(Dataset):
    """Dataset for box counting regression."""
    
    def __init__(self, data_dirs, split='train', transform=None):
        self.transform = transform
        self.samples = []
        
        split_map = {'train': 'train', 'valid': 'valid', 'val': 'valid', 'test': 'test'}
        actual_split = split_map.get(split, split)
        
        for data_dir in data_dirs:
            data_path = Path(data_dir)
            images_dir = data_path / actual_split / 'images'
            labels_dir = data_path / actual_split / 'labels'
            
            if not images_dir.exists():
                print(f"  Warning: {images_dir} not found")
                continue
            
            for img_path in images_dir.glob('*.jpg'):
                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        box_count = len([l for l in f.readlines() if l.strip()])
                    self.samples.append((img_path, box_count))
        
        print(f"  Loaded {len(self.samples)} samples from {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, box_count = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(box_count, dtype=torch.float32)


class BoxCountRegressor(nn.Module):
    """CNN Regression model for predicting box counts."""
    
    def __init__(self, backbone='resnet18', dropout=0.3):
        super().__init__()
        
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            num_features = self.backbone.fc.in_features
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.backbone(x).squeeze(-1)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Evaluating', leave=False):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    mae = np.mean(np.abs(all_preds - all_targets))
    exact_match = np.mean(np.round(all_preds) == all_targets)
    off_by_1 = np.mean(np.abs(np.round(all_preds) - all_targets) <= 1)
    off_by_3 = np.mean(np.abs(np.round(all_preds) - all_targets) <= 3)
    
    return {
        'loss': total_loss / len(loader),
        'mae': mae,
        'count_accuracy': exact_match,
        'off_by_1': off_by_1,
        'off_by_3': off_by_3,
        'predictions': all_preds,
        'targets': all_targets
    }


def main():
    parser = argparse.ArgumentParser(description='Train CNN regression for box counting')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda, mps, cpu)')
    args = parser.parse_args()
    
    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print("=" * 60)
    print("CNN REGRESSION TRAINING FOR BOX COUNTING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 60)
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    data_dirs = [
        PROJECT_ROOT / 'Boxes.v1i.yolov8',
        PROJECT_ROOT / 'Final_Object_Detection.v1i.yolov8'
    ]
    
    print("\nLoading datasets...")
    train_dataset = BoxCountDataset(data_dirs, split='train', transform=train_transform)
    val_dataset = BoxCountDataset(data_dirs, split='valid', transform=val_transform)
    test_dataset = BoxCountDataset(data_dirs, split='test', transform=val_transform)
    
    # DataLoaders (num_workers=0 to avoid multiprocessing issues)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    
    print(f"\nTrain: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Valid: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    # Model
    model = BoxCountRegressor(backbone='resnet18', dropout=0.3)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: ResNet18 Regression")
    print(f"Parameters: {total_params:,}")
    
    # Training setup
    criterion = nn.L1Loss()  # MAE
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_accuracy': []}
    best_val_mae = float('inf')
    
    MODELS_DIR.mkdir(exist_ok=True)
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_accuracy'].append(val_metrics['count_accuracy'])
        
        # Save best model
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            torch.save(model.state_dict(), MODELS_DIR / 'resnet18_regression_best.pt')
            
            # Also save as pickle
            import pickle
            model_data = {
                'state_dict': {k: v.cpu().numpy() for k, v in model.state_dict().items()},
                'config': {'backbone': 'resnet18', 'dropout': 0.3, 'input_size': 224}
            }
            with open(MODELS_DIR / 'resnet18_regression_best.pkl', 'wb') as f:
                pickle.dump(model_data, f)
        
        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val MAE: {val_metrics['mae']:.2f} | "
              f"Val Acc: {val_metrics['count_accuracy'] * 100:.1f}%")
    
    # Test evaluation
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    
    model.load_state_dict(torch.load(MODELS_DIR / 'resnet18_regression_best.pt', weights_only=True))
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  MAE: {test_metrics['mae']:.2f}")
    print(f"  Count Accuracy: {test_metrics['count_accuracy'] * 100:.1f}%")
    print(f"  Off-by-1 Accuracy: {test_metrics['off_by_1'] * 100:.1f}%")
    print(f"  Off-by-3 Accuracy: {test_metrics['off_by_3'] * 100:.1f}%")
    
    # Save summary
    DOCS_DIR.mkdir(exist_ok=True)
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model': 'ResNet18',
        'epochs': args.epochs,
        'batch_size': args.batch,
        'learning_rate': args.lr,
        'device': str(device),
        'test_mae': float(test_metrics['mae']),
        'test_count_accuracy': float(test_metrics['count_accuracy']),
        'test_off_by_1': float(test_metrics['off_by_1']),
        'test_off_by_3': float(test_metrics['off_by_3']),
        'best_val_mae': float(best_val_mae)
    }
    
    with open(DOCS_DIR / 'regression_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nModels saved to:")
    print(f"  • {MODELS_DIR / 'resnet18_regression_best.pt'}")
    print(f"  • {MODELS_DIR / 'resnet18_regression_best.pkl'}")
    print(f"\nSummary saved to:")
    print(f"  • {DOCS_DIR / 'regression_summary.json'}")


if __name__ == '__main__':
    main()

