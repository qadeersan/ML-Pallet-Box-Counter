"""
YOLOv8 Training Script for Pallet Box Detection

This script trains YOLOv8 object detection models for counting boxes on pallets.

Usage:
    python src/train_yolo.py --model yolov8n --epochs 50
    python src/train_yolo.py --model yolov8s --epochs 50 --batch 16

Arguments:
    --model: Model size (yolov8n, yolov8s, yolov8m)
    --epochs: Number of training epochs
    --batch: Batch size
    --data: Path to data configuration file
    --imgsz: Image size for training
"""

import argparse
from pathlib import Path
import shutil
import torch
from ultralytics import YOLO


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLOv8 for box detection')
    
    parser.add_argument('--model', type=str, default='yolov8n',
                        choices=['yolov8n', 'yolov8s', 'yolov8m'],
                        help='Model size to train')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--data', type=str, default='data/combined_data.yaml',
                        help='Path to data configuration file')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for training')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda, mps, cpu)')
    
    return parser.parse_args()


def get_device():
    """Determine the best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def train_model(args):
    """Train YOLOv8 model with specified parameters."""
    
    # Set up paths
    project_root = Path(__file__).parent.parent
    data_config = project_root / args.data
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Determine device
    device = args.device if args.device else get_device()
    
    print("=" * 60)
    print(f"TRAINING {args.model.upper()}")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch}")
    print(f"Image Size: {args.imgsz}")
    print(f"Device: {device}")
    print(f"Data Config: {data_config}")
    print("=" * 60)
    
    # Initialize model
    model = YOLO(f'{args.model}.pt')
    
    # Train
    results = model.train(
        data=str(data_config),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        device=device,
        project=str(project_root / 'runs' / 'detect'),
        name=f'{args.model}_boxes',
        exist_ok=True,
        verbose=True,
        plots=True,
        save=True,
        val=True
    )
    
    # Copy best model to models directory
    best_model_path = project_root / 'runs' / 'detect' / f'{args.model}_boxes' / 'weights' / 'best.pt'
    target_path = models_dir / f'{args.model}_boxes_best.pt'
    
    if best_model_path.exists():
        shutil.copy(best_model_path, target_path)
        print(f"\n✓ Best model saved to: {target_path}")
    else:
        print(f"\n⚠ Best model not found at: {best_model_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return results


def main():
    """Main entry point."""
    args = parse_args()
    train_model(args)


if __name__ == '__main__':
    main()

