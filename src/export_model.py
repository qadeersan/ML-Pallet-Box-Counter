"""
Model Export Utility

This script exports trained models to various formats including pickle (.pkl).

Usage:
    python src/export_model.py --model models/yolov8n_boxes_best.pt --format pkl
    python src/export_model.py --model models/resnet18_regression_best.pt --format pkl

Supported formats:
    - pkl: Pickle format (for sklearn-style workflows)
    - onnx: ONNX format (for cross-platform deployment)
    - torchscript: TorchScript (for production PyTorch)
"""

import argparse
import pickle
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def export_yolo_to_pickle(model_path, output_path=None):
    """Export YOLOv8 model metadata to pickle (model itself stays as .pt)."""
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    
    # Create metadata dict
    model_data = {
        'model_type': 'YOLOv8',
        'model_path': str(model_path),
        'task': 'detect',
        'names': model.names,
        'nc': len(model.names),
        'imgsz': 640,
        'conf_threshold': 0.4,
        'info': 'Load with: from ultralytics import YOLO; model = YOLO(model_path)'
    }
    
    if output_path is None:
        output_path = Path(model_path).with_suffix('.pkl')
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✓ Exported YOLOv8 metadata to: {output_path}")
    print(f"  Note: YOLOv8 models must be loaded from .pt files")
    print(f"  The .pkl file contains metadata only")
    
    return output_path


def export_regression_to_pickle(model_path, output_path=None):
    """Export PyTorch regression model to pickle."""
    import torch
    
    # Load state dict
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Convert tensors to numpy for pickle compatibility
    model_data = {
        'model_type': 'ResNet18_Regression',
        'state_dict': {k: v.numpy() for k, v in state_dict.items()},
        'config': {
            'backbone': 'resnet18',
            'dropout': 0.3,
            'input_size': 224,
            'output': 'box_count'
        },
        'preprocessing': {
            'resize': (224, 224),
            'normalize_mean': [0.485, 0.456, 0.406],
            'normalize_std': [0.229, 0.224, 0.225]
        }
    }
    
    if output_path is None:
        output_path = Path(model_path).with_suffix('.pkl')
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✓ Exported regression model to: {output_path}")
    
    return output_path


def export_to_onnx(model_path, output_path=None):
    """Export YOLOv8 model to ONNX format."""
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    
    if output_path is None:
        output_path = Path(model_path).with_suffix('.onnx')
    
    model.export(format='onnx', imgsz=640)
    print(f"✓ Exported to ONNX: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Export models to various formats')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--format', type=str, default='pkl', 
                        choices=['pkl', 'onnx', 'torchscript'],
                        help='Export format')
    parser.add_argument('--output', type=str, default=None, help='Output path')
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)
    
    # Determine model type and export
    if 'yolo' in model_path.stem.lower():
        if args.format == 'pkl':
            export_yolo_to_pickle(model_path, args.output)
        elif args.format == 'onnx':
            export_to_onnx(model_path, args.output)
    elif 'regression' in model_path.stem.lower() or 'resnet' in model_path.stem.lower():
        if args.format == 'pkl':
            export_regression_to_pickle(model_path, args.output)
    else:
        print(f"Unknown model type. Attempting generic pickle export...")
        export_regression_to_pickle(model_path, args.output)


if __name__ == '__main__':
    main()

