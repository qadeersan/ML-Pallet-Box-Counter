"""
YOLOv8 Training Script for Google Colab

Upload this script to Colab and run:
    !python colab_training.py

Prerequisites:
    1. Upload dataset.zip to Colab
    2. Run: !unzip dataset.zip
    3. Run: !pip install ultralytics
"""

import os
import shutil
from pathlib import Path

# Install ultralytics if not present
try:
    from ultralytics import YOLO
except ImportError:
    os.system('pip install ultralytics')
    from ultralytics import YOLO

# Configuration
EPOCHS = 30
BATCH_SIZE = 16
IMG_SIZE = 640
PATIENCE = 10

# Create data.yaml for Colab
DATA_YAML = """
path: /content
train:
  - Boxes.v1i.yolov8/train/images
  - Final_Object_Detection.v1i.yolov8/train/images
val:
  - Boxes.v1i.yolov8/valid/images
  - Final_Object_Detection.v1i.yolov8/valid/images
test:
  - Boxes.v1i.yolov8/test/images
  - Final_Object_Detection.v1i.yolov8/test/images
nc: 1
names:
  0: box
"""

def main():
    print("=" * 60)
    print("YOLOV8 TRAINING ON GOOGLE COLAB")
    print("=" * 60)
    
    # Write data.yaml
    with open('/content/data.yaml', 'w') as f:
        f.write(DATA_YAML)
    print("✓ Created /content/data.yaml")
    
    # Verify dataset
    for ds in ['Boxes.v1i.yolov8', 'Final_Object_Detection.v1i.yolov8']:
        train_path = Path(f'/content/{ds}/train/images')
        if train_path.exists():
            count = len(list(train_path.glob('*.jpg')))
            print(f"✓ {ds}: {count} training images")
        else:
            print(f"⚠️ {ds} not found!")
            return
    
    # Create output directory
    os.makedirs('/content/models', exist_ok=True)
    
    # Train YOLOv8n
    print("\n" + "=" * 60)
    print("TRAINING YOLOV8n (Nano)")
    print("=" * 60)
    
    model_n = YOLO('yolov8n.pt')
    model_n.train(
        data='/content/data.yaml',
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        patience=PATIENCE,
        device=0,
        project='/content/runs/detect',
        name='yolov8n_boxes',
        exist_ok=True,
        verbose=True,
        plots=True
    )
    
    # Copy best model
    src_n = '/content/runs/detect/yolov8n_boxes/weights/best.pt'
    dst_n = '/content/models/yolov8n_boxes_best.pt'
    if os.path.exists(src_n):
        shutil.copy(src_n, dst_n)
        print(f"✓ YOLOv8n saved to: {dst_n}")
    
    # Train YOLOv8s
    print("\n" + "=" * 60)
    print("TRAINING YOLOV8s (Small)")
    print("=" * 60)
    
    model_s = YOLO('yolov8s.pt')
    model_s.train(
        data='/content/data.yaml',
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        patience=PATIENCE,
        device=0,
        project='/content/runs/detect',
        name='yolov8s_boxes',
        exist_ok=True,
        verbose=True,
        plots=True
    )
    
    # Copy best model
    src_s = '/content/runs/detect/yolov8s_boxes/weights/best.pt'
    dst_s = '/content/models/yolov8s_boxes_best.pt'
    if os.path.exists(src_s):
        shutil.copy(src_s, dst_s)
        print(f"✓ YOLOv8s saved to: {dst_s}")
    
    # Evaluate both models
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    model_n = YOLO(dst_n)
    metrics_n = model_n.val(data='/content/data.yaml', split='test')
    
    model_s = YOLO(dst_s)
    metrics_s = model_s.val(data='/content/data.yaml', split='test')
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nYOLOv8n:")
    print(f"  mAP@0.5: {metrics_n.box.map50:.4f}")
    print(f"  Precision: {metrics_n.box.mp:.4f}")
    print(f"  Recall: {metrics_n.box.mr:.4f}")
    
    print(f"\nYOLOv8s:")
    print(f"  mAP@0.5: {metrics_s.box.map50:.4f}")
    print(f"  Precision: {metrics_s.box.mp:.4f}")
    print(f"  Recall: {metrics_s.box.mr:.4f}")
    
    # Create zip for download
    os.system('cd /content/models && zip -r /content/trained_models.zip *.pt')
    print("\n✓ Models zipped to /content/trained_models.zip")
    print("\nDownload with: from google.colab import files; files.download('/content/trained_models.zip')")


if __name__ == '__main__':
    main()

