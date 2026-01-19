"""
Inference Script for Pallet Box Counting

This script performs inference on images using trained YOLOv8 models.

Usage:
    python src/predict.py --image path/to/image.jpg
    python src/predict.py --image path/to/image.jpg --model models/yolov8s_boxes_best.pt
    python src/predict.py --image path/to/image.jpg --conf 0.5 --show

Arguments:
    --image: Path to input image
    --model: Path to trained model weights
    --conf: Confidence threshold for detections
    --show: Display the annotated image
    --save: Save the annotated image
"""

import argparse
from pathlib import Path
import json
import cv2
from ultralytics import YOLO


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict box count from image')
    
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model', type=str, default='models/yolov8s_boxes_best.pt',
                        help='Path to trained model weights')
    parser.add_argument('--conf', type=float, default=0.4,
                        help='Confidence threshold for detections')
    parser.add_argument('--show', action='store_true',
                        help='Display the annotated image')
    parser.add_argument('--save', action='store_true',
                        help='Save the annotated image')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for annotated image')
    
    return parser.parse_args()


def predict(image_path, model_path='models/yolov8s_boxes_best.pt', conf_threshold=0.4):
    """
    Predict box count from an image.
    
    Args:
        image_path: Path to input image
        model_path: Path to trained model weights
        conf_threshold: Confidence threshold for detections
        
    Returns:
        dict with prediction results
    """
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(image_path, conf=conf_threshold, verbose=False)
    
    # Extract results
    boxes = results[0].boxes
    
    if boxes is not None and len(boxes) > 0:
        detections = []
        for i, box in enumerate(boxes):
            detections.append({
                'id': i,
                'bbox': box.xyxy[0].tolist(),
                'confidence': float(box.conf[0]),
                'class': int(box.cls[0])
            })
        
        return {
            'box_count': len(boxes),
            'confidence_threshold': conf_threshold,
            'detections': detections,
            'audit_status': 'PASS' if len(boxes) > 0 else 'REVIEW'
        }
    else:
        return {
            'box_count': 0,
            'confidence_threshold': conf_threshold,
            'detections': [],
            'audit_status': 'REVIEW'
        }


def main():
    """Main entry point."""
    args = parse_args()
    
    # Resolve paths
    project_root = Path(__file__).parent.parent
    image_path = Path(args.image)
    model_path = project_root / args.model
    
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    # Run prediction
    result = predict(str(image_path), str(model_path), args.conf)
    
    # Display results
    print("=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    print(f"Image: {image_path.name}")
    print(f"Box Count: {result['box_count']}")
    print(f"Confidence Threshold: {result['confidence_threshold']}")
    print(f"Audit Status: {result['audit_status']}")
    print("=" * 50)
    
    # Show or save annotated image
    if args.show or args.save:
        model = YOLO(str(model_path))
        results = model.predict(str(image_path), conf=args.conf, verbose=False)
        annotated_img = results[0].plot()
        
        if args.save:
            output_path = args.output if args.output else f"output_{image_path.stem}.jpg"
            cv2.imwrite(output_path, annotated_img)
            print(f"Annotated image saved to: {output_path}")
        
        if args.show:
            cv2.imshow('Box Detection', annotated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    # Print JSON result
    print("\nJSON Output:")
    print(json.dumps(result, indent=2))
    
    return result


if __name__ == '__main__':
    main()

