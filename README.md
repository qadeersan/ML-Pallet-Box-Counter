# 📦 Automated Pallet Box Counting Using Computer Vision

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![GCP](https://img.shields.io/badge/GCP-Cloud%20Run-4285F4.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

An end-to-end machine learning system that automatically counts boxes on pallets from images using computer vision. Built with YOLOv8 object detection and deployed as a REST API on Google Cloud Run.

---

## 🎯 Problem Statement

### Business Problem

Manual pallet audits in warehouse operations are:
- **Slow**: Each pallet requires 30-60 seconds of manual counting
- **Error-prone**: Human counters achieve ~95% accuracy under ideal conditions, dropping significantly with fatigue
- **Costly**: Incorrect box counts lead to shipment discrepancies and chargebacks (~$50-200 per incident)

### Solution

This project builds an **automated visual audit system** that detects and counts cartons on a pallet from a single image, enabling faster and more reliable outbound verification.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Image     │────▶│  YOLOv8     │────▶│  Count      │────▶│  Audit      │
│   Input     │     │  Detection  │     │  Logic      │     │  Result     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

---

## 📊 Model Results

### Performance Comparison

| Model | Type | mAP@0.5 | Precision | Recall | Count MAE | Count Accuracy |
|-------|------|---------|-----------|--------|-----------|----------------|
| **YOLOv8s** ⭐ | Object Detection | **83.7%** | **80.2%** | **80.7%** | ~1.2 | ~75% |
| YOLOv8n | Object Detection | 82.3% | 79.2% | 78.6% | ~1.5 | ~70% |
| ResNet18 | CNN Regression | N/A | N/A | N/A | 1.66 | 49.1% |

### Key Findings

✅ **YOLOv8s is the best model** with 83.7% mAP@0.5 and balanced precision/recall

✅ **Object detection outperforms regression** for counting tasks - provides interpretable bounding boxes

✅ **Off-by-1 accuracy of 71.7%** for regression shows counting is challenging without localization

### Detailed Results

#### YOLOv8s (Recommended for Production)
```
mAP@0.5:     83.70%
mAP@0.5-95:  68.5%
Precision:   80.18%
Recall:      80.74%
Inference:   ~10ms (GPU) / ~100ms (CPU)
Model Size:  22.5 MB
```

#### YOLOv8n (Lightweight Alternative)
```
mAP@0.5:     82.27%
mAP@0.5-95:  65.4%
Precision:   79.17%
Recall:      78.58%
Inference:   ~5ms (GPU) / ~50ms (CPU)
Model Size:  6.2 MB
```

#### ResNet18 Regression (Baseline)
```
Test MAE:           1.66 boxes
Count Accuracy:     49.1% (exact match)
Off-by-1 Accuracy:  71.7%
Off-by-3 Accuracy:  85.6%
Model Size:         44 MB
```

---

## 📁 Project Structure

```
ML-Pallet-Box-Counter/
│
├── 📂 api/                          # FastAPI web service
│   └── app.py                       # Main API application with /count_boxes endpoint
│
├── 📂 data/                         # Data configuration
│   ├── combined_data.yaml           # Combined dataset config for YOLOv8
│   ├── download_data.py             # Dataset download script
│   └── redistribute_splits.py       # 70/20/10 split utility
│
├── 📂 Boxes.v1i.yolov8/             # Dataset 1 (5,006 images)
│   ├── train/images/ & labels/      # 3,504 training samples
│   ├── valid/images/ & labels/      # 1,001 validation samples
│   ├── test/images/ & labels/       # 501 test samples
│   └── data.yaml                    # Dataset configuration
│
├── 📂 Final_Object_Detection.v1i.yolov8/  # Dataset 2 (1,869 images)
│   └── (same structure as above)
│
├── 📂 notebooks/                    # Jupyter notebooks for analysis
│   ├── 01_eda.ipynb                 # Exploratory Data Analysis
│   ├── 02_yolo_training.ipynb       # YOLOv8 training experiments
│   └── 03_regression_training.ipynb # CNN regression baseline
│
├── 📂 src/                          # Source code
│   ├── train_yolo.py                # YOLOv8 training script
│   ├── train_regression.py          # ResNet18 regression training
│   ├── predict.py                   # Inference utilities
│   └── export_model.py              # Model export (ONNX, pickle)
│
├── 📂 models/                       # Trained model weights
│   ├── yolov8s_boxes_best.pt        # Best YOLOv8s model ⭐
│   ├── yolov8n_boxes_best.pt        # YOLOv8n model
│   ├── resnet18_regression_best.pt  # ResNet18 regression
│   └── resnet18_regression_best.pkl # Pickle format
│
├── 📂 configs/                      # Configuration files
│   ├── yolo_config.yaml             # YOLOv8 hyperparameters
│   └── regression_config.yaml       # CNN regression config
│
├── 📂 deploy/                       # Deployment scripts
│   ├── deploy_safe.sh               # Cost-safe GCP deployment
│   ├── deploy_gcp.sh                # Standard GCP deployment
│   └── test_deployment.sh           # Deployment testing
│
├── 📂 docker/                       # Containerization
│   └── Dockerfile                   # Docker image definition
│
├── 📂 docs/                         # Documentation & visualizations
│   ├── GCP_DEPLOYMENT_GUIDE.md      # Detailed GCP deployment guide
│   ├── regression_summary.json      # Regression model results
│   ├── eda_statistics.json          # EDA summary statistics
│   └── *.png                        # Visualization outputs
│
├── 📂 runs/                         # Training outputs (auto-generated)
│   └── detect/                      # YOLOv8 training runs
│
├── 📄 requirements.txt              # Python dependencies
├── 📄 Pipfile                       # Pipenv dependencies
├── 📄 colab_training.py             # Google Colab training script
├── 📄 STEPS.md                      # Step-by-step project guide
└── 📄 README.md                     # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- GPU recommended (CUDA or Apple Silicon MPS)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ML-Pallet-Box-Counter.git
cd ML-Pallet-Box-Counter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the API

```bash
# Start the FastAPI server
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Test the health endpoint
curl http://localhost:8000/health

# Count boxes in an image
curl -X POST "http://localhost:8000/count_boxes" \
  -F "image=@test_image.jpg"
```

### API Response Example

```json
{
  "box_count": 24,
  "confidence_threshold": 0.4,
  "detections": [
    {"id": 0, "bbox": [120.5, 80.2, 280.1, 220.8], "confidence": 0.92},
    {"id": 1, "bbox": [300.2, 85.1, 450.6, 225.3], "confidence": 0.89}
  ],
  "audit_status": "PASS",
  "processing_time_ms": 145.23
}
```

---

## 📓 Notebooks

| Notebook | Description | Key Outputs |
|----------|-------------|-------------|
| `01_eda.ipynb` | Exploratory Data Analysis | Box count distributions, bounding box analysis, image quality |
| `02_yolo_training.ipynb` | YOLOv8 model training | Training curves, mAP metrics, confidence optimization |
| `03_regression_training.ipynb` | CNN baseline comparison | MAE analysis, model comparison charts |

---

## 🎓 Train the Model Yourself

Want to train your own model? Follow these steps to download data from Roboflow and train locally or on Google Colab.

### Step 1: Download Dataset from Roboflow

1. **Create a free Roboflow account** at [roboflow.com](https://roboflow.com)

2. **Download the datasets:**
   - [Final Object Detection Dataset](https://universe.roboflow.com/vishnua/final_object_detection-pjovt) (1,869 images)
   - [Boxes Dataset](https://universe.roboflow.com/box-fh0kz/boxes-ou3c2) (5,006 images)

3. **Export in YOLOv8 format:**
   - Click "Download Dataset"
   - Select **YOLOv8** format
   - Download the ZIP file

4. **Or use the Roboflow Python API:**

```python
# Install roboflow
pip install roboflow

# Download via API
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")

# Dataset 1: Final Object Detection
project1 = rf.workspace("vishnua").project("final_object_detection-pjovt")
dataset1 = project1.version(1).download("yolov8")

# Dataset 2: Boxes
project2 = rf.workspace("box-fh0kz").project("boxes-ou3c2")
dataset2 = project2.version(1).download("yolov8")
```

### Step 2: Train on Google Colab (Recommended - Free T4 GPU)

1. **Open Google Colab:** [colab.research.google.com](https://colab.research.google.com)

2. **Enable GPU:** Runtime → Change runtime type → **T4 GPU**

3. **Run these cells:**

```python
# Cell 1: Upload your dataset
from google.colab import files
print("Upload your dataset.zip file...")
uploaded = files.upload()

# Cell 2: Extract and install
!unzip -q dataset.zip
!pip install -q ultralytics

# Cell 3: Create data.yaml
data_yaml = """
path: /content
train:
  - Final_Object_Detection.v1i.yolov8/train/images
val:
  - Final_Object_Detection.v1i.yolov8/valid/images
test:
  - Final_Object_Detection.v1i.yolov8/test/images
nc: 1
names:
  0: box
"""
with open('/content/data.yaml', 'w') as f:
    f.write(data_yaml)

# Cell 4: Train YOLOv8
from ultralytics import YOLO

model = YOLO('yolov8s.pt')  # or yolov8n.pt for faster training
results = model.train(
    data='/content/data.yaml',
    epochs=30,
    batch=16,
    imgsz=640,
    device=0  # GPU
)

# Cell 5: Download trained model
from google.colab import files
files.download('/content/runs/detect/train/weights/best.pt')
```

**Training Time:** ~1-2 hours on Colab T4 GPU

### Step 3: Train Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/ML-Pallet-Box-Counter.git
cd ML-Pallet-Box-Counter

# Create environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train YOLOv8s (best accuracy)
python src/train_yolo.py --model yolov8s --epochs 30 --batch 16

# Train YOLOv8n (fastest)
python src/train_yolo.py --model yolov8n --epochs 30 --batch 16

# Train CNN Regression (baseline comparison)
python src/train_regression.py --epochs 30 --batch 32
```

**Note:** Local training on CPU is slow (~10+ hours). Use a GPU or Google Colab for faster results.

### Step 4: Use Your Trained Model

```python
from ultralytics import YOLO

# Load your trained model
model = YOLO('path/to/your/best.pt')

# Run inference
results = model.predict('test_image.jpg', conf=0.4)

# Count boxes
box_count = len(results[0].boxes)
print(f"Detected {box_count} boxes")
```

---

## 🌐 Live API Demo (Google Cloud Run)

The model is deployed as a REST API on Google Cloud Run. You can test it directly!

### API Endpoint

```
Sample: https://pallet-counter-[YOUR-ID].run.app

Mine:   https://pallet-counter-3zzsrr34aq-uc.a.run.app
```

> **Note:** Replace `[YOUR-ID]` with your actual Cloud Run service ID after deployment.

### Test the Live API

#### Using cURL

```bash
# Health check
curl https://pallet-counter-xxxxx.run.app/health

# Count boxes in an image
curl -X POST "https://pallet-counter-xxxxx.run.app/count_boxes" \
  -F "image=@your_pallet_image.jpg" \
  -F "confidence_threshold=0.4"
```

#### Interactive API Documentation

Visit the Swagger UI for interactive testing:
```
https://pallet-counter-xxxxx.run.app/docs
```

### API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome page with API info |
| `/health` | GET | Health check & model status |
| `/model/info` | GET | Model details & configuration |
| `/count_boxes` | POST | Count boxes in uploaded image |
| `/count_boxes/batch` | POST | Process multiple images |
| `/docs` | GET | Swagger UI documentation |

### Response Format

```json
{
  "box_count": 24,
  "confidence_threshold": 0.4,
  "detections": [
    {
      "id": 0,
      "bbox": [120.5, 80.2, 280.1, 220.8],
      "confidence": 0.92,
      "class_name": "box"
    }
  ],
  "audit_status": "PASS",
  "processing_time_ms": 145.23,
  "image_size": [640, 640]
}
```

### Audit Status Logic

| Status | Condition |
|--------|-----------|
| `PASS` | Boxes detected successfully |
| `REVIEW` | No boxes detected or count differs from expected |
| `FAIL` | Count differs significantly from expected (if provided) |

---

## 🔬 ML vs Rules Breakdown

| Component | Responsibility |
|-----------|----------------|
| **ML (YOLOv8)** | Detect boxes, predict bounding boxes, assign confidence scores |
| **Rules (Post-processing)** | Filter low-confidence detections, count boxes, determine audit status |


---

## 📊 Dataset

| Dataset | Images | Annotations | Source |
|---------|--------|-------------|--------|
| Boxes.v1i.yolov8 | 5,006 | ~66,759 | [Roboflow](https://universe.roboflow.com/box-fh0kz/boxes-ou3c2) |
| Final_Object_Detection | 1,869 | ~18,000 | [Roboflow](https://universe.roboflow.com/vishnua/final_object_detection-pjovt) |
| **Combined** | **6,875** | **~84,759** | — |

**Split:** 70% Train / 20% Validation / 10% Test

**License:** CC BY 4.0

---

## ⚠️ Limitations

1. **Visible boxes only** — Cannot count fully occluded boxes
2. **Single pallet per image** — Multi-pallet detection not supported
3. **Static images** — No video/real-time support

---

## 🛠️ Technologies Used

- **Object Detection:** YOLOv8 (Ultralytics)
- **Deep Learning:** PyTorch, torchvision
- **API Framework:** FastAPI
- **Containerization:** Docker
- **Cloud:** Google Cloud Run
- **Data Processing:** NumPy, Pandas, OpenCV, Pillow

---

## 📚 References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)

---

## 📝 License

This project is licensed under the MIT License. Datasets are licensed under CC BY 4.0.

---

## 👤 Author

**Your Name**
- GitHub: [@qadeersan](https://github.com/qadeersan)
- LinkedIn: [Qadeer Assan](https://linkedin.com/in/qadeerassan)

---

## 🙏 Acknowledgments

- Roboflow for providing the datasets
- Ultralytics for YOLOv8
- ML Zoomcamp for project guidance
