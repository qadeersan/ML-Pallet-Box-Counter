"""
FastAPI Web Service for Pallet Box Counting

This service provides an API for counting boxes on pallets using YOLOv8.
Optimized for deployment on GCP Cloud Run.

Endpoints:
    GET  /              - Welcome message
    GET  /health        - Health check
    GET  /model/info    - Model information
    POST /count_boxes   - Count boxes in image

Usage:
    Local: uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
    Docker: docker run -p 8000:8000 pallet-box-counter
    GCP: gcloud run deploy ...

Author: Your Name
License: MIT
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Initialize FastAPI app
app = FastAPI(
    title="Pallet Box Counter API",
    description="""
    ## 📦 Automated Pallet Box Counting API
    
    This API uses YOLOv8 object detection to count boxes on pallets from images.
    
    ### Features:
    - **Fast inference** (~100-200ms per image)
    - **Adjustable confidence threshold**
    - **Detailed detection results**
    - **Audit status for quality control**
    
    ### Use Cases:
    - Warehouse inventory audits
    - Shipment verification
    - Automated quality control
    
    Built with ❤️ using FastAPI and YOLOv8
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable (loaded on startup)
model = None
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8s_boxes_best.pt"
DEFAULT_CONFIDENCE = 0.4


# Pydantic models for API responses
class Detection(BaseModel):
    """Single box detection"""
    id: int
    bbox: List[float]
    confidence: float
    class_name: str = "box"


class CountResponse(BaseModel):
    """Response for /count_boxes endpoint"""
    box_count: int
    confidence_threshold: float
    detections: List[Detection]
    audit_status: str
    processing_time_ms: float
    image_size: Optional[List[int]] = None


class HealthResponse(BaseModel):
    """Response for /health endpoint"""
    status: str
    model_loaded: bool
    model_path: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Response for /model/info endpoint"""
    model_name: str
    model_type: str
    version: str
    classes: List[str]
    default_confidence_threshold: float
    input_size: int


# Startup event - load model
@app.on_event("startup")
async def load_model():
    """Load YOLOv8 model on startup."""
    global model
    
    try:
        from ultralytics import YOLO
        
        if MODEL_PATH.exists():
            model = YOLO(str(MODEL_PATH))
            print(f"✓ Model loaded from: {MODEL_PATH}")
        else:
            # Try to find any available model
            models_dir = PROJECT_ROOT / "models"
            available_models = list(models_dir.glob("*.pt"))
            
            if available_models:
                model = YOLO(str(available_models[0]))
                print(f"✓ Model loaded from: {available_models[0]}")
            else:
                # Fall back to pretrained model
                print("⚠ No trained model found. Using pretrained yolov8n.pt")
                model = YOLO("yolov8n.pt")
                
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        model = None


# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    """Welcome page with API information."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pallet Box Counter API</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                border-radius: 16px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            h1 {
                color: #333;
                margin-bottom: 10px;
            }
            .subtitle {
                color: #666;
                font-size: 1.1em;
                margin-bottom: 30px;
            }
            .endpoint {
                background: #f8f9fa;
                border-left: 4px solid #667eea;
                padding: 15px;
                margin: 15px 0;
                border-radius: 0 8px 8px 0;
            }
            .method {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 0.85em;
                margin-right: 10px;
            }
            .get { background: #28a745; color: white; }
            .post { background: #007bff; color: white; }
            code {
                background: #e9ecef;
                padding: 2px 8px;
                border-radius: 4px;
                font-family: monospace;
            }
            a {
                color: #667eea;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
            .links {
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #eee;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📦 Pallet Box Counter API</h1>
            <p class="subtitle">Automated box counting using YOLOv8 object detection</p>
            
            <h3>Endpoints</h3>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <code>/health</code> - Health check
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <code>/model/info</code> - Model information
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <code>/count_boxes</code> - Count boxes in image
            </div>
            
            <div class="links">
                <p>
                    <a href="/docs">📚 Interactive API Documentation (Swagger)</a><br><br>
                    <a href="/redoc">📖 ReDoc Documentation</a>
                </p>
            </div>
        </div>
    </body>
    </html>
    """


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns:
        Health status and model information
    """
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_path=str(MODEL_PATH),
        timestamp=datetime.now().isoformat()
    )


# Model info endpoint
@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """
    Get information about the loaded model.
    
    Returns:
        Model configuration and capabilities
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_name="yolov8n_boxes",
        model_type="YOLOv8 Object Detection",
        version="1.0.0",
        classes=["box"],
        default_confidence_threshold=DEFAULT_CONFIDENCE,
        input_size=640
    )


# Main counting endpoint
@app.post("/count_boxes", response_model=CountResponse)
async def count_boxes(
    image: UploadFile = File(..., description="Image file (JPEG, PNG)"),
    confidence_threshold: float = Query(
        default=DEFAULT_CONFIDENCE,
        ge=0.1,
        le=0.9,
        description="Confidence threshold for detections (0.1-0.9)"
    ),
    expected_count: Optional[int] = Query(
        default=None,
        ge=0,
        description="Expected box count for audit comparison"
    )
):
    """
    Count boxes in an uploaded image.
    
    **Parameters:**
    - **image**: Image file (JPEG or PNG format)
    - **confidence_threshold**: Minimum confidence for detections (default: 0.4)
    - **expected_count**: Optional expected count for audit comparison
    
    **Returns:**
    - **box_count**: Number of detected boxes
    - **detections**: List of individual detections with bounding boxes
    - **audit_status**: PASS, REVIEW, or FAIL based on detection quality
    - **processing_time_ms**: Inference time in milliseconds
    
    **Example:**
    ```bash
    curl -X POST "http://localhost:8000/count_boxes" \\
      -H "accept: application/json" \\
      -F "image=@pallet.jpg" \\
      -F "confidence_threshold=0.4"
    ```
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    
    # Validate file type
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {image.content_type}. Expected image/jpeg or image/png"
        )
    
    start_time = time.time()
    
    try:
        # Save uploaded file temporarily
        suffix = ".jpg" if "jpeg" in (image.content_type or "") else ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await image.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Run inference
        results = model.predict(tmp_path, conf=confidence_threshold, verbose=False)
        
        # Extract detections
        boxes = results[0].boxes
        detections = []
        
        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                detections.append(Detection(
                    id=i,
                    bbox=box.xyxy[0].tolist(),
                    confidence=float(box.conf[0]),
                    class_name="box"
                ))
        
        # Get image size
        from PIL import Image as PILImage
        with PILImage.open(tmp_path) as img:
            image_size = list(img.size)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Determine audit status
        box_count = len(detections)
        if expected_count is not None:
            if box_count == expected_count:
                audit_status = "PASS"
            elif abs(box_count - expected_count) <= 2:
                audit_status = "REVIEW"
            else:
                audit_status = "FAIL"
        else:
            audit_status = "PASS" if box_count > 0 else "REVIEW"
        
        return CountResponse(
            box_count=box_count,
            confidence_threshold=confidence_threshold,
            detections=detections,
            audit_status=audit_status,
            processing_time_ms=round(processing_time, 2),
            image_size=image_size
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


# Batch counting endpoint (for multiple images)
@app.post("/count_boxes/batch")
async def count_boxes_batch(
    images: List[UploadFile] = File(..., description="Multiple image files"),
    confidence_threshold: float = Query(default=DEFAULT_CONFIDENCE, ge=0.1, le=0.9)
):
    """
    Count boxes in multiple images (batch processing).
    
    **Parameters:**
    - **images**: List of image files
    - **confidence_threshold**: Confidence threshold for all images
    
    **Returns:**
    - List of count results for each image
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(images) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    for img in images:
        result = await count_boxes(img, confidence_threshold)
        results.append({
            "filename": img.filename,
            **result.dict()
        })
    
    return {
        "total_images": len(results),
        "total_boxes": sum(r["box_count"] for r in results),
        "results": results
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# Run with uvicorn when executed directly
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False
    )

