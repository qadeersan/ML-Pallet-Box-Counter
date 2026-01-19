#!/bin/bash
# =============================================================================
# COST-SAFE GCP Cloud Run Deployment Script
# =============================================================================
# This script deploys your pallet box counter with settings that minimize costs.
#
# Usage:
#   ./deploy/deploy_safe.sh <GCP_PROJECT_ID> [REGION]
#
# Example:
#   ./deploy/deploy_safe.sh my-gcp-project us-central1
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
GCP_PROJECT_ID="${1}"
REGION="${2:-us-central1}"
SERVICE_NAME="pallet-counter"
IMAGE_NAME="pallet-box-counter"

# Validate input
if [ -z "$GCP_PROJECT_ID" ]; then
    echo -e "${RED}Error: GCP Project ID required${NC}"
    echo ""
    echo "Usage: $0 <GCP_PROJECT_ID> [REGION]"
    echo "Example: $0 my-gcp-project us-central1"
    exit 1
fi

echo "============================================================"
echo -e "${GREEN}🚀 COST-SAFE GCP CLOUD RUN DEPLOYMENT${NC}"
echo "============================================================"
echo ""
echo "Project:  ${GCP_PROJECT_ID}"
echo "Region:   ${REGION}"
echo "Service:  ${SERVICE_NAME}"
echo ""
echo -e "${YELLOW}⚠️  COST-SAFE SETTINGS:${NC}"
echo "   • min-instances: 0 (scales to zero = no idle costs)"
echo "   • max-instances: 1 (prevents runaway scaling)"
echo "   • memory: 1GB, cpu: 1 (minimum for YOLOv8)"
echo ""

# Confirm before proceeding
read -p "Continue with deployment? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI not installed${NC}"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Set project
echo ""
echo "📋 Setting GCP project..."
gcloud config set project ${GCP_PROJECT_ID}

# Enable required APIs
echo ""
echo "🔧 Enabling required APIs..."
gcloud services enable run.googleapis.com --quiet
gcloud services enable containerregistry.googleapis.com --quiet
gcloud services enable cloudbuild.googleapis.com --quiet

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# Check if model exists
if [ ! -f "models/yolov8s_boxes_best.pt" ] && [ ! -f "models/yolov8n_boxes_best.pt" ]; then
    echo -e "${YELLOW}⚠️  Warning: No trained model found in models/ directory${NC}"
    echo "The API will use pretrained weights instead."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled. Train a model first."
        exit 0
    fi
fi

# Build image using Cloud Build
echo ""
echo "📦 Building Docker image with Cloud Build..."
echo "   (This may take 5-10 minutes)"
gcloud builds submit --tag gcr.io/${GCP_PROJECT_ID}/${IMAGE_NAME}:latest .

# Deploy to Cloud Run with COST-SAFE settings
echo ""
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image gcr.io/${GCP_PROJECT_ID}/${IMAGE_NAME}:latest \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 60 \
  --min-instances 0 \
  --max-instances 1 \
  --concurrency 10 \
  --port 8000 \
  --set-env-vars "MODEL_PATH=/app/models/yolov8s_boxes_best.pt"

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
  --region ${REGION} \
  --format='value(status.url)')

echo ""
echo "============================================================"
echo -e "${GREEN}✅ DEPLOYMENT SUCCESSFUL!${NC}"
echo "============================================================"
echo ""
echo "🌐 Service URL: ${SERVICE_URL}"
echo ""
echo "📖 API Endpoints:"
echo "   GET  ${SERVICE_URL}/health     - Health check"
echo "   GET  ${SERVICE_URL}/docs       - API documentation"
echo "   POST ${SERVICE_URL}/count_boxes - Count boxes in image"
echo ""
echo "🧪 Test with:"
echo "   curl ${SERVICE_URL}/health"
echo ""
echo "============================================================"
echo -e "${YELLOW}⚠️  IMPORTANT REMINDERS:${NC}"
echo "============================================================"
echo ""
echo "1. 💰 SET A BUDGET ALERT:"
echo "   https://console.cloud.google.com/billing/budgets"
echo ""
echo "2. 🗑️  DELETE WHEN DONE:"
echo "   gcloud run services delete ${SERVICE_NAME} --region ${REGION}"
echo ""
echo "3. 📊 MONITOR COSTS:"
echo "   https://console.cloud.google.com/billing"
echo ""
echo "With these settings, idle costs = \$0 (scales to zero)"
echo "============================================================"

