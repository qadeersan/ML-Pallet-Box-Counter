#!/bin/bash
# GCP Cloud Run Deployment Script for Pallet Box Counter
#
# Prerequisites:
#   1. Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install
#   2. Create a GCP project and enable billing
#   3. Enable required APIs: gcloud services enable run.googleapis.com containerregistry.googleapis.com cloudbuild.googleapis.com
#
# Usage:
#   chmod +x deploy/deploy_gcp.sh
#   ./deploy/deploy_gcp.sh
#
# Environment variables (optional):
#   PROJECT_ID: GCP project ID (default: current project)
#   REGION: GCP region (default: us-central1)
#   SERVICE_NAME: Cloud Run service name (default: pallet-box-counter)

set -e  # Exit on error

# Configuration
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project)}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-pallet-box-counter}"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "============================================================"
echo "GCP Cloud Run Deployment - Pallet Box Counter"
echo "============================================================"
echo "Project ID: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service Name: ${SERVICE_NAME}"
echo "Image: ${IMAGE_NAME}"
echo "============================================================"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI not found. Please install Google Cloud SDK."
    echo "Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if authenticated
if ! gcloud auth print-identity-token &> /dev/null; then
    echo "Not authenticated. Running gcloud auth login..."
    gcloud auth login
fi

# Set project
echo ""
echo "Setting project to ${PROJECT_ID}..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo ""
echo "Enabling required APIs..."
gcloud services enable \
    run.googleapis.com \
    containerregistry.googleapis.com \
    cloudbuild.googleapis.com \
    --quiet

# Build and push container image using Cloud Build
echo ""
echo "Building and pushing container image..."
echo "This may take 5-10 minutes..."
gcloud builds submit \
    --tag ${IMAGE_NAME} \
    --timeout=20m \
    -f docker/Dockerfile \
    .

# Deploy to Cloud Run
echo ""
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --memory 2Gi \
    --cpu 2 \
    --timeout 60 \
    --concurrency 80 \
    --max-instances 10 \
    --allow-unauthenticated \
    --set-env-vars="PYTHONUNBUFFERED=1"

# Get service URL
echo ""
echo "============================================================"
echo "DEPLOYMENT COMPLETE!"
echo "============================================================"
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)')
echo ""
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "Test the deployment:"
echo "  curl ${SERVICE_URL}/health"
echo ""
echo "API Documentation:"
echo "  ${SERVICE_URL}/docs"
echo ""
echo "Count boxes:"
echo "  curl -X POST '${SERVICE_URL}/count_boxes' -F 'image=@test_image.jpg'"
echo "============================================================"

# Save URL to file for reference
echo "${SERVICE_URL}" > deploy/service_url.txt
echo "Service URL saved to deploy/service_url.txt"

