# 🚀 GCP Cloud Run Deployment Guide (Cost-Safe)

## ⚠️ Cost Protection First!

Before deploying anything, set up these safeguards to **prevent surprise bills**.

---

## 1. Set Up Billing Alerts & Budget (DO THIS FIRST!)

### Step 1: Create a Budget Alert
```bash
# Go to: https://console.cloud.google.com/billing/budgets
```

1. Click **"CREATE BUDGET"**
2. Set budget amount: **$10-25** (safe for learning)
3. Set alert thresholds:
   - 50% ($5-12.50)
   - 90% ($9-22.50)
   - 100% ($10-25)
4. Enable **"Email alerts to billing admins"**

### Step 2: Set Quotas (Optional but Recommended)
```bash
# Go to: https://console.cloud.google.com/iam-admin/quotas
```
- Search for "Cloud Run" and set request limits if needed

---

## 2. Model Size & Cost Analysis

### Your Model Specs
| Model | Size | Inference Time | Memory Needed |
|-------|------|----------------|---------------|
| YOLOv8s | ~22.5 MB | ~10-50ms (GPU), ~100-300ms (CPU) | ~512MB-1GB |

### Cloud Run Pricing (as of 2024)
| Resource | Free Tier | Price After |
|----------|-----------|-------------|
| **Requests** | 2M/month | $0.40 per million |
| **CPU** | 180,000 vCPU-seconds | $0.00002400/vCPU-sec |
| **Memory** | 360,000 GB-seconds | $0.00000250/GB-sec |

### Estimated Cost Per Request
| Config | CPU | Memory | Est. Cost/Request |
|--------|-----|--------|-------------------|
| **Minimal** | 1 vCPU | 1GB | ~$0.00003 |
| **Recommended** | 2 vCPU | 2GB | ~$0.00008 |

### Monthly Cost Estimates
| Usage | Minimal Config | Recommended Config |
|-------|----------------|-------------------|
| 100 requests/day | **FREE** | **FREE** |
| 1,000 requests/day | ~$0.90/month | ~$2.40/month |
| 10,000 requests/day | ~$9/month | ~$24/month |

---

## 3. Cost-Optimized Deployment Configuration

### Recommended Settings for Learning/Demo
```yaml
# These settings minimize cost while keeping functionality
min-instances: 0        # Scale to zero when not used (KEY!)
max-instances: 1        # Prevent runaway scaling
cpu: 1                  # Minimum CPU
memory: 1Gi             # 1GB is enough for YOLOv8s
timeout: 60             # Shorter timeout saves money
concurrency: 10         # Handle multiple requests per instance
```

---

## 4. Step-by-Step Deployment

### Prerequisites
```bash
# 1. Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# 2. Login and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 3. Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

### Step 1: Prepare Your Project

Make sure you have the trained model:
```bash
cd ~/workspace/ML-Pallet-Box-Counter

# Check model exists
 
```

### Step 2: Update Dockerfile for Production
```bash
# The Dockerfile is already set up, but let's verify
cat docker/Dockerfile
```

### Step 3: Build & Push Docker Image

```bash
# Set your project ID
export GCP_PROJECT_ID="ml-box-counter-capstone-2"
export IMAGE_NAME="pallet-box-counter"
export REGION="us-central1"

# Build the image using Cloud Build (no local Docker needed!)
gcloud builds submit --tag gcr.io/${GCP_PROJECT_ID}/${IMAGE_NAME}:latest .
```

### Step 4: Deploy to Cloud Run (COST-SAFE CONFIG)

```bash
gcloud run deploy pallet-counter \
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
```

### Step 5: Get Your Service URL
```bash
# Get the URL
gcloud run services describe pallet-counter --region ${REGION} --format='value(status.url)'
```

---

## 5. Test Your Deployment

```bash
# Save your service URL
export SERVICE_URL="https://pallet-counter-3zzsrr34aq-uc.a.run.app"

# Test health endpoint
curl ${SERVICE_URL}/health

# Test with an image
curl -X POST "https://pallet-counter-3zzsrr34aq-uc.a.run.app/count_boxes" \
  -F "image=@box_test.jpg"
```

---

## 6. Cost Monitoring Commands

### Check Current Usage
```bash
# View Cloud Run metrics
gcloud run services describe pallet-counter --region us-central1

# View recent requests
gcloud logging read "resource.type=cloud_run_revision" --limit 10
```

### View Billing
```bash
# Open billing dashboard
open https://console.cloud.google.com/billing
```

---

## 7. Emergency: Stop/Delete Service

If you see unexpected charges:

```bash
# Option 1: Set max instances to 0 (stops all traffic)
gcloud run services update pallet-counter \
  --region us-central1 \
  --max-instances 0

# Option 2: Delete the service entirely
gcloud run services delete pallet-counter --region us-central1

# Option 3: Delete the container image
gcloud container images delete gcr.io/${GCP_PROJECT_ID}/${IMAGE_NAME}:latest
```

---

## 8. Cost-Saving Tips

### ✅ DO
1. **Always use `--min-instances 0`** - Service scales to zero when idle
2. **Set `--max-instances 1`** for demos - Prevents runaway scaling
3. **Use budget alerts** - Get notified before overspending
4. **Delete when done** - Remove service after demo/testing
5. **Use smallest resources** - 1 CPU, 1GB memory is enough

### ❌ DON'T
1. **Don't use `--min-instances 1+`** unless needed (costs ~$25/month minimum)
2. **Don't forget to delete** after testing
3. **Don't use GPUs** on Cloud Run (expensive and not needed for YOLOv8)
4. **Don't set high max-instances** for demos

---

## 9. Quick Reference: Safe vs Expensive Configs

### 💚 SAFE: Learning/Demo Config (~$0-5/month)
```bash
gcloud run deploy pallet-counter \
  --min-instances 0 \
  --max-instances 1 \
  --memory 1Gi \
  --cpu 1 \
  --concurrency 10
```

### 🟡 MODERATE: Light Production (~$10-30/month)
```bash
gcloud run deploy pallet-counter \
  --min-instances 0 \
  --max-instances 3 \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 20
```

### 🔴 EXPENSIVE: Full Production ($$$/month)
```bash
# Don't use this for learning!
gcloud run deploy pallet-counter \
  --min-instances 1 \        # Always running = always charging
  --max-instances 10 \       # Can scale up fast
  --memory 4Gi \
  --cpu 4
```

---

## 10. One-Command Safe Deployment Script

Save this as `deploy_safe.sh`:

```bash
#!/bin/bash
set -e

# Configuration
GCP_PROJECT_ID="${1:-your-project-id}"
REGION="${2:-us-central1}"
SERVICE_NAME="pallet-counter"
IMAGE_NAME="pallet-box-counter"

echo "🚀 Deploying to GCP Cloud Run (COST-SAFE CONFIG)"
echo "Project: ${GCP_PROJECT_ID}"
echo "Region: ${REGION}"
echo ""

# Build image
echo "📦 Building Docker image..."
gcloud builds submit --tag gcr.io/${GCP_PROJECT_ID}/${IMAGE_NAME}:latest . --project ${GCP_PROJECT_ID}

# Deploy with SAFE settings
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
  --project ${GCP_PROJECT_ID}

# Get URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --project ${GCP_PROJECT_ID} --format='value(status.url)')

echo ""
echo "✅ Deployment complete!"
echo "🌐 Service URL: ${SERVICE_URL}"
echo ""
echo "⚠️  REMEMBER:"
echo "   - Budget alert set? Check: https://console.cloud.google.com/billing/budgets"
echo "   - Delete when done: gcloud run services delete ${SERVICE_NAME} --region ${REGION}"
```

---

## Summary: Your Deployment Checklist

- [ ] **1. Set budget alert** ($10-25) at https://console.cloud.google.com/billing/budgets
- [ ] **2. Copy trained model** to `models/yolov8s_boxes_best.pt`
- [ ] **3. Run deploy command** with safe settings
- [ ] **4. Test the endpoint**
- [ ] **5. Monitor costs** in billing dashboard
- [ ] **6. Delete when done** with your demo/testing

---

## Expected Costs for Your Use Case

| Scenario | Monthly Cost |
|----------|--------------|
| Just testing (< 100 requests) | **FREE** |
| Demo to bootcamp (< 1000 requests) | **FREE** |
| Light usage (< 50 requests/day) | **$0-2** |
| Forget to delete (idle) | **$0** (min-instances=0) |

**Bottom line:** With the safe config, you'll likely stay in the **FREE tier** for bootcamp demos! 🎉

