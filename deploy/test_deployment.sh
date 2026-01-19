#!/bin/bash
# Test script for deployed Pallet Box Counter API
#
# Usage:
#   ./deploy/test_deployment.sh [SERVICE_URL]
#
# If SERVICE_URL is not provided, reads from deploy/service_url.txt

set -e

# Get service URL
if [ -n "$1" ]; then
    SERVICE_URL="$1"
elif [ -f "deploy/service_url.txt" ]; then
    SERVICE_URL=$(cat deploy/service_url.txt)
else
    echo "Usage: $0 [SERVICE_URL]"
    echo "Or create deploy/service_url.txt with the service URL"
    exit 1
fi

echo "============================================================"
echo "Testing Pallet Box Counter API"
echo "Service URL: ${SERVICE_URL}"
echo "============================================================"

# Test 1: Health check
echo ""
echo "Test 1: Health Check"
echo "--------------------"
curl -s "${SERVICE_URL}/health" | python3 -m json.tool

# Test 2: Model info
echo ""
echo "Test 2: Model Info"
echo "------------------"
curl -s "${SERVICE_URL}/model/info" | python3 -m json.tool

# Test 3: Count boxes (if test image exists)
TEST_IMAGE="Boxes.v1i.yolov8/test/images/$(ls Boxes.v1i.yolov8/test/images/ 2>/dev/null | head -1)"
if [ -f "$TEST_IMAGE" ]; then
    echo ""
    echo "Test 3: Count Boxes"
    echo "-------------------"
    echo "Using image: $TEST_IMAGE"
    curl -s -X POST "${SERVICE_URL}/count_boxes" \
        -F "image=@${TEST_IMAGE}" \
        -F "confidence_threshold=0.4" | python3 -m json.tool
else
    echo ""
    echo "Test 3: Count Boxes (SKIPPED - no test image found)"
    echo "To test, run:"
    echo "  curl -X POST '${SERVICE_URL}/count_boxes' -F 'image=@your_image.jpg'"
fi

echo ""
echo "============================================================"
echo "All tests completed!"
echo "============================================================"
echo ""
echo "API Documentation: ${SERVICE_URL}/docs"

