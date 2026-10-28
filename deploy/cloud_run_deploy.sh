#!/bin/bash
# CineTrack Google Cloud Run Deployment Script
set -e

PROJECT_ID=${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}
REGION=${GCP_REGION:-"us-central1"}
SERVICE_NAME="cinetrack"
IMAGE_TAG="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

if [ -z "$PROJECT_ID" ]; then
    echo "ERROR: GCP_PROJECT_ID is not set and no active gcloud project found."
    echo "Usage: GCP_PROJECT_ID=your-project-id ./deploy/cloud_run_deploy.sh"
    exit 1
fi

echo "============================================================"
echo " Deploying CineTrack to Google Cloud Run"
echo " Project: $PROJECT_ID | Region: $REGION | Service: $SERVICE_NAME"
echo "============================================================"

# Enable required Google Cloud services
echo "1. Enabling required Google Cloud APIs (Cloud Run, Container Registry)..."
gcloud services enable run.googleapis.com containerregistry.googleapis.com --project "$PROJECT_ID"

# Build container image with Google Cloud Build
echo "2. Building container image via Google Cloud Build..."
gcloud builds submit --tag "$IMAGE_TAG" --project "$PROJECT_ID" .

# Deploy container image to Google Cloud Run
echo "3. Deploying service to Google Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
    --image "$IMAGE_TAG" \
    --platform managed \
    --region "$REGION" \
    --allow-unauthenticated \
    --port 8080 \
    --memory 2Gi \
    --cpu 1 \
    --set-env-vars "APP_MODE=all" \
    --project "$PROJECT_ID"

SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --platform managed --region "$REGION" --format 'value(status.url)' --project "$PROJECT_ID")

echo "============================================================"
echo " CineTrack deployed successfully!"
echo " Public Application URL: $SERVICE_URL"
echo "============================================================"
