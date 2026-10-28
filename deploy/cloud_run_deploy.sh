#!/bin/bash
# CineTrack Google Cloud Run Deployment Script
set -e

export PATH="/home/samarth/google-cloud-sdk/bin:$PATH"

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
echo "1. Enabling required Google Cloud APIs (Cloud Run, Cloud Build, Artifact Registry)..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --project "$PROJECT_ID" --quiet

# Deploy source directly to Google Cloud Run via Cloud Build
echo "2. Building and deploying service to Google Cloud Run from source..."
gcloud run deploy "$SERVICE_NAME" \
    --source . \
    --platform managed \
    --region "$REGION" \
    --allow-unauthenticated \
    --port 8080 \
    --memory 2Gi \
    --cpu 1 \
    --set-env-vars "APP_MODE=all" \
    --project "$PROJECT_ID" \
    --quiet

SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --platform managed --region "$REGION" --format 'value(status.url)' --project "$PROJECT_ID")

echo "============================================================"
echo " CineTrack deployed successfully!"
echo " Public Application URL: $SERVICE_URL"
echo "============================================================"
