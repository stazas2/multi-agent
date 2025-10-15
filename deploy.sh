"""
Deployment script for Multi-Agent System on GCP
"""

set -e  # Exit on error

# Configuration
PROJECT_ID=${1:-"your-project-id"}
REGION=${2:-"us-central1"}

echo "=== Deploying Multi-Agent System ==="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudfunctions.googleapis.com \
    pubsub.googleapis.com \
    firestore.googleapis.com \
    cloudtasks.googleapis.com \
    aiplatform.googleapis.com \
    secretmanager.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com

# Create Artifact Registry repository
echo "Creating Artifact Registry repository..."
gcloud artifacts repositories create agent-images \
    --repository-format=docker \
    --location=$REGION \
    --description="Multi-Agent System Docker images" \
    || echo "Repository already exists"

# Deploy Terraform infrastructure
echo "Deploying infrastructure with Terraform..."
cd terraform
terraform init
terraform plan -var="project_id=$PROJECT_ID" -var="region=$REGION"
terraform apply -var="project_id=$PROJECT_ID" -var="region=$REGION" -auto-approve
cd ..

# Build and deploy Orchestrator
echo "Building and deploying Orchestrator..."
cd orchestrator
docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/agent-images/orchestrator:latest .
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/agent-images/orchestrator:latest

gcloud run deploy orchestrator-service \
    --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/agent-images/orchestrator:latest \
    --region $REGION \
    --platform managed \
    --allow-unauthenticated \
    --set-env-vars PROJECT_ID=$PROJECT_ID \
    --service-account multi-agent-system@${PROJECT_ID}.iam.gserviceaccount.com
cd ..

# Deploy Research Agent
echo "Deploying Research Agent..."
cd agents/research
zip -r function.zip .
gsutil cp function.zip gs://${PROJECT_ID}-agent-code/research-agent.zip

gcloud functions deploy research-agent \
    --gen2 \
    --runtime python311 \
    --region $REGION \
    --source . \
    --entry-point handle_message \
    --trigger-topic agent-research-tasks \
    --set-env-vars PROJECT_ID=$PROJECT_ID \
    --service-account multi-agent-system@${PROJECT_ID}.iam.gserviceaccount.com \
    --memory 1GB \
    --timeout 300s
cd ../..

# Deploy Analysis Agent
echo "Deploying Analysis Agent..."
cd agents/analysis
zip -r function.zip .
gsutil cp function.zip gs://${PROJECT_ID}-agent-code/analysis-agent.zip

gcloud functions deploy analysis-agent \
    --gen2 \
    --runtime python311 \
    --region $REGION \
    --source . \
    --entry-point handle_message \
    --trigger-topic agent-analysis-tasks \
    --set-env-vars PROJECT_ID=$PROJECT_ID \
    --service-account multi-agent-system@${PROJECT_ID}.iam.gserviceaccount.com \
    --memory 1GB \
    --timeout 300s
cd ../..

# Deploy Code Agent
echo "Deploying Code Agent..."
cd agents/code
zip -r function.zip .
gsutil cp function.zip gs://${PROJECT_ID}-agent-code/code-agent.zip

gcloud functions deploy code-agent \
    --gen2 \
    --runtime python311 \
    --region $REGION \
    --source . \
    --entry-point handle_message \
    --trigger-topic agent-code-tasks \
    --set-env-vars PROJECT_ID=$PROJECT_ID \
    --service-account multi-agent-system@${PROJECT_ID}.iam.gserviceaccount.com \
    --memory 1GB \
    --timeout 300s
cd ../..

# Set up secrets
echo "Setting up secrets..."
echo "Please add your API keys to Secret Manager:"
echo "  - gemini-api-key: Your Gemini API key"
echo "  - github-token: Your GitHub personal access token (optional)"

# Get Orchestrator URL
ORCHESTRATOR_URL=$(gcloud run services describe orchestrator-service \
    --region $REGION \
    --format 'value(status.url)')

echo "=== Deployment Complete ==="
echo "Orchestrator URL: $ORCHESTRATOR_URL"
echo ""
echo "Test the system with:"
echo "curl -X POST ${ORCHESTRATOR_URL}/tasks \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"query\": \"Find top 5 AI projects on GitHub and analyze their tech stack\"}'"