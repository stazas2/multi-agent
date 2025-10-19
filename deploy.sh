#!/usr/bin/env bash
#
# Automated deployment script for the multi-agent system.
# Requirements:
#   - gcloud CLI authenticated (`gcloud auth login`)
#   - Application Default Credentials if using a service account JSON:
#         gcloud auth application-default login
#   - Cloud Build, Cloud Run, Cloud Functions Gen2 enabled in the project
#
# Usage:
#   export PROJECT_ID=<gcp-project>
#   export REGION=<gcp-region>            # optional, defaults to us-central1
#   export GEMINI_API_KEY=<gemini-key>    # required
#   bash deploy.sh
#

set -euo pipefail

# --- Helper functions -------------------------------------------------------

log() {
  printf "\n[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

ensure_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: '$1' not found in PATH" >&2
    exit 1
  fi
}

ensure_command gcloud
ensure_command python

# --- Configuration ----------------------------------------------------------

PROJECT_ID="${PROJECT_ID:-${1:-}}"
REGION="${REGION:-${2:-us-central1}}"
GEMINI_API_KEY="${GEMINI_API_KEY:-${3:-}}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "Error: PROJECT_ID not provided. Export PROJECT_ID or pass it as first argument." >&2
  exit 1
fi

if [[ -z "${GEMINI_API_KEY}" ]]; then
  echo "Error: GEMINI_API_KEY not provided. Export GEMINI_API_KEY or pass it as third argument." >&2
  exit 1
fi

SERVICE_ACCOUNT="multi-agent-system@${PROJECT_ID}.iam.gserviceaccount.com"
REPOSITORY="agent-images"
ORCHESTRATOR_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/orchestrator:latest"

log "Deploying Multi-Agent System"
log "Project: ${PROJECT_ID}"
log "Region: ${REGION}"

gcloud config set project "${PROJECT_ID}" >/dev/null

# --- Enable APIs ------------------------------------------------------------

log "Ensuring required APIs are enabled"
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  cloudfunctions.googleapis.com \
  pubsub.googleapis.com \
  firestore.googleapis.com \
  cloudtasks.googleapis.com \
  aiplatform.googleapis.com \
  secretmanager.googleapis.com \
  artifactregistry.googleapis.com \
  eventarc.googleapis.com \
  bigquery.googleapis.com \
  --project "${PROJECT_ID}"

# --- Service Account --------------------------------------------------------

log "Ensuring service account ${SERVICE_ACCOUNT} exists"
if ! gcloud iam service-accounts describe "${SERVICE_ACCOUNT}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud iam service-accounts create multi-agent-system \
    --display-name "Multi-Agent System Service Account" \
    --project "${PROJECT_ID}"
fi

log "Granting required IAM roles to ${SERVICE_ACCOUNT}"
roles=(
  "roles/datastore.user"
  "roles/pubsub.publisher"
  "roles/pubsub.subscriber"
  "roles/cloudtasks.enqueuer"
  "roles/aiplatform.user"
  "roles/secretmanager.secretAccessor"
  "roles/run.invoker"
)
for role in "${roles[@]}"; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="${role}" \
    --quiet >/dev/null
done

# --- Firestore --------------------------------------------------------------

log "Ensuring Firestore database exists"
if ! gcloud firestore databases list --project "${PROJECT_ID}" \
    --format="value(name)" | grep -q "(default)"; then
  gcloud firestore databases create \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --type=firestore-native
else
  log "Firestore database already present"
fi

# --- Pub/Sub Topics ---------------------------------------------------------

log "Ensuring Pub/Sub topics exist"
topics=(
  "agent-task-dispatch"
  "agent-task-results"
  "agent-research-tasks"
  "agent-analysis-tasks"
  "agent-code-tasks"
  "agent-validator-tasks"
)
for topic in "${topics[@]}"; do
  gcloud pubsub topics create "${topic}" --project "${PROJECT_ID}" >/dev/null 2>&1 || true
done

# --- Cloud Tasks Queue ------------------------------------------------------

log "Ensuring Cloud Tasks queue exists"
QUEUE="agent-task-queue"
if ! gcloud tasks queues describe "${QUEUE}" --location "${REGION}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud tasks queues create "${QUEUE}" \
    --location "${REGION}" \
    --project "${PROJECT_ID}"
fi

# --- Artifact Registry ------------------------------------------------------

log "Ensuring Artifact Registry repository exists"
gcloud artifacts repositories create "${REPOSITORY}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="Multi-Agent System Docker images" \
  --project "${PROJECT_ID}" >/dev/null 2>&1 || true

# --- Secret Manager ---------------------------------------------------------

log "Updating Secret Manager entry for gemini-api-key"
if ! gcloud secrets describe gemini-api-key --project "${PROJECT_ID}" >/dev/null 2>&1; then
  echo -n "${GEMINI_API_KEY}" | gcloud secrets create gemini-api-key \
    --replication-policy="automatic" \
    --data-file=- \
    --project "${PROJECT_ID}"
else
  echo -n "${GEMINI_API_KEY}" | gcloud secrets versions add gemini-api-key \
    --data-file=- \
    --project "${PROJECT_ID}"
fi

# --- Build and Deploy Orchestrator ------------------------------------------

log "Building orchestrator container via Cloud Build"
gcloud builds submit . \
  --tag "${ORCHESTRATOR_IMAGE}" \
  --project "${PROJECT_ID}"

log "Deploying orchestrator to Cloud Run"
gcloud run deploy orchestrator-service \
  --image "${ORCHESTRATOR_IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars PROJECT_ID="${PROJECT_ID}" \
  --update-secrets GEMINI_API_KEY=gemini-api-key:latest \
  --service-account "${SERVICE_ACCOUNT}" \
  --project "${PROJECT_ID}"

# --- Prepare Cloud Functions Packages ---------------------------------------

# --- Deploy Cloud Functions (Gen2) ------------------------------------------

deploy_function() {
  local name="$1"
  local topic="$2"
  pushd "agents/${name}" >/dev/null
  gcloud functions deploy "${name}-agent" \
    --gen2 \
    --runtime python311 \
    --region "${REGION}" \
    --source . \
    --entry-point handle_message \
    --trigger-topic "${topic}" \
    --set-env-vars PROJECT_ID="${PROJECT_ID}" \
    --set-secrets GEMINI_API_KEY=gemini-api-key:latest \
    --service-account "${SERVICE_ACCOUNT}" \
    --memory 1GB \
    --timeout 300s \
    --project "${PROJECT_ID}"
  popd >/dev/null
}

log "Deploying Cloud Functions agents"
deploy_function "research" "agent-research-tasks"
deploy_function "analysis" "agent-analysis-tasks"
deploy_function "code" "agent-code-tasks"
deploy_function "validator" "agent-validator-tasks"

# --- Output -----------------------------------------------------------------

ORCHESTRATOR_URL="$(gcloud run services describe orchestrator-service \
  --region "${REGION}" \
  --format 'value(status.url)' \
  --project "${PROJECT_ID}")"

log "Deployment complete"
log "Orchestrator URL: ${ORCHESTRATOR_URL}"
log "Example request:"
cat <<EOF
curl -X POST ${ORCHESTRATOR_URL}/tasks \\
  -H 'Content-Type: application/json' \\
  -d '{"query": "Find top 5 AI projects on GitHub and analyze their tech stack"}'
EOF
