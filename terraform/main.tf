variable "project_id" {
  description = "GCP Project ID"
}

variable "region" {
  default = "us-central1"
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "run.googleapis.com",
    "cloudfunctions.googleapis.com",
    "pubsub.googleapis.com",
    "firestore.googleapis.com",
    "cloudtasks.googleapis.com",
    "aiplatform.googleapis.com",
    "secretmanager.googleapis.com"
  ])
  
  project = var.project_id
  service = each.value
}

# Firestore database for state management
resource "google_firestore_database" "agent_state" {
  project     = var.project_id
  name        = "(default)"
  location_id = var.region
  type        = "FIRESTORE_NATIVE"
}

# Pub/Sub Topics
resource "google_pubsub_topic" "task_dispatch" {
  name = "agent-task-dispatch"
}

resource "google_pubsub_topic" "task_results" {
  name = "agent-task-results"
}

resource "google_pubsub_topic" "agent_research" {
  name = "agent-research-tasks"
}

resource "google_pubsub_topic" "agent_analysis" {
  name = "agent-analysis-tasks"
}

resource "google_pubsub_topic" "agent_code" {
  name = "agent-code-tasks"
}

resource "google_pubsub_topic" "agent_validator" {
  name = "agent-validator-tasks"
}

# Cloud Tasks Queue for delayed operations
resource "google_cloud_tasks_queue" "agent_tasks" {
  name     = "agent-task-queue"
  location = var.region

  rate_limits {
    max_concurrent_dispatches = 100
    max_dispatches_per_second = 10
  }

  retry_config {
    max_attempts = 5
    max_retry_duration = "3600s"
    max_backoff = "300s"
    min_backoff = "5s"
    max_doublings = 3
  }
}

# Service Account for agents
resource "google_service_account" "agent_sa" {
  account_id   = "multi-agent-system"
  display_name = "Multi-Agent System Service Account"
}

# IAM permissions for Service Account
resource "google_project_iam_member" "agent_permissions" {
  for_each = toset([
    "roles/firestore.user",
    "roles/pubsub.publisher",
    "roles/pubsub.subscriber",
    "roles/cloudtasks.enqueuer",
    "roles/aiplatform.user",
    "roles/secretmanager.secretAccessor"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.agent_sa.email}"
}

# Artifact Registry for container images
resource "google_artifact_registry_repository" "agent_images" {
  location      = var.region
  repository_id = "agent-images"
  format        = "DOCKER"
}

# Secret Manager for API keys
resource "google_secret_manager_secret" "gemini_api_key" {
  secret_id = "gemini-api-key"
  
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret" "github_token" {
  secret_id = "github-token"
  
  replication {
    auto {}
  }
}

# Cloud Run service for Orchestrator
resource "google_cloud_run_v2_service" "orchestrator" {
  name     = "orchestrator-service"
  location = var.region
  
  template {
    service_account = google_service_account.agent_sa.email
    
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/agent-images/orchestrator:latest"
      
      env {
        name  = "PROJECT_ID"
        value = var.project_id
      }
      
      env {
        name = "GEMINI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.gemini_api_key.secret_id
            version = "latest"
          }
        }
      }
      
      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }
    }
  }
}

# Cloud Functions for Agents
resource "google_cloudfunctions2_function" "research_agent" {
  name     = "research-agent"
  location = var.region
  
  build_config {
    runtime     = "python311"
    entry_point = "handle_message"
    source {
      storage_source {
        bucket = google_storage_bucket.agent_code.name
        object = google_storage_bucket_object.research_agent_code.name
      }
    }
  }
  
  service_config {
    max_instance_count = 10
    min_instance_count = 0
    available_memory   = "1Gi"
    timeout_seconds    = 300
    
    environment_variables = {
      PROJECT_ID = var.project_id
    }
    
    service_account_email = google_service_account.agent_sa.email
  }
  
  event_trigger {
    trigger_region = var.region
    event_type     = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic   = google_pubsub_topic.agent_research.id
  }
}

# Storage bucket for function code
resource "google_storage_bucket" "agent_code" {
  name     = "${var.project_id}-agent-code"
  location = var.region
}

resource "google_storage_bucket_object" "research_agent_code" {
  name   = "research-agent.zip"
  bucket = google_storage_bucket.agent_code.name
  source = "../agents/research/function.zip"
}

# BigQuery dataset for analytics
resource "google_bigquery_dataset" "agent_analytics" {
  dataset_id = "agent_analytics"
  location   = var.region
  
  labels = {
    env = "production"
  }
}

resource "google_bigquery_table" "task_logs" {
  dataset_id = google_bigquery_dataset.agent_analytics.dataset_id
  table_id   = "task_logs"
  
  schema = jsonencode([
    {
      name = "task_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "agent_name"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "status"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "timestamp"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "duration_ms"
      type = "INTEGER"
      mode = "NULLABLE"
    },
    {
      name = "error_message"
      type = "STRING"
      mode = "NULLABLE"
    }
  ])
}

# Outputs
output "orchestrator_url" {
  value = google_cloud_run_v2_service.orchestrator.uri
}

output "firestore_database" {
  value = google_firestore_database.agent_state.name
}

output "pubsub_topics" {
  value = {
    dispatch   = google_pubsub_topic.task_dispatch.name
    results    = google_pubsub_topic.task_results.name
    research   = google_pubsub_topic.agent_research.name
    analysis   = google_pubsub_topic.agent_analysis.name
    code       = google_pubsub_topic.agent_code.name
    validator  = google_pubsub_topic.agent_validator.name
  }
}