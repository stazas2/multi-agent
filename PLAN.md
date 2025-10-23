# Multi-Agent System Enhancements Plan

This plan tracks the next iteration of work on the multi-agent platform. Tasks are ordered to minimise cross-dependencies; complete each section before moving on.

## 1. Multi-Model Support per Agent

- Extend configuration (`.env`, `deploy.sh`, Terraform locals if needed) with dedicated variables: `MODEL_RESEARCH`, `MODEL_ANALYSIS`, `MODEL_CODE`, `MODEL_VALIDATOR`.
- Refactor `shared/utils.py`:
  - Introduce a `ModelConfig` helper that resolves per-agent model selection with sensible fallbacks (defaults to `GEMINI_MODEL`).
  - Update `GeminiManager` (cloud and local variants) to expose `get_client(agent_type: AgentType)` returning a cached client bound to the configured model.
  - Add placeholder hooks for non-Gemini providers to ease future integration.
- Update all agents to request their model through the new helper before calling `generate_content`.
- Adjust documentation (`README.md`) to describe the new env vars and defaults.
- Sanity checks:
  - Local mode (`LOCAL_MODE=1`) still works.
  - In cloud mode, changing a single model env var takes effect for the corresponding agent.

## 2. Generated Code Packages

- Define a shared schema for generated assets:
  - Add dataclasses (`GeneratedAsset`, `GeneratedPackage`) in `shared/models.py`.
  - Include validation helpers for file paths and duplicate entries.
- Update Code Agent:
  - Allow agents to output structured package data (list of files + metadata) when `parameters['package'] == True`.
  - Ensure backward compatibility (plain string output still supported).
- Enhance orchestrator aggregation:
  - Detect structured packages in `agent_results` and assemble an in-memory ZIP using Python’s `zipfile`.
  - Store the archive and expose download info in the final task result. For local runs, return base64 or a temporary file path; for cloud, upload to GCS (stubbed in local mode).
- Adjust Validator Agent so it can inspect package metadata (file naming, required manifests, etc.).
- Surface download functionality in API responses (e.g., `result.package_url`).
- Document usage in README and SDK examples.

## 3. Advanced Subtask Controls

- Extend `SubTask` model with fields:
  - `retry_count`, `last_error`, `priority`, `manual` (if manually triggered), timestamps.
- Add new orchestrator endpoints:
  - `POST /tasks/{task_id}/subtasks/{subtask_id}/retry`
  - `POST /tasks/{task_id}/subtasks/{subtask_id}/cancel`
  - `POST /tasks/{task_id}/subtasks/{subtask_id}/prioritize`
- Update orchestration logic:
  - Implement `retry_subtask`, `cancel_subtask`, `prioritize_subtask` helpers (with optimistic locking).
  - Ensure Pub/Sub dispatcher tags messages with retry metadata.
  - Improve dependency handling so retried subtasks respect finished prerequisites.
- Enhance Pub/Sub manager to deduplicate by `subtask_id` and log manual retries.
- Update documentation and SDK with new endpoints and examples.

## 4. UI / SDK Enhancements

- **SDK (`sdk/multi_agent_client.py`):**
  - Add methods `retry_subtask`, `cancel_subtask`, `prioritize_subtask`, `download_package`.
  - Optional: provide `watch_task` helper (long-poll or SSE stub).
- **Web UI (`web/`):**
  - Extend task detail page with:
    - Subtask list displaying status, retries, timestamps.
    - Action buttons (retry/cancel/prioritise).
    - Link to download generated package.
  - Display per-agent model information.
- Ensure REST API responses include the data required by both SDK and UI.
- Update README plus any end-user docs to showcase new UI actions.

## 5. Validation & Release

- Update automated checks (unit tests, `scripts/run_checks.ps1`) to cover:
  - Model selection logic.
  - Packaging validation helpers.
  - New API endpoints (FastAPI tests).
- Add at least one end-to-end smoke test in `test_system.py` covering package generation flow.
- Double-check Terraform/deployment scripts for new env vars and permissions (GCS bucket for packages, if applicable).
- Final pass on docs and changelog before release.

> Work sequentially through sections; treat each as a milestone. Once a section is fully implemented and tested, check it off in this plan.

