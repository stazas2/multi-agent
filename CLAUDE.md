# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-agent system with a microservices architecture. The system is designed to support multiple autonomous agents coordinated by a central orchestrator.

## Architecture

### Core Components

- **agents/**: Contains individual agent implementations. Each agent should be self-contained with its own logic, state management, and communication interfaces.

- **orchestrator/**: Central coordination service that manages agent lifecycle, task distribution, and inter-agent communication. This is the brain of the system that routes requests and manages agent workflows.

- **shared/**: Common libraries, utilities, data models, and interfaces used across agents and the orchestrator. Changes here affect multiple components.

- **sdk/**: Software development kit for building and integrating with the multi-agent system. Provides client libraries, API wrappers, and helper functions for external consumers.

- **monitoring/**: Observability infrastructure including logging, metrics, tracing, and health checks for the distributed system.

- **terraform/**: Infrastructure as Code definitions for deploying the multi-agent system to cloud environments.

- **testing/**: Integration tests, end-to-end tests, and test utilities that validate the entire system behavior.

## Development Workflow

### Deployment

Run the deployment script:
```bash
./deploy.sh
```

### Architecture Considerations

When making changes:
1. **Agent changes** should be isolated and communicate through well-defined interfaces
2. **Orchestrator changes** may affect all agents - test coordination logic thoroughly
3. **Shared library changes** require testing all dependent components
4. **SDK changes** maintain backward compatibility for external consumers

### Testing Strategy

- Unit tests belong in the same directory as the component being tested
- Integration tests for multi-component interactions go in **testing/**
- Test agent-orchestrator communication patterns thoroughly
- Validate error handling and failure scenarios across distributed components
