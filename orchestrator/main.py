import os
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
import json
from dotenv import load_dotenv

# Import shared utilities and models
from shared.models import (
    TaskContext, TaskStatus, AgentType, 
    AgentMessage, SubTask, GeneratedPackage
)
from shared.utils import (
    FirestoreManager, PubSubManager, 
    CloudTasksManager, GeminiManager, create_package_artifact
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env if present (useful for local mode)
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Multi-Agent Orchestrator")

# Configure CORS to enable UI integrations
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*")
if allowed_origins.strip() == "*":
    cors_origins = ["*"]
else:
    cors_origins = [origin.strip() for origin in allowed_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve compiled frontend bundle if available
_frontend_candidates = [
    Path(__file__).resolve().parent / "frontend",
    Path(__file__).resolve().parent.parent / "web" / "dist"
]

for candidate in _frontend_candidates:
    if candidate.exists():
        logger.info("Serving frontend assets from %s", candidate)
        app.mount("/ui", StaticFiles(directory=candidate, html=True), name="ui")
        break

# Initialize managers
project_id = os.environ.get('PROJECT_ID', 'your-project-id')
firestore_manager = FirestoreManager(project_id)
pubsub_manager = PubSubManager(project_id, firestore_manager=firestore_manager)
tasks_manager = CloudTasksManager(project_id)
gemini_manager = GeminiManager()

# Request/Response models
class TaskRequest(BaseModel):
    query: str
    metadata: Optional[Dict[str, Any]] = None

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str
    estimated_time_seconds: int = 30
    trace_id: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: float
    subtasks: List[Dict[str, Any]]
    result: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    trace_id: str
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    events: List[Dict[str, Any]] = Field(default_factory=list)


class TaskSummary(BaseModel):
    task_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    progress: float
    has_artifacts: bool
    result_preview: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskListResponse(BaseModel):
    items: List[TaskSummary]


class ManualActionRequest(BaseModel):
    reason: Optional[str] = Field(default=None, max_length=500)


class SubtaskPriorityRequest(ManualActionRequest):
    priority: int = Field(..., ge=0, le=100)

# Orchestrator logic

AGENT_TYPE_ALIASES = {
    "orchestrator": AgentType.ORCHESTRATOR,
    "research": AgentType.RESEARCH,
    "research agent": AgentType.RESEARCH,
    "researcher": AgentType.RESEARCH,
    "analysis": AgentType.ANALYSIS,
    "analysis agent": AgentType.ANALYSIS,
    "analyst": AgentType.ANALYSIS,
    "code": AgentType.CODE,
    "code agent": AgentType.CODE,
    "coder": AgentType.CODE,
    "developer": AgentType.CODE,
    "validator": AgentType.VALIDATOR,
    "validator agent": AgentType.VALIDATOR,
    "validation": AgentType.VALIDATOR,
}


def resolve_agent_type(agent_type_value: Any) -> AgentType:
    """Map incoming agent identifiers to AgentType enum."""
    if isinstance(agent_type_value, AgentType):
        return agent_type_value

    if isinstance(agent_type_value, str):
        normalized = " ".join(agent_type_value.strip().lower().replace("_", " ").replace("-", " ").split())
        if normalized in AGENT_TYPE_ALIASES:
            return AGENT_TYPE_ALIASES[normalized]

    raise ValueError(f"Unsupported agent type: {agent_type_value!r}")


class Orchestrator:
    """Main orchestrator that coordinates agent activities"""
    
    def __init__(self):
        self.firestore = firestore_manager
        self.pubsub = pubsub_manager
        self.tasks = tasks_manager
        self.gemini = gemini_manager
        self.project_id = project_id
        # In LOCAL_MODE the Pub/Sub manager delivers results via callback
        if hasattr(self.pubsub, "register_result_handler"):
            self.pubsub.register_result_handler(self.handle_agent_result)

    def _dependencies_completed(self, subtask: SubTask, subtasks: Optional[List[SubTask]] = None) -> bool:
        """Check if all dependencies for a subtask are completed."""
        if not subtask.dependencies:
            return True

        if subtasks is None:
            subtasks = self.firestore.get_subtasks_for_task(subtask.parent_task_id)

        subtask_map = {st.subtask_id: st for st in subtasks}
        for dep_id in subtask.dependencies:
            dep = subtask_map.get(dep_id)
            if not dep or dep.status != TaskStatus.COMPLETED:
                return False
        return True

    async def manual_retry_subtask(self, task_id: str, subtask_id: str, reason: Optional[str] = None) -> Dict[str, Any]:
        context = self.firestore.get_task_context(task_id)
        if not context:
            raise HTTPException(status_code=404, detail="Task not found")

        subtask = self.firestore.get_subtask(subtask_id)
        if not subtask or subtask.parent_task_id != task_id:
            raise HTTPException(status_code=404, detail="Subtask not found")

        if subtask.status == TaskStatus.IN_PROGRESS:
            raise HTTPException(status_code=409, detail="Subtask currently in progress and cannot be retried")

        previous_error = subtask.error or subtask.last_error
        subtask.last_error = previous_error
        subtask.error = None
        subtask.result = None
        subtask.status = TaskStatus.PENDING
        subtask.started_at = None
        subtask.completed_at = None
        subtask.retry_count += 1
        subtask.manual = True
        subtask.manual_triggered_at = datetime.utcnow()

        self.firestore.save_subtask(subtask)

        action_note = reason or "Subtask manually retried"
        context.errors.append({
            'timestamp': datetime.utcnow().isoformat(),
            'error': action_note,
            'subtask_id': subtask_id,
            'action': 'retry',
            'retry_count': subtask.retry_count,
        })
        self.firestore.save_task_context(context)

        subtasks = self.firestore.get_subtasks_for_task(task_id)
        if self._dependencies_completed(subtask, subtasks):
            await self.dispatch_subtask(subtask)
            status = "queued"
        else:
            status = "pending_dependencies"

        return {
            "status": status,
            "subtask_id": subtask_id,
            "retry_count": subtask.retry_count,
            "last_error": previous_error,
        }

    async def manual_cancel_subtask(self, task_id: str, subtask_id: str, reason: Optional[str] = None) -> Dict[str, Any]:
        context = self.firestore.get_task_context(task_id)
        if not context:
            raise HTTPException(status_code=404, detail="Task not found")

        subtask = self.firestore.get_subtask(subtask_id)
        if not subtask or subtask.parent_task_id != task_id:
            raise HTTPException(status_code=404, detail="Subtask not found")

        if subtask.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return {"status": subtask.status.value, "message": "Subtask already finished"}

        subtask.status = TaskStatus.CANCELLED
        subtask.error = reason or "Cancelled manually"
        subtask.manual = True
        subtask.manual_triggered_at = datetime.utcnow()
        subtask.last_error = subtask.error

        self.firestore.save_subtask(subtask)

        context.errors.append({
            'timestamp': datetime.utcnow().isoformat(),
            'error': subtask.error,
            'subtask_id': subtask_id,
            'action': 'cancel',
        })
        self.firestore.save_task_context(context)

        return {
            "status": "cancelled",
            "subtask_id": subtask_id,
            "message": subtask.error,
        }

    async def manual_prioritize_subtask(
        self,
        task_id: str,
        subtask_id: str,
        priority: int,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        context = self.firestore.get_task_context(task_id)
        if not context:
            raise HTTPException(status_code=404, detail="Task not found")

        subtask = self.firestore.get_subtask(subtask_id)
        if not subtask or subtask.parent_task_id != task_id:
            raise HTTPException(status_code=404, detail="Subtask not found")

        old_priority = subtask.priority
        subtask.priority = priority
        subtask.manual = True
        subtask.manual_triggered_at = datetime.utcnow()
        self.firestore.save_subtask(subtask)

        context.errors.append({
            'timestamp': datetime.utcnow().isoformat(),
            'error': reason or "Subtask priority changed",
            'subtask_id': subtask_id,
            'action': 'prioritize',
            'priority': priority,
        })
        self.firestore.save_task_context(context)

        return {
            "status": "updated",
            "subtask_id": subtask_id,
            "old_priority": old_priority,
            "new_priority": priority,
        }
        
    async def process_user_query(
        self,
        query: str,
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[TaskContext] = None
    ) -> TaskContext:
        """Process a new user query"""

        metadata = metadata or {}

        # Create or update task context
        if context is None:
            context = TaskContext(
                user_query=query,
                metadata=metadata,
                status=TaskStatus.IN_PROGRESS
            )
        else:
            context.user_query = query
            context.metadata = metadata or context.metadata
            context.status = TaskStatus.IN_PROGRESS

        # Save initial context
        self.firestore.save_task_context(context)
        logger.info("Created task %s trace=%s", context.task_id, context.trace_id)
        
        try:
            # Use Gemini to decompose the task
            subtasks_data = self.gemini.decompose_task(query)
            
            # Create SubTask objects
            subtasks = []
            pending_dependency_indices: Dict[str, List[int]] = {}
            for idx, subtask_data in enumerate(subtasks_data):
                subtask = SubTask(
                    parent_task_id=context.task_id,
                    agent_type=resolve_agent_type(subtask_data['agent_type']),
                    description=subtask_data['description'],
                    parameters=subtask_data.get('parameters', {}),
                    dependencies=[],
                )
                subtasks.append(subtask)
                pending_dependency_indices[subtask.subtask_id] = subtask_data.get('dependencies', [])

            # Resolve dependency indices to subtask IDs safely
            for subtask in subtasks:
                resolved_deps: List[str] = []
                for dep_idx in pending_dependency_indices.get(subtask.subtask_id, []):
                    if 0 <= dep_idx < len(subtasks):
                        resolved_deps.append(subtasks[dep_idx].subtask_id)
                    else:
                        logger.warning(
                            "Ignoring invalid dependency index %s for subtask %s",
                            dep_idx,
                            subtask.subtask_id,
                        )
                subtask.dependencies = resolved_deps
                self.firestore.save_subtask(subtask)
                
            context.subtasks = [st.to_dict() for st in subtasks]
            self.firestore.save_task_context(context)
            
            # Dispatch initial subtasks (those with no dependencies)
            for subtask in subtasks:
                if not subtask.dependencies:
                    await self.dispatch_subtask(subtask)
                    
        except Exception as e:
            logger.error("Failed to process task %s trace=%s: %s", context.task_id, context.trace_id, e)
            context.status = TaskStatus.FAILED
            context.errors.append({
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            })
            self.firestore.save_task_context(context)
            
        return context
        
    async def dispatch_subtask(self, subtask: SubTask) -> None:
        """Dispatch a subtask to the appropriate agent"""
        
        if not self._dependencies_completed(subtask):
            logger.info(
                "Subtask %s for task %s is waiting on dependencies, skipping dispatch",
                subtask.subtask_id,
                subtask.parent_task_id,
            )
            return

        logger.info(
            "Dispatching subtask %s for task %s to %s",
            subtask.subtask_id,
            subtask.parent_task_id,
            subtask.agent_type.value,
        )
        
        # Create message for the agent
        message = AgentMessage(
            task_id=subtask.parent_task_id,
            agent_type=subtask.agent_type,
            action="process",
            payload={
                'subtask_id': subtask.subtask_id,
                'description': subtask.description,
                'parameters': subtask.parameters,
                'priority': subtask.priority,
                'retry_count': subtask.retry_count,
            },
            source_agent="orchestrator",
            target_agent=subtask.agent_type.value,
            priority=subtask.priority,
            retry_count=subtask.retry_count,
        )
        
        # Publish to appropriate topic
        try:
            self.pubsub.dispatch_to_agent(subtask.agent_type, message)
            
            # Update subtask status
            subtask.status = TaskStatus.IN_PROGRESS
            subtask.started_at = datetime.utcnow()
            self.firestore.save_subtask(subtask)
            
        except Exception as e:
            logger.error(
                "Failed to dispatch subtask %s for task %s: %s",
                subtask.subtask_id,
                subtask.parent_task_id,
                e,
            )
            subtask.status = TaskStatus.FAILED
            subtask.error = str(e)
            subtask.last_error = str(e)
            self.firestore.save_subtask(subtask)
            
    async def handle_agent_result(self, message_data: Dict[str, Any]) -> None:
        """Handle results from agents"""
        
        task_id = message_data.get('task_id')
        subtask_id = message_data.get('subtask_id')
        agent_type = message_data.get('agent_type')
        result = message_data.get('result')
        error = message_data.get('error')
        
        logger.info(
            "Received result from %s for subtask %s (task %s)",
            agent_type,
            subtask_id,
            task_id,
        )
        
        subtask_obj = self.firestore.get_subtask(subtask_id)
        if subtask_obj:
            if subtask_obj.status == TaskStatus.CANCELLED:
                logger.info("Ignoring result for cancelled subtask %s", subtask_id)
                return

            if error:
                subtask_obj.status = TaskStatus.FAILED
                subtask_obj.error = error
                subtask_obj.last_error = error
            else:
                subtask_obj.status = TaskStatus.COMPLETED
                subtask_obj.result = result

            subtask_obj.completed_at = datetime.utcnow()
            self.firestore.save_subtask(subtask_obj)

            if result:
                self.firestore.update_agent_result(task_id, agent_type, result)

            await self.check_and_dispatch_dependencies(task_id, subtask_id)
            await self.check_task_completion(task_id)
            
    async def check_and_dispatch_dependencies(self, task_id: str, completed_subtask_id: str) -> None:
        """Check and dispatch subtasks that depend on the completed one"""
        
        # Get all subtasks for this task
        all_subtasks = self.firestore.get_subtasks_for_task(task_id)
        
        for subtask in all_subtasks:
            # If this subtask depends on the completed one and hasn't started yet
            if (completed_subtask_id in subtask.dependencies and 
                subtask.status == TaskStatus.PENDING):
                
                # Check if all dependencies are complete
                all_deps_complete = True
                for dep_id in subtask.dependencies:
                    dep_subtask = next((st for st in all_subtasks if st.subtask_id == dep_id), None)
                    if not dep_subtask or dep_subtask.status != TaskStatus.COMPLETED:
                        all_deps_complete = False
                        break
                        
                if all_deps_complete:
                    await self.dispatch_subtask(subtask)
                    
    async def check_task_completion(self, task_id: str) -> None:
        """Check if all subtasks are complete and finalize the task"""
        
        context = self.firestore.get_task_context(task_id)
        if not context:
            return
            
        all_subtasks = self.firestore.get_subtasks_for_task(task_id)
        
        # Check if all subtasks are done
        all_complete = all(
            st.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] 
            for st in all_subtasks
        )
        
        if all_complete:
            # Aggregate results
            final_result = await self.aggregate_results(context, all_subtasks)
            
            # Update context
            context.final_result = final_result
            context.status = TaskStatus.COMPLETED
            self.firestore.save_task_context(context)
            
            logger.info("Task %s completed successfully (trace=%s)", task_id, context.trace_id)
            
            # Optionally notify via webhook or email
            await self.send_completion_notification(context)
            
    async def aggregate_results(self, context: TaskContext, subtasks: List[SubTask]) -> str:
        """Aggregate results from all agents into final response"""
        
        # Gather all successful results
        results = []
        packages_recorded = context.artifacts.get("packages")
        packages_to_store: List[Dict[str, Any]] = [] if not packages_recorded else None

        for subtask in subtasks:
            if subtask.status == TaskStatus.COMPLETED and subtask.result:
                payload = subtask.result
                results.append({
                    'agent': subtask.agent_type.value,
                    'task': subtask.description,
                    'result': payload
                })

                if packages_to_store is not None and isinstance(payload, dict) and payload.get('package'):
                    try:
                        package = GeneratedPackage.from_dict(payload['package'])
                        artifact = create_package_artifact(
                            package,
                            self.project_id,
                            prefix=subtask.parent_task_id,
                        )
                        packages_to_store.append({
                            'agent': subtask.agent_type.value,
                            'subtask_id': subtask.subtask_id,
                            'summary': payload.get('summary'),
                            'artifact': artifact,
                            'created_at': datetime.utcnow().isoformat(),
                        })
                    except Exception as exc:
                        logger.error("Failed to persist package for subtask %s: %s", subtask.subtask_id, exc)
                        context.errors.append({
                            'timestamp': datetime.utcnow().isoformat(),
                            'error': f"Package processing failed for {subtask.subtask_id}: {exc}",
                        })
                
        if packages_to_store:
            context.artifacts['packages'] = packages_to_store

        # Use Gemini to synthesize the final response
        prompt = f"""
        User Query: {context.user_query}
        
        Agent Results:
        {json.dumps(results, indent=2)}
        
        Please synthesize these results into a comprehensive, clear response to the user's query.
        Focus on answering their question directly using the information gathered by the agents.
        """
        
        final_response = self.gemini.generate_response(prompt)
        return final_response
        
    async def send_completion_notification(self, context: TaskContext) -> None:
        """Send notification when task is complete"""
        # This could send email, webhook, or push notification
        logger.info("Task %s (trace=%s) completed. Result ready for user.", context.task_id, context.trace_id)

# Initialize orchestrator
orchestrator = Orchestrator()

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Multi-Agent Orchestrator is running"}

@app.get("/tasks", response_model=TaskListResponse)
async def list_tasks(limit: int = 20):
    """List recent tasks with summary information"""
    limit = max(1, min(limit, 100))
    contexts = orchestrator.firestore.list_recent_tasks(limit)
    summaries: List[TaskSummary] = []

    for context in contexts:
        subtask_records: List[Dict[str, Any]] = context.subtasks or []  # type: ignore[assignment]
        total = len(subtask_records)
        completed = sum(
            1 for record in subtask_records
            if (record.get("status") or "").lower() in {"completed", "failed"}
        )
        progress = (completed / total * 100) if total > 0 else 0.0
        summaries.append(
            TaskSummary(
                task_id=context.task_id,
                status=context.status.value if isinstance(context.status, TaskStatus) else str(context.status),
                created_at=context.created_at,
                updated_at=context.updated_at,
                progress=progress,
                has_artifacts=bool((context.artifacts or {}).get("packages")),
                result_preview=context.final_result[:200] if context.final_result else None,
                metadata=context.metadata or {},
            )
        )

    return TaskListResponse(items=summaries)

@app.post("/tasks", response_model=TaskResponse)
async def create_task(request: TaskRequest, background_tasks: BackgroundTasks):
    """Create a new task"""
    try:
        metadata = request.metadata or {}

        # Prepare context upfront so we can return task_id immediately
        context = TaskContext(
            user_query=request.query,
            metadata=metadata,
            status=TaskStatus.PENDING
        )
        orchestrator.firestore.save_task_context(context)

        # Process in background
        background_tasks.add_task(
            orchestrator.process_user_query,
            request.query,
            metadata,
            context
        )
        
        # Return immediate response with task details
        return TaskResponse(
            task_id=context.task_id,
            status="accepted",
            message="Task accepted for processing",
            estimated_time_seconds=30,
            trace_id=context.trace_id,
        )
        
    except Exception as e:
        logger.error("Failed to create task for query '%s': %s", request.query, e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get task status and results"""
    
    context = orchestrator.firestore.get_task_context(task_id)
    if not context:
        raise HTTPException(status_code=404, detail="Task not found")
        
    subtasks = orchestrator.firestore.get_subtasks_for_task(task_id)
    
    # Calculate progress
    total = len(subtasks)
    completed = sum(1 for st in subtasks if st.status in [TaskStatus.COMPLETED, TaskStatus.FAILED])
    progress = (completed / total * 100) if total > 0 else 0
    
    return TaskStatusResponse(
        task_id=task_id,
        status=context.status.value,
        progress=progress,
        subtasks=[st.to_dict() for st in subtasks],
        result=context.final_result,
        errors=[e.get('error', '') for e in context.errors],
        trace_id=context.trace_id,
        artifacts=context.artifacts,
        events=context.errors,
    )

@app.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel a running task"""
    
    context = orchestrator.firestore.get_task_context(task_id)
    if not context:
        raise HTTPException(status_code=404, detail="Task not found")
        
    if context.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
        return {"message": "Task already finished"}
        
    # Update status
    orchestrator.firestore.update_task_status(task_id, TaskStatus.CANCELLED)
    
    # Cancel all pending subtasks
    subtasks = orchestrator.firestore.get_subtasks_for_task(task_id)
    for subtask in subtasks:
        if subtask.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]:
            subtask.status = TaskStatus.CANCELLED
            subtask.error = "Cancelled with parent task"
            subtask.manual = True
            subtask.manual_triggered_at = datetime.utcnow()
            orchestrator.firestore.save_subtask(subtask)
            
    return {"message": "Task cancelled successfully"}

@app.post("/tasks/{task_id}/subtasks/{subtask_id}/retry")
async def retry_subtask_endpoint(
    task_id: str,
    subtask_id: str,
    payload: Optional[ManualActionRequest] = Body(default=None),
):
    return await orchestrator.manual_retry_subtask(task_id, subtask_id, payload.reason if payload else None)

@app.post("/tasks/{task_id}/subtasks/{subtask_id}/cancel")
async def cancel_subtask_endpoint(
    task_id: str,
    subtask_id: str,
    payload: Optional[ManualActionRequest] = Body(default=None),
):
    return await orchestrator.manual_cancel_subtask(task_id, subtask_id, payload.reason if payload else None)

@app.post("/tasks/{task_id}/subtasks/{subtask_id}/prioritize")
async def prioritize_subtask_endpoint(
    task_id: str,
    subtask_id: str,
    payload: SubtaskPriorityRequest,
):
    return await orchestrator.manual_prioritize_subtask(task_id, subtask_id, payload.priority, payload.reason)

@app.post("/webhook/agent-result")
async def handle_agent_result(data: Dict[str, Any]):
    """Webhook endpoint for agent results"""
    
    try:
        await orchestrator.handle_agent_result(data)
        return {"status": "success"}
    except Exception as e:
        logger.error("Failed to handle agent result payload %s: %s", data, e)
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "project_id": project_id
    }

# Run the server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
