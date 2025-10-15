import os
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import shared utilities and models
from shared.models import (
    TaskContext, TaskStatus, AgentType, 
    AgentMessage, SubTask
)
from shared.utils import (
    FirestoreManager, PubSubManager, 
    CloudTasksManager, GeminiManager
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Multi-Agent Orchestrator")

# Initialize managers
project_id = os.environ.get('PROJECT_ID', 'your-project-id')
firestore_manager = FirestoreManager(project_id)
pubsub_manager = PubSubManager(project_id)
tasks_manager = CloudTasksManager(project_id)
gemini_manager = GeminiManager()

# Request/Response models
class TaskRequest(BaseModel):
    query: str
    metadata: Optional[Dict[str, Any]] = {}

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str
    estimated_time_seconds: int = 30

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: float
    subtasks: List[Dict[str, Any]]
    result: Optional[str] = None
    errors: List[str] = []

# Orchestrator logic
class Orchestrator:
    """Main orchestrator that coordinates agent activities"""
    
    def __init__(self):
        self.firestore = firestore_manager
        self.pubsub = pubsub_manager
        self.tasks = tasks_manager
        self.gemini = gemini_manager
        
    async def process_user_query(self, query: str, metadata: Dict[str, Any] = {}) -> TaskContext:
        """Process a new user query"""
        
        # Create new task context
        context = TaskContext(
            user_query=query,
            metadata=metadata,
            status=TaskStatus.IN_PROGRESS
        )
        
        # Save initial context
        self.firestore.save_task_context(context)
        logger.info(f"Created task: {context.task_id}")
        
        try:
            # Use Gemini to decompose the task
            subtasks_data = self.gemini.decompose_task(query)
            
            # Create SubTask objects
            subtasks = []
            for idx, subtask_data in enumerate(subtasks_data):
                subtask = SubTask(
                    parent_task_id=context.task_id,
                    agent_type=AgentType(subtask_data['agent_type']),
                    description=subtask_data['description'],
                    parameters=subtask_data.get('parameters', {}),
                    dependencies=[subtasks[i].subtask_id for i in subtask_data.get('dependencies', [])]
                )
                subtasks.append(subtask)
                
                # Save subtask
                self.firestore.save_subtask(subtask)
                
            context.subtasks = [st.to_dict() for st in subtasks]
            self.firestore.save_task_context(context)
            
            # Dispatch initial subtasks (those with no dependencies)
            for subtask in subtasks:
                if not subtask.dependencies:
                    await self.dispatch_subtask(subtask)
                    
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            context.status = TaskStatus.FAILED
            context.errors.append({
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            })
            self.firestore.save_task_context(context)
            
        return context
        
    async def dispatch_subtask(self, subtask: SubTask) -> None:
        """Dispatch a subtask to the appropriate agent"""
        
        logger.info(f"Dispatching subtask {subtask.subtask_id} to {subtask.agent_type.value}")
        
        # Create message for the agent
        message = AgentMessage(
            task_id=subtask.parent_task_id,
            agent_type=subtask.agent_type,
            action="process",
            payload={
                'subtask_id': subtask.subtask_id,
                'description': subtask.description,
                'parameters': subtask.parameters
            },
            source_agent="orchestrator",
            target_agent=subtask.agent_type.value
        )
        
        # Publish to appropriate topic
        try:
            self.pubsub.dispatch_to_agent(subtask.agent_type, message)
            
            # Update subtask status
            subtask.status = TaskStatus.IN_PROGRESS
            subtask.started_at = datetime.utcnow()
            self.firestore.save_subtask(subtask)
            
        except Exception as e:
            logger.error(f"Failed to dispatch subtask: {e}")
            subtask.status = TaskStatus.FAILED
            subtask.error = str(e)
            self.firestore.save_subtask(subtask)
            
    async def handle_agent_result(self, message_data: Dict[str, Any]) -> None:
        """Handle results from agents"""
        
        task_id = message_data.get('task_id')
        subtask_id = message_data.get('subtask_id')
        agent_type = message_data.get('agent_type')
        result = message_data.get('result')
        error = message_data.get('error')
        
        logger.info(f"Received result from {agent_type} for subtask {subtask_id}")
        
        # Update subtask
        subtask = self.firestore.db.collection('subtasks').document(subtask_id).get()
        if subtask.exists:
            subtask_data = subtask.to_dict()
            subtask_obj = SubTask.from_dict(subtask_data)
            
            if error:
                subtask_obj.status = TaskStatus.FAILED
                subtask_obj.error = error
            else:
                subtask_obj.status = TaskStatus.COMPLETED
                subtask_obj.result = result
                
            subtask_obj.completed_at = datetime.utcnow()
            self.firestore.save_subtask(subtask_obj)
            
            # Update agent results in task context
            if result:
                self.firestore.update_agent_result(task_id, agent_type, result)
                
            # Check if we can dispatch dependent tasks
            await self.check_and_dispatch_dependencies(task_id, subtask_id)
            
            # Check if all subtasks are complete
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
            
            logger.info(f"Task {task_id} completed successfully")
            
            # Optionally notify via webhook or email
            await self.send_completion_notification(context)
            
    async def aggregate_results(self, context: TaskContext, subtasks: List[SubTask]) -> str:
        """Aggregate results from all agents into final response"""
        
        # Gather all successful results
        results = []
        for subtask in subtasks:
            if subtask.status == TaskStatus.COMPLETED and subtask.result:
                results.append({
                    'agent': subtask.agent_type.value,
                    'task': subtask.description,
                    'result': subtask.result
                })
                
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
        logger.info(f"Task {context.task_id} completed. Result ready for user.")

# Initialize orchestrator
orchestrator = Orchestrator()

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Multi-Agent Orchestrator is running"}

@app.post("/tasks", response_model=TaskResponse)
async def create_task(request: TaskRequest, background_tasks: BackgroundTasks):
    """Create a new task"""
    try:
        # Process in background
        background_tasks.add_task(
            orchestrator.process_user_query,
            request.query,
            request.metadata
        )
        
        # Return immediate response
        return TaskResponse(
            task_id="",  # Will be generated in background
            status="accepted",
            message="Task accepted for processing",
            estimated_time_seconds=30
        )
        
    except Exception as e:
        logger.error(f"Failed to create task: {e}")
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
        errors=[e.get('error', '') for e in context.errors]
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
            orchestrator.firestore.save_subtask(subtask)
            
    return {"message": "Task cancelled successfully"}

@app.post("/webhook/agent-result")
async def handle_agent_result(data: Dict[str, Any]):
    """Webhook endpoint for agent results"""
    
    try:
        await orchestrator.handle_agent_result(data)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to handle agent result: {e}")
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