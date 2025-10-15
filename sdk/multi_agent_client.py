"""
Python SDK Client for Multi-Agent System
Easy-to-use client library for interacting with the multi-agent system
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import websocket
import threading
from queue import Queue, Empty

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    """Task representation"""
    task_id: str
    query: str
    status: TaskStatus
    progress: float = 0.0
    result: Optional[str] = None
    errors: List[str] = None
    subtasks: List[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['status'] = self.status.value
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create from dictionary"""
        data['status'] = TaskStatus(data.get('status', 'pending'))
        if data.get('created_at'):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('completed_at'):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        return cls(**data)

@dataclass
class AgentResult:
    """Result from a specific agent"""
    agent_name: str
    status: str
    result: Dict[str, Any]
    duration_ms: Optional[float] = None
    error: Optional[str] = None

class MultiAgentClient:
    """Client for interacting with the Multi-Agent System"""
    
    def __init__(self, orchestrator_url: str, api_key: Optional[str] = None,
                 timeout: int = 30, max_retries: int = 3):
        """
        Initialize the client
        
        Args:
            orchestrator_url: URL of the orchestrator service
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.orchestrator_url = orchestrator_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        
        # Configure session with retries
        self.session = requests.Session()
        retry = Retry(
            total=max_retries,
            read=max_retries,
            connect=max_retries,
            backoff_factor=0.3,
            status_forcelist=(500, 502, 504)
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # Set headers
        self.headers = {'Content-Type': 'application/json'}
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
            
        # WebSocket for real-time updates (if supported)
        self.ws = None
        self.ws_callbacks = {}
        self.ws_queue = Queue()
        
    def submit_task(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> Task:
        """
        Submit a new task to the system
        
        Args:
            query: The query/task description
            metadata: Optional metadata for the task
            
        Returns:
            Task object with initial status
        """
        payload = {'query': query}
        if metadata:
            payload['metadata'] = metadata
            
        try:
            response = self.session.post(
                f"{self.orchestrator_url}/tasks",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Create Task object
            task = Task(
                task_id=data.get('task_id', ''),
                query=query,
                status=TaskStatus(data.get('status', 'pending')),
                metadata=metadata,
                created_at=datetime.now(),
                errors=[]
            )
            
            logger.info(f"Submitted task: {task.task_id}")
            return task
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to submit task: {e}")
            raise
            
    def get_task_status(self, task_id: str) -> Task:
        """
        Get the current status of a task
        
        Args:
            task_id: The task ID
            
        Returns:
            Task object with current status
        """
        try:
            response = self.session.get(
                f"{self.orchestrator_url}/tasks/{task_id}",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            task = Task(
                task_id=task_id,
                query=data.get('query', ''),
                status=TaskStatus(data.get('status', 'pending')),
                progress=data.get('progress', 0),
                result=data.get('result'),
                errors=data.get('errors', []),
                subtasks=data.get('subtasks', [])
            )
            
            return task
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get task status: {e}")
            raise
            
    def wait_for_completion(self, task_id: str, 
                          poll_interval: int = 5,
                          max_wait_seconds: int = 300,
                          progress_callback: Optional[Callable[[float], None]] = None) -> Task:
        """
        Wait for a task to complete
        
        Args:
            task_id: The task ID
            poll_interval: Seconds between status checks
            max_wait_seconds: Maximum time to wait
            progress_callback: Optional callback for progress updates
            
        Returns:
            Completed Task object
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait_seconds:
            task = self.get_task_status(task_id)
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(task.progress)
                
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                if task.status == TaskStatus.COMPLETED:
                    logger.info(f"Task {task_id} completed successfully")
                else:
                    logger.warning(f"Task {task_id} ended with status: {task.status.value}")
                    
                task.completed_at = datetime.now()
                return task
                
            time.sleep(poll_interval)
            
        raise TimeoutError(f"Task {task_id} did not complete within {max_wait_seconds} seconds")
        
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task
        
        Args:
            task_id: The task ID
            
        Returns:
            True if cancelled successfully
        """
        try:
            response = self.session.post(
                f"{self.orchestrator_url}/tasks/{task_id}/cancel",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            logger.info(f"Cancelled task: {task_id}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to cancel task: {e}")
            return False
            
    def process_query(self, query: str, 
                     metadata: Optional[Dict[str, Any]] = None,
                     wait: bool = True,
                     progress_callback: Optional[Callable[[float], None]] = None) -> Union[Task, str]:
        """
        Submit a query and optionally wait for results
        
        Args:
            query: The query to process
            metadata: Optional metadata
            wait: Whether to wait for completion
            progress_callback: Optional progress callback
            
        Returns:
            Task object if wait=False, result string if wait=True
        """
        task = self.submit_task(query, metadata)
        
        if not wait:
            return task
            
        completed_task = self.wait_for_completion(
            task.task_id, 
            progress_callback=progress_callback
        )
        
        if completed_task.status == TaskStatus.COMPLETED:
            return completed_task.result
        else:
            raise Exception(f"Task failed: {completed_task.errors}")
            
    def batch_process(self, queries: List[str], 
                     concurrent_limit: int = 5,
                     progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Task]:
        """
        Process multiple queries in batch
        
        Args:
            queries: List of queries to process
            concurrent_limit: Maximum concurrent tasks
            progress_callback: Callback for batch progress (completed, total)
            
        Returns:
            List of completed Task objects
        """
        tasks = []
        completed = []
        
        # Submit tasks with concurrency limit
        for i in range(0, len(queries), concurrent_limit):
            batch = queries[i:i + concurrent_limit]
            batch_tasks = []
            
            for query in batch:
                task = self.submit_task(query)
                batch_tasks.append(task)
                tasks.append(task)
                
            # Wait for batch to complete
            for task in batch_tasks:
                completed_task = self.wait_for_completion(task.task_id)
                completed.append(completed_task)
                
                if progress_callback:
                    progress_callback(len(completed), len(queries))
                    
        return completed
        
    def get_agent_results(self, task_id: str) -> Dict[str, AgentResult]:
        """
        Get detailed results from each agent
        
        Args:
            task_id: The task ID
            
        Returns:
            Dictionary mapping agent names to their results
        """
        task = self.get_task_status(task_id)
        agent_results = {}
        
        for subtask in task.subtasks or []:
            agent_name = subtask.get('agent_type', 'unknown')
            
            result = AgentResult(
                agent_name=agent_name,
                status=subtask.get('status', 'unknown'),
                result=subtask.get('result', {}),
                duration_ms=subtask.get('duration_ms'),
                error=subtask.get('error')
            )
            
            agent_results[agent_name] = result
            
        return agent_results
        
    def stream_task(self, task_id: str, 
                   callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Stream real-time updates for a task (if WebSocket is supported)
        
        Args:
            task_id: The task ID
            callback: Callback function for updates
        """
        if not self.ws:
            self._connect_websocket()
            
        # Register callback
        self.ws_callbacks[task_id] = callback
        
        # Subscribe to task updates
        self.ws.send(json.dumps({
            'action': 'subscribe',
            'task_id': task_id
        }))
        
    def _connect_websocket(self) -> None:
        """Connect to WebSocket for real-time updates"""
        ws_url = self.orchestrator_url.replace('http', 'ws') + '/ws'
        
        def on_message(ws, message):
            data = json.loads(message)
            task_id = data.get('task_id')
            
            if task_id in self.ws_callbacks:
                self.ws_callbacks[task_id](data)
                
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
            
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            header=self.headers
        )
        
        # Run in background thread
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
    def close(self) -> None:
        """Close the client and clean up resources"""
        if self.ws:
            self.ws.close()
        self.session.close()

# Async version of the client
class AsyncMultiAgentClient:
    """Asynchronous client for the Multi-Agent System"""
    
    def __init__(self, orchestrator_url: str, api_key: Optional[str] = None):
        self.orchestrator_url = orchestrator_url.rstrip('/')
        self.api_key = api_key
        self.headers = {'Content-Type': 'application/json'}
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
            
    async def submit_task(self, query: str, 
                         metadata: Optional[Dict[str, Any]] = None) -> Task:
        """Submit a task asynchronously"""
        import aiohttp
        
        payload = {'query': query}
        if metadata:
            payload['metadata'] = metadata
            
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.orchestrator_url}/tasks",
                json=payload,
                headers=self.headers
            ) as response:
                data = await response.json()
                
                return Task(
                    task_id=data.get('task_id', ''),
                    query=query,
                    status=TaskStatus(data.get('status', 'pending')),
                    metadata=metadata,
                    created_at=datetime.now()
                )
                
    async def wait_for_completion(self, task_id: str, 
                                 poll_interval: int = 5) -> Task:
        """Wait for task completion asynchronously"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            while True:
                async with session.get(
                    f"{self.orchestrator_url}/tasks/{task_id}",
                    headers=self.headers
                ) as response:
                    data = await response.json()
                    
                    task = Task(
                        task_id=task_id,
                        query=data.get('query', ''),
                        status=TaskStatus(data.get('status', 'pending')),
                        progress=data.get('progress', 0),
                        result=data.get('result'),
                        errors=data.get('errors', []),
                        subtasks=data.get('subtasks', [])
                    )
                    
                    if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                        return task
                        
                await asyncio.sleep(poll_interval)
                
    async def batch_process_async(self, queries: List[str]) -> List[Task]:
        """Process multiple queries concurrently"""
        tasks = []
        
        for query in queries:
            task = await self.submit_task(query)
            tasks.append(task)
            
        # Wait for all tasks concurrently
        completed_tasks = await asyncio.gather(
            *[self.wait_for_completion(task.task_id) for task in tasks]
        )
        
        return completed_tasks

# Helper functions
def create_client(orchestrator_url: str, api_key: Optional[str] = None) -> MultiAgentClient:
    """
    Create a Multi-Agent System client
    
    Args:
        orchestrator_url: URL of the orchestrator service
        api_key: Optional API key
        
    Returns:
        Configured client instance
    """
    return MultiAgentClient(orchestrator_url, api_key)

# Example usage
if __name__ == "__main__":
    # Initialize client
    client = create_client("https://orchestrator-xxxxx.run.app")
    
    # Example 1: Simple query
    print("Example 1: Simple Query")
    result = client.process_query(
        "What are the top 5 Python web frameworks?",
        wait=True
    )
    print(f"Result: {result}\n")
    
    # Example 2: Query with progress tracking
    print("Example 2: Query with Progress")
    def progress_callback(progress):
        print(f"Progress: {progress:.1f}%")
        
    result = client.process_query(
        "Research AI repositories on GitHub and analyze their trends",
        wait=True,
        progress_callback=progress_callback
    )
    print(f"Result: {result}\n")
    
    # Example 3: Batch processing
    print("Example 3: Batch Processing")
    queries = [
        "What is machine learning?",
        "Explain neural networks",
        "What is deep learning?"
    ]
    
    def batch_progress(completed, total):
        print(f"Batch progress: {completed}/{total}")
        
    results = client.batch_process(
        queries,
        concurrent_limit=2,
        progress_callback=batch_progress
    )
    
    for task in results:
        print(f"Query: {task.query[:50]}...")
        print(f"Status: {task.status.value}")
        print(f"Result: {task.result[:100] if task.result else 'No result'}...\n")
        
    # Example 4: Async processing
    print("Example 4: Async Processing")
    async def async_example():
        async_client = AsyncMultiAgentClient("https://orchestrator-xxxxx.run.app")
        
        # Submit multiple tasks concurrently
        queries = [
            "Find serverless platforms",
            "Analyze cloud providers",
            "Generate sample Lambda code"
        ]
        
        results = await async_client.batch_process_async(queries)
        
        for task in results:
            print(f"Async Task {task.task_id}: {task.status.value}")
            
    # Run async example
    asyncio.run(async_example())
    
    # Example 5: Get detailed agent results
    print("Example 5: Agent Results")
    task = client.submit_task("Research and analyze Python frameworks")
    completed_task = client.wait_for_completion(task.task_id)
    
    agent_results = client.get_agent_results(task.task_id)
    for agent_name, result in agent_results.items():
        print(f"{agent_name} Agent:")
        print(f"  Status: {result.status}")
        print(f"  Duration: {result.duration_ms}ms" if result.duration_ms else "  Duration: N/A")
        print(f"  Error: {result.error}" if result.error else "  Success!")
        
    # Cleanup
    client.close()