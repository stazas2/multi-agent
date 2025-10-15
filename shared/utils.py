import os
import logging
from typing import Optional, Dict, Any, List
from google.cloud import firestore, pubsub_v1, tasks_v2
from google.cloud.firestore import DocumentSnapshot
import google.generativeai as genai
import time
import hashlib
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FirestoreManager:
    """Manages Firestore operations for the multi-agent system"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.db = firestore.Client(project=project_id)
        
    def save_task_context(self, context: TaskContext) -> None:
        """Save or update task context"""
        context.updated_at = datetime.utcnow()
        doc_ref = self.db.collection('tasks').document(context.task_id)
        doc_ref.set(context.to_dict())
        logger.info(f"Saved task context: {context.task_id}")
        
    def get_task_context(self, task_id: str) -> Optional[TaskContext]:
        """Retrieve task context"""
        doc_ref = self.db.collection('tasks').document(task_id)
        doc = doc_ref.get()
        
        if doc.exists:
            return TaskContext.from_dict(doc.to_dict())
        return None
        
    def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        """Update task status"""
        doc_ref = self.db.collection('tasks').document(task_id)
        doc_ref.update({
            'status': status.value,
            'updated_at': datetime.utcnow().isoformat()
        })
        logger.info(f"Updated task {task_id} status to {status.value}")
        
    def save_subtask(self, subtask: SubTask) -> None:
        """Save subtask"""
        doc_ref = self.db.collection('subtasks').document(subtask.subtask_id)
        doc_ref.set(subtask.to_dict())
        logger.info(f"Saved subtask: {subtask.subtask_id}")
        
    def get_subtasks_for_task(self, task_id: str) -> List[SubTask]:
        """Get all subtasks for a task"""
        query = self.db.collection('subtasks').where('parent_task_id', '==', task_id)
        docs = query.stream()
        return [SubTask.from_dict(doc.to_dict()) for doc in docs]
        
    def update_agent_result(self, task_id: str, agent_type: str, result: Dict[str, Any]) -> None:
        """Update results from a specific agent"""
        doc_ref = self.db.collection('tasks').document(task_id)
        doc_ref.update({
            f'agent_results.{agent_type}': result,
            'updated_at': datetime.utcnow().isoformat()
        })
        logger.info(f"Updated {agent_type} results for task {task_id}")

class PubSubManager:
    """Manages Pub/Sub operations"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()
        
        # Topic paths
        self.topics = {
            'dispatch': f'projects/{project_id}/topics/agent-task-dispatch',
            'results': f'projects/{project_id}/topics/agent-task-results',
            'research': f'projects/{project_id}/topics/agent-research-tasks',
            'analysis': f'projects/{project_id}/topics/agent-analysis-tasks',
            'code': f'projects/{project_id}/topics/agent-code-tasks',
            'validator': f'projects/{project_id}/topics/agent-validator-tasks'
        }
        
    def publish_message(self, topic_name: str, message: AgentMessage) -> str:
        """Publish message to a topic"""
        topic_path = self.topics.get(topic_name)
        if not topic_path:
            raise ValueError(f"Unknown topic: {topic_name}")
            
        message_data = message.to_json().encode('utf-8')
        
        # Add attributes for filtering
        attributes = {
            'task_id': message.task_id,
            'agent_type': message.agent_type.value,
            'priority': str(message.priority)
        }
        
        future = self.publisher.publish(topic_path, message_data, **attributes)
        message_id = future.result()
        logger.info(f"Published message {message_id} to {topic_name}")
        return message_id
        
    def dispatch_to_agent(self, agent_type: AgentType, message: AgentMessage) -> str:
        """Dispatch message to specific agent topic"""
        topic_map = {
            AgentType.RESEARCH: 'research',
            AgentType.ANALYSIS: 'analysis',
            AgentType.CODE: 'code',
            AgentType.VALIDATOR: 'validator'
        }
        
        topic_name = topic_map.get(agent_type)
        if not topic_name:
            raise ValueError(f"Cannot dispatch to agent type: {agent_type}")
            
        return self.publish_message(topic_name, message)

class CloudTasksManager:
    """Manages Cloud Tasks for scheduled operations"""
    
    def __init__(self, project_id: str, location: str = 'us-central1'):
        self.project_id = project_id
        self.location = location
        self.client = tasks_v2.CloudTasksClient()
        self.queue_path = self.client.queue_path(project_id, location, 'agent-task-queue')
        
    def create_task(self, url: str, payload: Dict[str, Any], 
                   delay_seconds: int = 0) -> str:
        """Create a new task"""
        task = {
            'http_request': {
                'http_method': tasks_v2.HttpMethod.POST,
                'url': url,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps(payload).encode()
            }
        }
        
        if delay_seconds > 0:
            import time
            scheduled_time = time.time() + delay_seconds
            from google.protobuf import timestamp_pb2
            timestamp = timestamp_pb2.Timestamp()
            timestamp.FromSeconds(int(scheduled_time))
            task['schedule_time'] = timestamp
            
        response = self.client.create_task(
            request={'parent': self.queue_path, 'task': task}
        )
        
        logger.info(f"Created task: {response.name}")
        return response.name

class GeminiManager:
    """Manages Gemini API interactions"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not provided")
            
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
    def decompose_task(self, user_query: str) -> List[Dict[str, Any]]:
        """Use Gemini to break down user query into subtasks"""
        
        prompt = f"""
        You are a task orchestrator for a multi-agent AI system. 
        
        Given this user query: "{user_query}"
        
        Break it down into specific subtasks that can be handled by these specialized agents:
        1. Research Agent: Searches for information, queries databases, gathers data
        2. Analysis Agent: Analyzes data, finds patterns, generates statistics
        3. Code Agent: Writes code, analyzes repositories, understands programming languages
        4. Validator Agent: Validates results, checks quality, ensures accuracy
        
        Return the subtasks as a JSON array with this structure:
        [
            {{
                "agent_type": "research|analysis|code|validator",
                "description": "Clear description of what this agent should do",
                "parameters": {{...}},  // Any specific parameters for the task
                "dependencies": []  // Array of subtask indices this depends on (0-based)
            }}
        ]
        
        Be specific and ensure proper task ordering through dependencies.
        Return ONLY valid JSON, no additional text.
        """
        
        try:
            response = self.model.generate_content(prompt)
            subtasks_json = response.text.strip()
            
            # Clean up response if needed
            if subtasks_json.startswith('```json'):
                subtasks_json = subtasks_json[7:]
            if subtasks_json.endswith('```'):
                subtasks_json = subtasks_json[:-3]
                
            subtasks = json.loads(subtasks_json)
            return subtasks
            
        except Exception as e:
            logger.error(f"Failed to decompose task: {e}")
            # Fallback to simple decomposition
            return [{
                "agent_type": "research",
                "description": f"Research information for: {user_query}",
                "parameters": {"query": user_query},
                "dependencies": []
            }]
    
    def generate_response(self, prompt: str) -> str:
        """Generate a response using Gemini"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return f"Error generating response: {str(e)}"

# Caching utility for expensive operations
class CacheManager:
    """Simple cache manager using Firestore"""
    
    def __init__(self, db: firestore.Client):
        self.db = db
        self.cache_collection = 'cache'
        
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        doc_ref = self.db.collection(self.cache_collection).document(key)
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict()
            # Check if cache is still valid (default 1 hour)
            if 'expires_at' in data:
                expires_at = datetime.fromisoformat(data['expires_at'])
                if expires_at > datetime.utcnow():
                    return data.get('value')
        return None
        
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        """Set cached value with TTL"""
        expires_at = datetime.utcnow().timestamp() + ttl_seconds
        
        doc_ref = self.db.collection(self.cache_collection).document(key)
        doc_ref.set({
            'value': value,
            'expires_at': datetime.fromtimestamp(expires_at).isoformat(),
            'created_at': datetime.utcnow().isoformat()
        })
        
    def generate_key(self, *args) -> str:
        """Generate cache key from arguments"""
        key_str = '_'.join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()