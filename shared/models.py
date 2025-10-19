from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from enum import Enum
import uuid
import json

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AgentType(Enum):
    ORCHESTRATOR = "orchestrator"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    CODE = "code"
    VALIDATOR = "validator"

@dataclass
class TaskContext:
    """Shared context for a task across all agents"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_query: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    status: TaskStatus = TaskStatus.PENDING
    
    # Task breakdown
    subtasks: List[Dict[str, Any]] = field(default_factory=list)
    
    # Results from each agent
    agent_results: Dict[str, Any] = field(default_factory=dict)
    
    # Final aggregated result
    final_result: Optional[str] = None
    
    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Distributed tracing support
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Optimistic locking
    lock_version: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Firestore"""
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskContext':
        """Create from Firestore document"""
        data['status'] = TaskStatus(data.get('status', 'pending'))
        data['created_at'] = datetime.fromisoformat(data.get('created_at', datetime.utcnow().isoformat()))
        data['updated_at'] = datetime.fromisoformat(data.get('updated_at', datetime.utcnow().isoformat()))
        data['trace_id'] = data.get('trace_id') or str(uuid.uuid4())
        data['lock_version'] = data.get('lock_version', 0)
        return cls(**data)

@dataclass
class AgentMessage:
    """Message format for agent communication via Pub/Sub"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    agent_type: AgentType = AgentType.ORCHESTRATOR
    action: str = ""  # e.g., "process", "validate", "analyze"
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # For tracking message flow
    source_agent: Optional[str] = None
    target_agent: Optional[str] = None
    
    # Priority and retry information
    priority: int = 0  # 0 = normal, higher = more urgent
    retry_count: int = 0
    max_retries: int = 3
    
    def to_json(self) -> str:
        """Serialize for Pub/Sub"""
        data = asdict(self)
        data['agent_type'] = self.agent_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AgentMessage':
        """Deserialize from Pub/Sub"""
        data = json.loads(json_str)
        data['agent_type'] = AgentType(data.get('agent_type', 'orchestrator'))
        data['timestamp'] = datetime.fromisoformat(data.get('timestamp', datetime.utcnow().isoformat()))
        return cls(**data)

@dataclass
class SubTask:
    """Individual subtask assigned to an agent"""
    subtask_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_task_id: str = ""
    agent_type: AgentType = AgentType.RESEARCH
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # List of subtask_ids this depends on
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    lock_version: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['agent_type'] = self.agent_type.value
        data['status'] = self.status.value
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SubTask':
        data['agent_type'] = AgentType(data.get('agent_type', 'research'))
        data['status'] = TaskStatus(data.get('status', 'pending'))
        if data.get('started_at'):
            data['started_at'] = datetime.fromisoformat(data['started_at'])
        if data.get('completed_at'):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        data['lock_version'] = data.get('lock_version', 0)
        return cls(**data)
