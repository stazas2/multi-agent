from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from enum import Enum
from pathlib import PurePosixPath
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
    # Generated artifacts (e.g., packages)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    
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
        data['artifacts'] = data.get('artifacts', {})
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
    retry_count: int = 0
    last_error: Optional[str] = None
    priority: int = 0
    manual: bool = False
    manual_triggered_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['agent_type'] = self.agent_type.value
        data['status'] = self.status.value
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        if self.manual_triggered_at:
            data['manual_triggered_at'] = self.manual_triggered_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SubTask':
        data['agent_type'] = AgentType(data.get('agent_type', 'research'))
        data['status'] = TaskStatus(data.get('status', 'pending'))
        if data.get('started_at'):
            data['started_at'] = datetime.fromisoformat(data['started_at'])
        if data.get('completed_at'):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        if data.get('manual_triggered_at'):
            data['manual_triggered_at'] = datetime.fromisoformat(data['manual_triggered_at'])
        data['lock_version'] = data.get('lock_version', 0)
        data['retry_count'] = data.get('retry_count', 0)
        data['last_error'] = data.get('last_error')
        data['priority'] = data.get('priority', 0)
        data['manual'] = data.get('manual', False)
        return cls(**data)


def _normalise_path(path: str) -> str:
    posix = PurePosixPath(path.replace("\\", "/"))
    if posix.is_absolute():
        raise ValueError("Generated asset path must be relative")

    parts = []
    for part in posix.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            raise ValueError("Generated asset path cannot traverse upwards ('..')")
        parts.append(part)

    if not parts:
        raise ValueError("Generated asset path must contain at least one segment")

    return "/".join(parts)


@dataclass
class GeneratedAsset:
    """Represents a single file generated by the code agent."""

    path: str
    content: str
    executable: bool = False
    media_type: Optional[str] = None

    def __post_init__(self) -> None:
        self.path = _normalise_path(self.path)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "content": self.content,
            "executable": self.executable,
            "media_type": self.media_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratedAsset":
        return cls(
            path=data.get("path", ""),
            content=data.get("content", ""),
            executable=bool(data.get("executable", False)),
            media_type=data.get("media_type"),
        )


@dataclass
class GeneratedPackage:
    """A collection of generated assets plus metadata."""

    files: List[GeneratedAsset] = field(default_factory=list)
    name: str = "code-package"
    entrypoint: Optional[str] = None
    instructions: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = "code-package"
        self.validate()

    def validate(self) -> None:
        seen: Dict[str, GeneratedAsset] = {}
        for asset in self.files:
            if asset.path in seen:
                raise ValueError(f"Duplicate generated asset path: {asset.path}")
            seen[asset.path] = asset

        if self.entrypoint:
            normalised_entry = _normalise_path(self.entrypoint)
            if normalised_entry not in seen:
                raise ValueError("Package entrypoint must reference an existing file")
            self.entrypoint = normalised_entry

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "files": [asset.to_dict() for asset in self.files],
            "entrypoint": self.entrypoint,
            "instructions": self.instructions,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratedPackage":
        raw_files = data.get("files", [])
        files = [
            asset if isinstance(asset, GeneratedAsset) else GeneratedAsset.from_dict(asset)
            for asset in raw_files
        ]
        package = cls(
            files=files,
            name=data.get("name", "code-package"),
            entrypoint=data.get("entrypoint"),
            instructions=data.get("instructions"),
            metadata=data.get("metadata", {}),
        )
        return package
