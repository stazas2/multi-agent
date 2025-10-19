import os
import logging
import asyncio
import json
import time
import hashlib
import random
from typing import Optional, Dict, Any, List
from datetime import datetime

from shared.models import TaskContext, TaskStatus, SubTask, AgentType, AgentMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOCAL_MODE = str(os.environ.get("LOCAL_MODE", "0")).lower() in {"1", "true", "yes"}

if not LOCAL_MODE:
    from google.cloud import firestore, pubsub_v1, tasks_v2
    import google.generativeai as genai
else:
    firestore = None  # type: ignore
    pubsub_v1 = None  # type: ignore
    tasks_v2 = None  # type: ignore
    genai = None  # type: ignore


# === Firestore / Storage Management =====================================================

if not LOCAL_MODE:

    class FirestoreManager:
        """Manages Firestore operations for the multi-agent system"""

        def __init__(self, project_id: str):
            self.project_id = project_id
            self.db = firestore.Client(project=project_id)

        def save_task_context(self, context: TaskContext) -> None:
            context.updated_at = datetime.utcnow()
            doc_ref = self.db.collection("tasks").document(context.task_id)
            doc_ref.set(context.to_dict())
            logger.info("Saved task context: %s", context.task_id)

        def get_task_context(self, task_id: str) -> Optional[TaskContext]:
            doc = self.db.collection("tasks").document(task_id).get()
            if doc.exists:
                return TaskContext.from_dict(doc.to_dict())
            return None

        def update_task_status(self, task_id: str, status: TaskStatus) -> None:
            doc_ref = self.db.collection("tasks").document(task_id)
            doc_ref.update({
                "status": status.value,
                "updated_at": datetime.utcnow().isoformat()
            })
            logger.info("Updated task %s status to %s", task_id, status.value)

        def save_subtask(self, subtask: SubTask) -> None:
            doc_ref = self.db.collection("subtasks").document(subtask.subtask_id)
            doc_ref.set(subtask.to_dict())
            logger.info("Saved subtask: %s", subtask.subtask_id)

        def get_subtasks_for_task(self, task_id: str) -> List[SubTask]:
            query = self.db.collection("subtasks").where("parent_task_id", "==", task_id)
            docs = query.stream()
            return [SubTask.from_dict(doc.to_dict()) for doc in docs]

        def get_subtask(self, subtask_id: str) -> Optional[SubTask]:
            doc = self.db.collection("subtasks").document(subtask_id).get()
            if doc.exists:
                return SubTask.from_dict(doc.to_dict())
            return None

        def update_agent_result(self, task_id: str, agent_type: str, result: Dict[str, Any]) -> None:
            doc_ref = self.db.collection("tasks").document(task_id)
            doc_ref.update({
                f"agent_results.{agent_type}": result,
                "updated_at": datetime.utcnow().isoformat()
            })
            logger.info("Updated %s results for task %s", agent_type, task_id)

else:

    class FirestoreManager:
        """In-memory Firestore replacement for local mode"""

        def __init__(self, project_id: str):
            self.project_id = project_id
            self.tasks: Dict[str, TaskContext] = {}
            self.subtasks: Dict[str, SubTask] = {}

        def save_task_context(self, context: TaskContext) -> None:
            context.updated_at = datetime.utcnow()
            # Store a clone to avoid shared references
            self.tasks[context.task_id] = TaskContext.from_dict(context.to_dict())
            logger.debug("[LOCAL] Saved task context: %s", context.task_id)

        def get_task_context(self, task_id: str) -> Optional[TaskContext]:
            ctx = self.tasks.get(task_id)
            return TaskContext.from_dict(ctx.to_dict()) if ctx else None

        def update_task_status(self, task_id: str, status: TaskStatus) -> None:
            ctx = self.tasks.get(task_id)
            if ctx:
                ctx.status = status
                ctx.updated_at = datetime.utcnow()
                self.tasks[task_id] = ctx
                logger.debug("[LOCAL] Updated task %s status to %s", task_id, status.value)

        def save_subtask(self, subtask: SubTask) -> None:
            self.subtasks[subtask.subtask_id] = SubTask.from_dict(subtask.to_dict())
            logger.debug("[LOCAL] Saved subtask: %s", subtask.subtask_id)

        def get_subtasks_for_task(self, task_id: str) -> List[SubTask]:
            return [SubTask.from_dict(st.to_dict()) for st in self.subtasks.values() if st.parent_task_id == task_id]

        def get_subtask(self, subtask_id: str) -> Optional[SubTask]:
            st = self.subtasks.get(subtask_id)
            return SubTask.from_dict(st.to_dict()) if st else None

        def update_agent_result(self, task_id: str, agent_type: str, result: Dict[str, Any]) -> None:
            ctx = self.tasks.get(task_id)
            if ctx:
                ctx.agent_results[agent_type] = result
                ctx.updated_at = datetime.utcnow()
                self.tasks[task_id] = ctx
                logger.debug("[LOCAL] Updated %s results for task %s", agent_type, task_id)


# === Pub/Sub Management ================================================================

if not LOCAL_MODE:

    class PubSubManager:
        """Manages Pub/Sub operations"""

        def __init__(self, project_id: str, **_: Any):
            self.project_id = project_id
            self.publisher = pubsub_v1.PublisherClient()
            self.subscriber = pubsub_v1.SubscriberClient()
            self._result_handler = None

            self.topics = {
                "dispatch": f"projects/{project_id}/topics/agent-task-dispatch",
                "results": f"projects/{project_id}/topics/agent-task-results",
                "research": f"projects/{project_id}/topics/agent-research-tasks",
                "analysis": f"projects/{project_id}/topics/agent-analysis-tasks",
                "code": f"projects/{project_id}/topics/agent-code-tasks",
                "validator": f"projects/{project_id}/topics/agent-validator-tasks",
            }

        def register_result_handler(self, handler):  # pragma: no cover - only used in local mode
            self._result_handler = handler

        def publish_message(self, topic_name: str, message: AgentMessage) -> str:
            topic_path = self.topics.get(topic_name)
            if not topic_path:
                raise ValueError(f"Unknown topic: {topic_name}")

            message_data = message.to_json().encode("utf-8")
            attributes = {
                "task_id": message.task_id,
                "agent_type": message.agent_type.value,
                "priority": str(message.priority),
            }

            future = self.publisher.publish(topic_path, message_data, **attributes)
            message_id = future.result()
            logger.info("Published message %s to %s", message_id, topic_name)
            return message_id

        def dispatch_to_agent(self, agent_type: AgentType, message: AgentMessage) -> str:
            topic_map = {
                AgentType.RESEARCH: "research",
                AgentType.ANALYSIS: "analysis",
                AgentType.CODE: "code",
                AgentType.VALIDATOR: "validator",
            }
            topic_name = topic_map.get(agent_type)
            if not topic_name:
                raise ValueError(f"Cannot dispatch to agent type: {agent_type}")
            return self.publish_message(topic_name, message)

else:

    class PubSubManager:
        """In-process task dispatcher for local mode"""

        def __init__(self, project_id: str, firestore_manager: Optional[FirestoreManager] = None):
            self.project_id = project_id
            self.firestore = firestore_manager
            self._result_handler = None
            self._message_counter = 0

        def register_result_handler(self, handler):
            self._result_handler = handler

        def dispatch_to_agent(self, agent_type: AgentType, message: AgentMessage) -> str:
            self._message_counter += 1
            message_id = f"local-{self._message_counter}"
            asyncio.create_task(self._simulate_agent_execution(agent_type, message))
            logger.debug("[LOCAL] Dispatched %s to %s", message.payload.get("subtask_id"), agent_type.value)
            return message_id

        async def _simulate_agent_execution(self, agent_type: AgentType, message: AgentMessage) -> None:
            await asyncio.sleep(0.1)  # Simulate processing time

            result_data: Dict[str, Any]
            error: Optional[str] = None

            try:
                result_data = self._generate_agent_result(agent_type, message)
            except Exception as exc:  # pragma: no cover - unexpected local failures
                error = str(exc)
                result_data = {}

            if not self._result_handler:
                return

            payload = {
                "task_id": message.task_id,
                "subtask_id": message.payload.get("subtask_id"),
                "agent_type": agent_type.value,
                "result": result_data if not error else None,
                "error": error,
            }

            await self._result_handler(payload)

        def _generate_agent_result(self, agent_type: AgentType, message: AgentMessage) -> Dict[str, Any]:
            description = message.payload.get("description", "")
            task_id = message.task_id
            subtask_id = message.payload.get("subtask_id")
            now = datetime.utcnow().isoformat()

            research_result = {
                "type": "web_research",
                "query": description,
                "sources": [
                    f"https://example.com/{abs(hash(description)) % 1000}",
                    f"https://knowledge-base.internal/{abs(hash(description + 'kb')) % 1000}",
                ],
                "synthesis": f"Synthesised findings for '{description}'",
                "highlights": [
                    f"Key point #{i + 1} about {description}"
                    for i in range(3)
                ],
                "timestamp": now,
            }

            if agent_type == AgentType.RESEARCH:
                return research_result

            if agent_type == AgentType.ANALYSIS:
                context = self.firestore.get_task_context(task_id) if self.firestore else None
                research = context.agent_results.get("research") if context else research_result
                topics = research.get("highlights", []) if isinstance(research, dict) else []
                summary = "\n".join(topics) if topics else research.get("synthesis", "")
                return {
                    "type": "statistical_analysis",
                    "description": description,
                    "summary": summary or f"Analysis summary for {description}",
                    "insights": topics or [f"Insight #{i + 1} for {description}" for i in range(2)],
                    "timestamp": now,
                }

            if agent_type == AgentType.CODE:
                code_snippet = (
                    "def generated_solution(input_data):\n"
                    "    \"\"\"Auto-generated sample implementation\"\"\"\n"
                    f"    # Task: {description}\n"
                    "    return {\"status\": \"ok\", \"details\": input_data}\n"
                )
                return {
                    "type": "code_generation",
                    "description": description,
                    "language": "python",
                    "code": code_snippet,
                    "timestamp": now,
                }

            if agent_type == AgentType.VALIDATOR:
                return {
                    "type": "validation",
                    "description": description,
                    "quality_score": 0.9,
                    "passed": True,
                    "issues": [],
                    "timestamp": now,
                }

            raise ValueError(f"Unsupported agent type: {agent_type}")


# === Cloud Tasks Management =============================================================

if not LOCAL_MODE:

    class CloudTasksManager:
        """Manages Cloud Tasks for scheduled operations"""

        def __init__(self, project_id: str, location: str = "us-central1"):
            self.project_id = project_id
            self.location = location
            self.client = tasks_v2.CloudTasksClient()
            self.queue_path = self.client.queue_path(project_id, location, "agent-task-queue")

        def create_task(self, url: str, payload: Dict[str, Any], delay_seconds: int = 0) -> str:
            task: Dict[str, Any] = {
                "http_request": {
                    "http_method": tasks_v2.HttpMethod.POST,
                    "url": url,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps(payload).encode(),
                }
            }

            if delay_seconds > 0:
                scheduled_time = time.time() + delay_seconds
                from google.protobuf import timestamp_pb2

                timestamp = timestamp_pb2.Timestamp()
                timestamp.FromSeconds(int(scheduled_time))
                task["schedule_time"] = timestamp

            response = self.client.create_task(request={"parent": self.queue_path, "task": task})
            logger.info("Created task: %s", response.name)
            return response.name

else:

    class CloudTasksManager:
        """No-op Cloud Tasks replacement for local mode"""

        def __init__(self, project_id: str, location: str = "us-central1"):
            self.project_id = project_id
            self.location = location

        def create_task(self, url: str, payload: Dict[str, Any], delay_seconds: int = 0) -> str:
            logger.debug("[LOCAL] Scheduled task to %s with delay %s", url, delay_seconds)
            return f"local-task-{int(time.time() * 1000)}"


# === Gemini / LLM Management ============================================================

if not LOCAL_MODE:

    class GeminiManager:
        """Manages Gemini API interactions"""

        def __init__(self, api_key: Optional[str] = None):
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("Gemini API key not provided")

            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-1.5-pro")

        def decompose_task(self, user_query: str) -> List[Dict[str, Any]]:
            prompt = f"""
            You are a task orchestrator for a multi-agent AI system.

            Given this user query: "{user_query}"

            Break it down into specific subtasks that can be handled by these specialized agents:
            1. Research Agent
            2. Analysis Agent
            3. Code Agent
            4. Validator Agent

            Return ONLY valid JSON with fields agent_type, description, parameters, dependencies (list of indices).
            """

            try:
                response = self.model.generate_content(prompt)
                subtasks_json = response.text.strip()
                if subtasks_json.startswith("```json"):
                    subtasks_json = subtasks_json[7:]
                if subtasks_json.endswith("```"):
                    subtasks_json = subtasks_json[:-3]
                return json.loads(subtasks_json)
            except Exception as exc:
                logger.error("Failed to decompose task: %s", exc)
                return [
                    {
                        "agent_type": "research",
                        "description": f"Research information for: {user_query}",
                        "parameters": {"query": user_query},
                        "dependencies": [],
                    }
                ]

        def generate_response(self, prompt: str) -> str:
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as exc:
                logger.error("Gemini generation failed: %s", exc)
                return f"Error generating response: {exc}"

else:

    class GeminiManager:
        """Lightweight heuristic responses for local mode"""

        def __init__(self, api_key: Optional[str] = None):
            self.api_key = api_key

        def decompose_task(self, user_query: str) -> List[Dict[str, Any]]:
            return [
                {
                    "agent_type": "research",
                    "description": f"Gather key facts for: {user_query}",
                    "parameters": {},
                    "dependencies": [],
                },
                {
                    "agent_type": "analysis",
                    "description": "Summarise and analyse the findings",
                    "parameters": {},
                    "dependencies": [0],
                },
                {
                    "agent_type": "code",
                    "description": "Draft helper code or pseudo-code based on findings",
                    "parameters": {"language": "python"},
                    "dependencies": [0],
                },
                {
                    "agent_type": "validator",
                    "description": "Validate the aggregated outcome",
                    "parameters": {},
                    "dependencies": [0, 1, 2],
                },
            ]

        def generate_response(self, prompt: str) -> str:
            # Very small heuristic to keep responses readable in local mode
            lines = [line.strip() for line in prompt.splitlines() if line.strip()]
            summary_lines = [line for line in lines if line.lower().startswith("agent results")]
            conclusion = "Local synthesis: " + (summary_lines[0] if summary_lines else "Task processed successfully.")
            return conclusion + "\n(This response was generated in LOCAL_MODE without external LLMs.)"


# === Cache Management ==================================================================

if not LOCAL_MODE:

    class CacheManager:
        """Simple cache manager using Firestore"""

        def __init__(self, db: firestore.Client):
            self.db = db
            self.cache_collection = "cache"

        def get(self, key: str) -> Optional[Any]:
            doc = self.db.collection(self.cache_collection).document(key).get()
            if doc.exists:
                data = doc.to_dict()
                if "expires_at" in data:
                    expires_at = datetime.fromisoformat(data["expires_at"])
                    if expires_at > datetime.utcnow():
                        return data.get("value")
            return None

        def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
            expires_at = datetime.utcnow().timestamp() + ttl_seconds
            self.db.collection(self.cache_collection).document(key).set({
                "value": value,
                "expires_at": datetime.fromtimestamp(expires_at).isoformat(),
                "created_at": datetime.utcnow().isoformat(),
            })

        def generate_key(self, *args) -> str:
            key_str = "_".join(str(arg) for arg in args)
            return hashlib.md5(key_str.encode()).hexdigest()

else:

    class CacheManager:
        """In-memory cache for local mode"""

        def __init__(self):
            self._cache: Dict[str, Dict[str, Any]] = {}

        def get(self, key: str) -> Optional[Any]:
            item = self._cache.get(key)
            if not item:
                return None
            if item["expires_at"] and item["expires_at"] < time.time():
                self._cache.pop(key, None)
                return None
            return item["value"]

        def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
            self._cache[key] = {
                "value": value,
                "expires_at": time.time() + ttl_seconds if ttl_seconds else None,
            }

        def generate_key(self, *args) -> str:
            key_str = "_".join(str(arg) for arg in args)
            return hashlib.md5(key_str.encode()).hexdigest()
