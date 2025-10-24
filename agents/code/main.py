"""
Code Agent - Responsible for code analysis and generation
"""

import os
import json
import logging
import base64
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import functions_framework
from google.cloud import firestore
import google.generativeai as genai
import ast
import re
from collections import defaultdict
import requests

from dataclasses import dataclass, field
from pathlib import PurePosixPath


def _resolve_agent_model(agent_key: str) -> str:
    env_key = f"MODEL_{agent_key.upper()}"
    return os.environ.get(env_key) or os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")


def _normalise_path(path: str) -> str:
    posix = PurePosixPath(path.replace("\\", "/"))
    if posix.is_absolute():
        raise ValueError("Generated asset path must be relative")

    parts: List[str] = []
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
project_id = os.environ.get('PROJECT_ID')
firestore_client = firestore.Client(project=project_id)

# Initialize Gemini
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
MODEL_NAME = _resolve_agent_model("CODE")
model = genai.GenerativeModel(MODEL_NAME)
ORCHESTRATOR_URL = os.environ.get('ORCHESTRATOR_URL', '').rstrip('/')
LOCAL_MODE = str(os.environ.get("LOCAL_MODE", "0")).lower() in {"1", "true", "yes"}


def notify_orchestrator(task_id: str, subtask_id: str, agent_type: str, result: Optional[Dict[str, Any]], error: Optional[str]) -> None:
    """Post agent outcome back to the orchestrator webhook."""
    if not ORCHESTRATOR_URL:
        logger.warning("ORCHESTRATOR_URL not set; skipping orchestrator notification for %s", subtask_id)
        return
    if LOCAL_MODE:
        logger.debug("[LOCAL] Skipping orchestrator webhook notification for %s", subtask_id)
        return
    payload = {
        "task_id": task_id,
        "subtask_id": subtask_id,
        "agent_type": agent_type,
        "result": result,
        "error": error,
    }
    try:
        response = requests.post(f"{ORCHESTRATOR_URL}/webhook/agent-result", json=payload, timeout=10)
        response.raise_for_status()
        logger.info("Reported %s completion to orchestrator", subtask_id)
    except Exception as exc:
        logger.error("Failed to notify orchestrator for %s: %s", subtask_id, exc)

class CodeAgent:
    """Agent specialized in code analysis and generation"""
    
    def __init__(self):
        self.firestore = firestore_client
        self.gemini = model
        
    def process_task(self, subtask_id: str, description: str, 
                    parameters: Dict[str, Any], task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a code-related task"""
        
        logger.info(f"Processing code task: {subtask_id}")
        
        task_type = parameters.get('type', 'analyze')
        
        if task_type == 'analyze':
            return self.analyze_code(description, parameters, task_context)
        elif task_type == 'generate':
            return self.generate_code(description, parameters, task_context)
        elif task_type == 'review':
            return self.review_code(description, parameters, task_context)
        elif task_type == 'refactor':
            return self.refactor_code(description, parameters, task_context)
        else:
            return self.general_code_task(description, parameters, task_context)
            
    def analyze_code(self, description: str, parameters: Dict[str, Any], 
                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code from repositories or snippets"""
        
        analysis_results = {
            'languages': defaultdict(int),
            'frameworks': [],
            'dependencies': [],
            'patterns': [],
            'metrics': {}
        }
        
        # Get repositories from research agent results
        agent_results = context.get('agent_results', {})
        if 'research' in agent_results and 'results' in agent_results['research']:
            repos = agent_results['research'].get('results', [])
            
            for repo in repos:
                # Analyze language distribution
                if 'language' in repo and repo['language']:
                    analysis_results['languages'][repo['language']] += 1
                    
                # Extract frameworks from topics
                if 'topics' in repo:
                    for topic in repo['topics']:
                        if any(fw in topic.lower() for fw in ['react', 'vue', 'angular', 'django', 'flask', 'spring']):
                            if topic not in analysis_results['frameworks']:
                                analysis_results['frameworks'].append(topic)
                                
        # Use Gemini for deeper analysis
        gemini_analysis = self.gemini_code_analysis(repos if 'repos' in locals() else [], description)
        analysis_results['detailed_analysis'] = gemini_analysis
        
        # Calculate metrics
        analysis_results['metrics'] = {
            'total_repos_analyzed': len(repos) if 'repos' in locals() else 0,
            'unique_languages': len(analysis_results['languages']),
            'most_popular_language': max(analysis_results['languages'].items(), 
                                        key=lambda x: x[1])[0] if analysis_results['languages'] else None
        }
        
        return {
            'type': 'code_analysis',
            'description': description,
            'results': dict(analysis_results),
            'timestamp': datetime.utcnow().isoformat()
        }
        
    def generate_code(self, description: str, parameters: Dict[str, Any], 
                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code based on requirements"""
        
        language = parameters.get('language', 'python')
        framework = parameters.get('framework', '')
        package_flag = parameters.get('package')
        package_requested = True if package_flag is None else bool(package_flag)
        
        # Build context from other agents
        agent_results = context.get('agent_results', {})
        
        base_context = json.dumps(agent_results.get('research', {}), indent=2)[:2000]
        requirements = json.dumps(parameters.get('requirements', []), indent=2)

        if package_requested:
            prompt = f"""
            You are generating a multi-file project for the following request:
            {description}

            Target language: {language}
            Preferred framework: {framework if framework else 'any appropriate'}

            Research context:
            {base_context}

            Explicit requirements:
            {requirements}

            Return ONLY valid JSON with this structure:
            {{
              "summary": "<one paragraph description>",
              "package": {{
                "name": "<short name>",
                "entrypoint": "<relative path to main file>",
                "instructions": "<how to run the project>",
                "metadata": {{}},
                "files": [
                  {{"path": "src/main.py", "content": "...", "executable": false}}
                ]
              }}
            }}
            Each file content must be UTF-8 text. Do not include markdown fences.
            """
        else:
            prompt = f"""
            Generate code for: {description}

            Language: {language}
            Framework: {framework if framework else 'any appropriate'}

            Context from research:
            {base_context}

            Requirements:
            {requirements}

            Generate clean, well-commented, production-ready code.
            Include error handling and best practices.
            """
        
        try:
            response = self.gemini.generate_content(prompt)
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            if package_requested:
                package, summary = self.build_fallback_package("", language, description, note=str(e))
                return self.build_package_result(
                    package,
                    summary,
                    language,
                    framework,
                    description,
                    notes={"error": str(e)},
                )
            return {
                'type': 'code_generation',
                'description': description,
                'language': language,
                'framework': framework,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

        if package_requested:
            try:
                package, summary = self.parse_generated_package(
                    response.text,
                    description=description,
                    language=language,
                    framework=framework,
                    parameters=parameters,
                )
                return self.build_package_result(package, summary, language, framework, description)
            except Exception as parse_error:
                logger.warning("Failed to parse package output: %s", parse_error)
                package, summary = self.build_fallback_package(
                    response.text,
                    language,
                    description,
                    note=str(parse_error),
                )
                return self.build_package_result(
                    package,
                    summary,
                    language,
                    framework,
                    description,
                    notes={"warning": str(parse_error)},
                )

        # Non-package flow
        code_blocks = self.extract_code_blocks(response.text)
        
        return {
            'type': 'code_generation',
            'description': description,
            'language': language,
            'framework': framework,
            'code': code_blocks[0] if code_blocks else response.text,
            'all_code_blocks': code_blocks,
            'explanation': self.extract_explanation(response.text),
            'timestamp': datetime.utcnow().isoformat()
        }
        
    def review_code(self, description: str, parameters: Dict[str, Any], 
                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Review code for quality, security, and best practices"""
        
        code = parameters.get('code', '')
        if not code:
            # Try to get code from context
            agent_results = context.get('agent_results', {})
            if 'code' in agent_results and 'code' in agent_results['code']:
                code = agent_results['code']['code']
                
        if not code:
            return {
                'type': 'code_review',
                'error': 'No code provided for review',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        prompt = f"""
        Perform a comprehensive code review for:
        
        {code[:5000]}  # Limit code size
        
        Review for:
        1. Code quality and readability
        2. Security vulnerabilities
        3. Performance issues
        4. Best practices
        5. Potential bugs
        
        Provide specific, actionable feedback.
        Rate the code on a scale of 1-10.
        """
        
        try:
            response = self.gemini.generate_content(prompt)
            
            return {
                'type': 'code_review',
                'description': description,
                'review': response.text,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Code review failed: {e}")
            return {
                'type': 'code_review',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
            
    def refactor_code(self, description: str, parameters: Dict[str, Any], 
                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Refactor code for improvement"""
        
        original_code = parameters.get('code', '')
        refactor_goals = parameters.get('goals', ['improve readability', 'optimize performance'])
        
        prompt = f"""
        Refactor this code:
        
        {original_code[:5000]}
        
        Refactoring goals:
        {json.dumps(refactor_goals)}
        
        Provide:
        1. Refactored code
        2. Explanation of changes
        3. Benefits of refactoring
        """
        
        try:
            response = self.gemini.generate_content(prompt)
            
            code_blocks = self.extract_code_blocks(response.text)
            
            return {
                'type': 'code_refactoring',
                'description': description,
                'original_code': original_code[:1000] + '...' if len(original_code) > 1000 else original_code,
                'refactored_code': code_blocks[0] if code_blocks else '',
                'explanation': response.text,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Code refactoring failed: {e}")
            return {
                'type': 'code_refactoring',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    def parse_generated_package(
        self,
        response_text: str,
        description: str,
        language: str,
        framework: str,
        parameters: Dict[str, Any],
    ) -> Tuple[GeneratedPackage, str]:
        """Parse the LLM response into a GeneratedPackage."""
        text = response_text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].lstrip()

        # Extract JSON portion even if extra text surrounds it
        start = text.find("{")
        if start == -1:
            raise ValueError("Package response missing JSON payload")
        end = text.rfind("}")
        if end == -1 or end < start:
            raise ValueError("Package response contains malformed JSON")

        json_fragment = text[start:end + 1]
        payload = json.loads(json_fragment)
        package_payload = payload.get("package", payload)
        package = GeneratedPackage.from_dict(package_payload)
        package.metadata.setdefault("language", language)
        if framework:
            package.metadata.setdefault("framework", framework)
        if parameters.get("requirements"):
            package.metadata.setdefault("requirements", parameters.get("requirements"))
        summary = payload.get("summary") or package.instructions or description
        package.validate()
        return package, summary

    def build_package_result(
        self,
        package: GeneratedPackage,
        summary: str,
        language: str,
        framework: str,
        description: str,
        *,
        notes: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compose the response payload for a generated package."""
        primary_asset = package.files[0] if package.files else None
        code_blocks = [asset.content for asset in package.files]
        result: Dict[str, Any] = {
            'type': 'code_generation',
            'description': description,
            'language': language,
            'framework': framework,
            'code': primary_asset.content if primary_asset else '',
            'all_code_blocks': code_blocks,
            'package': package.to_dict(),
            'summary': summary,
            'explanation': summary,
            'instructions': package.instructions,
            'timestamp': datetime.utcnow().isoformat(),
        }
        if notes:
            result['notes'] = notes
        return result

    def build_fallback_package(
        self,
        raw_text: str,
        language: str,
        description: str,
        *,
        note: Optional[str] = None,
    ) -> Tuple[GeneratedPackage, str]:
        """Create a minimal package when parsing fails."""
        extension = self.guess_file_extension(language)
        filename = f"main.{extension}" if extension else "main.txt"
        content = raw_text or "# Unable to generate code content."
        asset = GeneratedAsset(path=filename, content=content)
        metadata = {"language": language, "fallback": True}
        if note:
            metadata["note"] = note
        package = GeneratedPackage(
            files=[asset],
            entrypoint=asset.path,
            instructions="Review and adjust the generated snippet.",
            metadata=metadata,
        )
        summary = f"Fallback package generated for: {description}"
        return package, summary

    def guess_file_extension(self, language: str) -> str:
        """Map language names to reasonable file extensions."""
        mapping = {
            'python': 'py',
            'javascript': 'js',
            'typescript': 'ts',
            'go': 'go',
            'rust': 'rs',
            'java': 'java',
            'csharp': 'cs',
            'c++': 'cpp',
            'c': 'c',
            'bash': 'sh',
            'shell': 'sh',
            'html': 'html',
            'css': 'css',
            'kotlin': 'kt',
            'swift': 'swift',
            'ruby': 'rb',
            'php': 'php',
        }
        return mapping.get(language.lower(), 'txt')

    def general_code_task(self, description: str, parameters: Dict[str, Any], 
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general code-related tasks"""
        
        prompt = f"""
        Code Task: {description}
        
        Parameters: {json.dumps(parameters)}
        
        Context: {json.dumps(context.get('agent_results', {}), indent=2)[:3000]}
        
        Please complete this code-related task and provide:
        1. Solution/Implementation
        2. Explanation
        3. Any relevant code snippets
        """
        
        try:
            response = self.gemini.generate_content(prompt)
            
            return {
                'type': 'general_code_task',
                'description': description,
                'result': response.text,
                'code_blocks': self.extract_code_blocks(response.text),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"General code task failed: {e}")
            return {
                'type': 'general_code_task',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
            
    def gemini_code_analysis(self, repos: List[Dict[str, Any]], description: str) -> str:
        """Use Gemini for detailed code analysis"""
        
        prompt = f"""
        Analyze these repositories for: {description}
        
        Repositories:
        {json.dumps(repos, indent=2)[:3000]}
        
        Provide insights on:
        1. Technology stack trends
        2. Architecture patterns
        3. Common dependencies
        4. Best practices observed
        5. Recommendations
        """
        
        try:
            response = self.gemini.generate_content(prompt)
            return response.text
        except:
            return "Unable to perform detailed analysis"
            
    def extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks from text"""
        
        # Pattern for code blocks with ``` markers
        pattern = r'```(?:\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        # If no code blocks found, try to find indented code
        if not matches:
            lines = text.split('\n')
            code_lines = []
            in_code = False
            
            for line in lines:
                if line.startswith('    '):  # 4 spaces = code
                    in_code = True
                    code_lines.append(line[4:])
                elif in_code and line.strip() == '':
                    code_lines.append('')
                elif in_code:
                    if code_lines:
                        matches.append('\n'.join(code_lines))
                    code_lines = []
                    in_code = False
                    
            if code_lines:
                matches.append('\n'.join(code_lines))
                
        return matches
        
    def extract_explanation(self, text: str) -> str:
        """Extract explanation text, excluding code blocks"""
        
        # Remove code blocks
        pattern = r'```(?:\w+)?\n.*?```'
        explanation = re.sub(pattern, '', text, flags=re.DOTALL)
        
        # Clean up extra whitespace
        explanation = re.sub(r'\n{3,}', '\n\n', explanation)
        
        return explanation.strip()

# Cloud Function entry point
@functions_framework.cloud_event
def handle_message(cloud_event):
    """Cloud Function entry point for Pub/Sub messages"""
    
    # Decode the Pub/Sub message
    message_data = base64.b64decode(cloud_event.data["message"]["data"]).decode()
    message = json.loads(message_data)
    
    logger.info(f"Code Agent received message: {message.get('subtask_id')}")
    
    # Extract task details
    subtask_id = message['payload']['subtask_id']
    description = message['payload']['description']
    parameters = message['payload']['parameters']
    task_id = message['task_id']
    
    # Get task context
    task_ref = firestore_client.collection('tasks').document(task_id)
    task_doc = task_ref.get()
    task_context = task_doc.to_dict() if task_doc.exists else {}
    
    # Initialize agent
    agent = CodeAgent()
    
    try:
        result = agent.process_task(subtask_id, description, parameters, task_context)
        
        # Update Firestore with results
        doc_ref = firestore_client.collection('subtasks').document(subtask_id)
        doc_ref.update({
            'status': 'completed',
            'result': result,
            'completed_at': datetime.utcnow().isoformat()
        })
        
        # Update task context
        task_ref.update({
            f'agent_results.code': result,
            'updated_at': datetime.utcnow().isoformat()
        })
        
        logger.info(f"Code task {subtask_id} completed successfully")
        notify_orchestrator(
            task_id=task_id,
            subtask_id=subtask_id,
            agent_type="code",
            result=result,
            error=None,
        )
        
    except Exception as e:
        logger.error(f"Code task {subtask_id} failed: {e}")
        
        # Update with error
        doc_ref = firestore_client.collection('subtasks').document(subtask_id)
        doc_ref.update({
            'status': 'failed',
            'error': str(e),
            'completed_at': datetime.utcnow().isoformat()
        })
        notify_orchestrator(
            task_id=task_id,
            subtask_id=subtask_id,
            agent_type="code",
            result=None,
            error=str(e),
        )
