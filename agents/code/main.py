"""
Code Agent - Responsible for code analysis and generation
"""

import os
import json
import logging
import base64
from typing import Dict, Any, List, Optional
from datetime import datetime
import functions_framework
from google.cloud import firestore
import google.generativeai as genai
import ast
import re
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
project_id = os.environ.get('PROJECT_ID')
firestore_client = firestore.Client(project=project_id)

# Initialize Gemini
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-pro')

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
        
        # Build context from other agents
        agent_results = context.get('agent_results', {})
        
        prompt = f"""
        Generate code for: {description}
        
        Language: {language}
        Framework: {framework if framework else 'any appropriate'}
        
        Context from research:
        {json.dumps(agent_results.get('research', {}), indent=2)[:2000]}
        
        Requirements:
        {json.dumps(parameters.get('requirements', []), indent=2)}
        
        Generate clean, well-commented, production-ready code.
        Include error handling and best practices.
        """
        
        try:
            response = self.gemini.generate_content(prompt)
            
            # Extract code from response
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
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return {
                'type': 'code_generation',
                'error': str(e),
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
        
    except Exception as e:
        logger.error(f"Code task {subtask_id} failed: {e}")
        
        # Update with error
        doc_ref = firestore_client.collection('subtasks').document(subtask_id)
        doc_ref.update({
            'status': 'failed',
            'error': str(e),
            'completed_at': datetime.utcnow().isoformat()
        })