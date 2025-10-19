"""
Validator Agent - Responsible for validating and verifying results
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
import hashlib
from dataclasses import dataclass
import re
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
project_id = os.environ.get('PROJECT_ID')
firestore_client = firestore.Client(project=project_id)

# Initialize Gemini
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
MODEL_NAME = os.environ.get('GEMINI_MODEL', 'gemini-2.5-pro')
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

@dataclass
class ValidationResult:
    """Structure for validation results"""
    is_valid: bool
    confidence: float  # 0.0 to 1.0
    issues: List[str]
    suggestions: List[str]
    metadata: Dict[str, Any]

class ValidatorAgent:
    """Agent specialized in validation and quality assurance"""
    
    def __init__(self):
        self.firestore = firestore_client
        self.gemini = model
        self.validation_rules = self.load_validation_rules()
        
    def load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules from configuration"""
        return {
            'min_confidence': 0.7,
            'required_fields': ['type', 'results', 'timestamp'],
            'max_error_rate': 0.1,
            'consistency_threshold': 0.8
        }
        
    def process_task(self, subtask_id: str, description: str, 
                    parameters: Dict[str, Any], task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a validation task"""
        
        logger.info(f"Processing validation task: {subtask_id}")
        
        validation_type = parameters.get('type', 'comprehensive')
        
        # Get all agent results to validate
        agent_results = task_context.get('agent_results', {})
        
        validation_results = {}
        
        # Validate each agent's output
        for agent_name, agent_data in agent_results.items():
            if agent_name != 'validator':  # Don't validate ourselves
                result = self.validate_agent_output(agent_name, agent_data, validation_type)
                validation_results[agent_name] = result.is_valid
                
        # Cross-validate results for consistency
        consistency_result = self.cross_validate_results(agent_results)
        
        # Fact-check key claims
        fact_check_result = self.fact_check_results(agent_results, task_context.get('user_query', ''))
        
        # Generate quality score
        quality_score = self.calculate_quality_score(validation_results, consistency_result, fact_check_result)
        
        # Compile final validation report
        validation_report = self.generate_validation_report(
            validation_results,
            consistency_result,
            fact_check_result,
            quality_score
        )
        
        return {
            'type': 'validation',
            'description': description,
            'validation_results': validation_results,
            'consistency_check': consistency_result,
            'fact_check': fact_check_result,
            'quality_score': quality_score,
            'report': validation_report,
            'passed': quality_score >= self.validation_rules['min_confidence'],
            'timestamp': datetime.utcnow().isoformat()
        }
        
    def validate_agent_output(self, agent_name: str, output: Dict[str, Any], 
                             validation_type: str) -> ValidationResult:
        """Validate output from a specific agent"""
        
        issues = []
        suggestions = []
        
        # Check required fields
        for field in self.validation_rules['required_fields']:
            if field not in output:
                issues.append(f"Missing required field: {field}")
                
        # Validate based on agent type
        if agent_name == 'research':
            return self.validate_research_output(output)
        elif agent_name == 'analysis':
            return self.validate_analysis_output(output)
        elif agent_name == 'code':
            return self.validate_code_output(output)
        else:
            return self.general_validation(output)
            
    def validate_research_output(self, output: Dict[str, Any]) -> ValidationResult:
        """Validate research agent output"""
        
        issues = []
        suggestions = []
        
        # Check for sources
        if 'sources' not in output or not output['sources']:
            issues.append("No sources provided")
            suggestions.append("Research should include verifiable sources")
            
        # Check for results
        if 'results' in output:
            results = output['results']
            if isinstance(results, list) and len(results) == 0:
                issues.append("No research results found")
            elif isinstance(results, dict) and not results:
                issues.append("Empty research results")
                
        # Validate synthesis
        if 'synthesis' in output:
            synthesis = output['synthesis']
            if len(synthesis) < 100:
                suggestions.append("Synthesis seems too brief, consider expanding")
                
        confidence = 1.0 - (len(issues) * 0.2)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence=max(0.0, confidence),
            issues=issues,
            suggestions=suggestions,
            metadata={'agent': 'research'}
        )
        
    def validate_analysis_output(self, output: Dict[str, Any]) -> ValidationResult:
        """Validate analysis agent output"""
        
        issues = []
        suggestions = []
        
        # Check for analysis results
        if 'results' not in output and 'analysis' not in output:
            issues.append("No analysis results provided")
            
        # Validate statistical analysis if present
        if 'results' in output and 'statistics' in str(output.get('type', '')):
            stats = output.get('results', {})
            
            # Check for NaN or infinite values
            for key, value in self._flatten_dict(stats).items():
                if isinstance(value, (int, float)):
                    if value != value:  # NaN check
                        issues.append(f"NaN value in {key}")
                    elif value == float('inf') or value == float('-inf'):
                        issues.append(f"Infinite value in {key}")
                        
        # Check for summary
        if 'summary' not in output:
            suggestions.append("Consider adding a summary of analysis findings")
            
        confidence = 1.0 - (len(issues) * 0.15)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence=max(0.0, confidence),
            issues=issues,
            suggestions=suggestions,
            metadata={'agent': 'analysis'}
        )
        
    def validate_code_output(self, output: Dict[str, Any]) -> ValidationResult:
        """Validate code agent output"""
        
        issues = []
        suggestions = []
        
        # Check for code content
        if 'code' not in output and 'code_blocks' not in output:
            issues.append("No code provided in output")
            
        # Validate code syntax if present
        if 'code' in output:
            code = output['code']
            language = output.get('language', 'python')
            
            syntax_issues = self.check_code_syntax(code, language)
            issues.extend(syntax_issues)
            
            # Check for common code issues
            if language == 'python':
                if 'import' not in code and len(code) > 100:
                    suggestions.append("Consider adding necessary imports")
                if 'def ' not in code and 'class ' not in code and len(code) > 50:
                    suggestions.append("Consider organizing code into functions or classes")
                    
        # Check for explanation
        if 'explanation' not in output and 'description' not in output:
            suggestions.append("Add explanation for the generated code")
            
        confidence = 1.0 - (len(issues) * 0.25)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence=max(0.0, confidence),
            issues=issues,
            suggestions=suggestions,
            metadata={'agent': 'code'}
        )
        
    def general_validation(self, output: Dict[str, Any]) -> ValidationResult:
        """General validation for any output"""
        
        issues = []
        suggestions = []
        
        # Check basic structure
        if not output:
            issues.append("Empty output")
        elif len(output) < 3:
            suggestions.append("Output seems minimal, consider adding more detail")
            
        # Check for error indicators
        if 'error' in output:
            issues.append(f"Error in output: {output['error']}")
            
        # Validate timestamp if present
        if 'timestamp' in output:
            try:
                datetime.fromisoformat(output['timestamp'].replace('Z', '+00:00'))
            except:
                issues.append("Invalid timestamp format")
                
        confidence = 1.0 - (len(issues) * 0.2)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence=max(0.0, confidence),
            issues=issues,
            suggestions=suggestions,
            metadata={'type': 'general'}
        )
        
    def cross_validate_results(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate results between agents for consistency"""
        
        consistency_checks = []
        
        # Check if research and analysis align
        if 'research' in agent_results and 'analysis' in agent_results:
            research_data = agent_results['research']
            analysis_data = agent_results['analysis']
            
            # Check if analysis used research data
            if 'results' in research_data and 'results' in analysis_data:
                research_items = research_data.get('results', [])
                analysis_text = str(analysis_data.get('results', ''))
                
                # Simple check: are research items mentioned in analysis?
                mentioned_count = 0
                for item in research_items[:5]:  # Check first 5 items
                    if isinstance(item, dict):
                        for value in item.values():
                            if str(value).lower() in analysis_text.lower():
                                mentioned_count += 1
                                break
                                
                consistency_score = mentioned_count / min(5, len(research_items)) if research_items else 0
                
                consistency_checks.append({
                    'check': 'research_analysis_alignment',
                    'score': consistency_score,
                    'passed': consistency_score >= 0.5
                })
                
        # Check if code aligns with requirements
        if 'code' in agent_results:
            code_data = agent_results['code']
            
            # Check against other agents' outputs
            alignment_score = self.check_code_alignment(code_data, agent_results)
            
            consistency_checks.append({
                'check': 'code_requirements_alignment',
                'score': alignment_score,
                'passed': alignment_score >= 0.6
            })
            
        return {
            'checks': consistency_checks,
            'overall_consistency': sum(c['score'] for c in consistency_checks) / len(consistency_checks) if consistency_checks else 1.0,
            'passed': all(c['passed'] for c in consistency_checks)
        }
        
    def fact_check_results(self, agent_results: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        """Fact-check key claims using Gemini"""
        
        # Extract key claims from results
        claims = self.extract_key_claims(agent_results)
        
        if not claims:
            return {
                'checked': False,
                'message': 'No specific claims to fact-check'
            }
            
        # Use Gemini to fact-check
        prompt = f"""
        Original Query: {user_query}
        
        Please fact-check these claims from the agent results:
        {json.dumps(claims, indent=2)}
        
        For each claim, determine:
        1. Is it factually accurate?
        2. Confidence level (0-1)
        3. Any corrections needed
        
        Return as JSON: {{"claims": [{{"claim": "...", "accurate": true/false, "confidence": 0.9, "correction": "..."}}]}}
        """
        
        try:
            response = self.gemini.generate_content(prompt)
            
            # Parse response
            fact_check_data = json.loads(response.text)
            
            accurate_count = sum(1 for c in fact_check_data['claims'] if c['accurate'])
            total_count = len(fact_check_data['claims'])
            
            return {
                'checked': True,
                'accurate_claims': accurate_count,
                'total_claims': total_count,
                'accuracy_rate': accurate_count / total_count if total_count > 0 else 1.0,
                'details': fact_check_data['claims']
            }
            
        except Exception as e:
            logger.error(f"Fact checking failed: {e}")
            return {
                'checked': False,
                'error': str(e)
            }
            
    def calculate_quality_score(self, validation_results: Dict[str, bool], 
                               consistency_result: Dict[str, Any],
                               fact_check_result: Dict[str, Any]) -> float:
        """Calculate overall quality score"""
        
        scores = []
        weights = []
        
        # Agent validation scores
        agent_score = sum(1 for v in validation_results.values() if v) / len(validation_results) if validation_results else 0
        scores.append(agent_score)
        weights.append(0.3)
        
        # Consistency score
        consistency_score = consistency_result.get('overall_consistency', 1.0)
        scores.append(consistency_score)
        weights.append(0.3)
        
        # Fact check score
        if fact_check_result.get('checked'):
            fact_score = fact_check_result.get('accuracy_rate', 0.5)
        else:
            fact_score = 0.8  # Default if not checked
        scores.append(fact_score)
        weights.append(0.4)
        
        # Weighted average
        quality_score = sum(s * w for s, w in zip(scores, weights))
        
        return min(1.0, max(0.0, quality_score))
        
    def generate_validation_report(self, validation_results: Dict[str, bool],
                                  consistency_result: Dict[str, Any],
                                  fact_check_result: Dict[str, Any],
                                  quality_score: float) -> str:
        """Generate human-readable validation report"""
        
        report_parts = []
        
        # Header
        report_parts.append(f"## Validation Report")
        report_parts.append(f"**Quality Score: {quality_score:.2%}**")
        report_parts.append(f"**Status: {'✅ PASSED' if quality_score >= 0.7 else '⚠️ NEEDS REVIEW'}**\n")
        
        # Agent validation results
        report_parts.append("### Agent Output Validation")
        for agent, valid in validation_results.items():
            status = "✅" if valid else "❌"
            report_parts.append(f"- {agent.capitalize()} Agent: {status}")
            
        # Consistency check
        report_parts.append("\n### Cross-Agent Consistency")
        if consistency_result['passed']:
            report_parts.append(f"✅ Consistency Score: {consistency_result['overall_consistency']:.2%}")
        else:
            report_parts.append(f"⚠️ Consistency Issues Detected: {consistency_result['overall_consistency']:.2%}")
            
        # Fact checking
        report_parts.append("\n### Fact Checking")
        if fact_check_result.get('checked'):
            accuracy = fact_check_result['accuracy_rate']
            report_parts.append(f"Accuracy Rate: {accuracy:.2%}")
            report_parts.append(f"Verified Claims: {fact_check_result['accurate_claims']}/{fact_check_result['total_claims']}")
        else:
            report_parts.append("Fact checking not performed or failed")
            
        # Recommendations
        if quality_score < 0.7:
            report_parts.append("\n### Recommendations")
            report_parts.append("- Review agent outputs for completeness")
            report_parts.append("- Verify consistency between different analyses")
            report_parts.append("- Double-check factual claims")
            
        return "\n".join(report_parts)
        
    def check_code_syntax(self, code: str, language: str) -> List[str]:
        """Basic syntax checking for code"""
        
        issues = []
        
        if language == 'python':
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError as e:
                issues.append(f"Python syntax error: {e}")
                
        elif language == 'javascript':
            # Basic JS syntax checks
            open_braces = code.count('{')
            close_braces = code.count('}')
            if open_braces != close_braces:
                issues.append("Mismatched braces in JavaScript code")
                
        return issues
        
    def check_code_alignment(self, code_data: Dict[str, Any], agent_results: Dict[str, Any]) -> float:
        """Check if code aligns with requirements from other agents"""
        
        # Simple heuristic: check if code mentions key terms from research/analysis
        code_text = str(code_data.get('code', '')) + str(code_data.get('explanation', ''))
        
        alignment_score = 0.0
        checks = 0
        
        # Check against research findings
        if 'research' in agent_results:
            research_terms = self.extract_key_terms(agent_results['research'])
            mentioned = sum(1 for term in research_terms if term.lower() in code_text.lower())
            if research_terms:
                alignment_score += mentioned / len(research_terms)
                checks += 1
                
        # Check against analysis recommendations
        if 'analysis' in agent_results:
            analysis_terms = self.extract_key_terms(agent_results['analysis'])
            mentioned = sum(1 for term in analysis_terms if term.lower() in code_text.lower())
            if analysis_terms:
                alignment_score += mentioned / len(analysis_terms)
                checks += 1
                
        return alignment_score / checks if checks > 0 else 0.5
        
    def extract_key_claims(self, agent_results: Dict[str, Any]) -> List[str]:
        """Extract key factual claims from agent results"""
        
        claims = []
        
        # Extract from research
        if 'research' in agent_results:
            if 'synthesis' in agent_results['research']:
                # Extract sentences that look like claims
                synthesis = agent_results['research']['synthesis']
                sentences = synthesis.split('.')
                for sentence in sentences[:5]:  # First 5 sentences
                    if len(sentence) > 20 and any(word in sentence.lower() for word in ['is', 'are', 'has', 'have', 'was', 'were']):
                        claims.append(sentence.strip())
                        
        # Extract from analysis
        if 'analysis' in agent_results:
            if 'summary' in agent_results['analysis']:
                summary = agent_results['analysis']['summary']
                # Look for statistical claims
                if re.search(r'\d+%|\d+\.\d+', summary):
                    claims.append(summary[:200])  # First 200 chars
                    
        return claims[:5]  # Limit to 5 claims
        
    def extract_key_terms(self, agent_data: Dict[str, Any], max_terms: int = 10) -> List[str]:
        """Extract key terms from agent data"""
        
        terms = []
        
        # Look for named entities, technical terms, etc.
        text = json.dumps(agent_data)
        
        # Simple extraction of capitalized words (likely proper nouns)
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        terms.extend(words)
        
        # Technical terms (words with numbers or special patterns)
        technical = re.findall(r'\b\w+\d+\w*\b|\b[A-Z]+\b', text)
        terms.extend(technical)
        
        # Deduplicate and limit
        unique_terms = list(set(terms))
        return unique_terms[:max_terms]
        
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        """Flatten nested dictionary for easier validation"""
        
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

# Cloud Function entry point
@functions_framework.cloud_event
def handle_message(cloud_event):
    """Cloud Function entry point for Pub/Sub messages"""
    
    # Decode the Pub/Sub message
    message_data = base64.b64decode(cloud_event.data["message"]["data"]).decode()
    message = json.loads(message_data)
    
    logger.info(f"Validator Agent received message: {message.get('subtask_id')}")
    
    # Extract task details
    subtask_id = message['payload']['subtask_id']
    description = message['payload']['description']
    parameters = message['payload']['parameters']
    task_id = message['task_id']
    
    # Get full task context
    task_ref = firestore_client.collection('tasks').document(task_id)
    task_doc = task_ref.get()
    task_context = task_doc.to_dict() if task_doc.exists else {}
    
    # Initialize agent
    agent = ValidatorAgent()
    
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
            f'agent_results.validator': result,
            'updated_at': datetime.utcnow().isoformat()
        })
        
        logger.info(f"Validation task {subtask_id} completed successfully")
        notify_orchestrator(
            task_id=task_id,
            subtask_id=subtask_id,
            agent_type="validator",
            result=result,
            error=None,
        )
        
    except Exception as e:
        logger.error(f"Validation task {subtask_id} failed: {e}")
        
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
            agent_type="validator",
            result=None,
            error=str(e),
        )
