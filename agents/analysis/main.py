"""
Analysis Agent - Responsible for data analysis and insights
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
import pandas as pd
import numpy as np
from collections import Counter
import statistics

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

class AnalysisAgent:
    """Agent specialized in data analysis and insights generation"""
    
    def __init__(self):
        self.firestore = firestore_client
        self.gemini = model
        
    def process_task(self, subtask_id: str, description: str, parameters: Dict[str, Any], 
                    task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process an analysis task"""
        
        logger.info(f"Processing analysis task: {subtask_id}")
        
        # Get data from other agents
        agent_results = task_context.get('agent_results', {})
        
        # Determine analysis type
        analysis_type = parameters.get('type', 'general')
        
        if analysis_type == 'statistical':
            return self.statistical_analysis(description, agent_results, parameters)
        elif analysis_type == 'pattern':
            return self.pattern_analysis(description, agent_results, parameters)
        elif analysis_type == 'comparison':
            return self.comparison_analysis(description, agent_results, parameters)
        else:
            return self.general_analysis(description, agent_results, parameters)
            
    def statistical_analysis(self, description: str, data: Dict[str, Any], 
                           parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis on data"""
        
        results = {}
        
        # Extract numerical data from agent results
        numerical_data = []
        
        # Example: Analyze GitHub repo statistics
        if 'research' in data and 'results' in data['research']:
            repos = data['research'].get('results', [])
            
            if repos and isinstance(repos, list):
                # Extract stars data
                stars = [r.get('stars', 0) for r in repos if 'stars' in r]
                forks = [r.get('forks', 0) for r in repos if 'forks' in r]
                
                if stars:
                    results['stars_analysis'] = {
                        'mean': statistics.mean(stars),
                        'median': statistics.median(stars),
                        'stdev': statistics.stdev(stars) if len(stars) > 1 else 0,
                        'min': min(stars),
                        'max': max(stars),
                        'total': sum(stars)
                    }
                    
                if forks:
                    results['forks_analysis'] = {
                        'mean': statistics.mean(forks),
                        'median': statistics.median(forks),
                        'stdev': statistics.stdev(forks) if len(forks) > 1 else 0,
                        'min': min(forks),
                        'max': max(forks),
                        'total': sum(forks)
                    }
                    
                # Language distribution
                languages = [r.get('language') for r in repos if r.get('language')]
                if languages:
                    lang_counts = Counter(languages)
                    results['language_distribution'] = dict(lang_counts.most_common(10))
                    
        return {
            'type': 'statistical_analysis',
            'description': description,
            'results': results,
            'summary': self.generate_statistical_summary(results),
            'timestamp': datetime.utcnow().isoformat()
        }
        
    def pattern_analysis(self, description: str, data: Dict[str, Any], 
                        parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Identify patterns in data"""
        
        patterns = []
        
        # Look for patterns in research data
        if 'research' in data:
            research_data = data['research']
            
            # Example: Technology stack patterns
            if 'results' in research_data:
                repos = research_data.get('results', [])
                
                # Analyze common technology combinations
                tech_stacks = []
                for repo in repos:
                    if 'language' in repo:
                        stack = {
                            'language': repo['language'],
                            'topics': repo.get('topics', [])
                        }
                        tech_stacks.append(stack)
                        
                # Find common patterns
                if tech_stacks:
                    # Group by language
                    lang_topics = {}
                    for stack in tech_stacks:
                        lang = stack['language']
                        if lang not in lang_topics:
                            lang_topics[lang] = []
                        lang_topics[lang].extend(stack['topics'])
                        
                    # Find most common topics per language
                    for lang, topics in lang_topics.items():
                        if topics:
                            common_topics = Counter(topics).most_common(3)
                            patterns.append({
                                'pattern': f"Common topics for {lang}",
                                'details': dict(common_topics)
                            })
                            
        # Use Gemini for advanced pattern recognition
        gemini_patterns = self.gemini_pattern_analysis(data, description)
        patterns.extend(gemini_patterns)
        
        return {
            'type': 'pattern_analysis',
            'description': description,
            'patterns': patterns,
            'summary': self.generate_pattern_summary(patterns),
            'timestamp': datetime.utcnow().isoformat()
        }
        
    def comparison_analysis(self, description: str, data: Dict[str, Any], 
                           parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Compare different data points or datasets"""
        
        comparisons = []
        
        # Example: Compare different aspects of repositories
        if 'research' in data and 'results' in data['research']:
            repos = data['research'].get('results', [])
            
            if len(repos) >= 2:
                # Compare top repos
                for i in range(min(3, len(repos) - 1)):
                    repo1 = repos[i]
                    repo2 = repos[i + 1]
                    
                    comparison = {
                        'item1': repo1.get('name', 'Unknown'),
                        'item2': repo2.get('name', 'Unknown'),
                        'metrics': {
                            'stars_ratio': repo1.get('stars', 0) / max(repo2.get('stars', 1), 1),
                            'forks_ratio': repo1.get('forks', 0) / max(repo2.get('forks', 1), 1),
                            'same_language': repo1.get('language') == repo2.get('language')
                        }
                    }
                    comparisons.append(comparison)
                    
        return {
            'type': 'comparison_analysis',
            'description': description,
            'comparisons': comparisons,
            'summary': self.generate_comparison_summary(comparisons),
            'timestamp': datetime.utcnow().isoformat()
        }
        
    def general_analysis(self, description: str, data: Dict[str, Any], 
                        parameters: Dict[str, Any]) -> Dict[str, Any]:
        """General analysis using Gemini"""
        
        prompt = f"""
        Analysis Task: {description}
        
        Available Data:
        {json.dumps(data, indent=2)}
        
        Parameters: {json.dumps(parameters)}
        
        Please provide a comprehensive analysis including:
        1. Key insights and findings
        2. Trends and patterns
        3. Recommendations based on the data
        4. Potential areas for further investigation
        
        Format as structured JSON with clear sections.
        """
        
        try:
            response = self.gemini.generate_content(prompt)
            
            # Try to parse as JSON
            try:
                analysis = json.loads(response.text)
            except:
                analysis = {'analysis': response.text}
                
            return {
                'type': 'general_analysis',
                'description': description,
                'analysis': analysis,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return {
                'type': 'general_analysis',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
            
    def gemini_pattern_analysis(self, data: Dict[str, Any], description: str) -> List[Dict[str, Any]]:
        """Use Gemini to identify complex patterns"""
        
        prompt = f"""
        Task: Identify patterns in this data
        Context: {description}
        
        Data:
        {json.dumps(data, indent=2)[:3000]}  # Limit data size
        
        Identify and describe 3-5 key patterns in this data.
        Return as JSON array with structure: [{"pattern": "name", "details": "description"}]
        """
        
        try:
            response = self.gemini.generate_content(prompt)
            patterns = json.loads(response.text)
            return patterns if isinstance(patterns, list) else []
        except:
            return []
            
    def generate_statistical_summary(self, stats: Dict[str, Any]) -> str:
        """Generate summary of statistical analysis"""
        
        summary_parts = []
        
        if 'stars_analysis' in stats:
            stars = stats['stars_analysis']
            summary_parts.append(
                f"Repository stars range from {stars['min']:,} to {stars['max']:,}, "
                f"with an average of {stars['mean']:,.0f}"
            )
            
        if 'language_distribution' in stats:
            langs = stats['language_distribution']
            top_lang = list(langs.keys())[0] if langs else 'Unknown'
            summary_parts.append(f"The most common programming language is {top_lang}")
            
        return ". ".join(summary_parts) if summary_parts else "Statistical analysis complete"
        
    def generate_pattern_summary(self, patterns: List[Dict[str, Any]]) -> str:
        """Generate summary of pattern analysis"""
        
        if not patterns:
            return "No significant patterns identified"
            
        return f"Identified {len(patterns)} patterns in the data, including technology correlations and usage trends"
        
    def generate_comparison_summary(self, comparisons: List[Dict[str, Any]]) -> str:
        """Generate summary of comparison analysis"""
        
        if not comparisons:
            return "No comparisons performed"
            
        return f"Performed {len(comparisons)} comparisons across different data points"

# Cloud Function entry point
@functions_framework.cloud_event
def handle_message(cloud_event):
    """Cloud Function entry point for Pub/Sub messages"""
    
    # Decode the Pub/Sub message
    message_data = base64.b64decode(cloud_event.data["message"]["data"]).decode()
    message = json.loads(message_data)
    
    logger.info(f"Analysis Agent received message: {message.get('subtask_id')}")
    
    # Extract task details
    subtask_id = message['payload']['subtask_id']
    description = message['payload']['description']
    parameters = message['payload']['parameters']
    task_id = message['task_id']
    
    # Get task context for access to other agent results
    task_ref = firestore_client.collection('tasks').document(task_id)
    task_doc = task_ref.get()
    task_context = task_doc.to_dict() if task_doc.exists else {}
    
    # Initialize agent
    agent = AnalysisAgent()
    
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
            f'agent_results.analysis': result,
            'updated_at': datetime.utcnow().isoformat()
        })
        
        logger.info(f"Analysis task {subtask_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis task {subtask_id} failed: {e}")
        
        # Update with error
        doc_ref = firestore_client.collection('subtasks').document(subtask_id)
        doc_ref.update({
            'status': 'failed',
            'error': str(e),
            'completed_at': datetime.utcnow().isoformat()
        })
