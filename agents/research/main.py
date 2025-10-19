"""
Research Agent - Responsible for information gathering
"""

import os
import json
import logging
import base64
from typing import Dict, Any, List, Optional
from datetime import datetime
import functions_framework
from google.cloud import bigquery, firestore
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import asyncio
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
project_id = os.environ.get('PROJECT_ID')
firestore_client = firestore.Client(project=project_id)
bigquery_client = bigquery.Client(project=project_id)

# Initialize Gemini
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
MODEL_NAME = os.environ.get('GEMINI_MODEL', 'gemini-2.5-pro')
model = genai.GenerativeModel(MODEL_NAME)

class ResearchAgent:
    """Agent specialized in research and information gathering"""
    
    def __init__(self):
        self.firestore = firestore_client
        self.bq = bigquery_client
        self.gemini = model
        
    async def process_task(self, subtask_id: str, description: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Process a research task"""
        
        logger.info(f"Processing research task: {subtask_id}")
        
        # Determine research type
        research_type = parameters.get('type', 'web')
        
        if research_type == 'web':
            return await self.web_research(description, parameters)
        elif research_type == 'bigquery':
            return await self.bigquery_research(description, parameters)
        elif research_type == 'github':
            return await self.github_research(description, parameters)
        else:
            return await self.general_research(description, parameters)
            
    async def web_research(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform web research using search APIs"""
        
        results = []
        
        # Use Google Custom Search API (you'd need to set this up)
        # For now, using a simple web scraping approach
        search_urls = parameters.get('urls', [])
        
        if not search_urls:
            # Generate search URLs based on query
            search_query = query.replace(' ', '+')
            search_urls = [
                f"https://www.google.com/search?q={search_query}",
                # Add more search engines as needed
            ]
            
        async with aiohttp.ClientSession() as session:
            for url in search_urls:
                try:
                    async with session.get(url, timeout=10) as response:
                        if response.status == 200:
                            content = await response.text()
                            # Parse content
                            soup = BeautifulSoup(content, 'html.parser')
                            
                            # Extract relevant information
                            extracted = self.extract_information(soup, query)
                            results.append({
                                'source': url,
                                'content': extracted
                            })
                            
                except Exception as e:
                    logger.error(f"Failed to fetch {url}: {e}")
                    
        # Use Gemini to synthesize findings
        synthesis = self.synthesize_research(results, query)
        
        return {
            'type': 'web_research',
            'query': query,
            'sources': [r['source'] for r in results],
            'raw_results': results,
            'synthesis': synthesis,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    async def bigquery_research(self, description: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query BigQuery datasets for research"""
        
        dataset_id = parameters.get('dataset', 'github_repos')
        
        # Example: Query GitHub data for popular AI projects
        if 'github' in description.lower() and 'ai' in description.lower():
            query = """
            SELECT 
                repo.name as repo_name,
                repo.description,
                repo.language,
                repo.stargazers_count,
                repo.forks_count,
                repo.created_at,
                repo.topics
            FROM `bigquery-public-data.github_repos.repos` repo
            WHERE 
                LOWER(repo.description) LIKE '%artificial intelligence%'
                OR LOWER(repo.description) LIKE '%machine learning%'
                OR LOWER(repo.description) LIKE '%deep learning%'
                OR ARRAY_LENGTH(repo.topics) > 0
            ORDER BY repo.stargazers_count DESC
            LIMIT 10
            """
            
            try:
                query_job = self.bq.query(query)
                results = list(query_job.result())
                
                # Format results
                repos = []
                for row in results:
                    repos.append({
                        'name': row.repo_name,
                        'description': row.description,
                        'language': row.language,
                        'stars': row.stargazers_count,
                        'forks': row.forks_count,
                        'created': row.created_at.isoformat() if row.created_at else None,
                        'topics': list(row.topics) if row.topics else []
                    })
                    
                return {
                    'type': 'bigquery_research',
                    'dataset': dataset_id,
                    'query': query,
                    'results': repos,
                    'count': len(repos),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"BigQuery query failed: {e}")
                return {
                    'type': 'bigquery_research',
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        return {
            'type': 'bigquery_research',
            'message': 'Query type not implemented',
            'timestamp': datetime.utcnow().isoformat()
        }
        
    async def github_research(self, description: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Research GitHub repositories"""
        
        # GitHub API endpoint
        github_token = os.environ.get('GITHUB_TOKEN')
        headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json'
        } if github_token else {}
        
        search_query = parameters.get('query', description)
        
        # Search repositories
        url = f"https://api.github.com/search/repositories?q={search_query}&sort=stars&order=desc&per_page=10"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    repos = []
                    for item in data.get('items', []):
                        repos.append({
                            'name': item['full_name'],
                            'description': item['description'],
                            'stars': item['stargazers_count'],
                            'language': item['language'],
                            'url': item['html_url'],
                            'topics': item.get('topics', []),
                            'updated': item['updated_at']
                        })
                        
                    return {
                        'type': 'github_research',
                        'query': search_query,
                        'results': repos,
                        'count': len(repos),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                else:
                    return {
                        'type': 'github_research',
                        'error': f'GitHub API returned status {response.status}',
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
    async def general_research(self, description: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """General research using Gemini"""
        
        prompt = f"""
        Research Task: {description}
        
        Parameters: {json.dumps(parameters)}
        
        Please provide comprehensive research on this topic including:
        1. Key findings
        2. Important facts and statistics
        3. Relevant sources and references
        4. Analysis and insights
        
        Format the response as structured JSON.
        """
        
        try:
            response = self.gemini.generate_content(prompt)
            
            # Try to parse as JSON, fallback to text
            try:
                result = json.loads(response.text)
            except:
                result = {'findings': response.text}
                
            return {
                'type': 'general_research',
                'description': description,
                'result': result,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Gemini research failed: {e}")
            return {
                'type': 'general_research',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
            
    def extract_information(self, soup: BeautifulSoup, query: str) -> str:
        """Extract relevant information from HTML"""
        
        # Simple extraction - can be enhanced
        text_content = soup.get_text()[:2000]  # Limit to first 2000 chars
        return text_content
        
    def synthesize_research(self, results: List[Dict[str, Any]], query: str) -> str:
        """Use Gemini to synthesize research findings"""
        
        prompt = f"""
        Research Query: {query}
        
        Research Results:
        {json.dumps(results, indent=2)}
        
        Please synthesize these findings into a clear, comprehensive summary.
        Focus on answering the research query directly.
        """
        
        try:
            response = self.gemini.generate_content(prompt)
            return response.text
        except:
            return "Unable to synthesize results"

# Cloud Function entry point
@functions_framework.cloud_event
def handle_message(cloud_event):
    """Cloud Function entry point for Pub/Sub messages"""
    
    # Decode the Pub/Sub message
    message_data = base64.b64decode(cloud_event.data["message"]["data"]).decode()
    message = json.loads(message_data)
    
    logger.info(f"Research Agent received message: {message.get('subtask_id')}")
    
    # Extract task details
    subtask_id = message['payload']['subtask_id']
    description = message['payload']['description']
    parameters = message['payload']['parameters']
    task_id = message['task_id']
    
    # Initialize agent
    agent = ResearchAgent()
    
    # Process task asynchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            agent.process_task(subtask_id, description, parameters)
        )
        
        # Update Firestore with results
        doc_ref = firestore_client.collection('subtasks').document(subtask_id)
        doc_ref.update({
            'status': 'completed',
            'result': result,
            'completed_at': datetime.utcnow().isoformat()
        })
        
        # Also update task context
        task_ref = firestore_client.collection('tasks').document(task_id)
        task_ref.update({
            f'agent_results.research': result,
            'updated_at': datetime.utcnow().isoformat()
        })
        
        logger.info(f"Research task {subtask_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Research task {subtask_id} failed: {e}")
        
        # Update with error
        doc_ref = firestore_client.collection('subtasks').document(subtask_id)
        doc_ref.update({
            'status': 'failed',
            'error': str(e),
            'completed_at': datetime.utcnow().isoformat()
        })
        
    finally:
        loop.close()
