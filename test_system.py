"""
Test script for Multi-Agent System
"""

import requests
import json
import time
import sys

def test_multi_agent_system(orchestrator_url: str):
    """Test the multi-agent system with a sample query"""
    
    # Test query
    test_query = {
        "query": "Find the top 5 most popular open-source AI projects on GitHub from the last month, analyze their technology stacks, and provide insights on the most common frameworks and languages used",
        "metadata": {
            "test": True,
            "requester": "test_script"
        }
    }
    
    print("=== Testing Multi-Agent System ===")
    print(f"Query: {test_query['query']}")
    print(f"Orchestrator URL: {orchestrator_url}")
    
    # Submit task
    print("\nSubmitting task...")
    response = requests.post(
        f"{orchestrator_url}/tasks",
        json=test_query
    )
    
    if response.status_code != 200:
        print(f"Error submitting task: {response.text}")
        return
        
    task_response = response.json()
    task_id = task_response.get('task_id')
    
    if not task_id:
        # For async processing, we might not get task_id immediately
        print("Task accepted for processing (async mode)")
        return
        
    print(f"Task ID: {task_id}")
    
    # Poll for results
    print("\nPolling for results...")
    max_attempts = 60  # 5 minutes max
    attempt = 0
    
    while attempt < max_attempts:
        response = requests.get(f"{orchestrator_url}/tasks/{task_id}")
        
        if response.status_code != 200:
            print(f"Error getting status: {response.text}")
            break
            
        status_data = response.json()
        status = status_data.get('status')
        progress = status_data.get('progress', 0)
        
        print(f"Status: {status} | Progress: {progress:.1f}%")
        
        if status == 'completed':
            print("\n=== Task Completed Successfully ===")
            print(f"Result: {status_data.get('result')}")
            
            # Show subtask details
            print("\nSubtask Details:")
            for subtask in status_data.get('subtasks', []):
                print(f"  - {subtask['agent_type']}: {subtask['status']}")
                
            break
            
        elif status == 'failed':
            print("\n=== Task Failed ===")
            print(f"Errors: {status_data.get('errors')}")
            break
            
        attempt += 1
        time.sleep(5)  # Poll every 5 seconds
        
    if attempt >= max_attempts:
        print("\nTimeout: Task did not complete within 5 minutes")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_system.py <orchestrator_url>")
        print("Example: python test_system.py https://orchestrator-service-xxxxx.run.app")
        sys.exit(1)
        
    orchestrator_url = sys.argv[1]
    test_multi_agent_system(orchestrator_url)