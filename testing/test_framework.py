"""
Comprehensive Testing Framework for Multi-Agent System
"""

import pytest
import asyncio
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import requests
from dataclasses import dataclass
import aiohttp
import pandas as pd

# Import our modules (assuming they're in the path)
from shared.models import TaskContext, TaskStatus, AgentType, SubTask, AgentMessage
from shared.utils import FirestoreManager, PubSubManager, GeminiManager
from orchestrator.main import Orchestrator

@dataclass
class TestCase:
    """Structure for test cases"""
    name: str
    query: str
    expected_agents: List[str]
    expected_subtasks: int
    max_duration_seconds: int
    validation_rules: Dict[str, Any]

class MultiAgentTestFramework:
    """Comprehensive test framework for the multi-agent system"""
    
    def __init__(self, project_id: str, orchestrator_url: str):
        self.project_id = project_id
        self.orchestrator_url = orchestrator_url
        self.test_results = []
        
    def run_integration_test(self, test_case: TestCase) -> Dict[str, Any]:
        """Run a complete integration test"""
        
        start_time = time.time()
        result = {
            'test_name': test_case.name,
            'status': 'pending',
            'errors': [],
            'duration': 0
        }
        
        try:
            # Submit task
            response = requests.post(
                f"{self.orchestrator_url}/tasks",
                json={"query": test_case.query}
            )
            
            if response.status_code != 200:
                result['errors'].append(f"Failed to submit task: {response.text}")
                result['status'] = 'failed'
                return result
                
            task_id = response.json().get('task_id')
            
            # Poll for completion
            completed = False
            attempts = 0
            max_attempts = test_case.max_duration_seconds // 5
            
            while not completed and attempts < max_attempts:
                time.sleep(5)
                status_response = requests.get(f"{self.orchestrator_url}/tasks/{task_id}")
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    
                    if status_data['status'] in ['completed', 'failed']:
                        completed = True
                        
                        # Validate results
                        validation_results = self.validate_results(
                            status_data, 
                            test_case
                        )
                        
                        if validation_results['passed']:
                            result['status'] = 'passed'
                        else:
                            result['status'] = 'failed'
                            result['errors'].extend(validation_results['errors'])
                            
                        result['task_data'] = status_data
                        
                attempts += 1
                
            if not completed:
                result['status'] = 'timeout'
                result['errors'].append(f"Test timed out after {test_case.max_duration_seconds} seconds")
                
        except Exception as e:
            result['status'] = 'error'
            result['errors'].append(str(e))
            
        result['duration'] = time.time() - start_time
        self.test_results.append(result)
        return result
        
    def validate_results(self, task_data: Dict[str, Any], 
                        test_case: TestCase) -> Dict[str, Any]:
        """Validate test results against expectations"""
        
        errors = []
        
        # Check if expected agents were used
        subtasks = task_data.get('subtasks', [])
        used_agents = {st['agent_type'] for st in subtasks}
        
        for expected_agent in test_case.expected_agents:
            if expected_agent not in used_agents:
                errors.append(f"Expected agent {expected_agent} was not used")
                
        # Check subtask count
        if len(subtasks) != test_case.expected_subtasks:
            errors.append(f"Expected {test_case.expected_subtasks} subtasks, got {len(subtasks)}")
            
        # Check if all subtasks completed
        incomplete = [st for st in subtasks if st['status'] != 'completed']
        if incomplete:
            errors.append(f"{len(incomplete)} subtasks did not complete")
            
        # Apply custom validation rules
        for rule_name, rule_value in test_case.validation_rules.items():
            if rule_name == 'has_result' and rule_value:
                if not task_data.get('result'):
                    errors.append("Expected final result but none found")
                    
            elif rule_name == 'min_result_length':
                result = task_data.get('result', '')
                if len(result) < rule_value:
                    errors.append(f"Result too short: {len(result)} < {rule_value}")
                    
        return {
            'passed': len(errors) == 0,
            'errors': errors
        }
        
    def run_test_suite(self) -> Dict[str, Any]:
        """Run a complete test suite"""
        
        test_cases = [
            TestCase(
                name="Simple Research Task",
                query="What are the top 3 Python web frameworks?",
                expected_agents=['research'],
                expected_subtasks=1,
                max_duration_seconds=60,
                validation_rules={'has_result': True, 'min_result_length': 100}
            ),
            TestCase(
                name="Research and Analysis",
                query="Find the top 5 AI repositories on GitHub and analyze their popularity trends",
                expected_agents=['research', 'analysis'],
                expected_subtasks=2,
                max_duration_seconds=120,
                validation_rules={'has_result': True}
            ),
            TestCase(
                name="Code Generation",
                query="Write a Python function to calculate fibonacci numbers",
                expected_agents=['code'],
                expected_subtasks=1,
                max_duration_seconds=60,
                validation_rules={'has_result': True}
            ),
            TestCase(
                name="Complete Pipeline",
                query="Research serverless platforms, analyze their features, and generate sample code for AWS Lambda",
                expected_agents=['research', 'analysis', 'code', 'validator'],
                expected_subtasks=4,
                max_duration_seconds=180,
                validation_rules={'has_result': True, 'min_result_length': 500}
            )
        ]
        
        suite_start = time.time()
        passed = 0
        failed = 0
        
        for test_case in test_cases:
            print(f"Running test: {test_case.name}")
            result = self.run_integration_test(test_case)
            
            if result['status'] == 'passed':
                passed += 1
                print(f"  ✅ PASSED ({result['duration']:.2f}s)")
            else:
                failed += 1
                print(f"  ❌ FAILED: {result['errors']}")
                
        suite_duration = time.time() - suite_start
        
        return {
            'total_tests': len(test_cases),
            'passed': passed,
            'failed': failed,
            'duration': suite_duration,
            'results': self.test_results
        }

        # Unit Tests
class TestAgentModels:
    """Unit tests for agent models"""
    
    def test_task_context_creation(self):
        """Test TaskContext creation and serialization"""
        context = TaskContext(user_query="Test query")
        
        assert context.task_id is not None
        assert context.user_query == "Test query"
        assert context.status == TaskStatus.PENDING
        
        # Test serialization
        data = context.to_dict()
        assert data['user_query'] == "Test query"
        assert data['status'] == 'pending'
        
        # Test deserialization
        context2 = TaskContext.from_dict(data)
        assert context2.task_id == context.task_id
        assert context2.user_query == context.user_query
        
    def test_agent_message(self):
        """Test AgentMessage creation and JSON conversion"""
        message = AgentMessage(
            task_id="test-123",
            agent_type=AgentType.RESEARCH,
            action="process",
            payload={"key": "value"}
        )
        
        # Test JSON serialization
        json_str = message.to_json()
        data = json.loads(json_str)
        
        assert data['task_id'] == "test-123"
        assert data['agent_type'] == 'research'
        assert data['action'] == "process"
        
        # Test JSON deserialization
        message2 = AgentMessage.from_json(json_str)
        assert message2.task_id == message.task_id
        assert message2.agent_type == message.agent_type
        
    def test_subtask_dependencies(self):
        """Test SubTask with dependencies"""
        subtask1 = SubTask(
            parent_task_id="parent-123",
            agent_type=AgentType.RESEARCH,
            description="Research task"
        )
        
        subtask2 = SubTask(
            parent_task_id="parent-123",
            agent_type=AgentType.ANALYSIS,
            description="Analysis task",
            dependencies=[subtask1.subtask_id]
        )
        
        assert len(subtask2.dependencies) == 1
        assert subtask1.subtask_id in subtask2.dependencies
        
class TestOrchestratorLogic:
    """Unit tests for Orchestrator logic"""
    
    @patch('orchestrator.main.GeminiManager')
    @patch('orchestrator.main.FirestoreManager')
    @patch('orchestrator.main.PubSubManager')
    def test_task_decomposition(self, mock_pubsub, mock_firestore, mock_gemini):
        """Test task decomposition logic"""
        
        # Mock Gemini response
        mock_gemini.return_value.decompose_task.return_value = [
            {
                "agent_type": "research",
                "description": "Research task",
                "parameters": {},
                "dependencies": []
            },
            {
                "agent_type": "analysis",
                "description": "Analysis task",
                "parameters": {},
                "dependencies": [0]
            }
        ]
        
        orchestrator = Orchestrator()
        orchestrator.gemini = mock_gemini.return_value
        orchestrator.firestore = mock_firestore.return_value
        orchestrator.pubsub = mock_pubsub.return_value
        
        # Test decomposition
        loop = asyncio.new_event_loop()
        context = loop.run_until_complete(
            orchestrator.process_user_query("Test query")
        )
        
        assert context.user_query == "Test query"
        assert len(context.subtasks) == 2
        
# Performance Tests
class PerformanceTests:
    """Performance and load testing"""
    
    def __init__(self, orchestrator_url: str):
        self.orchestrator_url = orchestrator_url
        
    async def load_test(self, num_requests: int, concurrent: int) -> Dict[str, Any]:
        """Run load test with concurrent requests"""
        
        async def submit_task(session: aiohttp.ClientSession, query: str) -> float:
            start = time.time()
            
            async with session.post(
                f"{self.orchestrator_url}/tasks",
                json={"query": query}
            ) as response:
                if response.status == 200:
                    return time.time() - start
                else:
                    return -1
                    
        async with aiohttp.ClientSession() as session:
            tasks = []
            queries = [
                f"Test query {i}: Find information about topic {i}"
                for i in range(num_requests)
            ]
            
            for i in range(0, num_requests, concurrent):
                batch = queries[i:i+concurrent]
                batch_tasks = [submit_task(session, q) for q in batch]
                results = await asyncio.gather(*batch_tasks)
                tasks.extend(results)
                
        successful = [t for t in tasks if t > 0]
        failed = len(tasks) - len(successful)
        
        return {
            'total_requests': num_requests,
            'successful': len(successful),
            'failed': failed,
            'average_latency': sum(successful) / len(successful) if successful else 0,
            'p95_latency': pd.Series(successful).quantile(0.95) if successful else 0,
            'p99_latency': pd.Series(successful).quantile(0.99) if successful else 0
        }
        
    def stress_test(self, duration_seconds: int, rps: int) -> Dict[str, Any]:
        """Run stress test at specified RPS for duration"""
        
        start_time = time.time()
        results = []
        
        while time.time() - start_time < duration_seconds:
            # Submit requests at specified rate
            batch_start = time.time()
            
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(
                self.load_test(rps, min(rps, 10))
            )
            results.append(result)
            
            # Wait to maintain RPS
            elapsed = time.time() - batch_start
            if elapsed < 1:
                time.sleep(1 - elapsed)
                
        return {
            'duration': duration_seconds,
            'target_rps': rps,
            'results': results
        }

# Run all tests
if __name__ == "__main__":
    project_id = "your-project-id"
    orchestrator_url = "https://orchestrator-xxxxx.run.app"
    
    print("=== Running Multi-Agent System Tests ===\n")
    
    # Integration tests
    print("1. Integration Tests")
    test_framework = MultiAgentTestFramework(project_id, orchestrator_url)
    suite_results = test_framework.run_test_suite()
    
    print(f"\nIntegration Test Results:")
    print(f"  Passed: {suite_results['passed']}/{suite_results['total_tests']}")
    print(f"  Duration: {suite_results['duration']:.2f}s\n")
    
    # Performance tests
    print("2. Performance Tests")
    perf_tests = PerformanceTests(orchestrator_url)
    
    # Load test
    loop = asyncio.new_event_loop()
    load_results = loop.run_until_complete(
        perf_tests.load_test(num_requests=50, concurrent=5)
    )
    
    print(f"\nLoad Test Results:")
    print(f"  Successful: {load_results['successful']}/{load_results['total_requests']}")
    print(f"  Avg Latency: {load_results['average_latency']:.2f}s")
    print(f"  P95 Latency: {load_results['p95_latency']:.2f}s")
    print(f"  P99 Latency: {load_results['p99_latency']:.2f}s")
