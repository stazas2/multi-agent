"""
Monitoring and Observability for Multi-Agent System
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from google.cloud import monitoring_v3, logging as cloud_logging, trace_v1
from google.cloud.monitoring_dashboard import v1 as dashboard_v1
import pandas as pd
import asyncio
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentMetrics:
    """Metrics for individual agent performance"""
    agent_name: str
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    average_duration_ms: float
    p95_duration_ms: float
    error_rate: float
    last_24h_tasks: int

@dataclass
class SystemHealth:
    """Overall system health status"""
    status: str  # healthy, degraded, unhealthy
    agents_online: int
    total_agents: int
    queue_depth: Dict[str, int]
    error_rate_24h: float
    average_latency_ms: float
    alerts: List[Dict[str, Any]]

class SystemMonitor:
    """Monitor for the multi-agent system"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.metrics_client = monitoring_v3.MetricServiceClient()
        self.logging_client = cloud_logging.Client(project=project_id)
        self.trace_client = trace_v1.TraceServiceClient()
        
        # Metric descriptors
        self.custom_metrics = {
            'agent_task_duration': 'custom.googleapis.com/agent/task_duration',
            'agent_task_count': 'custom.googleapis.com/agent/task_count',
            'agent_error_count': 'custom.googleapis.com/agent/error_count',
            'queue_depth': 'custom.googleapis.com/queue/depth',
            'orchestrator_latency': 'custom.googleapis.com/orchestrator/latency'
        }
        
    def create_custom_metrics(self) -> None:
        """Create custom metric descriptors in Cloud Monitoring"""
        
        project_name = f"projects/{self.project_id}"
        
        # Task duration metric
        duration_descriptor = monitoring_v3.MetricDescriptor(
            type=self.custom_metrics['agent_task_duration'],
            metric_kind=monitoring_v3.MetricDescriptor.MetricKind.GAUGE,
            value_type=monitoring_v3.MetricDescriptor.ValueType.DOUBLE,
            display_name="Agent Task Duration",
            description="Duration of agent task execution in milliseconds",
            labels=[
                monitoring_v3.LabelDescriptor(
                    key="agent_name",
                    value_type=monitoring_v3.LabelDescriptor.ValueType.STRING
                ),
                monitoring_v3.LabelDescriptor(
                    key="task_type",
                    value_type=monitoring_v3.LabelDescriptor.ValueType.STRING
                )
            ]
        )
        
        try:
            self.metrics_client.create_metric_descriptor(
                name=project_name,
                metric_descriptor=duration_descriptor
            )
            logger.info("Created task duration metric")
        except Exception as e:
            logger.warning(f"Metric descriptor might already exist: {e}")
            
        # Task count metric
        count_descriptor = monitoring_v3.MetricDescriptor(
            type=self.custom_metrics['agent_task_count'],
            metric_kind=monitoring_v3.MetricDescriptor.MetricKind.CUMULATIVE,
            value_type=monitoring_v3.MetricDescriptor.ValueType.INT64,
            display_name="Agent Task Count",
            description="Number of tasks processed by agents",
            labels=[
                monitoring_v3.LabelDescriptor(
                    key="agent_name",
                    value_type=monitoring_v3.LabelDescriptor.ValueType.STRING
                ),
                monitoring_v3.LabelDescriptor(
                    key="status",
                    value_type=monitoring_v3.LabelDescriptor.ValueType.STRING
                )
            ]
        )
        
        try:
            self.metrics_client.create_metric_descriptor(
                name=project_name,
                metric_descriptor=count_descriptor
            )
            logger.info("Created task count metric")
        except Exception as e:
            logger.warning(f"Metric descriptor might already exist: {e}")
            
    def write_metric(self, metric_type: str, value: float, 
                    labels: Dict[str, str] = {}) -> None:
        """Write a custom metric to Cloud Monitoring"""
        
        project_name = f"projects/{self.project_id}"
        
        series = monitoring_v3.TimeSeries()
        series.metric.type = metric_type
        
        # Add labels
        for key, val in labels.items():
            series.metric.labels[key] = val
            
        # Resource type
        series.resource.type = "global"
        
        # Create a data point
        now = time.time()
        seconds = int(now)
        nanos = int((now - seconds) * 10**9)
        
        interval = monitoring_v3.TimeInterval(
            {"end_time": {"seconds": seconds, "nanos": nanos}}
        )
        
        point = monitoring_v3.Point({
            "interval": interval,
            "value": {"double_value": value} if isinstance(value, float) else {"int64_value": value}
        })
        
        series.points = [point]
        
        # Write the time series
        self.metrics_client.create_time_series(
            name=project_name,
            time_series=[series]
        )
        
    async def get_agent_metrics(self, agent_name: str, 
                               time_range_hours: int = 24) -> AgentMetrics:
        """Get metrics for a specific agent"""
        
        project_name = f"projects/{self.project_id}"
        
        # Time range
        interval = monitoring_v3.TimeInterval({
            "end_time": {"seconds": int(time.time())},
            "start_time": {"seconds": int(time.time() - time_range_hours * 3600)}
        })
        
        # Query for task count
        task_count_filter = f'metric.type="{self.custom_metrics["agent_task_count"]}" AND metric.labels.agent_name="{agent_name}"'
        
        results = self.metrics_client.list_time_series(
            request={
                "name": project_name,
                "filter": task_count_filter,
                "interval": interval
            }
        )
        
        total_tasks = 0
        successful_tasks = 0
        failed_tasks = 0
        
        for result in results:
            for point in result.points:
                total_tasks += point.value.int64_value
                if result.metric.labels.get('status') == 'success':
                    successful_tasks += point.value.int64_value
                elif result.metric.labels.get('status') == 'failure':
                    failed_tasks += point.value.int64_value
                    
        # Query for task duration
        duration_filter = f'metric.type="{self.custom_metrics["agent_task_duration"]}" AND metric.labels.agent_name="{agent_name}"'
        
        duration_results = self.metrics_client.list_time_series(
            request={
                "name": project_name,
                "filter": duration_filter,
                "interval": interval
            }
        )
        
        durations = []
        for result in duration_results:
            for point in result.points:
                durations.append(point.value.double_value)
                
        # Calculate metrics
        avg_duration = sum(durations) / len(durations) if durations else 0
        p95_duration = pd.Series(durations).quantile(0.95) if durations else 0
        error_rate = failed_tasks / total_tasks if total_tasks > 0 else 0
        
        return AgentMetrics(
            agent_name=agent_name,
            total_tasks=total_tasks,
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            average_duration_ms=avg_duration,
            p95_duration_ms=p95_duration,
            error_rate=error_rate,
            last_24h_tasks=total_tasks
        )
        
    async def get_system_health(self) -> SystemHealth:
        """Get overall system health status"""
        
        agents = ['research', 'analysis', 'code', 'validator']
        agents_online = 0
        total_error_count = 0
        total_task_count = 0
        all_durations = []
        alerts = []
        
        # Check each agent
        for agent in agents:
            try:
                metrics = await self.get_agent_metrics(agent, time_range_hours=1)
                
                if metrics.last_24h_tasks > 0:
                    agents_online += 1
                    
                total_error_count += metrics.failed_tasks
                total_task_count += metrics.total_tasks
                
                if metrics.error_rate > 0.2:
                    alerts.append({
                        'type': 'high_error_rate',
                        'agent': agent,
                        'error_rate': metrics.error_rate
                    })
                    
                if metrics.p95_duration_ms > 30000:  # 30 seconds
                    alerts.append({
                        'type': 'high_latency',
                        'agent': agent,
                        'p95_ms': metrics.p95_duration_ms
                    })
                    
                all_durations.append(metrics.average_duration_ms)
                
            except Exception as e:
                logger.error(f"Failed to get metrics for {agent}: {e}")
                alerts.append({
                    'type': 'agent_unreachable',
                    'agent': agent,
                    'error': str(e)
                })
                
        # Calculate overall metrics
        error_rate_24h = total_error_count / total_task_count if total_task_count > 0 else 0
        avg_latency = sum(all_durations) / len(all_durations) if all_durations else 0
        
        # Determine health status
        if agents_online < 2:
            status = 'unhealthy'
        elif error_rate_24h > 0.15 or len(alerts) > 2:
            status = 'degraded'
        else:
            status = 'healthy'
            
        # Get queue depths (simplified - would query Pub/Sub in production)
        queue_depths = {
            'research': 0,
            'analysis': 0,
            'code': 0,
            'validator': 0
        }
        
        return SystemHealth(
            status=status,
            agents_online=agents_online,
            total_agents=len(agents),
            queue_depth=queue_depths,
            error_rate_24h=error_rate_24h,
            average_latency_ms=avg_latency,
            alerts=alerts
        )
        
    def create_dashboard(self) -> str:
        """Create a Cloud Monitoring dashboard for the system"""
        
        dashboard_client = dashboard_v1.DashboardsServiceClient()
        project_name = f"projects/{self.project_id}"
        
        dashboard = dashboard_v1.Dashboard(
            display_name="Multi-Agent System Dashboard",
            grid_layout=dashboard_v1.GridLayout(
                widgets=[
                    # Agent task count widget
                    dashboard_v1.Widget(
                        title="Agent Task Count",
                        xy_chart=dashboard_v1.XyChart(
                            data_sets=[
                                dashboard_v1.XyChart.DataSet(
                                    time_series_query=dashboard_v1.XyChart.TimeSeriesQuery(
                                        time_series_filter=dashboard_v1.XyChart.TimeSeriesFilter(
                                            filter=f'metric.type="{self.custom_metrics["agent_task_count"]}"',
                                            aggregation=dashboard_v1.Aggregation(
                                                alignment_period={"seconds": 60},
                                                per_series_aligner=dashboard_v1.Aggregation.Aligner.ALIGN_RATE
                                            )
                                        )
                                    )
                                )
                            ]
                        )
                    ),
                    # Agent task duration widget
                    dashboard_v1.Widget(
                        title="Agent Task Duration (p95)",
                        xy_chart=dashboard_v1.XyChart(
                            data_sets=[
                                dashboard_v1.XyChart.DataSet(
                                    time_series_query=dashboard_v1.XyChart.TimeSeriesQuery(
                                        time_series_filter=dashboard_v1.XyChart.TimeSeriesFilter(
                                            filter=f'metric.type="{self.custom_metrics["agent_task_duration"]}"',
                                            aggregation=dashboard_v1.Aggregation(
                                                alignment_period={"seconds": 300},
                                                per_series_aligner=dashboard_v1.Aggregation.Aligner.ALIGN_PERCENTILE_95
                                            )
                                        )
                                    )
                                )
                            ]
                        )
                    ),
                    # Error rate widget
                    dashboard_v1.Widget(
                        title="Error Rate by Agent",
                        xy_chart=dashboard_v1.XyChart(
                            data_sets=[
                                dashboard_v1.XyChart.DataSet(
                                    time_series_query=dashboard_v1.XyChart.TimeSeriesQuery(
                                        time_series_filter=dashboard_v1.XyChart.TimeSeriesFilter(
                                            filter=f'metric.type="{self.custom_metrics["agent_error_count"]}"',
                                            aggregation=dashboard_v1.Aggregation(
                                                alignment_period={"seconds": 300},
                                                per_series_aligner=dashboard_v1.Aggregation.Aligner.ALIGN_RATE
                                            )
                                        )
                                    )
                                )
                            ]
                        )
                    )
                ]
            )
        )
        
        result = dashboard_client.create_dashboard(
            parent=project_name,
            dashboard=dashboard
        )
        
        logger.info(f"Created dashboard: {result.name}")
        return result.name
        
    def create_alerts(self) -> None:
        """Create alerting policies"""
        
        alert_client = monitoring_v3.AlertPolicyServiceClient()
        notification_client = monitoring_v3.NotificationChannelServiceClient()
        project_name = f"projects/{self.project_id}"
        
        # High error rate alert
        error_rate_policy = monitoring_v3.AlertPolicy(
            display_name="High Agent Error Rate",
            conditions=[
                monitoring_v3.AlertPolicy.Condition(
                    display_name="Error rate > 20%",
                    condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                        filter=f'metric.type="{self.custom_metrics["agent_error_count"]}"',
                        aggregations=[
                            monitoring_v3.Aggregation(
                                alignment_period={"seconds": 300},
                                per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_RATE
                            )
                        ],
                        comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                        threshold_value=0.2
                    )
                )
            ],
            alert_strategy=monitoring_v3.AlertPolicy.AlertStrategy(
                rate={
                    "period": {"seconds": 300}
                }
            )
        )
        
        try:
            alert_client.create_alert_policy(
                name=project_name,
                alert_policy=error_rate_policy
            )
            logger.info("Created error rate alert policy")
        except Exception as e:
            logger.warning(f"Alert policy might already exist: {e}")
            
        # High latency alert
        latency_policy = monitoring_v3.AlertPolicy(
            display_name="High Agent Latency",
            conditions=[
                monitoring_v3.AlertPolicy.Condition(
                    display_name="P95 latency > 30s",
                    condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                        filter=f'metric.type="{self.custom_metrics["agent_task_duration"]}"',
                        aggregations=[
                            monitoring_v3.Aggregation(
                                alignment_period={"seconds": 600},
                                per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_PERCENTILE_95
                            )
                        ],
                        comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                        threshold_value=30000  # 30 seconds in milliseconds
                    )
                )
            ]
        )
        
        try:
            alert_client.create_alert_policy(
                name=project_name,
                alert_policy=latency_policy
            )
            logger.info("Created latency alert policy")
        except Exception as e:
            logger.warning(f"Alert policy might already exist: {e}")