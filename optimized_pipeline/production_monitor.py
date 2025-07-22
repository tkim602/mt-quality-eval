#!/usr/bin/env python3
"""
Production Monitoring System for MT Quality Pipeline
Provides real-time monitoring, alerting, and performance tracking
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
from collections import deque, defaultdict
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Quality metrics for monitoring"""
    timestamp: str
    pass_rate: float
    soft_pass_rate: float
    strict_pass_rate: float
    avg_confidence: float
    avg_processing_time: float
    error_rate: float
    throughput: float  # items per second

@dataclass
class AlertConfig:
    """Alert configuration"""
    pass_rate_threshold: float = 0.25  # Alert if pass rate < 25%
    confidence_threshold: float = 0.60  # Alert if avg confidence < 60%
    processing_time_threshold: float = 15.0  # Alert if avg time > 15s
    error_rate_threshold: float = 0.05  # Alert if error rate > 5%
    throughput_threshold: float = 0.1  # Alert if throughput < 0.1 items/s

class QualityMonitor:
    def __init__(self, alert_config: AlertConfig = None, window_size: int = 100):
        self.alert_config = alert_config or AlertConfig()
        self.window_size = window_size
        
        # Sliding windows for metrics
        self.recent_results = deque(maxlen=window_size)
        self.processing_times = deque(maxlen=window_size)
        self.error_count = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        # Alert state
        self.active_alerts = set()
        self.alert_history = []
        
        # Performance tracking
        self.start_time = time.time()
        self.total_processed = 0
        
    def record_result(self, result: Dict[str, Any], processing_time: float, error: bool = False):
        """Record a processing result for monitoring"""
        timestamp = datetime.now().isoformat()
        
        self.recent_results.append(result)
        self.processing_times.append(processing_time)
        self.error_count.append(1 if error else 0)
        self.timestamps.append(time.time())
        self.total_processed += 1
        
        # Check for alerts
        self._check_alerts()
    
    def get_current_metrics(self) -> QualityMetrics:
        """Get current quality metrics"""
        if not self.recent_results:
            return QualityMetrics(
                timestamp=datetime.now().isoformat(),
                pass_rate=0, soft_pass_rate=0, strict_pass_rate=0,
                avg_confidence=0, avg_processing_time=0, error_rate=0, throughput=0
            )
        
        # Calculate rates
        total = len(self.recent_results)
        strict_passes = sum(1 for r in self.recent_results if r.get('tag') == 'strict_pass')
        soft_passes = sum(1 for r in self.recent_results if r.get('tag') == 'soft_pass')
        all_passes = strict_passes + soft_passes
        
        # Calculate confidence (if available)
        confidences = [r.get('confidence', 0) for r in self.recent_results if r.get('confidence') is not None]
        avg_confidence = statistics.mean(confidences) if confidences else 0
        
        # Calculate performance metrics
        avg_processing_time = statistics.mean(self.processing_times) if self.processing_times else 0
        error_rate = sum(self.error_count) / len(self.error_count) if self.error_count else 0
        
        # Calculate throughput (items per second over last minute)
        current_time = time.time()
        recent_timestamps = [t for t in self.timestamps if current_time - t <= 60]
        throughput = len(recent_timestamps) / 60.0 if recent_timestamps else 0
        
        return QualityMetrics(
            timestamp=datetime.now().isoformat(),
            pass_rate=all_passes / total,
            soft_pass_rate=soft_passes / total,
            strict_pass_rate=strict_passes / total,
            avg_confidence=avg_confidence,
            avg_processing_time=avg_processing_time,
            error_rate=error_rate,
            throughput=throughput
        )
    
    def _check_alerts(self):
        """Check if any alert conditions are met"""
        if len(self.recent_results) < 10:  # Need minimum data for reliable alerts
            return
        
        metrics = self.get_current_metrics()
        new_alerts = set()
        
        # Check pass rate
        if metrics.pass_rate < self.alert_config.pass_rate_threshold:
            new_alerts.add("low_pass_rate")
        
        # Check confidence
        if metrics.avg_confidence < self.alert_config.confidence_threshold:
            new_alerts.add("low_confidence")
        
        # Check processing time
        if metrics.avg_processing_time > self.alert_config.processing_time_threshold:
            new_alerts.add("slow_processing")
        
        # Check error rate
        if metrics.error_rate > self.alert_config.error_rate_threshold:
            new_alerts.add("high_error_rate")
        
        # Check throughput
        if metrics.throughput < self.alert_config.throughput_threshold:
            new_alerts.add("low_throughput")
        
        # Trigger new alerts
        for alert in new_alerts - self.active_alerts:
            self._trigger_alert(alert, metrics)
        
        # Clear resolved alerts
        for alert in self.active_alerts - new_alerts:
            self._clear_alert(alert, metrics)
        
        self.active_alerts = new_alerts
    
    def _trigger_alert(self, alert_type: str, metrics: QualityMetrics):
        """Trigger an alert"""
        alert_info = {
            "type": alert_type,
            "timestamp": metrics.timestamp,
            "metrics": asdict(metrics),
            "status": "triggered"
        }
        
        self.alert_history.append(alert_info)
        
        # Log the alert
        logger.warning(f"ALERT TRIGGERED: {alert_type}")
        self._log_alert_details(alert_type, metrics)
    
    def _clear_alert(self, alert_type: str, metrics: QualityMetrics):
        """Clear a resolved alert"""
        alert_info = {
            "type": alert_type,
            "timestamp": metrics.timestamp,
            "metrics": asdict(metrics),
            "status": "cleared"
        }
        
        self.alert_history.append(alert_info)
        logger.info(f"ALERT CLEARED: {alert_type}")
    
    def _log_alert_details(self, alert_type: str, metrics: QualityMetrics):
        """Log detailed information about an alert"""
        details = {
            "low_pass_rate": f"Pass rate ({metrics.pass_rate:.1%}) below threshold ({self.alert_config.pass_rate_threshold:.1%})",
            "low_confidence": f"Avg confidence ({metrics.avg_confidence:.2f}) below threshold ({self.alert_config.confidence_threshold:.2f})",
            "slow_processing": f"Avg processing time ({metrics.avg_processing_time:.1f}s) above threshold ({self.alert_config.processing_time_threshold:.1f}s)",
            "high_error_rate": f"Error rate ({metrics.error_rate:.1%}) above threshold ({self.alert_config.error_rate_threshold:.1%})",
            "low_throughput": f"Throughput ({metrics.throughput:.2f} items/s) below threshold ({self.alert_config.throughput_threshold:.2f} items/s)"
        }
        
        logger.warning(f"  Details: {details.get(alert_type, 'Unknown alert type')}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        metrics = self.get_current_metrics()
        
        return {
            "current_metrics": asdict(metrics),
            "active_alerts": list(self.active_alerts),
            "recent_alerts": self.alert_history[-10:],  # Last 10 alerts
            "system_status": "degraded" if self.active_alerts else "healthy",
            "uptime_hours": (time.time() - self.start_time) / 3600,
            "total_processed": self.total_processed,
            "alert_config": asdict(self.alert_config)
        }
    
    def save_monitoring_report(self, output_file: Path):
        """Save monitoring report to file"""
        dashboard_data = self.get_dashboard_data()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Monitoring report saved to {output_file}")

class AsyncQualityProcessor:
    """Async wrapper for quality processing with monitoring"""
    
    def __init__(self, monitor: QualityMonitor):
        self.monitor = monitor
        
    async def process_batch(self, items: List[Dict], process_func) -> List[Dict]:
        """Process a batch of items asynchronously with monitoring"""
        results = []
        
        # Process items concurrently
        tasks = []
        for item in items:
            task = asyncio.create_task(self._process_single_item(item, process_func))
            tasks.append(task)
        
        # Wait for all tasks to complete
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and record metrics
        for item, result in zip(items, completed_results):
            if isinstance(result, Exception):
                logger.error(f"Error processing item {item.get('key', 'unknown')}: {result}")
                self.monitor.record_result({}, 0, error=True)
            else:
                results.append(result)
        
        return results
    
    async def _process_single_item(self, item: Dict, process_func) -> Dict:
        """Process a single item with timing"""
        start_time = time.time()
        
        try:
            # Run the processing function (make it async if it isn't)
            if asyncio.iscoroutinefunction(process_func):
                result = await process_func(item)
            else:
                result = process_func(item)
            
            processing_time = time.time() - start_time
            self.monitor.record_result(result, processing_time, error=False)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.monitor.record_result({}, processing_time, error=True)
            raise e

def main():
    """Example usage of the monitoring system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MT Quality Pipeline Monitor")
    parser.add_argument("--output", type=Path, default=Path("monitoring_report.json"),
                       help="Output file for monitoring report")
    parser.add_argument("--window-size", type=int, default=100,
                       help="Size of monitoring window")
    
    args = parser.parse_args()
    
    # Create monitor with custom alert config
    alert_config = AlertConfig(
        pass_rate_threshold=0.20,  # Alert if < 20% pass
        confidence_threshold=0.65,  # Alert if < 65% confidence
        processing_time_threshold=12.0,  # Alert if > 12s processing
        error_rate_threshold=0.03  # Alert if > 3% errors
    )
    
    monitor = QualityMonitor(alert_config, args.window_size)
    
    # Example: simulate some processing results
    import random
    
    print("Simulating quality processing with monitoring...")
    
    for i in range(50):
        # Simulate processing result
        result = {
            "key": f"test_key_{i}",
            "tag": random.choice(["strict_pass", "soft_pass", "fail"]),
            "confidence": random.uniform(0.5, 0.95)
        }
        
        # Simulate processing time
        processing_time = random.uniform(1.0, 20.0)
        
        # Simulate occasional errors
        error = random.random() < 0.02  # 2% error rate
        
        monitor.record_result(result, processing_time, error)
        
        # Print dashboard every 10 items
        if (i + 1) % 10 == 0:
            dashboard = monitor.get_dashboard_data()
            metrics = dashboard["current_metrics"]
            print(f"\nMetrics after {i+1} items:")
            print(f"  Pass Rate: {metrics['pass_rate']:.1%}")
            print(f"  Avg Confidence: {metrics['avg_confidence']:.2f}")
            print(f"  Avg Processing Time: {metrics['avg_processing_time']:.1f}s")
            print(f"  Active Alerts: {dashboard['active_alerts']}")
    
    # Save final report
    monitor.save_monitoring_report(args.output)
    print(f"\nMonitoring simulation complete. Report saved to {args.output}")

if __name__ == "__main__":
    main()
