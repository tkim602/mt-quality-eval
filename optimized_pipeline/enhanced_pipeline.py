#!/usr/bin/env python3
"""
Enhanced Pipeline Runner with Monitoring and Advanced Quality Controls
Integrates all quality improvements for production-ready MT evaluation
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import logging

# Import our modules
import cfg
from production_monitor import QualityMonitor, AlertConfig, AsyncQualityProcessor
from failure_analyzer import FailureAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPipeline:
    def __init__(self, config_file: str = "cfg.py", enable_monitoring: bool = True):
        self.config_file = config_file
        self.enable_monitoring = enable_monitoring
        
        # Setup monitoring
        if enable_monitoring:
            alert_config = AlertConfig(
                pass_rate_threshold=0.25,
                confidence_threshold=0.65,
                processing_time_threshold=12.0,
                error_rate_threshold=0.03
            )
            self.monitor = QualityMonitor(alert_config)
            self.async_processor = AsyncQualityProcessor(self.monitor)
        else:
            self.monitor = None
            self.async_processor = None
    
    def enhanced_quality_check(self, item: Dict) -> Dict:
        """Enhanced quality check with confidence scoring and business rules"""
        start_time = time.time()
        
        try:
            # Extract metrics
            cos = item.get('cos', 0.0)
            comet = item.get('comet', 0.0)
            gemba = item.get('gemba', 0.0)
            bucket = item.get('bucket', 'medium')
            key = item.get('key', '')
            
            # Apply enhanced quality decision
            tag, passed, failed, confidence = cfg.make_quality_decision_enhanced(
                cos, comet, gemba, bucket, key
            )
            
            # Add enhanced information to result
            enhanced_result = item.copy()
            enhanced_result.update({
                'tag': tag,
                'passed_checks': passed,
                'failed_checks': failed,
                'confidence': confidence,
                'string_type': self._get_string_type(key),
                'business_rule_applied': self._get_business_rule_info(key),
                'processing_time': time.time() - start_time
            })
            
            # Record metrics if monitoring is enabled
            if self.monitor:
                self.monitor.record_result(enhanced_result, time.time() - start_time)
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error in quality check for {item.get('key', 'unknown')}: {e}")
            if self.monitor:
                self.monitor.record_result({}, time.time() - start_time, error=True)
            raise
    
    def _get_string_type(self, key: str) -> str:
        """Determine string type for business rules"""
        return cfg.get_string_type(key)
    
    def _get_business_rule_info(self, key: str) -> Dict:
        """Get information about applied business rule"""
        string_type = self._get_string_type(key)
        rule = cfg.BUSINESS_RULES.get(string_type, cfg.BUSINESS_RULES["default"])
        
        return {
            'type': string_type,
            'required_confidence': rule['required_confidence'],
            'has_adjustments': any(k.endswith(('_boost', '_penalty')) for k in rule.keys())
        }
    
    async def process_batch_async(self, items: List[Dict]) -> List[Dict]:
        """Process a batch of items asynchronously"""
        if self.async_processor:
            return await self.async_processor.process_batch(items, self.enhanced_quality_check)
        else:
            # Fallback to synchronous processing
            return [self.enhanced_quality_check(item) for item in items]
    
    def run_quality_analysis(self, data_file: Path, output_dir: Path) -> Dict[str, Any]:
        """Run comprehensive quality analysis on processed data"""
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load processed data
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'input_file': str(data_file),
            'total_items': len(data),
            'quality_summary': self._analyze_quality_distribution(data),
            'confidence_analysis': self._analyze_confidence_scores(data),
            'business_rules_impact': self._analyze_business_rules_impact(data)
        }
        
        # Run failure analysis if we have failures
        failures = [item for item in data if item.get('tag') == 'fail']
        if failures:
            logger.info("Running failure pattern analysis...")
            
            # Create a custom failure analyzer that uses the processed data directly
            failure_analysis = self._run_custom_failure_analysis(data)
            results['failure_analysis'] = failure_analysis
        
        # Save results
        output_file = output_dir / "enhanced_quality_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Enhanced quality analysis saved to {output_file}")
        return results
    
    def _analyze_quality_distribution(self, data: List[Dict]) -> Dict:
        """Analyze distribution of quality decisions"""
        total = len(data)
        if total == 0:
            return {}
        
        tag_counts = {}
        confidence_by_tag = {}
        
        for item in data:
            tag = item.get('tag', 'unknown')
            confidence = item.get('confidence', 0)
            
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
            if tag not in confidence_by_tag:
                confidence_by_tag[tag] = []
            confidence_by_tag[tag].append(confidence)
        
        return {
            'distribution': {tag: count/total for tag, count in tag_counts.items()},
            'counts': tag_counts,
            'avg_confidence_by_tag': {
                tag: sum(confs)/len(confs) if confs else 0 
                for tag, confs in confidence_by_tag.items()
            }
        }
    
    def _analyze_confidence_scores(self, data: List[Dict]) -> Dict:
        """Analyze confidence score distribution"""
        confidences = [item.get('confidence', 0) for item in data if item.get('confidence') is not None]
        
        if not confidences:
            return {'no_confidence_data': True}
        
        import statistics
        
        return {
            'mean': statistics.mean(confidences),
            'median': statistics.median(confidences),
            'std': statistics.stdev(confidences) if len(confidences) > 1 else 0,
            'min': min(confidences),
            'max': max(confidences),
            'low_confidence_count': sum(1 for c in confidences if c < 0.5),
            'high_confidence_count': sum(1 for c in confidences if c > 0.8),
            'distribution': {
                'very_low': sum(1 for c in confidences if c < 0.3) / len(confidences),
                'low': sum(1 for c in confidences if 0.3 <= c < 0.5) / len(confidences),
                'medium': sum(1 for c in confidences if 0.5 <= c < 0.7) / len(confidences),
                'high': sum(1 for c in confidences if 0.7 <= c < 0.9) / len(confidences),
                'very_high': sum(1 for c in confidences if c >= 0.9) / len(confidences)
            }
        }
    
    def _analyze_business_rules_impact(self, data: List[Dict]) -> Dict:
        """Analyze impact of business rules"""
        string_types = {}
        
        for item in data:
            string_type = item.get('string_type', 'default')
            tag = item.get('tag', 'unknown')
            confidence = item.get('confidence', 0)
            
            if string_type not in string_types:
                string_types[string_type] = {
                    'count': 0,
                    'tags': {},
                    'avg_confidence': 0,
                    'confidences': []
                }
            
            string_types[string_type]['count'] += 1
            string_types[string_type]['tags'][tag] = string_types[string_type]['tags'].get(tag, 0) + 1
            string_types[string_type]['confidences'].append(confidence)
        
        # Calculate averages
        for string_type, stats in string_types.items():
            if stats['confidences']:
                stats['avg_confidence'] = sum(stats['confidences']) / len(stats['confidences'])
            del stats['confidences']  # Remove raw data
            
            # Convert counts to percentages
            total = stats['count']
            stats['tag_distribution'] = {
                tag: count/total for tag, count in stats['tags'].items()
            }
        
        return string_types
    
    def _run_custom_failure_analysis(self, data: List[Dict]) -> Dict:
        """Run custom failure analysis on processed data"""
        import pandas as pd
        df = pd.DataFrame(data)
        
        failures = df[df['tag'] == 'fail']
        total = len(df)
        
        if len(failures) == 0:
            return {'no_failures': True}
        
        analysis = {
            'failure_overview': {
                'total_failures': len(failures),
                'failure_rate': len(failures) / total,
                'avg_confidence_failures': failures['confidence'].mean() if 'confidence' in failures.columns else 0,
                'avg_confidence_all': df['confidence'].mean() if 'confidence' in df.columns else 0
            },
            'by_bucket': {},
            'by_string_type': {},
            'confidence_analysis': {}
        }
        
        # Analyze by bucket
        for bucket in df['bucket'].unique():
            bucket_data = df[df['bucket'] == bucket]
            bucket_failures = failures[failures['bucket'] == bucket]
            
            analysis['by_bucket'][bucket] = {
                'failure_rate': len(bucket_failures) / len(bucket_data),
                'count': len(bucket_data),
                'failures': len(bucket_failures)
            }
        
        # Analyze by string type
        if 'string_type' in df.columns:
            for string_type in df['string_type'].unique():
                type_data = df[df['string_type'] == string_type]
                type_failures = failures[failures['string_type'] == string_type]
                
                analysis['by_string_type'][string_type] = {
                    'failure_rate': len(type_failures) / len(type_data) if len(type_data) > 0 else 0,
                    'count': len(type_data),
                    'failures': len(type_failures)
                }
        
        # Confidence analysis
        if 'confidence' in df.columns:
            low_conf_failures = failures[failures['confidence'] < 0.5]
            analysis['confidence_analysis'] = {
                'low_confidence_failures': len(low_conf_failures),
                'low_conf_failure_rate': len(low_conf_failures) / len(failures) if len(failures) > 0 else 0,
                'avg_confidence_by_tag': df.groupby('tag')['confidence'].mean().to_dict()
            }
        
        return analysis
    
    def get_monitoring_dashboard(self) -> Dict:
        """Get current monitoring dashboard data"""
        if self.monitor:
            return self.monitor.get_dashboard_data()
        else:
            return {'monitoring_disabled': True}
    
    def save_monitoring_report(self, output_file: Path):
        """Save monitoring report"""
        if self.monitor:
            self.monitor.save_monitoring_report(output_file)
        else:
            logger.warning("Monitoring is disabled")

def main():
    """Main function for running enhanced pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced MT Quality Pipeline")
    parser.add_argument("--input", type=Path, required=True,
                       help="Input data file (JSON)")
    parser.add_argument("--output-dir", type=Path, default=Path("enhanced_output"),
                       help="Output directory for results")
    parser.add_argument("--disable-monitoring", action="store_true",
                       help="Disable production monitoring")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Create enhanced pipeline
    pipeline = EnhancedPipeline(enable_monitoring=not args.disable_monitoring)
    
    # Load input data
    logger.info(f"Loading data from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    logger.info(f"Processing {len(input_data)} items with enhanced quality pipeline...")
    
    # Process data in batches
    async def process_all():
        results = []
        for i in range(0, len(input_data), args.batch_size):
            batch = input_data[i:i + args.batch_size]
            logger.info(f"Processing batch {i//args.batch_size + 1}/{(len(input_data)-1)//args.batch_size + 1}")
            
            batch_results = await pipeline.process_batch_async(batch)
            results.extend(batch_results)
        
        return results
    
    # Run async processing
    start_time = time.time()
    processed_data = asyncio.run(process_all())
    processing_time = time.time() - start_time
    
    logger.info(f"Processing completed in {processing_time:.1f} seconds")
    
    # Save processed results
    args.output_dir.mkdir(exist_ok=True, parents=True)
    output_file = args.output_dir / "enhanced_results.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Enhanced results saved to {output_file}")
    
    # Run comprehensive analysis
    logger.info("Running comprehensive quality analysis...")
    analysis_results = pipeline.run_quality_analysis(output_file, args.output_dir)
    
    # Save monitoring report if enabled
    if not args.disable_monitoring:
        pipeline.save_monitoring_report(args.output_dir / "monitoring_report.json")
        
        # Print monitoring summary
        dashboard = pipeline.get_monitoring_dashboard()
        print("\n" + "="*50)
        print("MONITORING SUMMARY")
        print("="*50)
        metrics = dashboard["current_metrics"]
        print(f"Pass Rate: {metrics['pass_rate']:.1%}")
        print(f"Average Confidence: {metrics['avg_confidence']:.2f}")
        print(f"Processing Time: {metrics['avg_processing_time']:.1f}s avg")
        print(f"System Status: {dashboard['system_status'].upper()}")
        if dashboard['active_alerts']:
            print(f"Active Alerts: {', '.join(dashboard['active_alerts'])}")
        print("="*50)

if __name__ == "__main__":
    main()
