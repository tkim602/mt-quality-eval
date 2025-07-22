#!/usr/bin/env python3
"""
Failure Pattern Analyzer for MT Quality Evaluation
Analyzes why translations fail quality checks and provides actionable insights
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging
from collections import Counter
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FailureAnalyzer:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        
    def load_data(self) -> pd.DataFrame:
        """Load pipeline data for analysis"""
        # Try to load the most complete dataset (APE > GEMBA > Filter)
        for filename in ["ape_evidence.json", "gemba.json", "filtered.json"]:
            file_path = self.data_dir / filename
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    df = pd.DataFrame(data)
                    logger.info(f"Loaded {len(df)} records from {filename}")
                    return df
        
        raise FileNotFoundError("No pipeline data found")
    
    def analyze_terminology_issues(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze terminology consistency issues"""
        try:
            import cfg
            termbase = getattr(cfg, 'TERMBASE', {})
        except:
            termbase = {
                "스레드": "thread",
                "이벤트": "event", 
                "세션": "session",
                "취약점": "vulnerability",
                "보안": "security"
            }
        
        issues = []
        
        for _, row in df.iterrows():
            if row.get('tag') == 'fail':
                src_text = row['src'].lower()
                mt_text = row['mt'].lower()
                
                # Check for terminology violations
                for ko_term, en_term in termbase.items():
                    if ko_term in src_text:
                        if en_term.lower() not in mt_text:
                            issues.append({
                                'key': row['key'],
                                'missing_term': en_term,
                                'ko_term': ko_term,
                                'severity': 'high' if 'security' in en_term or 'authentication' in en_term else 'medium'
                            })
        
        return {
            'total_terminology_violations': len(issues),
            'high_severity_violations': len([i for i in issues if i['severity'] == 'high']),
            'most_common_violations': Counter([i['missing_term'] for i in issues]).most_common(5),
            'details': issues[:10]  # Top 10 for detailed review
        }
    
    def analyze_length_issues(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze length-related quality issues"""
        failures = df[df['tag'] == 'fail'].copy()
        
        if len(failures) == 0:
            return {'no_failures': True}
        
        # Calculate length statistics for failures
        # Handle different possible column names for lengths
        src_len_col = None
        mt_len_col = None
        
        for col in ['src_len', 'source_length', 'src_length']:
            if col in df.columns:
                src_len_col = col
                break
        
        for col in ['mt_len', 'target_length', 'mt_length']:
            if col in df.columns:
                mt_len_col = col
                break
        
        # If length columns don't exist, calculate them from text
        if src_len_col is None:
            failures['src_len'] = failures['src'].str.len()
            src_len_col = 'src_len'
        
        if mt_len_col is None:
            failures['mt_len'] = failures['mt'].str.len()
            mt_len_col = 'mt_len'
        
        failures['length_ratio'] = failures[mt_len_col] / failures[src_len_col]
        
        # Also calculate for all data if length columns didn't exist
        if 'length_ratio' not in df.columns:
            df_copy = df.copy()
            if src_len_col not in df_copy.columns:
                df_copy['src_len'] = df_copy['src'].str.len()
            if mt_len_col not in df_copy.columns:
                df_copy['mt_len'] = df_copy['mt'].str.len()
            df_copy['length_ratio'] = df_copy['mt_len'] / df_copy['src_len']
        else:
            df_copy = df
        
        analysis = {
            'extreme_length_ratios': {
                'too_short': len(failures[failures['length_ratio'] < 0.3]),
                'too_long': len(failures[failures['length_ratio'] > 3.0]),
                'avg_ratio_failures': float(failures['length_ratio'].mean()),
                'avg_ratio_all': float(df_copy['length_ratio'].mean())
            },
            'by_bucket': {}
        }
        
        # Analyze by length bucket
        for bucket in failures['bucket'].unique():
            bucket_failures = failures[failures['bucket'] == bucket]
            bucket_all = df_copy[df_copy['bucket'] == bucket]
            
            analysis['by_bucket'][bucket] = {
                'failure_rate': len(bucket_failures) / len(bucket_all),
                'avg_cos_failure': float(bucket_failures['cos'].mean()),
                'avg_comet_failure': float(bucket_failures['comet'].mean()),
                'avg_cos_all': float(bucket_all['cos'].mean()),
                'avg_comet_all': float(bucket_all['comet'].mean())
            }
        
        return analysis
    
    def analyze_semantic_issues(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze semantic drift and meaning preservation issues"""
        failures = df[df['tag'] == 'fail'].copy()
        
        if len(failures) == 0:
            return {'no_failures': True}
        
        # Find cases where cosine is much lower than COMET or vice versa
        failures['cos_comet_diff'] = abs(failures['cos'] - failures['comet'])
        
        semantic_issues = {
            'high_disagreement_cases': len(failures[failures['cos_comet_diff'] > 0.2]),
            'low_cosine_high_comet': len(failures[(failures['cos'] < 0.7) & (failures['comet'] > 0.8)]),
            'high_cosine_low_comet': len(failures[(failures['cos'] > 0.8) & (failures['comet'] < 0.7)]),
            'avg_disagreement': float(failures['cos_comet_diff'].mean()),
            'patterns': []
        }
        
        # Find specific patterns
        high_disagreement = failures[failures['cos_comet_diff'] > 0.3].head(5)
        for _, row in high_disagreement.iterrows():
            semantic_issues['patterns'].append({
                'key': row['key'],
                'src': row['src'][:100] + "..." if len(row['src']) > 100 else row['src'],
                'mt': row['mt'][:100] + "..." if len(row['mt']) > 100 else row['mt'],
                'cos': float(row['cos']),
                'comet': float(row['comet']),
                'disagreement': float(row['cos_comet_diff'])
            })
        
        return semantic_issues
    
    def analyze_fluency_issues(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze fluency and naturalness issues"""
        if 'gemba_fluency' not in df.columns:
            return {'no_gemba_data': True}
        
        failures = df[df['tag'] == 'fail'].copy()
        
        if len(failures) == 0:
            return {'no_failures': True}
        
        fluency_analysis = {
            'low_fluency_cases': len(failures[failures['gemba_fluency'] < 60]),
            'fluency_vs_adequacy': {
                'fluency_worse': len(failures[failures['gemba_fluency'] < failures['gemba_adequacy']]),
                'adequacy_worse': len(failures[failures['gemba_adequacy'] < failures['gemba_fluency']]),
                'similar': len(failures[abs(failures['gemba_fluency'] - failures['gemba_adequacy']) < 10])
            },
            'avg_fluency_failures': float(failures['gemba_fluency'].mean()),
            'avg_adequacy_failures': float(failures['gemba_adequacy'].mean()),
            'avg_fluency_all': float(df['gemba_fluency'].mean()),
            'avg_adequacy_all': float(df['gemba_adequacy'].mean())
        }
        
        return fluency_analysis
    
    def generate_improvement_recommendations(self, analysis_results: Dict) -> Dict[str, List[str]]:
        """Generate actionable improvement recommendations"""
        recommendations = {
            'immediate_actions': [],
            'threshold_adjustments': [],
            'process_improvements': [],
            'data_quality': []
        }
        
        # Terminology recommendations
        if 'terminology' in analysis_results:
            term_analysis = analysis_results['terminology']
            if term_analysis['high_severity_violations'] > 0:
                recommendations['immediate_actions'].append(
                    f"Fix {term_analysis['high_severity_violations']} high-severity terminology violations"
                )
            
            if term_analysis['total_terminology_violations'] > 10:
                recommendations['process_improvements'].append(
                    "Implement terminology validation in the filter stage"
                )
        
        # Length-based recommendations
        if 'length_issues' in analysis_results and 'by_bucket' in analysis_results['length_issues']:
            for bucket, stats in analysis_results['length_issues']['by_bucket'].items():
                if stats['failure_rate'] > 0.7:
                    recommendations['threshold_adjustments'].append(
                        f"Lower thresholds for '{bucket}' bucket (current failure rate: {stats['failure_rate']:.1%})"
                    )
        
        # Semantic recommendations
        if 'semantic_issues' in analysis_results:
            sem_analysis = analysis_results['semantic_issues']
            if sem_analysis.get('high_disagreement_cases', 0) > 5:
                recommendations['data_quality'].append(
                    "Review cases with high cosine-COMET disagreement for potential data quality issues"
                )
        
        # Fluency recommendations
        if 'fluency_issues' in analysis_results and not analysis_results['fluency_issues'].get('no_gemba_data'):
            fluency = analysis_results['fluency_issues']['fluency_vs_adequacy']
            if fluency['fluency_worse'] > fluency['adequacy_worse']:
                recommendations['process_improvements'].append(
                    "Focus APE on improving fluency rather than adequacy"
                )
        
        return recommendations
    
    def generate_failure_report(self, output_file: Path):
        """Generate comprehensive failure analysis report"""
        df = self.load_data()
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'dataset_summary': {
                'total_records': len(df),
                'failure_rate': len(df[df['tag'] == 'fail']) / len(df) if 'tag' in df.columns else 0,
                'soft_pass_rate': len(df[df['tag'] == 'soft_pass']) / len(df) if 'tag' in df.columns else 0,
                'strict_pass_rate': len(df[df['tag'] == 'strict_pass']) / len(df) if 'tag' in df.columns else 0
            }
        }
        
        # Run all analyses
        logger.info("Analyzing terminology issues...")
        report['terminology'] = self.analyze_terminology_issues(df)
        
        logger.info("Analyzing length issues...")
        report['length_issues'] = self.analyze_length_issues(df)
        
        logger.info("Analyzing semantic issues...")
        report['semantic_issues'] = self.analyze_semantic_issues(df)
        
        logger.info("Analyzing fluency issues...")
        report['fluency_issues'] = self.analyze_fluency_issues(df)
        
        # Generate recommendations
        logger.info("Generating recommendations...")
        report['recommendations'] = self.generate_improvement_recommendations(report)
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Failure analysis report saved to {output_file}")
        
        # Print summary
        self._print_summary(report)
        
        return report
    
    def _print_summary(self, report: Dict):
        """Print a summary of the failure analysis"""
        print("\n" + "="*60)
        print("FAILURE PATTERN ANALYSIS SUMMARY")
        print("="*60)
        
        summary = report['dataset_summary']
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   Total Records: {summary['total_records']}")
        print(f"   Failure Rate: {summary['failure_rate']:.1%}")
        print(f"   Soft Pass Rate: {summary['soft_pass_rate']:.1%}")
        print(f"   Strict Pass Rate: {summary['strict_pass_rate']:.1%}")
        
        if 'terminology' in report:
            term = report['terminology']
            print(f"\n🔤 TERMINOLOGY ISSUES:")
            print(f"   Total Violations: {term['total_terminology_violations']}")
            print(f"   High Severity: {term['high_severity_violations']}")
        
        if 'length_issues' in report and 'by_bucket' in report['length_issues']:
            print(f"\n📏 LENGTH BUCKET ANALYSIS:")
            for bucket, stats in report['length_issues']['by_bucket'].items():
                print(f"   {bucket}: {stats['failure_rate']:.1%} failure rate")
        
        print(f"\n💡 TOP RECOMMENDATIONS:")
        recs = report['recommendations']
        for action in recs['immediate_actions'][:3]:
            print(f"   • {action}")
        for action in recs['threshold_adjustments'][:2]:
            print(f"   • {action}")
        
        print("\n" + "="*60)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Failure Pattern Analysis for MT Pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("out/v3"),
                       help="Directory containing pipeline output files")
    parser.add_argument("--output", type=Path, default=Path("failure_analysis.json"),
                       help="Output file for failure analysis report")
    
    args = parser.parse_args()
    
    analyzer = FailureAnalyzer(args.data_dir)
    analyzer.generate_failure_report(args.output)

if __name__ == "__main__":
    main()
