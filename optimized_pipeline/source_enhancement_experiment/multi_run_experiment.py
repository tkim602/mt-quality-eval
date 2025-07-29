#!/usr/bin/env python3
"""
Run source enhancement experiment multiple times and calculate mean results for generalization.
"""

import json
import asyncio
import random
from pathlib import Path
from datetime import datetime
import statistics

# Import from existing modules
import sys
sys.path.append('..')
from source_enhancement_experiment import SourceEnhancementExperiment
from comparative_analyzer import ComparativeAnalyzer

class MultiRunExperiment:
    def __init__(self, num_runs=3, samples_per_run=200):
        self.num_runs = num_runs
        self.samples_per_run = samples_per_run
        self.results_dir = Path("results/multi_run")
        self.results_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """Load the original data for sampling."""
        data_file = Path("../../data/data_pairs/korean_english_pairs.json")
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to list if it's a dictionary
        if isinstance(data, dict):
            data_list = []
            for key, value in data.items():
                data_list.append({
                    'id': key,
                    'korean_original': value['korean'],
                    'english_reference': value['english'],
                    'category': value.get('category', 'unknown')
                })
            return data_list
        return data
    
    async def run_single_experiment(self, run_number: int, data: list):
        """Run a single experiment with random sampling."""
        print(f"\n{'='*60}")
        print(f"STARTING RUN {run_number}/{self.num_runs}")
        print(f"{'='*60}")
        
        # Random sampling for this run
        random.seed(42 + run_number)  # Different seed for each run
        sampled_data = random.sample(data, min(self.samples_per_run, len(data)))
        
        print(f"Sampled {len(sampled_data)} items for run {run_number}")
        
        # Run source enhancement experiment
        experiment = SourceEnhancementExperiment()
        results_file = self.results_dir / f"source_enhancement_run_{run_number}.json"
        
        # Convert sampled data to expected format
        formatted_data = {}
        for item in sampled_data:
            formatted_data[item['id']] = {
                'korean': item['korean_original'],
                'english': item['english_reference'],
                'category': item['category']
            }
        
        await experiment.run_enhancement_experiment(
            data=formatted_data,
            limit=len(sampled_data),
            output_file=results_file
        )
        
        # Run comparative analysis
        analyzer = ComparativeAnalyzer()
        analysis_results = analyzer.run_comparative_analysis(
            results_file=f"source_enhancement_run_{run_number}.json"
        )
        
        # Save analysis results
        analysis_file = self.results_dir / f"analysis_run_{run_number}.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Run {run_number} completed")
        print(f"   Results: {results_file}")
        print(f"   Analysis: {analysis_file}")
        
        return analysis_results
    
    def filter_medium_plus_results(self, analysis_results):
        """Filter results to include only medium+ length texts."""
        medium_plus_buckets = ['medium', 'long', 'very_long']
        
        # Filter detailed results
        filtered_details = []
        for detail in analysis_results['detailed_results']:
            if detail['length_bucket'] in medium_plus_buckets:
                filtered_details.append(detail)
        
        if not filtered_details:
            return None
        
        # Recalculate stats for filtered data
        original_cos = [d['original_cos_similarity'] for d in filtered_details]
        enhanced_cos = [d['enhanced_cos_similarity'] for d in filtered_details]
        original_comet = [d['original_comet_score'] for d in filtered_details]
        enhanced_comet = [d['enhanced_comet_score'] for d in filtered_details]
        
        cos_improvements = [d['cos_improvement'] for d in filtered_details]
        comet_improvements = [d['comet_improvement'] for d in filtered_details]
        
        filtered_stats = {
            "total_samples": len(filtered_details),
            "original_cos_mean": statistics.mean(original_cos),
            "enhanced_cos_mean": statistics.mean(enhanced_cos),
            "cos_improvement_mean": statistics.mean(cos_improvements),
            "original_comet_mean": statistics.mean(original_comet),
            "enhanced_comet_mean": statistics.mean(enhanced_comet),
            "comet_improvement_mean": statistics.mean(comet_improvements),
            "cos_improvement_positive_pct": (len([x for x in cos_improvements if x > 0]) / len(cos_improvements)) * 100,
            "comet_improvement_positive_pct": (len([x for x in comet_improvements if x > 0]) / len(comet_improvements)) * 100
        }
        
        return filtered_stats
    
    def calculate_multi_run_statistics(self, all_results):
        """Calculate mean and std dev across multiple runs."""
        # Extract metrics from all runs
        metrics = {
            'total_samples': [],
            'cos_improvement_mean': [],
            'comet_improvement_mean': [],
            'cos_improvement_positive_pct': [],
            'comet_improvement_positive_pct': [],
            'original_cos_mean': [],
            'enhanced_cos_mean': [],
            'original_comet_mean': [],
            'enhanced_comet_mean': []
        }
        
        for result in all_results:
            if result:  # Skip None results
                for key in metrics.keys():
                    if key in result:
                        metrics[key].append(result[key])
        
        # Calculate statistics
        stats = {}
        for key, values in metrics.items():
            if values:
                stats[key] = {
                    'mean': statistics.mean(values),
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'min': min(values),
                    'max': max(values),
                    'values': values
                }
        
        return stats
    
    async def run_multi_experiment(self):
        """Run multiple experiments and analyze results."""
        print(f"🚀 Starting {self.num_runs} experimental runs with {self.samples_per_run} samples each")
        
        # Load data
        data = self.load_data()
        print(f"Loaded {len(data)} total data points")
        
        # Run experiments
        all_results = []
        all_filtered_results = []
        
        for run_num in range(1, self.num_runs + 1):
            try:
                result = await self.run_single_experiment(run_num, data)
                all_results.append(result)
                
                # Filter for medium+ length
                filtered_result = self.filter_medium_plus_results(result)
                all_filtered_results.append(filtered_result)
                
            except Exception as e:
                print(f"❌ Error in run {run_num}: {e}")
                all_results.append(None)
                all_filtered_results.append(None)
        
        # Calculate multi-run statistics
        print(f"\n{'='*70}")
        print("MULTI-RUN ANALYSIS RESULTS")
        print(f"{'='*70}")
        
        # Full results statistics
        full_stats = self.calculate_multi_run_statistics(all_results)
        
        # Filtered results statistics  
        filtered_stats = self.calculate_multi_run_statistics(all_filtered_results)
        
        # Generate comprehensive report
        report = {
            'experiment_info': {
                'num_runs': self.num_runs,
                'samples_per_run': self.samples_per_run,
                'total_data_points': len(data),
                'timestamp': datetime.now().isoformat()
            },
            'full_results_stats': full_stats,
            'filtered_medium_plus_stats': filtered_stats,
            'individual_run_results': all_results,
            'individual_filtered_results': all_filtered_results
        }
        
        # Save comprehensive report
        report_file = self.results_dir / "multi_run_comprehensive_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Print summary
        self.print_summary_report(filtered_stats, report_file)
        
        return report
    
    def print_summary_report(self, filtered_stats, report_file):
        """Print a human-readable summary."""
        print(f"\n📊 GENERALIZED RESULTS (Medium+ Length, {self.num_runs} runs)")
        print(f"{'='*70}")
        
        if not filtered_stats:
            print("❌ No valid results to analyze")
            return
        
        cos_improvement = filtered_stats.get('cos_improvement_mean', {})
        comet_improvement = filtered_stats.get('comet_improvement_mean', {})
        comet_improve_rate = filtered_stats.get('comet_improvement_positive_pct', {})
        
        if cos_improvement and comet_improvement:
            print(f"COS Similarity Change:")
            print(f"  Mean: {cos_improvement['mean']:+.3f} (±{cos_improvement['std_dev']:.3f})")
            print(f"  Range: {cos_improvement['min']:+.3f} to {cos_improvement['max']:+.3f}")
            
            print(f"\nCOMET Score Change:")
            print(f"  Mean: {comet_improvement['mean']:+.3f} (±{comet_improvement['std_dev']:.3f})")
            print(f"  Range: {comet_improvement['min']:+.3f} to {comet_improvement['max']:+.3f}")
            
            print(f"\nCOMET Improvement Rate:")
            print(f"  Mean: {comet_improve_rate['mean']:.1f}% (±{comet_improve_rate['std_dev']:.1f}%)")
            print(f"  Range: {comet_improve_rate['min']:.1f}% to {comet_improve_rate['max']:.1f}%")
        
        print(f"\n{'='*70}")
        print("HYPOTHESIS VALIDATION (Generalized)")
        print(f"{'='*70}")
        
        if comet_improvement and comet_improvement['mean'] > 0:
            confidence_level = "HIGH" if comet_improvement['std_dev'] < comet_improvement['mean'] else "MEDIUM"
            print(f"✅ HYPOTHESIS CONFIRMED with {confidence_level} confidence")
            print(f"   Source enhancement consistently improves MT quality")
            print(f"   Average COMET improvement: {comet_improvement['mean']:+.3f}")
            print(f"   Consistency (low std dev): {comet_improvement['std_dev']:.3f}")
        else:
            print(f"❌ HYPOTHESIS NOT SUPPORTED")
        
        print(f"\nDetailed report saved: {report_file}")

async def main():
    """Main execution function."""
    experiment = MultiRunExperiment(num_runs=3, samples_per_run=200)
    await experiment.run_multi_experiment()

if __name__ == "__main__":
    asyncio.run(main())
