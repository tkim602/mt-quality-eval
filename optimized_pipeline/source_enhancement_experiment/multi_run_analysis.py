#!/usr/bin/env python3
"""
Simple multi-run experiment using existing functions.
"""

import json
import random
import statistics
from pathlib import Path
from datetime import datetime

def load_data():
    """Load the original data."""
    # Try different possible data file locations
    possible_paths = [
        Path("../../data/data_pairs/korean_english_pairs.json"),
        Path("../data/data_pairs/korean_english_pairs.json"),
        Path("../../data/data_pairs/true_pairs.json"),  # Use true_pairs.json as alternative
        Path("../../data/data_pairs/false_pairs.json"),  # Use false_pairs.json as alternative
    ]
    
    data_file = None
    for path in possible_paths:
        if path.exists():
            data_file = path
            break
    
    if not data_file:
        # Let's just use the existing results and create variations
        print("No original data file found. Will use existing experiment results to create variations.")
        return load_existing_experiment_data()
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded data from: {data_file}")
    
    # Handle different data formats
    data_list = []
    
    if isinstance(data, list):
        # Format: [[english, korean], [english, korean], ...]
        for i, pair in enumerate(data):
            if isinstance(pair, list) and len(pair) >= 2:
                english = pair[0]
                korean = pair[1]
                data_list.append({
                    'id': f"pair_{i}",
                    'korean_original': korean,
                    'english_reference': english,
                    'category': 'unknown'
                })
    elif isinstance(data, dict):
        # Format: {id: {korean: ..., english: ...}}
        for key, value in data.items():
            if isinstance(value, dict):
                korean = value.get('korean', value.get('source', ''))
                english = value.get('english', value.get('target', ''))
            else:
                continue
                
            data_list.append({
                'id': key,
                'korean_original': korean,
                'english_reference': english,
                'category': value.get('category', 'unknown')
            })
    
    print(f"Processed {len(data_list)} data pairs")
    return data_list

def load_existing_experiment_data():
    """Load data from existing experiment results."""
    results_file = Path("results/source_enhancement_results_200.json")
    if not results_file.exists():
        raise FileNotFoundError("No existing experiment results found")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Convert results back to data format
    data_list = []
    for result in results:
        data_list.append({
            'id': result['id'],
            'korean_original': result['korean_original'],
            'english_reference': result['english_reference'],
            'category': result.get('category', 'unknown')
        })
    
    print(f"Loaded {len(data_list)} items from existing experiment results")
    return data_list

def sample_data_for_run(data, run_number, samples_per_run=200):
    """Sample data for a specific run."""
    random.seed(42 + run_number)  # Different seed for each run
    return random.sample(data, min(samples_per_run, len(data)))

def create_sampled_data_file(sampled_data, run_number):
    """Create a data file for the sampled data."""
    results_dir = Path("results/multi_run")
    results_dir.mkdir(exist_ok=True)
    
    # Convert back to dictionary format
    data_dict = {}
    for item in sampled_data:
        data_dict[item['id']] = {
            'korean': item['korean_original'],
            'english': item['english_reference'],
            'category': item['category']
        }
    
    # Save sampled data
    sample_file = results_dir / f"sampled_data_run_{run_number}.json"
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)
    
    return sample_file

def filter_medium_plus_results(analysis_file):
    """Filter analysis results to medium+ length and return stats."""
    with open(analysis_file, 'r', encoding='utf-8') as f:
        analysis_results = json.load(f)
    
    medium_plus_buckets = ['medium', 'long', 'very_long']
    
    # Filter detailed results
    filtered_details = []
    for detail in analysis_results['detailed_results']:
        if detail['length_bucket'] in medium_plus_buckets:
            filtered_details.append(detail)
    
    if not filtered_details:
        return None
    
    # Calculate stats
    original_cos = [d['original_cos_similarity'] for d in filtered_details]
    enhanced_cos = [d['enhanced_cos_similarity'] for d in filtered_details]
    original_comet = [d['original_comet_score'] for d in filtered_details]
    enhanced_comet = [d['enhanced_comet_score'] for d in filtered_details]
    
    cos_improvements = [d['cos_improvement'] for d in filtered_details]
    comet_improvements = [d['comet_improvement'] for d in filtered_details]
    
    return {
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

def calculate_multi_run_stats(all_results):
    """Calculate statistics across multiple runs."""
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

def print_instructions_for_runs():
    """Print instructions for manual execution of runs."""
    print(f"{'='*70}")
    print("MULTI-RUN EXPERIMENT SETUP")
    print(f"{'='*70}")
    print("Due to the complexity of the async operations, please run the following manually:")
    print()
    
    for run_num in range(1, 4):
        print(f"RUN {run_num}:")
        print(f"1. Use the sampled data file: results/multi_run/sampled_data_run_{run_num}.json")
        print(f"2. Run: python source_enhancement_experiment.py")
        print(f"   (modify the script to use sampled_data_run_{run_num}.json as input)")
        print(f"3. Run: python comparative_analyzer.py")
        print(f"4. Save results as: results/multi_run/analysis_run_{run_num}.json")
        print()
    
    print("5. Then run: python multi_run_analysis.py")
    print(f"{'='*70}")

def analyze_existing_runs():
    """Analyze results from completed runs."""
    results_dir = Path("results/multi_run")
    
    # Look for existing analysis files
    analysis_files = []
    for run_num in range(1, 4):
        analysis_file = results_dir / f"analysis_run_{run_num}.json"
        if analysis_file.exists():
            analysis_files.append(analysis_file)
    
    if not analysis_files:
        print("❌ No analysis files found. Please complete the runs first.")
        return
    
    print(f"📊 Found {len(analysis_files)} completed runs. Analyzing...")
    
    # Process each run
    all_filtered_results = []
    for i, analysis_file in enumerate(analysis_files, 1):
        print(f"Processing run {i}: {analysis_file.name}")
        filtered_result = filter_medium_plus_results(analysis_file)
        all_filtered_results.append(filtered_result)
    
    # Calculate multi-run statistics
    filtered_stats = calculate_multi_run_stats(all_filtered_results)
    
    # Generate report
    report = {
        'experiment_info': {
            'num_runs': len(analysis_files),
            'samples_per_run': 200,
            'timestamp': datetime.now().isoformat()
        },
        'filtered_medium_plus_stats': filtered_stats,
        'individual_filtered_results': all_filtered_results
    }
    
    # Save report
    report_file = results_dir / "multi_run_comprehensive_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print_summary_report(filtered_stats, len(analysis_files))
    print(f"\nDetailed report saved: {report_file}")

def print_summary_report(filtered_stats, num_runs):
    """Print summary of multi-run results."""
    print(f"\n📊 GENERALIZED RESULTS (Medium+ Length, {num_runs} runs)")
    print(f"{'='*70}")
    
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
        
        if comet_improvement['mean'] > 0:
            if comet_improvement['std_dev'] < comet_improvement['mean']:
                confidence = "HIGH"
            elif comet_improvement['std_dev'] < 2 * comet_improvement['mean']:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
                
            print(f"✅ HYPOTHESIS CONFIRMED with {confidence} confidence")
            print(f"   Source enhancement consistently improves MT quality")
            print(f"   Average COMET improvement: {comet_improvement['mean']:+.3f}")
            print(f"   Consistency (std dev): {comet_improvement['std_dev']:.3f}")
        else:
            print(f"❌ HYPOTHESIS NOT SUPPORTED")

def main():
    """Main function to setup or analyze multi-run experiment."""
    results_dir = Path("results/multi_run")
    results_dir.mkdir(exist_ok=True)
    
    # Check if we should analyze existing results
    analysis_files = list(results_dir.glob("analysis_run_*.json"))
    
    if analysis_files:
        print(f"Found {len(analysis_files)} existing analysis files.")
        choice = input("Analyze existing results? (y/n): ").lower().strip()
        if choice == 'y':
            analyze_existing_runs()
            return
    
    # Setup new experiment
    try:
        data = load_data()
        print(f"Loaded {len(data)} data points")
        
        # Create sampled data files for each run
        for run_num in range(1, 4):
            sampled_data = sample_data_for_run(data, run_num, 200)
            sample_file = create_sampled_data_file(sampled_data, run_num)
            print(f"Created sample file for run {run_num}: {sample_file}")
        
        print_instructions_for_runs()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
