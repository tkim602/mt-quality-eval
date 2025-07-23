import json
import pandas as pd
import numpy as np
from pathlib import Path
from threshold_optimizer import ThresholdOptimizer
import cfg

def analyze_current_performance(file_path: str = None):
    """Quick analysis of current threshold performance"""
    
    optimizer = ThresholdOptimizer()
    
    if file_path is None:
        file_path = optimizer.find_latest_output()
        if file_path is None:
            print("No pipeline output found. Run 'python run_pipeline.py' first.")
            return
    
    print(f"Analyzing pipeline output: {file_path}")
    
    try:
        df = optimizer.load_pipeline_results(file_path)
    except Exception as e:
        print(f"Error loading results: {e}")
        return
    
    print(f"\nDataset Summary:")
    print(f"Total records: {len(df)}")
    print(f"Records by bucket: {df['bucket'].value_counts().to_dict()}")
    
    # Current performance
    current_f1_scores = optimizer.calculate_f1_scores(df, cfg.COS_THR, cfg.COMET_THR)
    
    print(f"\nCurrent Threshold Performance:")
    print("-" * 60)
    print(f"{'Bucket':<12} {'Records':<8} {'Current F1':<12} {'COS Thr':<8} {'COMET Thr':<10}")
    print("-" * 60)
    
    total_weighted_f1 = 0
    total_records = 0
    
    for bucket in ['very_short', 'short', 'medium', 'long', 'very_long']:
        bucket_count = len(df[df['bucket'] == bucket])
        f1_score = current_f1_scores.get(bucket, 0) * 100
        cos_thr = cfg.COS_THR[bucket]
        comet_thr = cfg.COMET_THR[bucket]
        
        if bucket_count > 0:
            total_weighted_f1 += f1_score * bucket_count
            total_records += bucket_count
            
        print(f"{bucket:<12} {bucket_count:<8} {f1_score:<12.2f} {cos_thr:<8.3f} {comet_thr:<10.3f}")
    
    if total_records > 0:
        avg_f1 = total_weighted_f1 / total_records
        print(f"\nWeighted Average F1: {avg_f1:.2f}%")
    
    # Show tag distribution
    if 'tag' in df.columns:
        tag_dist = df['tag'].value_counts()
        total = len(df)
        
        print(f"\nCurrent Decision Distribution:")
        print("-" * 30)
        for tag, count in tag_dist.items():
            percentage = (count / total) * 100
            print(f"{tag:<15} {count:>6} ({percentage:5.1f}%)")
    
    # Quick optimization preview
    print(f"\nQuick Optimization Preview:")
    print("-" * 40)
    
    for bucket in ['very_short', 'short', 'medium', 'long', 'very_long']:
        bucket_data = df[df['bucket'] == bucket]
        if len(bucket_data) == 0:
            continue
            
        best_cos, best_comet, best_f1 = optimizer.grid_search_thresholds(df, bucket)
        current_f1 = current_f1_scores.get(bucket, 0)
        improvement = (best_f1 - current_f1) * 100
        
        if improvement > 0.5:  # Only show if significant improvement
            print(f"{bucket}: F1 {current_f1*100:.1f}% → {best_f1*100:.1f}% (+{improvement:.1f}%)")
            print(f"  Suggest: COS {cfg.COS_THR[bucket]:.3f} → {best_cos:.3f}, "
                  f"COMET {cfg.COMET_THR[bucket]:.3f} → {best_comet:.3f}")

def compare_with_optimal_reference():
    """Compare current thresholds with the reference optimal values"""
    
    # Reference optimal thresholds from your data
    reference_optimal = {
        'cos': {
            'very_short': 0.840,
            'short': 0.850,
            'medium': 0.820,
            'long': 0.820,
            'very_long': 0.830
        },
        'comet': {
            'very_short': 0.850,
            'short': 0.830,
            'medium': 0.840,
            'long': 0.830,
            'very_long': 0.830
        }
    }
    
    print(f"\nComparison with Reference Optimal Thresholds:")
    print("-" * 70)
    print(f"{'Bucket':<12} {'Current COS':<12} {'Optimal COS':<12} {'Current COMET':<14} {'Optimal COMET':<14}")
    print("-" * 70)
    
    for bucket in ['very_short', 'short', 'medium', 'long', 'very_long']:
        current_cos = cfg.COS_THR[bucket]
        optimal_cos = reference_optimal['cos'][bucket]
        current_comet = cfg.COMET_THR[bucket]
        optimal_comet = reference_optimal['comet'][bucket]
        
        cos_diff = current_cos - optimal_cos
        comet_diff = current_comet - optimal_comet
        
        print(f"{bucket:<12} {current_cos:<12.3f} {optimal_cos:<12.3f} "
              f"{current_comet:<14.3f} {optimal_comet:<14.3f}")
        
        if abs(cos_diff) > 0.005 or abs(comet_diff) > 0.005:
            print(f"             Difference:  {cos_diff:+.3f}           {comet_diff:+.3f}")

def main():
    print("Threshold Analysis Tool")
    print("="*50)
    
    # Analyze current performance
    analyze_current_performance()
    
    # Compare with reference optimal
    compare_with_optimal_reference()
    
    print(f"\nRecommendations:")
    print("1. Run 'python threshold_optimizer.py' for single optimization")
    print("2. Run 'python iterative_optimizer.py' for automatic iterative optimization")
    print("3. Run 'python run_pipeline.py' to generate new data for optimization")

if __name__ == "__main__":
    main()
