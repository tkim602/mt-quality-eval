#!/usr/bin/env python3
"""
Filter analysis results to include only medium+ length texts and recalculate statistics.
"""

import json
from pathlib import Path

def filter_medium_plus_analysis():
    """Filter and recalculate analysis for medium+ length texts only."""
    
    # Load the full analysis results
    results_file = Path("results/comparative_analysis_results.json")
    with open(results_file, 'r', encoding='utf-8') as f:
        full_results = json.load(f)
    
    # Filter to keep only medium, long, and very_long buckets
    medium_plus_buckets = ['medium', 'long', 'very_long']
    
    # Filter detailed results
    filtered_details = []
    for detail in full_results['detailed_results']:
        if detail['length_bucket'] in medium_plus_buckets:
            filtered_details.append(detail)
    
    print(f"Filtered from {len(full_results['detailed_results'])} to {len(filtered_details)} samples")
    print(f"Removed buckets: very_short ({full_results['bucket_stats']['very_short']['count']} samples)")
    print(f"Removed buckets: short ({full_results['bucket_stats']['short']['count']} samples)")
    
    # Recalculate overall statistics
    if not filtered_details:
        print("No samples remaining after filtering!")
        return
    
    # Calculate new overall stats
    original_cos_scores = [d['original_cos_similarity'] for d in filtered_details]
    enhanced_cos_scores = [d['enhanced_cos_similarity'] for d in filtered_details]
    original_comet_scores = [d['original_comet_score'] for d in filtered_details]
    enhanced_comet_scores = [d['enhanced_comet_score'] for d in filtered_details]
    
    cos_improvements = [d['cos_improvement'] for d in filtered_details]
    comet_improvements = [d['comet_improvement'] for d in filtered_details]
    
    new_overall_stats = {
        "original_cos_mean": sum(original_cos_scores) / len(original_cos_scores),
        "enhanced_cos_mean": sum(enhanced_cos_scores) / len(enhanced_cos_scores),
        "cos_improvement_mean": sum(cos_improvements) / len(cos_improvements),
        "original_comet_mean": sum(original_comet_scores) / len(original_comet_scores),
        "enhanced_comet_mean": sum(enhanced_comet_scores) / len(enhanced_comet_scores),
        "comet_improvement_mean": sum(comet_improvements) / len(comet_improvements),
        "cos_improvement_positive_pct": (len([x for x in cos_improvements if x > 0]) / len(cos_improvements)) * 100,
        "comet_improvement_positive_pct": (len([x for x in comet_improvements if x > 0]) / len(comet_improvements)) * 100
    }
    
    # Filter bucket stats
    filtered_bucket_stats = {}
    for bucket in medium_plus_buckets:
        if bucket in full_results['bucket_stats']:
            filtered_bucket_stats[bucket] = full_results['bucket_stats'][bucket]
    
    # Create filtered results
    filtered_results = {
        "total_samples": len(filtered_details),
        "filter_applied": "medium_plus_only",
        "excluded_buckets": ["very_short", "short"],
        "excluded_samples": full_results['total_samples'] - len(filtered_details),
        "overall_stats": new_overall_stats,
        "bucket_stats": filtered_bucket_stats,
        "detailed_results": filtered_details
    }
    
    # Save filtered results
    filtered_file = Path("results/filtered_medium_plus_analysis.json")
    with open(filtered_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print("FILTERED ANALYSIS RESULTS (Medium+ Length Only)")
    print(f"{'='*70}")
    print(f"Total samples analyzed: {filtered_results['total_samples']}")
    print(f"Excluded samples: {filtered_results['excluded_samples']} (very_short + short)")
    
    print(f"\nOVERALL COMPARISON:")
    print(f"COS Similarity:  {new_overall_stats['original_cos_mean']:.3f} → {new_overall_stats['enhanced_cos_mean']:.3f} ({new_overall_stats['cos_improvement_mean']:+.3f})")
    print(f"COMET Score:     {new_overall_stats['original_comet_mean']:.3f} → {new_overall_stats['enhanced_comet_mean']:.3f} ({new_overall_stats['comet_improvement_mean']:+.3f})")
    
    print(f"\nIMPROVEMENT RATES:")
    print(f"COS Similarity improved:  {new_overall_stats['cos_improvement_positive_pct']:.1f}% of samples")
    print(f"COMET Score improved:     {new_overall_stats['comet_improvement_positive_pct']:.1f}% of samples")
    
    print(f"\nBUCKET-WISE ANALYSIS:")
    print(f"{'-'*85}")
    print(f"{'Bucket':<12} {'Count':<6} {'Original→Enhanced COS':<20} {'Original→Enhanced COMET':<22} {'Improve%'}")
    print(f"{'-'*85}")
    
    for bucket_name, bucket_data in filtered_bucket_stats.items():
        improve_pct = bucket_data['improvement_positive_pct']
        print(f"{bucket_name:<12} {bucket_data['count']:<6} "
              f"{bucket_data['original_cos_mean']:.3f}→{bucket_data['enhanced_cos_mean']:.3f}      "
              f"{bucket_data['original_comet_mean']:.3f}→{bucket_data['enhanced_comet_mean']:.3f}          "
              f"{improve_pct:.1f}%")
    
    print(f"\nDetailed results saved to: {filtered_file.absolute()}")
    
    print(f"\n{'='*70}")
    print("HYPOTHESIS VALIDATION (Medium+ Length Only)")
    print(f"{'='*70}")
    
    cos_change = new_overall_stats['cos_improvement_mean']
    comet_change = new_overall_stats['comet_improvement_mean']
    
    if comet_change > 0.01:
        status = "✅ HYPOTHESIS STRONGLY SUPPORTED"
    elif comet_change > 0.005:
        status = "✅ HYPOTHESIS SUPPORTED"
    elif comet_change > 0:
        status = "⚠️  HYPOTHESIS WEAKLY SUPPORTED"
    else:
        status = "❌ HYPOTHESIS NOT SUPPORTED"
    
    print(f"{status}: Enhanced Korean source improves MT quality for medium+ length texts")
    print(f"   COS Similarity change: {cos_change:+.3f}")
    print(f"   COMET Score change: {comet_change:+.3f}")
    
    return filtered_results

if __name__ == "__main__":
    filter_medium_plus_analysis()
