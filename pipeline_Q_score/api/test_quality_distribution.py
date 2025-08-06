import json
import sys
import os
from pathlib import Path

# Add the current directory to the path so we can import main
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the functions and classes we need
from main import (
    DATA_FILE, compute_global_stats, calculate_weighted_q_score, 
    get_quality_grade_by_scores, get_quality_distribution_before_after
)

def test_before_ape_distribution():
    print(f"Loading data from: {DATA_FILE}")
    
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'records' in data:
            evaluation_data = data['records']
            print(f"Loaded {len(evaluation_data)} evaluation records (with delta_gemba)")
        else:
            evaluation_data = data
            print(f"Loaded {len(evaluation_data)} evaluation records (original format)")
        
        if not evaluation_data:
            print("No evaluation data loaded!")
            return
        
        # Compute global stats
        global_stats = compute_global_stats(evaluation_data)
        print('Global stats computed:', global_stats)
        
        # Test quality grading for first few records
        print("\n=== Testing Quality Grading for First 10 Records ===")
        before_grades = {}
        
        for i, record in enumerate(evaluation_data[:10]):
            gemba = record.get('gemba', 0)
            comet = record.get('comet', 0)
            cos = record.get('cos', 0)
            tag = record.get('tag', None)
            gemba_reason = record.get('flag', {}).get('gemba_reason')
            q_score = record.get('q_score', None)
            
            # Calculate Q-score if not available
            if q_score is None:
                q_score = calculate_weighted_q_score(gemba, comet, cos, gemba_reason)
            
            grade = get_quality_grade_by_scores(gemba, comet, cos, q_score, tag, gemba_reason)
            
            print(f"Record {i+1}: G:{gemba:.0f}/C:{comet:.3f}/S:{cos:.3f}, Q:{q_score:.3f}, Tag:{tag}, Grade:{grade}")
            
            before_grades[grade] = before_grades.get(grade, 0) + 1
        
        print(f"\nSample grade distribution: {before_grades}")
        
        # Test the full quality distribution function
        import main
        main.evaluation_data = evaluation_data
        main.global_stats = global_stats
        
        quality_dist = get_quality_distribution_before_after()
        
        print('\n=== Full Quality Distribution ===')
        print('Before APE:')
        for grade, count in quality_dist['before'].items():
            if count > 0:
                pct = (count / quality_dist['total_records']) * 100
                print(f"  {grade}: {count} ({pct:.1f}%)")
        
        print('\nAfter APE:')
        for grade, count in quality_dist['after'].items():
            if count > 0:
                pct = (count / quality_dist['total_records']) * 100
                print(f"  {grade}: {count} ({pct:.1f}%)")
        
        print(f"\nTotal records: {quality_dist['total_records']}")
        print(f"APE applied count: {quality_dist['ape_applied_count']}")
        
        print("\n✅ Quality distribution analysis complete!")
        
    except FileNotFoundError:
        print(f"❌ Data file not found: {DATA_FILE}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_before_ape_distribution()
