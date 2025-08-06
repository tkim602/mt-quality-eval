import json
import sys
import os
from pathlib import Path

# Add the current directory to the path so we can import main
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the functions and classes we need
from main import (
    DATA_FILE, compute_global_stats_from_data, calculate_weighted_q_score, 
    get_quality_grade_by_scores, get_quality_distribution_before_after
)

def test():
    # Load data directly
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
        global_stats = compute_global_stats_from_data(evaluation_data)
        print('Global stats computed:', global_stats)
        
        # Test first record
        record = evaluation_data[0]
        print(f"First record keys: {list(record.keys())}")
        
        gemba = record.get('gemba', 0)
        comet = record.get('comet', 0) 
        cos = record.get('cos', 0)
        tag = record.get('tag', None)
        
        print(f'Test record: GEMBA:{gemba}, COMET:{comet:.3f}, COS:{cos:.3f}, TAG:{tag}')
        
        # Test Q-score calculation
        q_score = calculate_weighted_q_score(gemba, comet, cos, None)
        grade = get_quality_grade_by_scores(gemba, comet, cos, q_score, tag, None)
        
        print(f'Q-score: {q_score:.3f}, Grade: {grade}')
        
        # Test quality distribution with mock global vars
        import main
        main.evaluation_data = evaluation_data
        main.global_stats = global_stats
        
        quality_dist = get_quality_distribution_before_after()
        print('Quality Distribution:')
        print(json.dumps(quality_dist, indent=2))
        
        print("\n✅ All syntax checks passed!")
        
    except FileNotFoundError:
        print(f"❌ Data file not found: {DATA_FILE}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
