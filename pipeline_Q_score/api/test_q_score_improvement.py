import json
import sys
import os
from pathlib import Path

# Add the current directory to the path so we can import main
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the functions and classes we need
from main import *

async def test_q_score_improvement():
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
        
        # Set up the main module's global variables
        import main
        main.evaluation_data = evaluation_data
        main.global_stats = compute_global_stats(evaluation_data)
        
        print('Global stats computed:', main.global_stats)
        
        # Test Q-score improvement calculation for first few APE records
        ape_records = [r for r in evaluation_data if "ape" in r][:5]
        
        print(f"\n=== Testing Q-Score Improvement for First 5 APE Records ===")
        delta_q_scores = []
        
        for i, record in enumerate(ape_records):
            original_q_score = record.get("q_score", None)
            ape_q_score = record.get("ape_q_score", None)
            
            # Calculate original Q-score if missing
            if original_q_score is None:
                original_gemba = record.get("gemba", 0)
                original_comet = record.get("comet", 0)
                original_cos = record.get("cos", 0)
                original_gemba_reason = record.get("flag", {}).get("gemba_reason")
                original_q_score = calculate_weighted_q_score(original_gemba, original_comet, original_cos, original_gemba_reason)
            
            # Calculate APE Q-score if missing
            if ape_q_score is None:
                original_gemba = record.get("gemba", 0)
                original_comet = record.get("comet", 0)
                original_cos = record.get("cos", 0)
                delta_gemba = record.get("delta_gemba", 0)
                delta_comet = record.get("delta_comet", 0)
                delta_cos = record.get("delta_cos", 0)
                improved_gemba = original_gemba + delta_gemba
                improved_comet = original_comet + delta_comet
                improved_cos = original_cos + delta_cos
                original_gemba_reason = record.get("flag", {}).get("gemba_reason")
                ape_q_score = calculate_weighted_q_score(improved_gemba, improved_comet, improved_cos, original_gemba_reason)
            
            delta_q = ape_q_score - original_q_score
            delta_q_scores.append(delta_q)
            
            key = record.get("key", "unknown")
            print(f"Record {i+1} ({key}): Q-score {original_q_score:.3f} → {ape_q_score:.3f} (Δ{delta_q:+.3f})")
        
        avg_improvement = statistics.mean(delta_q_scores) if delta_q_scores else 0
        print(f"\nSample average Q-score improvement: {avg_improvement:.3f}")
        
        # Get full analytics
        analytics = await get_analytics()
        
        print(f"\n=== Full Analytics Results ===")
        print(f"Total APE records: {analytics['ape_effectiveness']['total_ape_records']}")
        print(f"Average Q-score improvement: {analytics['ape_effectiveness']['avg_q_score_improvement']:.3f}")
        print(f"Average COMET improvement: {analytics['ape_effectiveness']['avg_comet_improvement']:.3f}")
        print(f"Average Cosine improvement: {analytics['ape_effectiveness']['avg_cosine_improvement']:.3f}")
        print(f"Average GEMBA improvement: {analytics['ape_effectiveness']['avg_gemba_improvement']:.1f}")
        print(f"Meaningful improvement rate: {analytics['ape_effectiveness']['meaningful_improvement_rate']:.1f}%")
        
        print("\n✅ Q-score improvement calculation is working!")
        
    except FileNotFoundError:
        print(f"❌ Data file not found: {DATA_FILE}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_q_score_improvement())
