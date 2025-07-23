#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from pathlib import Path
import json
import cfg
from typing import Dict, Tuple

class PrecisionOptimizer:
    """
    Threshold optimizer that prioritizes precision over recall to avoid false positives.
    Uses a weighted score: 0.6 * precision + 0.4 * recall instead of pure F1.
    """
    
    def __init__(self):
        self.current_thresholds = {
            'cos': cfg.COS_THR,
            'comet': cfg.COMET_THR
        }
    
    def load_pipeline_results(self, file_path: str = None) -> pd.DataFrame:
        if file_path is None:
            # Find the latest version directory
            out_dir = Path(cfg.OUT_DIR)
            version_dirs = [d for d in out_dir.iterdir() if d.is_dir() and d.name.startswith('v')]
            if not version_dirs:
                raise FileNotFoundError("No version directories found in output")
            
            latest_version = max(version_dirs, key=lambda x: int(x.name[1:]))
            file_path = latest_version / cfg.APE_OUTPUT_FILENAME
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return pd.DataFrame(data)
    
    def precision_recall_score(self, y_true, y_pred, precision_weight=0.6, recall_weight=0.4):
        """
        Custom scoring function that weights precision higher than recall.
        Default: 60% precision, 40% recall
        """
        if len(set(y_pred)) <= 1:  # No variation in predictions
            return 0.0
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        return precision_weight * precision + recall_weight * recall
    
    def grid_search_thresholds(self, df: pd.DataFrame, bucket: str, 
                             precision_weight: float = 0.6) -> Tuple[float, float, float]:
        bucket_data = df[df['bucket'] == bucket].copy()
        if len(bucket_data) == 0:
            return self.current_thresholds['cos'][bucket], self.current_thresholds['comet'][bucket], 0.0
        
        # Higher threshold range for precision focus: 0.70-0.90
        cos_range = np.arange(0.70, 0.90, 0.01)
        comet_range = np.arange(0.70, 0.90, 0.01)
        
        best_score = 0
        best_cos = self.current_thresholds['cos'][bucket]
        best_comet = self.current_thresholds['comet'][bucket]
        
        if 'tag' not in bucket_data.columns:
            return best_cos, best_comet, 0.0
            
        bucket_data['actual_pass'] = bucket_data['tag'].isin(['strict_pass', 'soft_pass'])
        
        if bucket_data['actual_pass'].nunique() <= 1:
            return best_cos, best_comet, 0.0
        
        for cos_thr in cos_range:
            for comet_thr in comet_range:
                bucket_data['predicted_pass'] = (
                    (bucket_data['cos'] >= cos_thr) & 
                    (bucket_data['comet'] >= comet_thr) &
                    (bucket_data.get('gemba', 0) >= cfg.GEMBA_PASS)
                )
                
                if bucket_data['predicted_pass'].nunique() > 1:
                    score = self.precision_recall_score(
                        bucket_data['actual_pass'], 
                        bucket_data['predicted_pass'],
                        precision_weight=precision_weight
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_cos = cos_thr
                        best_comet = comet_thr
        
        return best_cos, best_comet, best_score
    
    def optimize_thresholds(self, file_path: str = None, precision_weight: float = 0.6) -> Dict:
        df = self.load_pipeline_results(file_path)
        
        print(f"Loaded {len(df)} records for precision-focused optimization")
        print(f"Distribution by bucket: {df['bucket'].value_counts().to_dict()}")
        print(f"Precision weight: {precision_weight:.1f}, Recall weight: {1-precision_weight:.1f}")
        print("Optimizing thresholds by bucket:")
        
        results = {
            'optimized_thresholds': {'cos': {}, 'comet': {}},
            'bucket_results': {},
            'total_records': len(df)
        }
        
        overall_before_scores = []
        overall_after_scores = []
        
        for bucket in ['very_short', 'short', 'medium', 'long', 'very_long']:
            bucket_data = df[df['bucket'] == bucket].copy()
            if len(bucket_data) == 0:
                continue
                
            # Current performance
            current_cos = self.current_thresholds['cos'][bucket]
            current_comet = self.current_thresholds['comet'][bucket]
            
            bucket_data['actual_pass'] = bucket_data['tag'].isin(['strict_pass', 'soft_pass'])
            bucket_data['current_predicted'] = (
                (bucket_data['cos'] >= current_cos) & 
                (bucket_data['comet'] >= current_comet) &
                (bucket_data.get('gemba', 0) >= cfg.GEMBA_PASS)
            )
            
            current_score = self.precision_recall_score(
                bucket_data['actual_pass'], 
                bucket_data['current_predicted'],
                precision_weight=precision_weight
            ) if bucket_data['current_predicted'].nunique() > 1 else 0
            
            # Optimize
            optimized_cos, optimized_comet, optimized_score = self.grid_search_thresholds(
                df, bucket, precision_weight
            )
            
            improvement = optimized_score - current_score
            
            print(f"Processing {bucket}...")
            print(f"  Current:   COS={current_cos:.3f}, COMET={current_comet:.3f}, Score={current_score:.2f}")
            print(f"  Optimized: COS={optimized_cos:.3f}, COMET={optimized_comet:.3f}, Score={optimized_score:.2f}")
            print(f"  Improvement: +{improvement:.2f}")
            
            results['optimized_thresholds']['cos'][bucket] = optimized_cos
            results['optimized_thresholds']['comet'][bucket] = optimized_comet
            results['bucket_results'][bucket] = {
                'current_cos': current_cos,
                'current_comet': current_comet,
                'current_score': current_score,
                'optimized_cos': optimized_cos,
                'optimized_comet': optimized_comet,
                'optimized_score': optimized_score,
                'improvement': improvement,
                'records_count': len(bucket_data)
            }
            
            overall_before_scores.append(current_score)
            overall_after_scores.append(optimized_score)
        
        overall_improvement = np.mean(overall_after_scores) - np.mean(overall_before_scores)
        
        print("=" * 60)
        print("PRECISION-FOCUSED OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"Overall score improvement: +{overall_improvement:.2f}")
        print(f"Average score: {np.mean(overall_before_scores):.2f} → {np.mean(overall_after_scores):.2f}")
        
        results['overall_improvement'] = overall_improvement
        results['avg_score_before'] = np.mean(overall_before_scores)
        results['avg_score_after'] = np.mean(overall_after_scores)
        
        return results
    
    def update_config_file(self, optimized_thresholds: Dict):
        cfg_path = Path(__file__).parent / "cfg.py"
        backup_path = cfg_path.with_suffix('.py.bak')
        
        # Create backup
        if backup_path.exists():
            backup_path.unlink()
        cfg_path.rename(backup_path)
        print(f"Backed up original config to {backup_path.name}")
        
        # Read original file
        with open(backup_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update COS_THR
        cos_dict_str = "{\n"
        for bucket, value in optimized_thresholds['cos'].items():
            cos_dict_str += f'    "{bucket}": {value:.3f},\n'
        cos_dict_str += "}"
        
        # Update COMET_THR  
        comet_dict_str = "{\n"
        for bucket, value in optimized_thresholds['comet'].items():
            comet_dict_str += f'    "{bucket}": {value:.3f},\n'
        comet_dict_str += "}"
        
        # Replace in content
        import re
        content = re.sub(
            r'COS_THR = \{[^}]*\}',
            f'COS_THR = {cos_dict_str}',
            content,
            flags=re.DOTALL
        )
        content = re.sub(
            r'COMET_THR = \{[^}]*\}',
            f'COMET_THR = {comet_dict_str}',
            content,
            flags=re.DOTALL
        )
        
        # Write updated file
        with open(cfg_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("Updated cfg.py with precision-optimized thresholds")

def main():
    optimizer = PrecisionOptimizer()
    
    print("Precision-Focused Threshold Optimizer")
    print("=" * 50)
    print("This optimizer prioritizes precision to reduce false positives.")
    
    precision_weight = float(input("Enter precision weight (0.6-0.8 recommended): ") or "0.7")
    
    try:
        results = optimizer.optimize_thresholds(precision_weight=precision_weight)
        
        # Save results
        report_path = Path("precision_optimization_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Optimization report saved to {report_path}")
        
        # Ask to update config
        update = input("\nDo you want to update cfg.py with these precision-optimized thresholds? (y/n): ")
        if update.lower() == 'y':
            optimizer.update_config_file(results['optimized_thresholds'])
            print("Thresholds updated! Run the pipeline again to see the improvement.")
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
