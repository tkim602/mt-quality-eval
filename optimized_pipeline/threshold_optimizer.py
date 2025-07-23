import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import ParameterGrid
import cfg
from collections import defaultdict

class ThresholdOptimizer:
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.current_thresholds = {
            'cos': cfg.COS_THR.copy(),
            'comet': cfg.COMET_THR.copy()
        }
        self.optimization_history = []
        
    def load_pipeline_results(self, file_path: str = None) -> pd.DataFrame:
        if file_path is None:
            latest_file = self.find_latest_output()
            if latest_file is None:
                raise FileNotFoundError("No pipeline output found")
            file_path = latest_file
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return pd.DataFrame(data)
    
    def find_latest_output(self) -> str:
        out_dir = Path(cfg.OUT_DIR)
        if not out_dir.exists():
            return None
        
        # Look for versioned directories first (v1, v2, etc.)
        version_dirs = [d for d in out_dir.iterdir() if d.is_dir() and d.name.startswith('v')]
        if version_dirs:
            # Get latest version directory
            latest_version = max(version_dirs, key=lambda p: p.stat().st_mtime)
            # Look for APE evidence file first, then gemba, then filtered
            for filename in ["ape_evidence.json", "gemba.json", "filtered.json"]:
                file_path = latest_version / filename
                if file_path.exists():
                    return str(file_path)
        
        # Fallback to root directory files
        json_files = list(out_dir.glob("filtered_*.json"))
        if not json_files:
            json_files = list(out_dir.glob("*.json"))
            
        if json_files:
            return str(max(json_files, key=lambda p: p.stat().st_mtime))
        return None
    
    def calculate_f1_scores(self, df: pd.DataFrame, cos_thr: Dict, comet_thr: Dict) -> Dict[str, float]:
        f1_scores = {}
        
        for bucket in ['very_short', 'short', 'medium', 'long', 'very_long']:
            bucket_data = df[df['bucket'] == bucket].copy()
            if len(bucket_data) == 0:
                continue
                
            bucket_data['predicted_pass'] = (
                (bucket_data['cos'] >= cos_thr[bucket]) & 
                (bucket_data['comet'] >= comet_thr[bucket]) &
                (bucket_data.get('gemba', 0) >= cfg.GEMBA_PASS)
            )
            
            if 'tag' in bucket_data.columns:
                bucket_data['actual_pass'] = bucket_data['tag'].isin(['strict_pass', 'soft_pass'])
                
                if bucket_data['actual_pass'].nunique() > 1:
                    f1 = f1_score(bucket_data['actual_pass'], bucket_data['predicted_pass'])
                    f1_scores[bucket] = f1
                    
        return f1_scores
    
    def grid_search_thresholds(self, df: pd.DataFrame, bucket: str) -> Tuple[float, float, float]:
        bucket_data = df[df['bucket'] == bucket].copy()
        if len(bucket_data) == 0:
            return self.current_thresholds['cos'][bucket], self.current_thresholds['comet'][bucket], 0.0
        
        cos_range = np.arange(0.75, 0.95, 0.01)
        comet_range = np.arange(0.75, 0.95, 0.01)
        
        best_f1 = 0
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
                    f1 = f1_score(bucket_data['actual_pass'], bucket_data['predicted_pass'])
                    if f1 > best_f1:
                        best_f1 = f1
                        best_cos = cos_thr
                        best_comet = comet_thr
        
        return best_cos, best_comet, best_f1
    
    def optimize_thresholds(self, file_path: str = None) -> Dict:
        df = self.load_pipeline_results(file_path)
        
        print(f"Loaded {len(df)} records for optimization")
        print(f"Distribution by bucket: {df['bucket'].value_counts().to_dict()}")
        
        optimized_thresholds = {
            'cos': {},
            'comet': {}
        }
        
        optimization_results = {
            'bucket_results': {},
            'overall_improvement': 0,
            'total_records': len(df)
        }
        
        current_f1_scores = self.calculate_f1_scores(df, self.current_thresholds['cos'], self.current_thresholds['comet'])
        
        print("\nOptimizing thresholds by bucket:")
        for bucket in ['very_short', 'short', 'medium', 'long', 'very_long']:
            print(f"\nProcessing {bucket}...")
            
            best_cos, best_comet, best_f1 = self.grid_search_thresholds(df, bucket)
            
            optimized_thresholds['cos'][bucket] = round(best_cos, 3)
            optimized_thresholds['comet'][bucket] = round(best_comet, 3)
            
            current_f1 = current_f1_scores.get(bucket, 0)
            improvement = best_f1 - current_f1
            
            optimization_results['bucket_results'][bucket] = {
                'current_cos': self.current_thresholds['cos'][bucket],
                'current_comet': self.current_thresholds['comet'][bucket],
                'current_f1': round(current_f1 * 100, 2),
                'optimized_cos': optimized_thresholds['cos'][bucket],
                'optimized_comet': optimized_thresholds['comet'][bucket],
                'optimized_f1': round(best_f1 * 100, 2),
                'improvement': round(improvement * 100, 2),
                'records_count': len(df[df['bucket'] == bucket])
            }
            
            print(f"  Current:   COS={self.current_thresholds['cos'][bucket]:.3f}, COMET={self.current_thresholds['comet'][bucket]:.3f}, F1={current_f1*100:.2f}%")
            print(f"  Optimized: COS={best_cos:.3f}, COMET={best_comet:.3f}, F1={best_f1*100:.2f}%")
            print(f"  Improvement: {improvement*100:+.2f}%")
        
        new_f1_scores = self.calculate_f1_scores(df, optimized_thresholds['cos'], optimized_thresholds['comet'])
        
        avg_current_f1 = np.mean(list(current_f1_scores.values())) if current_f1_scores else 0
        avg_new_f1 = np.mean(list(new_f1_scores.values())) if new_f1_scores else 0
        overall_improvement = avg_new_f1 - avg_current_f1
        
        optimization_results['overall_improvement'] = round(overall_improvement * 100, 2)
        optimization_results['avg_current_f1'] = round(avg_current_f1 * 100, 2)
        optimization_results['avg_optimized_f1'] = round(avg_new_f1 * 100, 2)
        
        self.optimization_history.append(optimization_results)
        
        return {
            'optimized_thresholds': optimized_thresholds,
            'results': optimization_results
        }
    
    def update_config_file(self, optimized_thresholds: Dict):
        cfg_path = Path('cfg.py')
        with open(cfg_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        cos_thr_str = "COS_THR = {\n"
        for bucket, threshold in optimized_thresholds['cos'].items():
            cos_thr_str += f'    "{bucket}": {threshold},\n'
        cos_thr_str += "}"
        
        comet_thr_str = "COMET_THR = {\n"
        for bucket, threshold in optimized_thresholds['comet'].items():
            comet_thr_str += f'    "{bucket}": {threshold},\n'
        comet_thr_str += "}"
        
        import re
        content = re.sub(r'COS_THR = \{[^}]+\}', cos_thr_str, content, flags=re.DOTALL)
        content = re.sub(r'COMET_THR = \{[^}]+\}', comet_thr_str, content, flags=re.DOTALL)
        
        backup_path = cfg_path.with_suffix('.py.bak')
        cfg_path.rename(backup_path)
        print(f"Backed up original config to {backup_path}")
        
        with open(cfg_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("Updated cfg.py with optimized thresholds")
    
    def save_optimization_report(self, results: Dict, output_path: str = None):
        if output_path is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"optimization_report_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Optimization report saved to {output_path}")
    
    def print_optimization_summary(self, results: Dict):
        print("\n" + "="*60)
        print("THRESHOLD OPTIMIZATION SUMMARY")
        print("="*60)
        
        bucket_results = results['results']['bucket_results']
        
        print(f"Total records analyzed: {results['results']['total_records']}")
        print(f"Overall F1 improvement: {results['results']['overall_improvement']:+.2f}%")
        print(f"Average F1: {results['results']['avg_current_f1']:.2f}% → {results['results']['avg_optimized_f1']:.2f}%")
        
        print("\nBucket-wise Results:")
        print("-" * 80)
        print(f"{'Bucket':<12} {'Current F1':<10} {'New F1':<10} {'Improve':<8} {'COS':<15} {'COMET':<15}")
        print("-" * 80)
        
        for bucket, data in bucket_results.items():
            if data['records_count'] > 0:
                cos_change = f"{data['current_cos']:.3f}→{data['optimized_cos']:.3f}"
                comet_change = f"{data['current_comet']:.3f}→{data['optimized_comet']:.3f}"
                print(f"{bucket:<12} {data['current_f1']:<10.2f} {data['optimized_f1']:<10.2f} "
                      f"{data['improvement']:<+8.2f} {cos_change:<15} {comet_change:<15}")

def main():
    optimizer = ThresholdOptimizer()
    
    try:
        results = optimizer.optimize_thresholds()
        
        optimizer.print_optimization_summary(results)
        optimizer.save_optimization_report(results)
        
        print(f"\nDo you want to update cfg.py with these optimized thresholds? (y/n): ", end="")
        response = input().strip().lower()
        
        if response in ['y', 'yes']:
            optimizer.update_config_file(results['optimized_thresholds'])
            print("Thresholds updated! Run the pipeline again to see the improvement.")
        else:
            print("Thresholds not updated. You can manually copy them from the report.")
            
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
