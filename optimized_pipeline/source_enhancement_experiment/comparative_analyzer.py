#!/usr/bin/env python3
"""
Source Enhancement Comparative Analysis

This script takes the results from the source enhancement experiment
and runs both original and enhanced translations through the same
LaBSE/COMET evaluation pipeline to quantitatively compare quality.

Author: MT Quality Evaluation Pipeline  
Date: July 23, 2025
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from comet import download_model, load_from_checkpoint
import torch
from typing import List, Dict, Any, Tuple
import sys

# Add parent directory to import main pipeline modules
sys.path.append(str(Path(__file__).parent.parent))
import cfg

class SourceEnhancementAnalyzer:
    def __init__(self):
        self.experiment_dir = Path(__file__).parent
        self.results_dir = self.experiment_dir / "results"
        self.device = torch.device("cpu")  # Use CPU like main pipeline
        
        print("Loading models...")
        self.labse_model = SentenceTransformer(cfg.LABSE_MODEL, device=self.device)
        
        # Load COMET model
        comet_model_path = download_model(cfg.COMET_CKPT)
        self.comet_model = load_from_checkpoint(comet_model_path)
        print("Models loaded successfully")
    
    def load_experiment_results(self, filename: str) -> List[Dict[str, Any]]:
        """Load the source enhancement experiment results."""
        
        results_file = self.results_dir / filename
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"Loaded {len(results)} experiment results")
        return results
    
    def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity using LaBSE embeddings."""
        
        embeddings = self.labse_model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)
    
    def calculate_comet_score(self, source: str, translation: str, reference: str) -> float:
        """Calculate COMET score for translation quality."""
        
        try:
            comet_input = [{
                'src': source,
                'mt': translation, 
                'ref': reference
            }]
            
            # Use CPU prediction like main pipeline
            scores = self.comet_model.predict(comet_input, batch_size=1)
            
            # Handle COMET Prediction object format
            if hasattr(scores, 'system_score'):
                return float(scores.system_score)
            elif hasattr(scores, 'scores') and isinstance(scores.scores, list):
                return float(scores.scores[0])
            elif isinstance(scores, list) and len(scores) > 0:
                score = scores[0]
                if hasattr(score, 'item'):
                    return float(score.item())
                elif isinstance(score, (list, tuple)) and len(score) > 0:
                    return float(score[0])
                else:
                    return float(score)
            else:
                return float(scores)
            
        except Exception as e:
            print(f"COMET calculation error: {e}")
            return 0.0
    
    def categorize_by_length(self, text: str) -> str:
        """Categorize text by length using same logic as main pipeline."""
        
        length = len(text.split())
        
        if length <= 5:
            return "very_short"
        elif length <= 15:
            return "short"
        elif length <= 30:
            return "medium"
        elif length <= 50:
            return "long"
        else:
            return "very_long"
    
    def analyze_single_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single experiment result with quality metrics."""
        
        analysis = {
            'id': result['id'],
            'category': result['category'],
            'korean_original': result['korean_original'],
            'korean_enhanced': result['korean_enhanced'],
            'english_reference': result['english_reference'],
            'english_from_original': result['english_from_original'],
            'english_from_enhanced': result['english_from_enhanced']
        }
        
        # Categorize by length
        analysis['length_bucket'] = self.categorize_by_length(result['korean_original'])
        
        # Calculate metrics for original translation
        analysis['original_cos_similarity'] = self.calculate_cosine_similarity(
            result['english_from_original'], 
            result['english_reference']
        )
        
        analysis['original_comet_score'] = self.calculate_comet_score(
            result['korean_original'],
            result['english_from_original'],
            result['english_reference']
        )
        
        # Calculate metrics for enhanced translation
        analysis['enhanced_cos_similarity'] = self.calculate_cosine_similarity(
            result['english_from_enhanced'],
            result['english_reference']
        )
        
        analysis['enhanced_comet_score'] = self.calculate_comet_score(
            result['korean_enhanced'],
            result['english_from_enhanced'], 
            result['english_reference']
        )
        
        # Calculate improvements
        analysis['cos_improvement'] = (
            analysis['enhanced_cos_similarity'] - analysis['original_cos_similarity']
        )
        
        analysis['comet_improvement'] = (
            analysis['enhanced_comet_score'] - analysis['original_comet_score']
        )
        
        return analysis
    
    def run_comparative_analysis(self, experiment_filename: str) -> Dict[str, Any]:
        """Run complete comparative analysis."""
        
        print("=" * 70)
        print("SOURCE ENHANCEMENT COMPARATIVE ANALYSIS")
        print("=" * 70)
        
        # Load experiment results
        experiment_results = self.load_experiment_results(experiment_filename)
        
        # Analyze each result
        print("Calculating quality metrics...")
        analyzed_results = []
        
        for i, result in enumerate(experiment_results):
            print(f"Analyzing {i+1}/{len(experiment_results)}: {result['id']}")
            analysis = self.analyze_single_result(result)
            analyzed_results.append(analysis)
        
        # Aggregate statistics
        df = pd.DataFrame(analyzed_results)
        
        stats = {
            'total_samples': len(analyzed_results),
            'overall_stats': {
                'original_cos_mean': df['original_cos_similarity'].mean(),
                'enhanced_cos_mean': df['enhanced_cos_similarity'].mean(),
                'cos_improvement_mean': df['cos_improvement'].mean(),
                'original_comet_mean': df['original_comet_score'].mean(),
                'enhanced_comet_mean': df['enhanced_comet_score'].mean(),
                'comet_improvement_mean': df['comet_improvement'].mean(),
                'cos_improvement_positive_pct': (df['cos_improvement'] > 0).mean() * 100,
                'comet_improvement_positive_pct': (df['comet_improvement'] > 0).mean() * 100
            },
            'bucket_stats': {},
            'detailed_results': analyzed_results
        }
        
        # Bucket-wise analysis
        for bucket in ['very_short', 'short', 'medium', 'long', 'very_long']:
            bucket_data = df[df['length_bucket'] == bucket]
            if len(bucket_data) > 0:
                stats['bucket_stats'][bucket] = {
                    'count': len(bucket_data),
                    'original_cos_mean': bucket_data['original_cos_similarity'].mean(),
                    'enhanced_cos_mean': bucket_data['enhanced_cos_similarity'].mean(),
                    'cos_improvement_mean': bucket_data['cos_improvement'].mean(),
                    'original_comet_mean': bucket_data['original_comet_score'].mean(),
                    'enhanced_comet_mean': bucket_data['enhanced_comet_score'].mean(),
                    'comet_improvement_mean': bucket_data['comet_improvement'].mean(),
                    'improvement_positive_pct': (bucket_data['cos_improvement'] > 0).mean() * 100
                }
        
        # Print results
        self.print_analysis_results(stats)
        
        # Save results
        output_file = self.results_dir / "comparative_analysis_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {output_file}")
        return stats
    
    def print_analysis_results(self, stats: Dict[str, Any]):
        """Print formatted analysis results."""
        
        print("\n" + "=" * 70)
        print("ANALYSIS RESULTS")
        print("=" * 70)
        
        overall = stats['overall_stats']
        
        print(f"Total samples analyzed: {stats['total_samples']}")
        print("\nOVERALL COMPARISON:")
        print(f"COS Similarity:  {overall['original_cos_mean']:.3f} → {overall['enhanced_cos_mean']:.3f} "
              f"(+{overall['cos_improvement_mean']:.3f})")
        print(f"COMET Score:     {overall['original_comet_mean']:.3f} → {overall['enhanced_comet_mean']:.3f} "
              f"(+{overall['comet_improvement_mean']:.3f})")
        
        print(f"\nIMPROVEMENT RATES:")
        print(f"COS Similarity improved:  {overall['cos_improvement_positive_pct']:.1f}% of samples")
        print(f"COMET Score improved:     {overall['comet_improvement_positive_pct']:.1f}% of samples")
        
        print(f"\nBUCKET-WISE ANALYSIS:")
        print("-" * 85)
        print("Bucket       Count  Original→Enhanced COS    Original→Enhanced COMET   Improve%")
        print("-" * 85)
        
        for bucket, bucket_stats in stats['bucket_stats'].items():
            if bucket_stats['count'] > 0:
                print(f"{bucket:12} {bucket_stats['count']:5}  "
                      f"{bucket_stats['original_cos_mean']:.3f}→{bucket_stats['enhanced_cos_mean']:.3f}      "
                      f"{bucket_stats['original_comet_mean']:.3f}→{bucket_stats['enhanced_comet_mean']:.3f}        "
                      f"{bucket_stats['improvement_positive_pct']:6.1f}%")

def main():
    """Run the comparative analysis."""
    
    analyzer = SourceEnhancementAnalyzer()
    
    try:
        # Look for experiment results file
        results_files = list(analyzer.results_dir.glob("source_enhancement_results_*.json"))
        
        if not results_files:
            print("No experiment results found. Please run source_enhancement_experiment.py first.")
            return
        
        # Use the most recent results file
        latest_file = max(results_files, key=lambda x: x.stat().st_mtime)
        print(f"Using results file: {latest_file.name}")
        
        # Run analysis
        stats = analyzer.run_comparative_analysis(latest_file.name)
        
        print("\n" + "=" * 70)
        print("HYPOTHESIS VALIDATION")
        print("=" * 70)
        
        cos_improvement = stats['overall_stats']['cos_improvement_mean']
        comet_improvement = stats['overall_stats']['comet_improvement_mean']
        
        if cos_improvement > 0 and comet_improvement > 0:
            print("✅ HYPOTHESIS SUPPORTED: Source enhancement improves MT quality")
        elif cos_improvement > 0 or comet_improvement > 0:
            print("⚠️  HYPOTHESIS PARTIALLY SUPPORTED: Mixed improvement results")
        else:
            print("❌ HYPOTHESIS NOT SUPPORTED: No significant improvement detected")
        
        print(f"   COS Similarity change: {cos_improvement:+.3f}")
        print(f"   COMET Score change: {comet_improvement:+.3f}")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
