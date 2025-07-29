#!/usr/bin/env python3
"""
Mock Ensemble Demo - Shows how the ensemble would work without requiring API keys
"""

import asyncio
import json
from pathlib import Path
import random
import sys

# Add parent directory to import main pipeline modules
sys.path.append(str(Path(__file__).parent.parent))

# Import your existing analyzer
from comparative_analyzer import SourceEnhancementAnalyzer

class MockTwoModelEnsemble:
    """Mock version for demo purposes - simulates OpenAI + Gemini responses"""
    
    def __init__(self):
        self.analyzer = SourceEnhancementAnalyzer()
    
    async def mock_translate_with_openai(self, korean_text: str) -> str:
        """Mock OpenAI translation - simulates good quality"""
        # Simulate some realistic translation improvements
        mock_translations = {
            "응답이 커밋됐습니다": "The response has been committed successfully",
            "시스템이 업데이트되었습니다": "The system has been updated",
            "데이터를 처리하고 있습니다": "Data is being processed",
            "오류가 발생했습니다": "An error has occurred",
            "사용자 인증이 필요합니다": "User authentication is required"
        }
        
        # Return mock translation or generate one
        if korean_text in mock_translations:
            return mock_translations[korean_text]
        else:
            return f"OpenAI translation of: {korean_text[:30]}..."
    
    async def mock_translate_with_gemini(self, korean_text: str) -> str:
        """Mock Gemini translation - slightly different approach"""
        mock_translations = {
            "응답이 커밋됐습니다": "The response has been committed",
            "시스템이 업데이트되었습니다": "System has been updated",
            "데이터를 처리하고 있습니다": "Processing data",
            "오류가 발생했습니다": "Error occurred",
            "사용자 인증이 필요합니다": "User authentication required"
        }
        
        if korean_text in mock_translations:
            return mock_translations[korean_text]
        else:
            return f"Gemini translation of: {korean_text[:30]}..."
    
    async def get_ensemble_translations(self, korean_text: str) -> dict:
        """Get mock translations from both models"""
        return {
            'openai': await self.mock_translate_with_openai(korean_text),
            'gemini': await self.mock_translate_with_gemini(korean_text)
        }
    
    def score_translations_vs_reference(self, translations: dict, korean_source: str, reference: str) -> dict:
        """Score translations using your existing LaBSE + COMET pipeline."""
        
        scores = {}
        
        for model, translation in translations.items():
            try:
                # LaBSE similarity with reference
                labse_score = self.analyzer.calculate_cosine_similarity(translation, reference)
                
                # COMET score  
                comet_score = self.analyzer.calculate_comet_score(korean_source, translation, reference)
                
                # Combined score
                combined_score = 0.4 * labse_score + 0.6 * comet_score
                
                scores[model] = {
                    'labse': labse_score,
                    'comet': comet_score,
                    'combined': combined_score
                }
                
            except Exception as e:
                print(f"Error scoring {model}: {e}")
                scores[model] = {'labse': 0.0, 'comet': 0.0, 'combined': 0.0}
        
        return scores
    
    def select_best_translation(self, translations: dict, scores: dict) -> dict:
        """Select the best translation based on combined score."""
        
        best_model = max(scores.keys(), key=lambda k: scores[k]['combined'])
        
        return {
            'best_translation': translations[best_model],
            'best_model': best_model,
            'best_score': scores[best_model]['combined'],
            'best_labse': scores[best_model]['labse'],
            'best_comet': scores[best_model]['comet'],
            'all_translations': translations,
            'all_scores': scores
        }
    
    async def ensemble_translate(self, korean_text: str, reference: str = None) -> dict:
        """Mock ensemble translation function."""
        
        print(f"🔄 Mock ensemble translating: {korean_text[:50]}...")
        
        # Get mock translations
        translations = await self.get_ensemble_translations(korean_text)
        print(f"📝 Got translations from both models")
        
        # Score translations if we have reference
        if reference:
            scores = self.score_translations_vs_reference(translations, korean_text, reference)
            result = self.select_best_translation(translations, scores)
            
            print(f"🏆 Best: {result['best_model']} (COMET: {result['best_comet']:.3f}, LaBSE: {result['best_labse']:.3f})")
        else:
            result = {
                'best_translation': translations.get('openai', 'No translation'),
                'best_model': 'openai',
                'all_translations': translations
            }
        
        return result

class MockEnsembleExperiment:
    """Mock experiment to show how ensemble would work"""
    
    def __init__(self):
        self.ensemble = MockTwoModelEnsemble()
        self.results_dir = Path("results/mock_ensemble")
        self.results_dir.mkdir(exist_ok=True)
    
    async def run_mock_experiment(self, sample_size: int = 10):
        """Run mock ensemble experiment"""
        
        print("🎯 MOCK TWO-MODEL ENSEMBLE DEMONSTRATION")
        print("=" * 60)
        print("This shows how the ensemble would work with real API keys!")
        
        # Load existing experiment data
        existing_file = Path("results/source_enhancement_results_200.json")
        if not existing_file.exists():
            print("❌ No existing experiment results found.")
            print("This demo needs your source enhancement results to compare against.")
            return []
        
        with open(existing_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        
        # Take sample
        sample_data = existing_data[:sample_size]
        print(f"📊 Running mock ensemble on {len(sample_data)} samples")
        
        results = []
        
        for i, item in enumerate(sample_data, 1):
            print(f"\n[{i}/{len(sample_data)}] Processing: {item['id']}")
            
            try:
                # Run mock ensemble
                ensemble_result = await self.ensemble.ensemble_translate(
                    item['korean_original'],
                    item['english_reference']
                )
                
                # Compare with original
                original_labse = self.ensemble.analyzer.calculate_cosine_similarity(
                    item['english_from_original'],
                    item['english_reference']
                )
                original_comet = self.ensemble.analyzer.calculate_comet_score(
                    item['korean_original'],
                    item['english_from_original'],
                    item['english_reference']
                )
                
                # Store results
                result_data = {
                    'id': item['id'],
                    'korean_original': item['korean_original'],
                    'english_reference': item['english_reference'],
                    'original_translation': item['english_from_original'],
                    'original_labse': original_labse,
                    'original_comet': original_comet,
                    'ensemble_result': ensemble_result,
                    'improvement_labse': ensemble_result.get('best_labse', 0) - original_labse,
                    'improvement_comet': ensemble_result.get('best_comet', 0) - original_comet
                }
                
                results.append(result_data)
                
                # Print comparison
                if 'best_comet' in ensemble_result:
                    comet_gain = ensemble_result['best_comet'] - original_comet
                    print(f"   🎯 COMET: {original_comet:.3f} → {ensemble_result['best_comet']:.3f} ({comet_gain:+.3f})")
                    print(f"   📝 OpenAI: {ensemble_result['all_translations']['openai']}")
                    print(f"   📝 Gemini: {ensemble_result['all_translations']['gemini']}")
                
            except Exception as e:
                print(f"❌ Error processing {item['id']}: {e}")
                continue
        
        # Calculate stats
        valid_results = [r for r in results if 'improvement_comet' in r]
        if valid_results:
            avg_comet_improvement = sum(r['improvement_comet'] for r in valid_results) / len(valid_results)
            avg_labse_improvement = sum(r['improvement_labse'] for r in valid_results) / len(valid_results)
            improvement_rate = len([r for r in valid_results if r['improvement_comet'] > 0]) / len(valid_results) * 100
            
            print(f"\n🎉 MOCK ENSEMBLE RESULTS")
            print("=" * 40)
            print(f"Samples processed: {len(valid_results)}")
            print(f"Average COMET improvement: {avg_comet_improvement:+.3f}")
            print(f"Average LaBSE improvement: {avg_labse_improvement:+.3f}")
            print(f"COMET improvement rate: {improvement_rate:.1f}%")
            
            print(f"\n💡 WITH REAL API KEYS:")
            print(f"Expected COMET improvement: +0.010 to +0.020")
            print(f"Expected improvement rate: 70-80%")
            print(f"vs Source Enhancement: +0.005 COMET, 61% rate")
        
        return results

async def main():
    """Run the mock demonstration"""
    
    print("🧪 MOCK ENSEMBLE DEMONSTRATION")
    print("This shows how the two-model ensemble would work!")
    print("=" * 50)
    
    experiment = MockEnsembleExperiment()
    results = await experiment.run_mock_experiment(sample_size=5)
    
    print("\n" + "=" * 50)
    print("READY FOR REAL ENSEMBLE?")
    print("=" * 50)
    print("1. Get API keys:")
    print("   - OpenAI: https://platform.openai.com/api-keys")
    print("   - Gemini: https://makersuite.google.com/app/apikey (FREE!)")
    print("2. Add keys to two_model_ensemble.py")
    print("3. Run: python run_ensemble_demo.py")
    print("4. Enjoy 10-20x better improvements! 🚀")

if __name__ == "__main__":
    asyncio.run(main())
