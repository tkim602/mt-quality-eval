#!/usr/bin/env python3
"""
Source Enhancement Experiment

Hypothesis: Post-editing Korean source text to better Korean will result in better machine translation output.

This experiment:
1. Takes the same dataset used in the main pipeline
2. Enhances Korean source text using GPT-4
3. Re-translates enhanced Korean to English
4. Compares MT quality metrics before/after source enhancement

Author: MT Quality Evaluation Pipeline
Date: July 23, 2025
"""

import json
import asyncio
import aiohttp
import random
from pathlib import Path
from typing import List, Dict, Any
import sys
import os

# Add parent directory to path to import main pipeline modules
sys.path.append(str(Path(__file__).parent.parent))
import cfg

class SourceEnhancementExperiment:
    def __init__(self, sample_size: int = 200):
        self.sample_size = sample_size
        self.experiment_dir = Path(__file__).parent
        self.output_dir = self.experiment_dir / "results"
        self.output_dir.mkdir(exist_ok=True)
        
    async def enhance_korean_source(self, korean_text: str, context: str = "") -> str:
        """
        Use GPT-4 to enhance Korean source text for better translation.
        
        Args:
            korean_text: Original Korean text
            context: Additional context if available
            
        Returns:
            Enhanced Korean text
        """
        
        enhancement_prompt = f"""당신은 한국어 전문 에디터입니다. 주어진 한국어 텍스트를 기계번역이 더 잘 이해할 수 있도록 개선해주세요.

개선 원칙:
1. 문법적 오류 수정
2. 애매한 표현을 명확하게 수정  
3. 전문 용어 표준화
4. 문장 구조 개선 (길고 복잡한 문장을 명확하게)
5. 대명사 명확화 (this, that → 구체적 명사)
6. 수동태를 능동태로 가능한 경우 변경
7. 의미 변화 없이 번역하기 쉬운 표현으로 개선

원문: {korean_text}

개선된 한국어 텍스트만 출력하세요 (설명 없이):"""

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": "당신은 한국어 텍스트 개선 전문가입니다."},
                            {"role": "user", "content": enhancement_prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 1000
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        enhanced_text = result['choices'][0]['message']['content'].strip()
                        return enhanced_text
                    else:
                        print(f"API Error: {response.status}")
                        return korean_text  # Return original if enhancement fails
                        
        except Exception as e:
            print(f"Enhancement error: {e}")
            return korean_text
    
    async def translate_enhanced_korean(self, enhanced_korean: str) -> str:
        """
        Translate enhanced Korean to English using the same model as the main pipeline.
        """
        
        translation_prompt = f"""Translate the following Korean text to English:

Korean: {enhanced_korean}

Provide only the English translation:"""

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4o-mini",  # Same model as main pipeline
                        "messages": [
                            {"role": "system", "content": "You are a professional Korean-English translator."},
                            {"role": "user", "content": translation_prompt}
                        ],
                        "temperature": 0.1,  # Low temperature for consistent translation
                        "max_tokens": 1000
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        translation = result['choices'][0]['message']['content'].strip()
                        return translation
                    else:
                        print(f"Translation API Error: {response.status}")
                        return ""
                        
        except Exception as e:
            print(f"Translation error: {e}")
            return ""
    
    def load_original_dataset(self) -> List[Dict[str, Any]]:
        """Load the same dataset used in the main pipeline."""
        
        print("Loading original dataset...")
        
        # Load Korean source data (dictionary format)
        with open(cfg.KO_JSON, 'r', encoding='utf-8') as f:
            ko_data = json.load(f)
        
        # Load English reference data (dictionary format)
        with open(cfg.EN_JSON, 'r', encoding='utf-8') as f:
            en_data = json.load(f)
        
        # Convert to list format with matching IDs
        combined_data = []
        for ko_id, ko_content in ko_data.items():
            if ko_id in en_data:
                en_content = en_data[ko_id]
                combined_data.append({
                    'id': ko_id,
                    'korean_original': ko_content,
                    'english_reference': en_content,
                    'category': ko_id.split('.')[0] if '.' in ko_id else 'unknown'
                })
        
        # Random sample
        if len(combined_data) > self.sample_size:
            combined_data = random.sample(combined_data, self.sample_size)
        
        print(f"Loaded {len(combined_data)} records for experiment")
        return combined_data
    
    async def run_enhancement_experiment(self) -> List[Dict[str, Any]]:
        """
        Run the complete source enhancement experiment.
        
        Returns:
            List of experiment results with original and enhanced versions
        """
        
        print("=" * 60)
        print("SOURCE ENHANCEMENT EXPERIMENT")
        print("=" * 60)
        print(f"Sample size: {self.sample_size}")
        print("Hypothesis: Enhanced Korean source → Better MT output")
        
        # Load dataset
        dataset = self.load_original_dataset()
        
        results = []
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        
        async def process_item(item):
            async with semaphore:
                result = {
                    'id': item['id'],
                    'korean_original': item['korean_original'],
                    'english_reference': item['english_reference'],
                    'category': item['category']
                }
                
                print(f"Processing item {item['id']}...")
                
                # Step 1: Enhance Korean source
                result['korean_enhanced'] = await self.enhance_korean_source(
                    item['korean_original']
                )
                
                # Step 2: Translate enhanced Korean  
                result['english_from_enhanced'] = await self.translate_enhanced_korean(
                    result['korean_enhanced']
                )
                
                # Step 3: Translate original Korean for comparison
                result['english_from_original'] = await self.translate_enhanced_korean(
                    item['korean_original']
                )
                
                return result
        
        # Process all items concurrently
        tasks = [process_item(item) for item in dataset]
        results = await asyncio.gather(*tasks)
        
        # Save results
        output_file = self.output_dir / f"source_enhancement_results_{self.sample_size}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nExperiment completed!")
        print(f"Results saved to: {output_file}")
        
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the enhancement experiment results.
        """
        
        print("\n" + "=" * 60)
        print("EXPERIMENT ANALYSIS")
        print("=" * 60)
        
        analysis = {
            'total_samples': len(results),
            'enhancement_examples': [],
            'translation_comparisons': []
        }
        
        # Sample some examples for manual review
        sample_indices = random.sample(range(len(results)), min(5, len(results)))
        
        for i in sample_indices:
            result = results[i]
            
            example = {
                'id': result['id'],
                'korean_original': result['korean_original'],
                'korean_enhanced': result['korean_enhanced'],
                'english_reference': result['english_reference'],
                'english_from_original': result['english_from_original'], 
                'english_from_enhanced': result['english_from_enhanced']
            }
            
            analysis['enhancement_examples'].append(example)
            
            # Print for immediate review
            print(f"\nExample {i+1} (ID: {result['id']}):")
            print(f"Korean Original:  {result['korean_original']}")
            print(f"Korean Enhanced:  {result['korean_enhanced']}")
            print(f"English Reference: {result['english_reference']}")
            print(f"MT from Original:  {result['english_from_original']}")
            print(f"MT from Enhanced:  {result['english_from_enhanced']}")
            print("-" * 60)
        
        # Save analysis
        analysis_file = self.output_dir / f"enhancement_analysis_{self.sample_size}.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        print(f"\nAnalysis saved to: {analysis_file}")
        return analysis

async def main():
    """Run the source enhancement experiment."""
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    experiment = SourceEnhancementExperiment(sample_size=200)
    
    try:
        # Run experiment
        results = await experiment.run_enhancement_experiment()
        
        # Analyze results
        analysis = experiment.analyze_results(results)
        
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("1. Review the enhancement examples above")
        print("2. Run the enhanced data through the main pipeline")
        print("3. Compare LaBSE/COMET scores: original vs enhanced")
        print("4. Statistical analysis of quality improvements")
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
