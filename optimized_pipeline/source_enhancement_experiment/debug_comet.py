#!/usr/bin/env python3
"""Debug COMET calculation issues."""

import json
from pathlib import Path
from comet import download_model, load_from_checkpoint

def test_comet():
    """Test COMET calculation with a single sample."""
    
    # Load one sample
    results_file = Path("results/source_enhancement_results_200.json")
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    sample = results[0]
    print("Sample data:")
    print(f"Korean original: {sample['korean_original']}")
    print(f"English reference: {sample['english_reference']}")
    print(f"English from original: {sample['english_from_original']}")
    print(f"English from enhanced: {sample['english_from_enhanced']}")
    print()
    
    # Load COMET model
    print("Loading COMET model...")
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    comet_model = load_from_checkpoint(model_path)
    print("COMET model loaded")
    
    # Test original MT
    print("\nTesting original MT:")
    comet_input = [{
        'src': sample['korean_original'],
        'mt': sample['english_from_original'], 
        'ref': sample['english_reference']
    }]
    
    try:
        scores = comet_model.predict(comet_input, batch_size=1)
        print(f"Raw scores: {scores}")
        print(f"Type: {type(scores)}")
        if isinstance(scores, list) and len(scores) > 0:
            print(f"First score: {scores[0]}")
            print(f"First score type: {type(scores[0])}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test enhanced MT
    print("\nTesting enhanced MT:")
    comet_input = [{
        'src': sample['korean_enhanced'],
        'mt': sample['english_from_enhanced'], 
        'ref': sample['english_reference']
    }]
    
    try:
        scores = comet_model.predict(comet_input, batch_size=1)
        print(f"Raw scores: {scores}")
        print(f"Type: {type(scores)}")
        if isinstance(scores, list) and len(scores) > 0:
            print(f"First score: {scores[0]}")
            print(f"First score type: {type(scores[0])}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_comet()
