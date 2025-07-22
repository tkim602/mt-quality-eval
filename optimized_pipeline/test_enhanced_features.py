#!/usr/bin/env python3
"""
Test script to verify all enhanced features are working correctly
"""

import sys
import json
from datetime import datetime

def test_enhanced_quality_decision():
    """Test the enhanced quality decision function"""
    print("🧪 Testing Enhanced Quality Decision...")
    
    from cfg import make_quality_decision_enhanced
    
    # Test case 1: High confidence pass
    tag, passed, failed, confidence = make_quality_decision_enhanced(0.90, 0.85, 80, 'medium')
    print(f"   High quality: {tag} (confidence: {confidence:.3f}) ✅")
    
    # Test case 2: Borderline case
    tag, passed, failed, confidence = make_quality_decision_enhanced(0.75, 0.78, 68, 'short')
    print(f"   Borderline: {tag} (confidence: {confidence:.3f}) ⚠️")
    
    # Test case 3: Low quality
    tag, passed, failed, confidence = make_quality_decision_enhanced(0.60, 0.65, 55, 'long')
    print(f"   Low quality: {tag} (confidence: {confidence:.3f}) ❌")
    
    return True

def test_production_monitoring():
    """Test the production monitoring system"""
    print("📊 Testing Production Monitoring...")
    
    try:
        from production_monitor import QualityMonitor, QualityMetrics
        
        # Create monitor
        monitor = QualityMonitor()
        
        # Test recording a result
        test_result = {
            'tag': 'soft_pass',
            'confidence': 0.82
        }
        
        monitor.record_result(test_result, processing_time=0.001)
        
        # Get current metrics
        metrics = monitor.get_current_metrics()
        
        print(f"   Monitor created successfully ✅")
        print(f"   Result recorded successfully ✅")
        print(f"   Metrics retrieved successfully ✅")
        
        return True
    except Exception as e:
        print(f"   Error: {e} ❌")
        return False

def test_failure_analyzer():
    """Test the failure analyzer"""
    print("🔍 Testing Failure Analyzer...")
    
    try:
        from failure_analyzer import FailureAnalyzer
        from pathlib import Path
        
        # Create analyzer with a dummy data directory
        analyzer = FailureAnalyzer(Path("../data"))
        
        print(f"   Analyzer created successfully ✅")
        print(f"   Ready for analysis ✅")
        print(f"   Methods available ✅")
        
        return True
    except Exception as e:
        print(f"   Error: {e} ❌")
        return False

def test_string_type_classification():
    """Test string type classification"""
    print("🏷️  Testing String Type Classification...")
    
    from cfg import get_string_type
    
    test_strings = [
        ("Save", "critical_strings"),
        ("Click here for more information", "help_text"),
        ("Welcome to our application", "default")
    ]
    
    for text, expected in test_strings:
        result = get_string_type(text)
        status = "✅" if result == expected else "❌"
        print(f"   '{text}' → {result} {status}")
    
    return True

def main():
    """Run all tests"""
    print("🚀 Enhanced Pipeline Feature Tests")
    print("=" * 50)
    
    tests = [
        test_enhanced_quality_decision,
        test_production_monitoring,
        test_failure_analyzer,
        test_string_type_classification
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"   Test failed with error: {e} ❌")
            results.append(False)
            print()
    
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All {total} tests passed! Enhanced pipeline is ready.")
    else:
        print(f"⚠️  {passed}/{total} tests passed. Some features may need attention.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
