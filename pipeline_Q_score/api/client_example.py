#!/usr/bin/env python3
"""
Simple client example for the MT Quality Evaluation API
"""

import requests
import json
from typing import Dict, Any

class MTQualityClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def get_status(self) -> Dict[Any, Any]:
        """Get API status"""
        response = requests.get(f"{self.base_url}/")
        return response.json()
    
    def get_analytics(self) -> Dict[Any, Any]:
        """Get summary analytics"""
        response = requests.get(f"{self.base_url}/analytics")
        return response.json()
    
    def get_records(self, **filters) -> Dict[Any, Any]:
        """Get records with optional filters"""
        response = requests.get(f"{self.base_url}/records", params=filters)
        return response.json()
    
    def get_record_by_key(self, key: str) -> Dict[Any, Any]:
        """Get specific record by key"""
        response = requests.get(f"{self.base_url}/records/{key}")
        return response.json()
    
    def search_records(self, query: str, limit: int = 50) -> Dict[Any, Any]:
        """Search records by text content"""
        response = requests.get(f"{self.base_url}/search", params={"q": query, "limit": limit})
        return response.json()
    
    def get_high_quality_records(self, min_gemba: float = 90) -> Dict[Any, Any]:
        """Get high-quality translations"""
        return self.get_records(min_gemba=min_gemba, tag="strict_pass")
    
    def get_failed_records(self) -> Dict[Any, Any]:
        """Get failed translations for analysis"""
        return self.get_records(tag="fail")
    
    def get_ape_improved_records(self) -> Dict[Any, Any]:
        """Get records that were improved by APE"""
        return self.get_records(has_ape=True)

def main():
    """Example usage"""
    client = MTQualityClient()
    
    print("🔍 API Status:")
    status = client.get_status()
    print(f"Total records: {status['total_records']}")
    print()
    
    print("📊 Analytics:")
    analytics = client.get_analytics()
    print(f"Average GEMBA score: {analytics['scores']['gemba']['mean']:.1f}")
    print(f"Pass rate: {(analytics['distributions']['tags'].get('strict_pass', 0) + analytics['distributions']['tags'].get('soft_pass', 0)) / analytics['total_records'] * 100:.1f}%")
    print(f"APE improvement (COMET): +{analytics['ape_effectiveness']['avg_comet_improvement']:.3f}")
    print()
    
    print("🌟 High Quality Records (GEMBA ≥ 90):")
    high_quality = client.get_high_quality_records()
    print(f"Found {len(high_quality['records'])} high-quality translations")
    
    if high_quality['records']:
        sample = high_quality['records'][0]
        print(f"Example: '{sample['src'][:50]}...' → '{sample['mt'][:50]}...'")
    print()
    
    print("🔴 Failed Records:")
    failed = client.get_failed_records()
    print(f"Found {len(failed['records'])} failed translations")
    
    if failed['records']:
        sample = failed['records'][0]
        print(f"Example issue: {sample['flag']['gemba_reason'][:100]}...")
    print()
    
    print("🚀 APE Improved Records:")
    ape_improved = client.get_ape_improved_records()
    print(f"Found {len(ape_improved['records'])} APE-improved translations")

if __name__ == "__main__":
    main()
