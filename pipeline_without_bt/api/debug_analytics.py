#!/usr/bin/env python3
"""
Debug script to check analytics data structure
"""

import requests
import json

def main():
    base_url = "http://localhost:8000"
    
    print("=== Analytics Data Structure ===")
    response = requests.get(f"{base_url}/analytics")
    data = response.json()
    
    print("Full analytics data:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    
    print("\n=== Bucket Distribution ===")
    if 'distributions' in data and 'buckets' in data['distributions']:
        buckets = data['distributions']['buckets']
        print("Buckets found:")
        for bucket, count in buckets.items():
            print(f"  {bucket}: {count}")
    else:
        print("No bucket distribution found")
    
    print("\n=== APE Effectiveness ===")
    if 'ape_effectiveness' in data:
        ape_data = data['ape_effectiveness']
        print("APE data found:")
        for key, value in ape_data.items():
            print(f"  {key}: {value}")
    else:
        print("No APE effectiveness data found")
    
    print("\n=== Sample Records with APE ===")
    response = requests.get(f"{base_url}/records", params={"has_ape": True, "limit": 3})
    records_data = response.json()
    
    if 'records' in records_data and records_data['records']:
        for i, record in enumerate(records_data['records'][:3]):
            print(f"\nRecord {i+1}:")
            print(f"  Key: {record.get('key', 'N/A')}")
            print(f"  Has APE: {'ape' in record}")
            print(f"  Delta COMET: {record.get('delta_comet', 'N/A')}")
            print(f"  Delta COS: {record.get('delta_cos', 'N/A')}")
            print(f"  Bucket: {record.get('bucket', 'N/A')}")
    else:
        print("No APE records found")

if __name__ == "__main__":
    main()
