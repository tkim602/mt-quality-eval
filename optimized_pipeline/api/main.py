#!/usr/bin/env python3
"""
MT Quality Evaluation API
Serves ape_evidence.json data through REST endpoints
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
import json
import statistics
from pathlib import Path
from enum import Enum

# Load data
DATA_FILE = Path("../out/v13/ape_evidence.json")

class BucketType(str, Enum):
    very_short = "very_short"
    short = "short" 
    medium = "medium"
    long = "long"
    very_long = "very_long"

class TagType(str, Enum):
    strict_pass = "strict_pass"
    soft_pass = "soft_pass"
    fail = "fail"

# Global data storage
evaluation_data: List[Dict[Any, Any]] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global evaluation_data
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            evaluation_data = json.load(f)
        print(f"✅ Loaded {len(evaluation_data)} evaluation records")
    except FileNotFoundError:
        print(f"❌ Data file not found: {DATA_FILE}")
        evaluation_data = []
    
    yield

app = FastAPI(
    title="MT Quality Evaluation API",
    description="API for accessing machine translation quality evaluation results",
    version="1.0.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """API status and basic info"""
    return {
        "message": "MT Quality Evaluation API",
        "total_records": len(evaluation_data),
        "version": "1.0.0",
        "endpoints": [
            "/records - Get all records with filtering",
            "/records/{key} - Get specific record by key", 
            "/analytics - Get summary analytics",
            "/buckets/{bucket} - Get records by text length bucket",
            "/tags/{tag} - Get records by quality tag"
        ]
    }

@app.get("/records")
async def get_records(
    bucket: Optional[BucketType] = None,
    tag: Optional[TagType] = None,
    min_gemba: Optional[float] = Query(None, ge=0, le=100),
    max_gemba: Optional[float] = Query(None, ge=0, le=100),
    min_comet: Optional[float] = Query(None, ge=0, le=1),
    has_ape: Optional[bool] = None,
    limit: Optional[int] = Query(100, ge=1, le=1000),
    offset: Optional[int] = Query(0, ge=0)
):
    """Get evaluation records with optional filtering"""
    if not evaluation_data:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    filtered_data = evaluation_data
    
    if bucket:
        filtered_data = [r for r in filtered_data if r.get("bucket") == bucket]
    
    if tag:
        filtered_data = [r for r in filtered_data if r.get("tag") == tag]
    
    if min_gemba is not None:
        filtered_data = [r for r in filtered_data if r.get("gemba", 0) >= min_gemba]
    
    if max_gemba is not None:
        filtered_data = [r for r in filtered_data if r.get("gemba", 100) <= max_gemba]
    
    if min_comet is not None:
        filtered_data = [r for r in filtered_data if r.get("comet", 0) >= min_comet]
    
    if has_ape is not None:
        if has_ape:
            filtered_data = [r for r in filtered_data if "ape" in r]
        else:
            filtered_data = [r for r in filtered_data if "ape" not in r]
    
    total = len(filtered_data)
    paginated_data = filtered_data[offset:offset + limit]
    
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "records": paginated_data
    }

@app.get("/records/{key}")
async def get_record_by_key(key: str):
    """Get a specific record by its key"""
    if not evaluation_data:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    record = next((r for r in evaluation_data if r.get("key") == key), None)
    
    if not record:
        raise HTTPException(status_code=404, detail=f"Record with key '{key}' not found")
    
    return record

@app.get("/analytics")
async def get_analytics():
    """Get summary analytics of all evaluation data"""
    if not evaluation_data:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Basic statistics
    gemba_scores = [r.get("gemba", 0) for r in evaluation_data if "gemba" in r]
    comet_scores = [r.get("comet", 0) for r in evaluation_data if "comet" in r]
    cos_scores = [r.get("cos", 0) for r in evaluation_data if "cos" in r]
    
    # Tag distribution
    tag_dist = {}
    for record in evaluation_data:
        tag = record.get("tag", "unknown")
        tag_dist[tag] = tag_dist.get(tag, 0) + 1
    
    # Bucket distribution
    bucket_dist = {}
    for record in evaluation_data:
        bucket = record.get("bucket", "unknown")
        bucket_dist[bucket] = bucket_dist.get(bucket, 0) + 1
    
    # APE effectiveness
    ape_records = [r for r in evaluation_data if "ape" in r]
    ape_improvements = {
        "delta_comet": [r.get("delta_comet", 0) for r in ape_records if "delta_comet" in r],
        "delta_cos": [r.get("delta_cos", 0) for r in ape_records if "delta_cos" in r]
    }
    
    return {
        "total_records": len(evaluation_data),
        "scores": {
            "gemba": {
                "mean": statistics.mean(gemba_scores) if gemba_scores else 0,
                "median": statistics.median(gemba_scores) if gemba_scores else 0,
                "stdev": statistics.stdev(gemba_scores) if len(gemba_scores) > 1 else 0,
                "min": min(gemba_scores) if gemba_scores else 0,
                "max": max(gemba_scores) if gemba_scores else 0
            },
            "comet": {
                "mean": statistics.mean(comet_scores) if comet_scores else 0,
                "median": statistics.median(comet_scores) if comet_scores else 0,
                "stdev": statistics.stdev(comet_scores) if len(comet_scores) > 1 else 0
            },
            "cosine": {
                "mean": statistics.mean(cos_scores) if cos_scores else 0,
                "median": statistics.median(cos_scores) if cos_scores else 0,
                "stdev": statistics.stdev(cos_scores) if len(cos_scores) > 1 else 0
            }
        },
        "distributions": {
            "tags": tag_dist,
            "buckets": bucket_dist
        },
        "ape_effectiveness": {
            "total_ape_records": len(ape_records),
            "avg_comet_improvement": statistics.mean(ape_improvements["delta_comet"]) if ape_improvements["delta_comet"] else 0,
            "avg_cosine_improvement": statistics.mean(ape_improvements["delta_cos"]) if ape_improvements["delta_cos"] else 0
        }
    }

@app.get("/buckets/{bucket}")
async def get_records_by_bucket(bucket: BucketType):
    """Get all records for a specific text length bucket"""
    if not evaluation_data:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    bucket_records = [r for r in evaluation_data if r.get("bucket") == bucket]
    
    if not bucket_records:
        raise HTTPException(status_code=404, detail=f"No records found for bucket '{bucket}'")
    
    # Compute bucket statistics
    gemba_scores = [r.get("gemba", 0) for r in bucket_records]
    
    return {
        "bucket": bucket,
        "total_records": len(bucket_records),
        "avg_gemba": statistics.mean(gemba_scores) if gemba_scores else 0,
        "pass_rate": len([r for r in bucket_records if r.get("tag") in ["strict_pass", "soft_pass"]]) / len(bucket_records) * 100,
        "records": bucket_records
    }

@app.get("/tags/{tag}")
async def get_records_by_tag(tag: TagType):
    """Get all records for a specific quality tag"""
    if not evaluation_data:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    tag_records = [r for r in evaluation_data if r.get("tag") == tag]
    
    if not tag_records:
        raise HTTPException(status_code=404, detail=f"No records found for tag '{tag}'")
    
    return {
        "tag": tag,
        "total_records": len(tag_records),
        "records": tag_records
    }

@app.get("/search")
async def search_records(
    q: str = Query(..., min_length=2, description="Search query for source or translation text"),
    limit: int = Query(50, ge=1, le=200)
):
    """Search records by text content in source or translation"""
    if not evaluation_data:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    query = q.lower()
    matches = []
    
    for record in evaluation_data:
        src = record.get("src", "").lower()
        mt = record.get("mt", "").lower()
        ape = record.get("ape", "").lower()
        
        if query in src or query in mt or query in ape:
            matches.append(record)
        
        if len(matches) >= limit:
            break
    
    return {
        "query": q,
        "total_matches": len(matches),
        "records": matches
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
