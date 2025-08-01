# MT Quality Evaluation API

A REST API for accessing machine translation quality evaluation results from the ape_evidence.json data.

## 🚀 Quick Start

### Installation
```bash
cd api
pip install -r requirements.txt
```

### Run the API
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Interactive Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📡 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API status and info |
| `/records` | GET | Get all records with filtering |
| `/records/{key}` | GET | Get specific record by key |
| `/analytics` | GET | Summary analytics |
| `/search` | GET | Search records by text content |

### Filtered Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/buckets/{bucket}` | GET | Records by text length |
| `/tags/{tag}` | GET | Records by quality tag |

## 🔍 Query Parameters

### `/records` endpoint supports:
- `bucket`: Filter by text length (very_short, short, medium, long, very_long)
- `tag`: Filter by quality (strict_pass, soft_pass, fail)
- `min_gemba`: Minimum GEMBA score (0-100)
- `max_gemba`: Maximum GEMBA score (0-100)
- `min_comet`: Minimum COMET score (0-1)
- `has_ape`: Filter records with/without APE improvements
- `limit`: Number of records to return (default: 100)
- `offset`: Pagination offset (default: 0)

## 💡 Example Usage

### Python Client
```python
import requests

# Get high-quality translations
response = requests.get("http://localhost:8000/records", params={
    "min_gemba": 90,
    "tag": "strict_pass",
    "limit": 10
})

data = response.json()
print(f"Found {data['total']} high-quality records")
```

### cURL Examples
```bash
# Get API status
curl http://localhost:8000/

# Get analytics
curl http://localhost:8000/analytics

# Search for specific text
curl "http://localhost:8000/search?q=buffer&limit=5"

# Get failed translations
curl http://localhost:8000/tags/fail

# Get medium-length texts
curl http://localhost:8000/buckets/medium
```

## 📊 Response Examples

### Analytics Response
```json
{
  "total_records": 100,
  "scores": {
    "gemba": {
      "mean": 78.9,
      "median": 80.0,
      "stdev": 11.9
    }
  },
  "distributions": {
    "tags": {
      "strict_pass": 53,
      "soft_pass": 17,
      "fail": 30
    },
    "buckets": {
      "medium": 36,
      "long": 31,
      "very_long": 22
    }
  },
  "ape_effectiveness": {
    "total_ape_records": 47,
    "avg_comet_improvement": 0.264,
    "avg_cosine_improvement": 0.247
  }
}
```

### Record Response
```json
{
  "key": "rule.sca.l.common.6268.name",
  "src": "NAIST-2003 라이선스 컴포넌트 사용",
  "mt": "Use of Components Licensed Under the NAIST-2003",
  "bucket": "medium",
  "gemba": 85.0,
  "gemba_adequacy": 90.0,
  "gemba_fluency": 80.0,
  "comet": 0.825,
  "cos": 0.830,
  "tag": "strict_pass",
  "flag": {
    "passed": ["cosine", "comet", "gemba"],
    "failed": [],
    "gemba_reason": "Good translation, but could be more natural..."
  }
}
```

## 🔧 Features

✅ **Easy to use**: RESTful API with intuitive endpoints  
✅ **Rich filtering**: Multiple filter options for data exploration  
✅ **Analytics**: Built-in statistics and aggregations  
✅ **Search**: Full-text search across translations  
✅ **Pagination**: Handle large datasets efficiently  
✅ **Type safety**: Pydantic models with validation  
✅ **Documentation**: Auto-generated OpenAPI docs  
✅ **Performance**: Fast JSON responses  

## 🎯 Use Cases

- **Quality Dashboard**: Build web dashboards with real-time metrics
- **Analysis Tools**: Create data analysis scripts and notebooks  
- **Integration**: Connect with other translation tools
- **Monitoring**: Track translation quality over time
- **Research**: Academic research on MT quality evaluation
