# Pipeline Without Back-Translation

A clean, simple MT evaluation pipeline focused on core functionality without unnecessary complexity.

## Overview

A streamlined machine translation evaluation pipeline that provides:
- **APE (Automatic Post-Editing)**: Improves MT quality using GPT-4
- **GEMBA Evaluation**: Quality assessment using LLM-based metrics  
- **COMET/Cosine Scoring**: Traditional MT evaluation metrics
- **Delta GEMBA**: APE improvement measurement

## Key Features

### ✨ Simple & Clean
- **No Back-Translation**: Removed unused back-translation logic
- **No Complex Monitoring**: Focused on core evaluation pipeline
- **Single Output**: Clean APE results with delta GEMBA scores

### 🎯 Core Evaluation
- **3-Stage Pipeline**: Filter → GEMBA → APE
- **APE Improvement**: Post-editing with improvement measurement
- **Delta Scores**: Before/after comparison for quality improvement

### 🚀 Performance
- **Async Processing**: Concurrent execution for better throughput
- **Batch Processing**: Efficient handling of large datasets
- **Reproducible**: Seed-based randomization control

## File Structure

```
pipeline_without_bt/
├── run_pipeline.py           # Main pipeline orchestrator
├── ape.py                    # APE module with delta GEMBA
├── gemba_batch.py           # GEMBA evaluation
├── filter.py                # Data filtering
├── cfg.py                   # Configuration
├── validation.py            # Data validation
├── requirements.txt         # Dependencies
├── README.md               # This file
└── api/                    # Web interface
    ├── main.py
    ├── quality_analyzer.py
    └── templates/
```

## Usage

### Quick Start

Run the complete pipeline:
```bash
python run_pipeline.py
```

### Individual Stages

Run specific stages:
```bash
# Filter data only
python run_pipeline.py --stage filter

# GEMBA evaluation only
python run_pipeline.py --stage gemba

# APE processing only
python run_pipeline.py --stage ape
```

## Configuration

Key settings in `cfg.py`:

```python
# Batch sizes
GEMBA_BATCH = 10
APE_BATCH = 5

# Model configurations
GEMBA_MODEL = "gpt-4o-mini"
APE_MODEL = "gpt-4o-mini"
COMET_CKPT = "Unbabel/wmt22-cometkiwi-da"
COS_MODEL = "sentence-transformers/LaBSE"

# File names
FILTER_OUTPUT_FILENAME = "filtered.json"
GEMBA_OUTPUT_FILENAME = "gemba.json"
APE_OUTPUT_FILENAME = "ape_evidence.json"
```

## Pipeline Stages

1. **Filter**: Initial data filtering and validation
2. **GEMBA**: Quality evaluation using GEMBA-MQM metrics
3. **APE**: Automatic post-editing with delta GEMBA calculation

## Output Format

Final results in `ape_evidence.json`:

```json
[
  {
    "key": "unique_id",
    "src": "source text",
    "mt": "machine translation", 
    "ape": "post-edited text",
    "validation": "soft_pass|hard_pass|fail",
    "cos": 0.85,
    "ape_cos": 0.92,
    "delta_cos": 0.07,
    "comet": 0.78,
    "ape_comet": 0.85,
    "delta_comet": 0.07,
    "gemba": 0.82,
    "ape_gemba": 0.91,
    "delta_gemba": 0.09
  }
]
```

## Performance

Typical performance on modern hardware:
- **Filter Stage**: ~100 items/second
- **GEMBA Stage**: ~5-10 items/second (API dependent)
- **APE Stage**: ~3-8 items/second (API dependent)

## Dependencies

Core requirements:
- Python 3.9+
- OpenAI API key
- PyTorch
- sentence-transformers
- COMET

Install with:
```bash
pip install -r requirements.txt
```

## Migration from optimized_pipeline

This pipeline simplifies the previous `optimized_pipeline`:

### Removed
- ❌ Back-translation logic (unused)
- ❌ Complex quality monitoring
- ❌ Failure analyzers
- ❌ Enhanced analytics
- ❌ Alert systems

### Kept
- ✅ Core APE functionality
- ✅ GEMBA evaluation
- ✅ COMET/Cosine scoring
- ✅ Delta score calculation
- ✅ Clean pipeline orchestration

## Troubleshooting

### Common Issues

**"Import comet could not be resolved"**
```bash
pip install unbabel-comet
```

**"OpenAI API rate limit"**
- Check API quota and billing
- Reduce batch sizes in cfg.py

**"CUDA out of memory"**
- Set DEVICE="cpu" in cfg.py
- Use smaller batch sizes

---

**Created**: 2025-01-23  
**Version**: 2.0 (Simplified)  
**Based on**: Enhanced pipeline (complexity removed)
