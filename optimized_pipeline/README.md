# Enhanced MT Evaluation Pipeline with Automatic Threshold Optimization

This folder contains the **enhanced** machine translation evaluation pipeline with advanced quality controls, monitoring, analysis capabilities, and **automatic threshold optimization**.

## 🎯 **NEW: Automatic Threshold Optimization**

The pipeline now includes sophisticated threshold optimization tools that can automatically find optimal COS and COMET thresholds for maximum F1 performance.

### **Threshold Optimization Tools:**

1. **`threshold_analysis.py`** - Quick performance analysis
   ```bash
   python threshold_analysis.py
   ```
   - Analyzes current threshold performance
   - Shows F1 scores by bucket
   - Compares with reference optimal values

2. **`threshold_optimizer.py`** - Single optimization run
   ```bash
   python threshold_optimizer.py
   ```
   - Grid search optimization on latest pipeline output
   - Finds optimal thresholds for each length bucket
   - Option to update cfg.py automatically

3. **`iterative_optimizer.py`** - Automated iterative optimization
   ```bash
   python iterative_optimizer.py
   ```
   - Runs pipeline multiple times with threshold refinement
   - Automatically converges to optimal thresholds
   - Complete automation of optimization process

4. **`apply_optimal_thresholds.py`** - Apply reference optimal values
   ```bash
   python apply_optimal_thresholds.py
   ```
   - Applies reference optimal thresholds from 100-dataset analysis
   - Achieves F1 scores: 87.39% - 98.39%

### **Reference Optimal Thresholds:**

| Length Bucket | Optimal COS | Optimal COMET | F1 Score |
|---------------|-------------|---------------|----------|
| very_short    | 0.840       | 0.850         | 98.39%   |
| short         | 0.850       | 0.830         | 95.65%   |
| medium        | 0.820       | 0.840         | 87.39%   |
| long          | 0.820       | 0.830         | 87.80%   |
| very_long     | 0.830       | 0.830         | 93.33%   |

### **Optimization Workflow:**
1. Run pipeline: `python run_pipeline.py`
2. Optimize thresholds: `python threshold_optimizer.py`
3. Or use full automation: `python iterative_optimizer.py`

## 🚀 Key Enhancements

### **Enhanced Quality Decision Logic**
- **Confidence Scoring**: Each decision includes a confidence score (0-1) based on metric agreement
- **Business Rules**: Different thresholds for different string types (critical strings, help text, default)
- **Optimized Thresholds**: Data-driven threshold adjustments, especially for 'short' bucket (reduced from 71% failure rate)
- **GEMBA Threshold**: Lowered from 70 to 65 for better balance

### **Production Monitoring System**
- **Real-time Metrics**: Pass rate, confidence, processing time, error rate, throughput
- **Alert System**: Configurable thresholds with automatic triggering/clearing
- **Performance Tracking**: Sliding window analysis
- **Dashboard Ready**: Integration-ready monitoring data

### **Failure Pattern Analysis**
- **Terminology Analysis**: Detects missing critical terms from termbase
- **Length Issue Detection**: Identifies problematic length ratios by bucket
- **Semantic Analysis**: Finds cosine-COMET disagreement cases
- **Actionable Recommendations**: Specific improvement suggestions

### **Enhanced Pipeline Integration**
- **Async Processing**: Efficient batch processing with monitoring
- **Comprehensive Analysis**: Automatic failure analysis and reporting
- **Business Impact Analysis**: Tracks impact of business rules

## Files Included

### Core Pipeline Scripts
- `run_pipeline.py` - Main orchestrator with enhanced analysis
- `filter.py` - Enhanced filtering with confidence scoring and business rules
- `gemba_batch.py` - GEMBA quality assessment step  
- `ape+back_translation.py` - APE step with back-translation

### Enhanced Analysis & Monitoring
- `enhanced_pipeline.py` - Advanced pipeline runner with monitoring
- `production_monitor.py` - Real-time monitoring and alerting system
- `failure_analyzer.py` - Comprehensive failure pattern analysis

### Configuration & Support
- `cfg.py` - Enhanced configuration with business rules and optimized thresholds
- `validation.py` - Complete validation functions with quality checks
- `requirements.txt` - All necessary Python dependencies
- `test_setup.py` - Setup verification script

## 📊 Performance Improvements

Based on our v3 analysis:
- **Confidence-Aware Decisions**: 65% of items now have confidence scores
- **Business Rules Impact**: Different performance for different string types
- **Optimized Thresholds**: Reduced failure rates in problematic buckets
- **Real-time Monitoring**: Immediate alerts on quality degradation

### Recommended Threshold Adjustments Applied
```python
COS_THR = {
    "short": 0.800,       # Reduced from 0.850 (was 71% failure rate)
    "very_short": 0.820,  # Reduced from 0.840
    # Other buckets optimized based on analysis
}

COMET_THR = {
    "short": 0.780,       # Reduced from 0.830 (was 71% failure rate)  
    "very_short": 0.830,  # Reduced from 0.850
    # Other buckets optimized based on analysis
}

GEMBA_PASS = 65  # Lowered from 70 for better balance
```

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

### 3. Verify Setup
```bash
python test_setup.py
```

### 4. Run the Complete Enhanced Pipeline
```bash
python run_pipeline.py
```

### 5. Run Enhanced Analysis on Existing Data
```bash
python enhanced_pipeline.py --input out/v3/gemba.json --output-dir enhanced_analysis
```

### 6. Run Failure Analysis
```bash
python failure_analyzer.py --data-dir out/v3 --output failure_analysis.json
```

### 7. Run Individual Steps (Optional)
```bash
# Filter step only
python run_pipeline.py --skip-gemba --skip-ape

# GEMBA step only  
python run_pipeline.py --skip-filter --skip-ape

# APE step only
python run_pipeline.py --skip-filter --skip-gemba
```

## Output

### Standard Pipeline Output
The pipeline creates versioned output directories in `out/v1/`, `out/v2/`, etc.:
- `filtered.json` - Enhanced filtering results with confidence scores
- `gemba.json` - GEMBA quality assessment with business rules applied
- `ape_evidence.json` - Final results with APE improvements

### Enhanced Analysis Output
Additional analysis files in each run directory:
- `enhanced_quality_analysis.json` - Comprehensive quality analysis
- `monitoring_report.json` - Production monitoring report
- `failure_analysis.json` - Detailed failure pattern analysis (when failures exist)

## Configuration

### Enhanced Settings in `cfg.py`
- **Business Rules**: Different thresholds for critical_strings, help_text, and default
- **Confidence Calculation**: Metric agreement and bucket-specific weights
- **Optimized Thresholds**: Data-driven threshold adjustments
- **Monitoring Config**: Alert thresholds and performance tracking

### Business Rules
```python
BUSINESS_RULES = {
    "critical_strings": {  # UI labels, error messages
        "cos_boost": 0.05,
        "comet_boost": 0.05,
        "required_confidence": 0.90
    },
    "help_text": {  # Help text, descriptions
        "cos_penalty": -0.03,
        "comet_penalty": -0.03,
        "required_confidence": 0.70
    },
    "default": {
        "required_confidence": 0.75
    }
}
```

## Data Requirements

The pipeline expects these data files (paths configured in cfg.py):
- Korean source: `ko_checker_dedup.json`
- English target: `en-US_checker.json` 
- Term base: `term_base_simple.json`

## Monitoring & Alerts

The enhanced pipeline includes production-ready monitoring:
- **Pass Rate Alerts**: Triggers when pass rate drops below 25%
- **Confidence Alerts**: Triggers when average confidence drops below 65%
- **Performance Alerts**: Monitors processing time and throughput
- **Dashboard Integration**: Ready for external monitoring systems

## Next Steps

1. **Deploy Enhanced Monitoring**: Set up dashboard integration
2. **Apply Threshold Adjustments**: Use failure analysis recommendations
3. **Implement Confidence-Based APE**: Skip APE for high-confidence items
4. **Scale Testing**: Run on larger datasets to validate improvements

This enhanced pipeline provides significantly better quality control, monitoring, and insights while maintaining the core functionality and performance of the original system.
