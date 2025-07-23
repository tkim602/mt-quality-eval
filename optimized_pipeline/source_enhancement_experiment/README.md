# Source Enhancement Experiment

## Hypothesis
**"Post-editing Korean source text to better Korean will result in better machine translation output."**

## Experimental Design

### Methodology
1. **Source Enhancement**: Use GPT-4 to improve Korean source text quality
   - Fix grammatical errors
   - Clarify ambiguous expressions
   - Standardize technical terminology
   - Improve sentence structure
   - Replace pronouns with specific nouns

2. **Translation Comparison**: 
   - Translate original Korean → English
   - Translate enhanced Korean → English
   - Compare both against reference English

3. **Quality Measurement**:
   - LaBSE cosine similarity (semantic similarity)
   - COMET scores (translation quality)
   - Statistical analysis across length buckets

### Files Structure
```
source_enhancement_experiment/
├── README.md                           # This file
├── source_enhancement_experiment.py    # Main experiment script
├── comparative_analyzer.py             # Quality analysis script
└── results/                           # Output directory
    ├── source_enhancement_results_200.json
    ├── enhancement_analysis_200.json
    └── comparative_analysis_results.json
```

## Running the Experiment

### Step 1: Run Source Enhancement
```bash
cd source_enhancement_experiment
python source_enhancement_experiment.py
```

### Step 2: Run Comparative Analysis
```bash
python comparative_analyzer.py
```

## Expected Outcomes

### If Hypothesis is Correct:
- Enhanced translations show higher LaBSE similarity to reference
- Enhanced translations show higher COMET scores
- Improvement is consistent across length buckets
- Statistical significance in quality metrics

### If Hypothesis is Incorrect:
- No significant improvement or degradation
- Quality metrics remain similar
- Possible cases where enhancement confuses MT system

## Enhancement Principles

The experiment uses these Korean text enhancement principles:

1. **Grammar Correction**: Fix subject-verb agreement, particles
2. **Clarity Improvement**: Replace vague terms with specific ones
3. **Structure Optimization**: Break complex sentences, improve flow
4. **Terminology Standardization**: Use consistent technical terms
5. **Pronoun Resolution**: Replace 이것/그것 with concrete nouns
6. **Active Voice**: Convert passive constructions where appropriate

## Analysis Metrics

### Primary Metrics:
- **LaBSE Cosine Similarity**: Semantic similarity (0-1 scale)
- **COMET Score**: Overall translation quality (-2 to +2 scale)

### Secondary Analysis:
- Improvement rate by length bucket
- Statistical significance testing
- Qualitative examples of enhancement impact

## Research Context

This experiment contributes to research on:
- **Source-side preprocessing** in MT pipelines
- **Controlled authoring** for translation quality
- **Human-in-the-loop** MT optimization
- **Korean→English** specific translation challenges

## Usage Notes

1. Requires OPENAI_API_KEY environment variable
2. Uses same models as main pipeline (LaBSE, COMET)
3. Sample size configurable (default: 200)
4. Results saved in JSON format for further analysis
