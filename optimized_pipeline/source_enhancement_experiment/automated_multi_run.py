#!/usr/bin/env python3
"""
Automated multi-run experiment runner.
"""

import json
import asyncio
import subprocess
import sys
from pathlib import Path

async def run_experiment_for_sample(run_number: int):
    """Run experiment for a specific sample file."""
    print(f"\n🚀 Starting Run {run_number}/3")
    print(f"{'='*50}")
    
    # Modify the source_enhancement_experiment.py to use the specific sample
    sample_file = f"results/multi_run/sampled_data_run_{run_number}.json"
    
    # Create a temporary experiment script for this run
    temp_script = f"temp_experiment_run_{run_number}.py"
    
    # Read the original experiment script
    with open("source_enhancement_experiment.py", 'r', encoding='utf-8') as f:
        original_script = f.read()
    
    # Modify the script to use our specific data file
    modified_script = original_script.replace(
        'data_file = Path("../data/data_pairs/korean_english_pairs.json")',
        f'data_file = Path("{sample_file}")'
    ).replace(
        'output_file = results_dir / "source_enhancement_results_200.json"',
        f'output_file = results_dir / "source_enhancement_run_{run_number}_results.json"'
    )
    
    # Write temporary script
    with open(temp_script, 'w', encoding='utf-8') as f:
        f.write(modified_script)
    
    try:
        # Run the experiment
        print(f"Running source enhancement for run {run_number}...")
        result = subprocess.run([sys.executable, temp_script], 
                              capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        if result.returncode != 0:
            print(f"❌ Error in experiment run {run_number}:")
            print(result.stderr)
            return False
        
        print(f"✅ Source enhancement completed for run {run_number}")
        
        # Now run comparative analysis
        print(f"Running comparative analysis for run {run_number}...")
        
        # Modify comparative analyzer to use the right results file
        analysis_script = f"temp_analyzer_run_{run_number}.py"
        
        with open("comparative_analyzer.py", 'r', encoding='utf-8') as f:
            original_analyzer = f.read()
        
        modified_analyzer = original_analyzer.replace(
            'results_file = results_file or "source_enhancement_results_200.json"',
            f'results_file = "source_enhancement_run_{run_number}_results.json"'
        ).replace(
            'analysis_file = results_dir / "comparative_analysis_results.json"',
            f'analysis_file = results_dir / "analysis_run_{run_number}.json"'
        )
        
        with open(analysis_script, 'w', encoding='utf-8') as f:
            f.write(modified_analyzer)
        
        result = subprocess.run([sys.executable, analysis_script], 
                              capture_output=True, text=True, timeout=1800)
        
        if result.returncode != 0:
            print(f"❌ Error in analysis run {run_number}:")
            print(result.stderr)
            return False
        
        print(f"✅ Comparative analysis completed for run {run_number}")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout in run {run_number}")
        return False
        
    finally:
        # Clean up temporary files
        for temp_file in [temp_script, analysis_script]:
            if Path(temp_file).exists():
                Path(temp_file).unlink()

async def run_all_experiments():
    """Run all 3 experiments sequentially."""
    print("🚀 Starting Multi-Run Source Enhancement Experiment")
    print("=" * 60)
    
    results = []
    for run_num in range(1, 4):
        success = await run_experiment_for_sample(run_num)
        results.append(success)
    
    # Check results
    successful_runs = sum(results)
    print(f"\n📊 Experiment Summary:")
    print(f"Successful runs: {successful_runs}/3")
    
    if successful_runs >= 2:
        print("✅ Sufficient runs completed. Running final analysis...")
        
        # Run final multi-run analysis
        result = subprocess.run([sys.executable, "multi_run_analysis.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Multi-run analysis completed!")
            print(result.stdout)
        else:
            print("❌ Error in final analysis:")
            print(result.stderr)
    else:
        print("❌ Insufficient successful runs for analysis")

if __name__ == "__main__":
    asyncio.run(run_all_experiments())
