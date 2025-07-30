# run_pipeline_multistyle.py - Multi-Style MT Evaluation Pipeline
# Orchestrates the complete multi-style evaluation pipeline

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import cfg

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiStylePipelineRunner:
    """Multi-style MT evaluation pipeline orchestrator"""
    
    def __init__(self, run_dir: Optional[Path] = None):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = run_dir or cfg.OUT_DIR / f"multistyle_run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variable for child processes
        os.environ['RUN_DIR'] = str(self.run_dir)
        
        logger.info(f"Multi-style pipeline run directory: {self.run_dir}")
    
    async def run_stage(self, script_name: str, description: str) -> bool:
        """Run a pipeline stage"""
        logger.info(f"Starting {description}...")
        
        script_path = Path(__file__).parent / script_name
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False
        
        try:
            # Run the script using the same Python interpreter
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(script_path),
                cwd=script_path.parent,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            
            output, _ = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"✓ {description} completed successfully")
                if output:
                    logger.debug(f"Output: {output.decode()}")
                return True
            else:
                logger.error(f"✗ {description} failed with return code {process.returncode}")
                if output:
                    logger.error(f"Error output: {output.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"✗ {description} failed with exception: {e}")
            return False
    
    def get_stage_files(self) -> Dict[str, Path]:
        """Get expected output files for each stage"""
        return {
            "filter": self.run_dir / cfg.FILTER_OUTPUT_FILENAME,
            "gemba": self.run_dir / cfg.GEMBA_OUTPUT_FILENAME.replace('.json', '_multistyle.json'),
            "ape": self.run_dir / cfg.APE_OUTPUT_FILENAME.replace('.json', '_multistyle.json'),
        }
    
    def check_stage_output(self, stage: str) -> bool:
        """Check if stage output file exists"""
        expected_files = self.get_stage_files()
        if stage not in expected_files:
            return False
        
        output_file = expected_files[stage]
        exists = output_file.exists()
        
        if exists:
            logger.info(f"✓ Found {stage} output: {output_file}")
        else:
            logger.warning(f"✗ Missing {stage} output: {output_file}")
        
        return exists
    
    def print_style_statistics(self):
        """Print statistics about detected text styles"""
        try:
            gemba_file = self.get_stage_files()["gemba"]
            if not gemba_file.exists():
                logger.warning("GEMBA output file not found, cannot show style statistics")
                return
            
            with open(gemba_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            style_counts = {}
            style_scores = {}
            
            for item in data:
                style = item.get("detected_style", "unknown")
                style_counts[style] = style_counts.get(style, 0) + 1
                
                if style not in style_scores:
                    style_scores[style] = []
                
                if "gemba" in item:
                    style_scores[style].append(item["gemba"])
            
            logger.info("\n" + "="*50)
            logger.info("TEXT STYLE STATISTICS")
            logger.info("="*50)
            
            total_items = len(data)
            for style in sorted(style_counts.keys()):
                count = style_counts[style]
                percentage = (count / total_items) * 100
                
                if style_scores[style]:
                    avg_score = sum(style_scores[style]) / len(style_scores[style])
                    logger.info(f"{style.upper():>12}: {count:>4} items ({percentage:>5.1f}%) - Avg GEMBA: {avg_score:>5.1f}")
                else:
                    logger.info(f"{style.upper():>12}: {count:>4} items ({percentage:>5.1f}%)")
            
            logger.info("="*50)
            
        except Exception as e:
            logger.error(f"Error calculating style statistics: {e}")
    
    def print_comparison_summary(self):
        """Print summary comparing original and APE results"""
        try:
            ape_file = self.get_stage_files()["ape"]
            if not ape_file.exists():
                logger.warning("APE output file not found, cannot show comparison summary")
                return
            
            with open(ape_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            improvements = {'cos': 0, 'comet': 0, 'gemba': 0}
            degradations = {'cos': 0, 'comet': 0, 'gemba': 0}
            total_deltas = {'cos': [], 'comet': [], 'gemba': []}
            
            for item in data:
                if "ape" in item:  # Only count items that were actually APE'd
                    for metric in ['cos', 'comet', 'gemba']:
                        delta_key = f"delta_{metric}"
                        if delta_key in item:
                            delta = item[delta_key]
                            total_deltas[metric].append(delta)
                            
                            if delta > 0:
                                improvements[metric] += 1
                            elif delta < 0:
                                degradations[metric] += 1
            
            logger.info("\n" + "="*60)
            logger.info("MULTI-STYLE APE IMPROVEMENT SUMMARY")
            logger.info("="*60)
            
            for metric in ['cos', 'comet', 'gemba']:
                if total_deltas[metric]:
                    avg_delta = sum(total_deltas[metric]) / len(total_deltas[metric])
                    total_ape = len(total_deltas[metric])
                    improved = improvements[metric]
                    degraded = degradations[metric]
                    unchanged = total_ape - improved - degraded
                    
                    logger.info(f"{metric.upper():>6}: {improved:>4} improved, {degraded:>4} degraded, {unchanged:>4} unchanged")
                    logger.info(f"         Average delta: {avg_delta:>+7.3f}")
                    logger.info("")
            
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Error calculating comparison summary: {e}")
    
    async def run_complete_pipeline(self) -> bool:
        """Run the complete multi-style evaluation pipeline"""
        
        logger.info("="*60)
        logger.info("STARTING MULTI-STYLE MT EVALUATION PIPELINE")
        logger.info("="*60)
        
        stages = [
            ("filter.py", "Filtering stage"),
            ("gemba_batch_multistyle.py", "Multi-style GEMBA evaluation"),
            ("ape_multistyle.py", "Multi-style APE post-editing"),
        ]
        
        success = True
        
        for script, description in stages:
            stage_success = await self.run_stage(script, description)
            if not stage_success:
                logger.error(f"Pipeline failed at stage: {description}")
                success = False
                break
        
        if success:
            logger.info("\n" + "="*60)
            logger.info("✓ MULTI-STYLE PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            
            # Print statistics and summaries
            self.print_style_statistics()
            self.print_comparison_summary()
            
            # Print final output locations
            logger.info("\nFINAL OUTPUT FILES:")
            stage_files = self.get_stage_files()
            for stage, filepath in stage_files.items():
                if filepath.exists():
                    logger.info(f"  {stage.upper():>6}: {filepath}")
                else:
                    logger.warning(f"  {stage.upper():>6}: {filepath} (MISSING)")
        
        else:
            logger.error("\n" + "="*60)
            logger.error("✗ MULTI-STYLE PIPELINE FAILED")
            logger.error("="*60)
        
        return success

async def main():
    """Main entry point for multi-style pipeline"""
    
    # Check if run directory is specified
    run_dir = None
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline = MultiStylePipelineRunner(run_dir)
    success = await pipeline.run_complete_pipeline()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
