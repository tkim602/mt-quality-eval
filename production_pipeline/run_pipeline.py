#!/usr/bin/env python3
"""
MT Pipeline Orchestrator
Runs the complete pipeline: filter -> gemba -> ape+bt
"""

import asyncio
import logging
import time
import os
import random
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from typing import Optional
import json
import cfg


# Set global random seeds for reproducibility
if hasattr(cfg, 'SEED') and cfg.SEED is not None:
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    # For additional reproducibility in some environments
    os.environ['PYTHONHASHSEED'] = str(cfg.SEED)
    print(f"🎲 Global random seeds set to {cfg.SEED} for pipeline reproducibility")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineRunner:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.python_exe = sys.executable
        self.run_dir = None
        
    def create_run_directory(self) -> Path:
        """Create a unique run directory with incrementing version number"""
        out_dir = self.base_dir / "out"
        out_dir.mkdir(exist_ok=True)
        
        # Find the next version number
        version_num = 1
        while (out_dir / f"v{version_num}").exists():
            version_num += 1
        
        version_dir = out_dir / f"v{version_num}"
        version_dir.mkdir()
        
        logger.info(f"Created version directory: {version_dir}")
        self.run_dir = version_dir
        
        # Set environment variable for scripts to use
        os.environ['RUN_DIR'] = str(version_dir)
        
        return version_dir
        
    def run_script(self, script_name: str, timeout: Optional[int] = None) -> bool:
        script_path = self.base_dir / script_name
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False
        
        logger.info(f"--- Starting {script_name} ---")
        start_time = time.time()
        
        try:
            subprocess.run(
                [self.python_exe, str(script_path)],
                text=True,
                timeout=timeout,
                cwd=self.base_dir,
                check=True
            )
            
            duration = time.time() - start_time
            logger.info(f"--- {script_name} completed in {duration:.1f}s ---")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"{script_name} timed out after {timeout}s")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"{script_name} failed with return code {e.returncode}")
            return False
        except Exception as e:
            logger.error(f"Error running {script_name}: {e}")
            return False
    
    async def run_async_script(self, script_name: str, timeout: Optional[int] = None) -> bool:
        script_path = self.base_dir / script_name
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False
        
        logger.info(f"--- Starting {script_name} (async) ---")
        start_time = time.time()
        
        try:
            process = await asyncio.create_subprocess_exec(
                self.python_exe, str(script_path),
                stdout=None,
                stderr=None,
                cwd=self.base_dir
            )
            
            await asyncio.wait_for(process.wait(), timeout=timeout)
            
            duration = time.time() - start_time
            
            if process.returncode != 0:
                logger.error(f"{script_name} failed with return code {process.returncode}")
                return False
            
            logger.info(f"--- {script_name} completed in {duration:.1f}s ---")
            return True
            
        except asyncio.TimeoutError:
            logger.error(f"{script_name} timed out after {timeout}s")
            return False
        except Exception as e:
            logger.error(f"Error running {script_name}: {e}")
            return False
    
    def check_prerequisites(self) -> bool:
        required_scripts = [
            "filter.py",
            "gemba_batch.py", 
            "ape+back_translation.py",
            "cfg.py"
        ]
        
        missing = []
        for script in required_scripts:
            if not (self.base_dir / script).exists():
                missing.append(script)
        
        if missing:
            logger.error(f"Missing required scripts: {missing}")
            return False
        
        return True
    
    def validate_outputs(self) -> dict:
        if not self.run_dir:
            return {"error": "No run directory set"}
            
        results = {}
        
        # Import cfg to get filenames
        import cfg
        
        # Check filter output
        filter_file = self.run_dir / cfg.FILTER_OUTPUT_FILENAME
        if filter_file.exists():
            try:
                with open(filter_file, 'r', encoding='utf-8') as f:
                    filter_data = json.load(f)
                results["filter"] = {
                    "success": True,
                    "records": len(filter_data),
                    "file_size": filter_file.stat().st_size
                }
            except Exception as e:
                results["filter"] = {"success": False, "error": str(e)}
        else:
            results["filter"] = {"success": False, "error": "Output file not found"}
        
        # Check GEMBA output
        gemba_file = self.run_dir / cfg.GEMBA_OUTPUT_FILENAME
        if gemba_file.exists():
            try:
                with open(gemba_file, 'r', encoding='utf-8') as f:
                    gemba_data = json.load(f)
                results["gemba"] = {
                    "success": True,
                    "records": len(gemba_data),
                    "file_size": gemba_file.stat().st_size
                }
            except Exception as e:
                results["gemba"] = {"success": False, "error": str(e)}
        else:
            results["gemba"] = {"success": False, "error": "Output file not found"}
        
        # Check APE output
        ape_file = self.run_dir / cfg.APE_OUTPUT_FILENAME
        if ape_file.exists():
            try:
                with open(ape_file, 'r', encoding='utf-8') as f:
                    ape_data = json.load(f)
                results["ape"] = {
                    "success": True,
                    "records": len(ape_data),
                    "file_size": ape_file.stat().st_size
                }
            except Exception as e:
                results["ape"] = {"success": False, "error": str(e)}
        else:
            results["ape"] = {"success": False, "error": "Output file not found"}
        
        return results
    
    async def run_full_pipeline(self, 
                              use_optimized: bool = False,
                              skip_filter: bool = False,
                              skip_gemba: bool = False,
                              skip_ape: bool = False) -> bool:
        
        if not self.check_prerequisites():
            return False
        
        # Create unique run directory
        run_dir = self.create_run_directory()
        
        pipeline_start = time.time()
        logger.info("Starting MT evaluation pipeline...")
        logger.info(f"Output will be saved to: {run_dir}")
        
        if not skip_filter:
            if not self.run_script("filter.py", timeout=3600):  
                logger.error("Filter step failed")
                return False
        
        if not skip_gemba:
            if not await self.run_async_script("gemba_batch.py", timeout=7200):  
                logger.error("GEMBA step failed")
                return False
        
        if not skip_ape:
            if not await self.run_async_script("ape+back_translation.py", timeout=7200): 
                logger.error("APE step failed")
                return False
        
        # Run enhanced quality analysis
        logger.info("Running enhanced quality analysis...")
        from enhanced_pipeline import EnhancedPipeline
        import cfg
        
        enhanced_pipeline = EnhancedPipeline(enable_monitoring=True)
        
        # Find the latest output file for analysis
        latest_file = None
        for filename in [cfg.APE_OUTPUT_FILENAME, cfg.GEMBA_OUTPUT_FILENAME, cfg.FILTER_OUTPUT_FILENAME]:
            file_path = run_dir / filename
            if file_path.exists():
                latest_file = file_path
                break
        
        if latest_file:
            analysis_results = enhanced_pipeline.run_quality_analysis(latest_file, run_dir)
            enhanced_pipeline.save_monitoring_report(run_dir / "monitoring_report.json")
            logger.info("Enhanced analysis completed")
        else:
            logger.warning("No output files found for enhanced analysis")
        
        pipeline_duration = time.time() - pipeline_start
        logger.info(f"Pipeline completed successfully in {pipeline_duration:.1f}s")
        
        results = self.validate_outputs()
        logger.info("Pipeline validation results:")
        for stage, result in results.items():
            if result["success"]:
                logger.info(f"  {stage}: ✓ {result['records']} records ({result['file_size']} bytes)")
            else:
                logger.error(f"  {stage}: ✗ {result['error']}")
        
        # Suggest threshold optimization
        self.suggest_threshold_optimization()
        
        return all(r["success"] for r in results.values())
    
    def suggest_threshold_optimization(self):
        """Suggest threshold optimization to the user"""
        logger.info("\n" + "="*50)
        logger.info("THRESHOLD OPTIMIZATION SUGGESTIONS")
        logger.info("="*50)
        logger.info("You can now optimize your thresholds using the new pipeline data:")
        logger.info("1. Quick analysis:     python threshold_analysis.py")
        logger.info("2. Single optimization: python threshold_optimizer.py") 
        logger.info("3. Auto-optimization:  python iterative_optimizer.py")
        logger.info("="*50)

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="MT Pipeline Runner")
    parser.add_argument("--skip-filter", action="store_true", help="Skip filter step")
    parser.add_argument("--skip-gemba", action="store_true", help="Skip GEMBA step")
    parser.add_argument("--skip-ape", action="store_true", help="Skip APE step")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).parent,
                       help="Base directory for pipeline scripts")
    
    args = parser.parse_args()
    
    runner = PipelineRunner(args.base_dir)
    success = await runner.run_full_pipeline(
        skip_filter=args.skip_filter,
        skip_gemba=args.skip_gemba,
        skip_ape=args.skip_ape
    )
    
    if success:
        logger.info("Pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
