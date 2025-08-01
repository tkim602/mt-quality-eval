
import asyncio
import logging
import os
import random
import subprocess
import sys
import time
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Optional

import cfg

if hasattr(cfg, 'SEED') and cfg.SEED is not None:
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    os.environ['PYTHONHASHSEED'] = str(cfg.SEED)
    print(f"🎲 Global random seeds set to {cfg.SEED} for pipeline reproducibility")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimplePipelineRunner:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.python_exe = sys.executable
        self.run_dir = None
    
    def create_run_directory(self) -> Path:
        out_dir = self.base_dir / "out"
        out_dir.mkdir(exist_ok=True)
        
        version_num = 1
        while (out_dir / f"v{version_num}").exists():
            version_num += 1
        
        version_dir = out_dir / f"v{version_num}"
        version_dir.mkdir()
        
        logger.info(f"Created version directory: {version_dir}")
        self.run_dir = version_dir
        
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
            # 실시간 출력을 위해 stdout/stderr을 상속
            subprocess.run(
                [self.python_exe, str(script_path)],
                text=True,
                timeout=timeout,
                cwd=self.base_dir,
                check=True,
                stdout=None,  # 실시간 출력
                stderr=None   # 실시간 출력
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
            # 실시간 출력을 위해 stdout/stderr을 상속
            process = await asyncio.create_subprocess_exec(
                self.python_exe, str(script_path),
                cwd=self.base_dir,
                stdout=None,  # 실시간 출력
                stderr=None   # 실시간 출력
            )
            
            # 프로세스 완료 대기
            returncode = await asyncio.wait_for(
                process.wait(), 
                timeout=timeout
            )
            
            duration = time.time() - start_time
            
            if returncode == 0:
                logger.info(f"--- {script_name} completed in {duration:.1f}s ---")
                return True
            else:
                logger.error(f"{script_name} failed with return code {returncode}")
                return False
                
        except asyncio.TimeoutError:
            logger.error(f"{script_name} timed out after {timeout}s")
            if 'process' in locals():
                process.terminate()
                await process.wait()
            return False
        except Exception as e:
            logger.error(f"Error running {script_name}: {e}")
            return False
    
    async def run_complete_pipeline(self) -> bool:
        try:
            run_dir = self.create_run_directory()
            logger.info(f"Starting complete pipeline in: {run_dir}")
            
            logger.info("Stage 1: Running filter...")
            if not self.run_script("filter.py", timeout=600):
                logger.error("Filter stage failed")
                return False
            
            logger.info("Stage 2: Running GEMBA evaluation...")
            if not await self.run_async_script("gemba_batch.py", timeout=3600):
                logger.error("GEMBA stage failed")
                return False
            
            logger.info("Stage 3: Running APE processing...")
            if not await self.run_async_script("ape.py", timeout=3600):
                logger.error("APE stage failed")
                return False
            
            logger.info("Pipeline completed successfully!")
            logger.info(f"Results saved in: {run_dir}")
            logger.info(f"Final output: {run_dir / cfg.APE_OUTPUT_FILENAME}")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False
    
    def run_single_stage(self, stage: str) -> bool:
        stage_map = {
            'filter': lambda: self.run_script("filter.py", timeout=600),
            'gemba': lambda: asyncio.run(self.run_async_script("gemba_batch.py", timeout=3600)),
            'ape': lambda: asyncio.run(self.run_async_script("ape.py", timeout=3600))
        }
        
        if stage not in stage_map:
            logger.error(f"Unknown stage: {stage}. Available: {list(stage_map.keys())}")
            return False
        
        logger.info(f"Running single stage: {stage}")
        return stage_map[stage]()

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple MT Pipeline Runner')
    parser.add_argument('--stage', choices=['filter', 'gemba', 'ape'], 
                       help='Run only a specific stage')
    parser.add_argument('--base-dir', type=Path, default=Path.cwd(),
                       help='Base directory for pipeline')
    
    args = parser.parse_args()
    
    pipeline = SimplePipelineRunner(base_dir=args.base_dir)
    
    if args.stage:
        success = pipeline.run_single_stage(args.stage)
    else:
        success = await pipeline.run_complete_pipeline()
    
    if not success:
        logger.error("Pipeline execution failed")
        sys.exit(1)
    
    logger.info("Pipeline execution completed successfully")

if __name__ == "__main__":
    asyncio.run(main())
