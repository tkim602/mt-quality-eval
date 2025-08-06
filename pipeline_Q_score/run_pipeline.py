
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
    print(f"SEED: {cfg.SEED}")

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
        
        logger.info(f"Starting {script_name}")
        start_time = time.time()
        
        try:
            subprocess.run(
                [self.python_exe, str(script_path)],
                text=True,
                timeout=timeout,
                cwd=self.base_dir,
                check=True,
                stdout=None,  
                stderr=None   
            )
            
            duration = time.time() - start_time
            logger.info(f"------------------------ {script_name} completed in {duration:.1f}s ------------------------")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"{script_name} timed out after {timeout}s")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"{script_name} failed: {e.returncode}")
            return False
        except Exception as e:
            logger.error(f"Error on {script_name}: {e}")
            return False
    
    async def run_async_script(self, script_name: str, timeout: Optional[int] = None) -> bool:
        script_path = self.base_dir / script_name
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False

        logger.info(f"------------------------ Starting {script_name} ------------------------")
        start_time = time.time()
        
        try:
            process = await asyncio.create_subprocess_exec(
                self.python_exe, str(script_path),
                cwd=self.base_dir,
                stdout=None,  
                stderr=None 
            )
            
            returncode = await asyncio.wait_for(
                process.wait(), 
                timeout=timeout
            )
            
            duration = time.time() - start_time
            
            if returncode == 0:
                logger.info(f"--------------- {script_name} completed in {duration:.1f}s ---------------")
                return True
            else:
                logger.error(f"{script_name} failed: {returncode}")
                return False
                
        except asyncio.TimeoutError:
            logger.error(f"{script_name} timeout after {timeout}s")
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
            logger.info(f"Starting {run_dir}")

            logger.info("------------------------ filter.py ------------------------")
            if not self.run_script("filter.py", timeout=3600):
                logger.error("Filter stage failed")
                return False

            logger.info("------------------------ gemba_batch.py ------------------------")
            if not await self.run_async_script("gemba_batch.py", timeout=7200):
                logger.error("GEMBA stage failed")
                return False

            logger.info("------------------------ ape.py ------------------------")
            if not await self.run_async_script("ape.py", timeout=7200):
                logger.error("APE stage failed")
                return False
            
            logger.info("------------------------ DONE ---------------------------")
            logger.info(f"saved to {run_dir}")
            logger.info(f"OUTPUT: {run_dir / cfg.APE_OUTPUT_FILENAME}")
            
            return True
            
        except Exception as e:
            logger.error(f"--------------------- pipeline failed ---------------------: {e}")
            return False
    
    async def run_single_stage(self, stage: str, use_existing_dir: str = None) -> bool:
        if use_existing_dir:
            existing_dir = Path(use_existing_dir)
            if existing_dir.exists():
                self.run_dir = existing_dir
                os.environ['RUN_DIR'] = str(existing_dir)
                logger.info(f"Using existing directory: {existing_dir}")
            else:
                logger.error(f"Existing directory not found: {existing_dir}")
                return False
        else:
            if not self.run_dir:
                self.run_dir = self.create_run_directory()
        
        if stage == 'filter':
            return self.run_script("filter.py", timeout=600)
        elif stage == 'gemba':
            return await self.run_async_script("gemba_batch.py", timeout=3600)
        elif stage == 'ape':
            return await self.run_async_script("ape.py", timeout=3600)
        else:
            logger.error(f"Unknown stage: {stage}. Available: ['filter', 'gemba', 'ape']")
            return False

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple MT Pipeline Runner')
    parser.add_argument('--stage', choices=['filter', 'gemba', 'ape'], 
                       help='Run only a specific stage')
    parser.add_argument('--base-dir', type=Path, default=Path.cwd(),
                       help='Base directory for pipeline')
    parser.add_argument('--use-existing-dir', type=str,
                       help='Use existing output directory (absolute path)')
    
    args = parser.parse_args()
    
    pipeline = SimplePipelineRunner(base_dir=args.base_dir)
    
    if args.stage:
        logger.info(f"Running single stage: {args.stage}")
        success = await pipeline.run_single_stage(args.stage, args.use_existing_dir)
    else:
        success = await pipeline.run_complete_pipeline()
    
    if not success:
        logger.error("------------------------ FAILED ------------------------")
        sys.exit(1)
    
    logger.info("--------------------------- SUCCESS ------------------------")

if __name__ == "__main__":
    asyncio.run(main())
