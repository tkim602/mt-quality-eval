import subprocess
import sys
import json
import time
import os
from pathlib import Path
from threshold_optimizer import ThresholdOptimizer
import cfg

class IterativeOptimizer:
    def __init__(self, max_iterations: int = 5, min_improvement: float = 0.5):
        self.max_iterations = max_iterations
        self.min_improvement = min_improvement  
        self.iteration_results = []
        
    def run_pipeline(self) -> str:
        print("Running pipeline...")
        try:
            env = os.environ.copy()
            env['FORCE_FRESH'] = 'true'
            
            result = subprocess.run([
                sys.executable, "run_pipeline.py"
            ], capture_output=True, text=True, timeout=3600, env=env)
            
            if result.returncode != 0:
                print(f"Pipeline failed with return code {result.returncode}")
                print(f"STDERR: {result.stderr}")
                return None
                
            print("Pipeline completed successfully")
            return "success"
            
        except subprocess.TimeoutExpired:
            print("Pipeline timed out after 1 hour")
            return None
        except Exception as e:
            print(f"Error running pipeline: {e}")
            return None
    
    def run_optimization_cycle(self) -> dict:
        print(f"\n{'='*60}")
        print(f"STARTING OPTIMIZATION CYCLE")
        print(f"{'='*60}")
        
        iteration_data = {
            'iterations': [],
            'final_thresholds': None,
            'total_improvement': 0,
            'converged': False
        }
        
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n--- ITERATION {iteration}/{self.max_iterations} ---")
            
            # Run the pipeline
            pipeline_result = self.run_pipeline()
            if pipeline_result is None:
                print(f"Pipeline failed in iteration {iteration}")
                break
            
            # Optimize thresholds
            try:
                optimizer = ThresholdOptimizer()
                results = optimizer.optimize_thresholds()
                
                improvement = results['results']['overall_improvement']
                
                iteration_info = {
                    'iteration': iteration,
                    'improvement': improvement,
                    'avg_f1_before': results['results']['avg_current_f1'],
                    'avg_f1_after': results['results']['avg_optimized_f1'],
                    'thresholds': results['optimized_thresholds'],
                    'bucket_results': results['results']['bucket_results']
                }
                
                iteration_data['iterations'].append(iteration_info)
                
                print(f"Iteration {iteration} Results:")
                print(f"  Overall F1 improvement: {improvement:+.2f}%")
                print(f"  Average F1: {results['results']['avg_current_f1']:.2f}% → {results['results']['avg_optimized_f1']:.2f}%")
                
                # Check convergence
                if improvement < self.min_improvement:
                    print(f"Improvement ({improvement:.2f}%) below threshold ({self.min_improvement}%). Stopping.")
                    iteration_data['converged'] = True
                    break
                
                # Update config with new thresholds
                optimizer.update_config_file(results['optimized_thresholds'])
                iteration_data['final_thresholds'] = results['optimized_thresholds']
                
                # Save iteration report
                iteration_report_path = f"iteration_{iteration}_report.json"
                optimizer.save_optimization_report(results, iteration_report_path)
                
                print(f"Updated thresholds for next iteration")
                time.sleep(2)  # Brief pause between iterations
                
            except Exception as e:
                print(f"Optimization failed in iteration {iteration}: {e}")
                break
        
        # Calculate total improvement
        if iteration_data['iterations']:
            total_improvement = sum(iter_data['improvement'] for iter_data in iteration_data['iterations'])
            iteration_data['total_improvement'] = total_improvement
        
        return iteration_data
    
    def print_final_summary(self, iteration_data: dict):
        print(f"\n{'='*60}")
        print(f"FINAL OPTIMIZATION SUMMARY")
        print(f"{'='*60}")
        
        iterations = iteration_data['iterations']
        if not iterations:
            print("No successful iterations completed.")
            return
        
        print(f"Completed iterations: {len(iterations)}")
        print(f"Total F1 improvement: {iteration_data['total_improvement']:+.2f}%")
        print(f"Converged: {'Yes' if iteration_data['converged'] else 'No'}")
        
        print(f"\nIteration Summary:")
        print("-" * 50)
        print(f"{'Iter':<5} {'F1 Before':<10} {'F1 After':<10} {'Improvement':<12}")
        print("-" * 50)
        
        for iter_data in iterations:
            print(f"{iter_data['iteration']:<5} "
                  f"{iter_data['avg_f1_before']:<10.2f} "
                  f"{iter_data['avg_f1_after']:<10.2f} "
                  f"{iter_data['improvement']:<+12.2f}")
        
        if iteration_data['final_thresholds']:
            print(f"\nFinal Optimized Thresholds:")
            print("-" * 40)
            final_cos = iteration_data['final_thresholds']['cos']
            final_comet = iteration_data['final_thresholds']['comet']
            
            print(f"{'Bucket':<12} {'COS':<8} {'COMET':<8}")
            print("-" * 30)
            for bucket in ['very_short', 'short', 'medium', 'long', 'very_long']:
                print(f"{bucket:<12} {final_cos[bucket]:<8.3f} {final_comet[bucket]:<8.3f}")
    
    def save_final_report(self, iteration_data: dict):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = f"iterative_optimization_report_{timestamp}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(iteration_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nComplete optimization report saved to: {report_path}")

def main():
    print("Starting Iterative Threshold Optimization")
    print("This will run the pipeline multiple times to find optimal thresholds.")
    
    max_iterations = 5
    min_improvement = 0.5  # Stop if improvement is less than 0.5%
    
    print(f"Configuration:")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Min improvement threshold: {min_improvement}%")
    print(f"  Dataset limit: {cfg.LIMIT}")
    
    print(f"\nProceed with optimization? (y/n): ", end="")
    response = input().strip().lower()
    
    if response not in ['y', 'yes']:
        print("Optimization cancelled.")
        return
    
    optimizer = IterativeOptimizer(max_iterations, min_improvement)
    
    start_time = time.time()
    iteration_data = optimizer.run_optimization_cycle()
    end_time = time.time()
    
    total_time = end_time - start_time
    print(f"\nTotal optimization time: {total_time/60:.1f} minutes")
    
    optimizer.print_final_summary(iteration_data)
    optimizer.save_final_report(iteration_data)
    
    print(f"\nOptimization complete! Your thresholds have been automatically tuned.")
    print(f"Run 'python run_pipeline.py' to test the final optimized thresholds.")

if __name__ == "__main__":
    main()
