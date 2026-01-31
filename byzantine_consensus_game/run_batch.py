#!/usr/bin/env python3
"""
Run Individual Batches
Run specific batches one at a time, with full control over what to run.

Usage:
    # List all available batches
    python run_batch.py --list
    
    # Run a specific batch by name
    python run_batch.py --batch Q1_qwen3-8b_4agents_batch_0 --gpu 0
    
    # Run multiple batches
    python run_batch.py --batch Q1_qwen3-8b_4agents_batch_0 Q1_qwen3-8b_4agents_batch_1 --gpu 0
    
    # Run all batches for a specific model
    python run_batch.py --model qwen3-8b --gpu 0
    
    # Run all batches for a specific model and agent count
    python run_batch.py --model qwen3-8b --agents 4 --gpu 0
"""

import os
import sys
import json
import argparse
import time
import signal
import atexit
from datetime import datetime
from typing import Dict, List, Any

# Configuration: 5 runs per batch
RUNS_PER_BATCH = 5

# Flag to track if cleanup has been done
_cleanup_done = False


def _signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    print(f"\n\n⚠️ Received signal {signum}, cleaning up...")
    _final_cleanup()
    sys.exit(1)


def _final_cleanup():
    """Ensure vLLM is cleaned up on exit."""
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True
    
    print("Performing final vLLM cleanup...")
    try:
        from vllm_agent import VLLMAgent
        VLLMAgent.shutdown()
    except Exception:
        pass
    
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    print("✓ Cleanup complete")


# Register cleanup handlers
atexit.register(_final_cleanup)
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def generate_all_batches() -> Dict[str, Dict[str, Any]]:
    """Generate all possible batch configurations for Q1 and Q2."""
    batches = {}
    
    # Quick test batch: 4 honest + 1 byzantine (single run)
    batches["test_4h_1b"] = {
        "experiment": "test",
        "model": "qwen3-8b",
        "num_agents": 5,  # 4 honest + 1 byzantine
        "byzantine_count": 1,
        "batch_id": 0,
        "runs": 1,  # Single run for quick test
    }
    
    # Q1: 4 models × 3 agent counts × 5 batches
    models = ["qwen3-8b", "qwen3-14b", "qwen3-32b", "mistral-22b"]
    agent_counts = [4, 8, 16]
    
    for model in models:
        for agents in agent_counts:
            for batch_id in range(5):
                batch_name = f"Q1_{model}_{agents}agents_batch_{batch_id}"
                batches[batch_name] = {
                    "experiment": "Q1",
                    "model": model,
                    "num_agents": agents,
                    "byzantine_count": 0,
                    "batch_id": batch_id,
                    "runs": RUNS_PER_BATCH,
                }
    
    # Q2: Same as Q1 but with byzantine agents
    # 4 models × 4 byzantine counts × 5 batches = 80 batches per model
    byzantine_counts = [1, 2, 3, 4]  # Number of byzantine agents to add
    
    for model in models:
        for byz in byzantine_counts:
            for batch_id in range(5):  # 5 batches like Q1
                batch_name = f"Q2_{model}_8h_{byz}byz_batch_{batch_id}"
                batches[batch_name] = {
                    "experiment": "Q2",
                    "model": model,
                    "num_agents": 8 + byz,  # 8 honest + N byzantine
                    "byzantine_count": byz,
                    "batch_id": batch_id,
                    "runs": RUNS_PER_BATCH,
                }
    
    # Q3: Prompt engineering - 7 honest + 3 byzantine with different prompt versions
    # 35 runs per prompt version = 7 batches of 5 runs
    prompt_versions = ["minimal", "standard", "detailed"]
    for prompt in prompt_versions:
        for batch_id in range(7):  # 7 batches × 5 runs = 35 runs
            batch_name = f"Q3_{prompt}_batch_{batch_id}"
            batches[batch_name] = {
                "experiment": "Q3",
                "model": "qwen3-8b",
                "num_agents": 10,  # 7 honest + 3 byzantine
                "byzantine_count": 3,
                "prompt_version": prompt,  # "minimal", "standard", "detailed"
                "batch_id": batch_id,
                "runs": RUNS_PER_BATCH,
            }
    
    return batches


def list_batches(batches: Dict[str, Dict], filter_model: str = None, 
                 filter_agents: int = None, filter_exp: str = None):
    """Pretty print available batches."""
    print(f"\n{'='*70}")
    print("Available Batches")
    print(f"{'='*70}\n")
    
    filtered = {}
    for name, config in batches.items():
        if filter_model and config["model"] != filter_model:
            continue
        if filter_agents and config["num_agents"] != filter_agents:
            continue
        if filter_exp and config["experiment"] != filter_exp:
            continue
        filtered[name] = config
    
    # Group by experiment
    by_exp = {}
    for name, config in filtered.items():
        exp = config["experiment"]
        if exp not in by_exp:
            by_exp[exp] = []
        by_exp[exp].append((name, config))
    
    for exp in sorted(by_exp.keys()):
        print(f"[{exp}]")
        for name, config in sorted(by_exp[exp]):
            byz_info = f", {config['byzantine_count']} byz" if config['byzantine_count'] > 0 else ""
            print(f"  {name}")
            print(f"    model={config['model']}, agents={config['num_agents']}{byz_info}")
        print()
    
    total = len(filtered)
    total_runs = sum(c["runs"] for c in filtered.values())
    print(f"Total: {total} batches, {total_runs} runs")
    print()


def get_next_run_number(results_dir: str, batch_name: str, experiment: str = "other") -> int:
    """Get the next available run number for a batch."""
    # Look in experiment subfolder
    experiment_dir = os.path.join(results_dir, experiment)
    os.makedirs(experiment_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(experiment_dir) 
                      if f.startswith(f"{batch_name}_") and f.endswith(".json")]
    if not existing_files:
        return 1
    
    run_numbers = []
    for f in existing_files:
        # Extract run number from filename like "batch_name_001.json"
        try:
            num_str = f.replace(f"{batch_name}_", "").replace(".json", "")
            run_numbers.append(int(num_str))
        except ValueError:
            continue
    
    return max(run_numbers) + 1 if run_numbers else 1


def cleanup_vllm_engine():
    """Force cleanup of vLLM engine and GPU memory."""
    global _cleanup_done
    
    try:
        from vllm_agent import VLLMAgent
        VLLMAgent.shutdown()
    except Exception as e:
        print(f"Warning: vLLM cleanup error: {e}")
    
    # Additional cleanup
    import gc
    gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    
    # Mark cleanup as done
    _cleanup_done = True


def run_batch(batch_name: str, batch_config: Dict[str, Any], gpu: int, 
              results_dir: str = "results/batches") -> Dict[str, Any]:
    """Run a single batch of simulations."""
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    # Import after setting GPU
    from config import MODEL_PRESETS
    from main import run_simulation
    
    # Get experiment for organizing results
    experiment = batch_config.get("experiment", "other")
    
    # Get next run number to avoid overwriting
    run_number = get_next_run_number(results_dir, batch_name, experiment)
    
    # Get prompt version for Q3 experiments
    prompt_version = batch_config.get("prompt_version", "standard")
    
    print(f"\n{'='*70}")
    print(f"Running Batch: {batch_name} (run #{run_number:03d})")
    print(f"  Experiment: {batch_config['experiment']}")
    print(f"  Model: {batch_config['model']}")
    print(f"  Agents: {batch_config['num_agents']}")
    print(f"  Byzantine: {batch_config['byzantine_count']}")
    print(f"  Prompt Version: {prompt_version}")
    print(f"  Runs: {batch_config['runs']}")
    print(f"  GPU: {gpu}")
    print(f"{'='*70}\n")
    
    model_name = MODEL_PRESETS[batch_config["model"]]
    
    # Import VLLM_CONFIG to capture quantization setting
    from config import VLLM_CONFIG
    
    batch_results = {
        "batch_name": batch_name,
        "run_number": run_number,
        "config": batch_config,
        "vllm_config": {
            "quantization": VLLM_CONFIG.get("quantization"),
            "max_model_len": VLLM_CONFIG.get("max_model_len"),
            "gpu_memory_utilization": VLLM_CONFIG.get("gpu_memory_utilization"),
        },
        "start_time": datetime.now().isoformat(),
        "runs": [],
    }
    
    batch_start = time.time()
    
    for run_id in range(batch_config["runs"]):
        print(f"\n--- Run {run_id + 1}/{batch_config['runs']} ---")
        run_start = time.time()
        
        try:
            result = run_simulation(
                n_agents=batch_config["num_agents"],
                max_rounds=15,
                model_name=model_name,
                byzantine_count=batch_config["byzantine_count"],
                prompt_version=prompt_version,
            )
            
            run_time = time.time() - run_start
            
            # Extract key metrics
            run_data = {
                "run_id": run_id,
                "success": True,
                "time_seconds": run_time,
                "metrics": result.get("metrics", {}),
            }
            
            print(f"  ✅ Completed in {run_time:.1f}s")
            
        except Exception as e:
            run_time = time.time() - run_start
            run_data = {
                "run_id": run_id,
                "success": False,
                "time_seconds": run_time,
                "error": str(e),
            }
            print(f"  ❌ Failed: {e}")
            
            # On failure, do emergency cleanup to recover GPU resources
            # This helps recover from engine crashes
            error_str = str(e).lower()
            if "engine" in error_str or "cuda" in error_str or "gpu" in error_str or "memory" in error_str:
                print("  ⚠️ GPU/Engine error detected, performing emergency cleanup...")
                cleanup_vllm_engine()
                import time as t
                t.sleep(2)  # Give GPU time to release resources
        
        batch_results["runs"].append(run_data)
    
    batch_results["end_time"] = datetime.now().isoformat()
    batch_results["total_time_seconds"] = time.time() - batch_start
    
    # Save batch results with run number to avoid overwriting
    # Organize by experiment (Q1, Q2, Q3 subfolders)
    experiment = batch_config.get("experiment", "other")
    experiment_dir = os.path.join(results_dir, experiment)
    os.makedirs(experiment_dir, exist_ok=True)
    output_file = os.path.join(experiment_dir, f"{batch_name}_{run_number:03d}.json")
    with open(output_file, "w") as f:
        json.dump(batch_results, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"Batch Complete: {batch_name} (run #{run_number:03d})")
    successful = sum(1 for r in batch_results["runs"] if r["success"])
    print(f"  Successful runs: {successful}/{batch_config['runs']}")
    print(f"  Total time: {batch_results['total_time_seconds']:.1f}s")
    print(f"  Saved to: {output_file}")
    print(f"{'='*70}\n")
    
    # Clean up vLLM engine after batch to free GPU memory
    cleanup_vllm_engine()
    
    return batch_results


def main():
    parser = argparse.ArgumentParser(description="Run individual experiment batches")
    parser.add_argument("--list", action="store_true", help="List all available batches")
    parser.add_argument("--batch", nargs="+", type=str, help="Batch name(s) to run")
    parser.add_argument("--model", type=str, help="Filter/run by model")
    parser.add_argument("--agents", type=int, help="Filter/run by agent count")
    parser.add_argument("--experiment", type=str, choices=["Q1", "Q2"], help="Filter/run by experiment")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--results-dir", type=str, default="results/batches", help="Output directory")
    
    args = parser.parse_args()
    
    all_batches = generate_all_batches()
    
    # List mode
    if args.list:
        list_batches(all_batches, args.model, args.agents, args.experiment)
        return 0
    
    # Determine which batches to run
    batches_to_run = []
    
    if args.batch:
        # Run specific batch(es) by name
        for batch_name in args.batch:
            if batch_name not in all_batches:
                print(f"Error: Unknown batch '{batch_name}'")
                print("Use --list to see available batches")
                return 1
            batches_to_run.append((batch_name, all_batches[batch_name]))
    elif args.model or args.agents or args.experiment:
        # Run batches matching filters
        for name, config in all_batches.items():
            if args.model and config["model"] != args.model:
                continue
            if args.agents and config["num_agents"] != args.agents:
                continue
            if args.experiment and config["experiment"] != args.experiment:
                continue
            batches_to_run.append((name, config))
    else:
        print("Error: Specify --batch, --model, --agents, or --experiment")
        print("Use --list to see available batches")
        return 1
    
    if not batches_to_run:
        print("No batches match the specified filters")
        return 1
    
    print(f"\nWill run {len(batches_to_run)} batch(es)")
    for name, _ in batches_to_run:
        print(f"  - {name}")
    print()
    
    # Run each batch with guaranteed cleanup
    try:
        for batch_name, batch_config in batches_to_run:
            run_batch(batch_name, batch_config, args.gpu, args.results_dir)
        
        print("\n" + "="*70)
        print(f"All done! Ran {len(batches_to_run)} batch(es)")
        print("="*70 + "\n")
        
        return 0
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        return 1
    finally:
        # Ensure cleanup happens no matter what
        cleanup_vllm_engine()


if __name__ == "__main__":
    sys.exit(main())
