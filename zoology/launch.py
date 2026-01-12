import os
import sys
import importlib.util
from datetime import datetime
import multiprocessing as mp
import click
import torch

from zoology.train import train
def worker_fn(gpu_id, task_queue, sweep_name):
    # 1. Isolate GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 2. Redirect stdout/stderr to a log file
    log_dir = os.path.join(os.getcwd(), "logs", sweep_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = open(os.path.join(log_dir, f"gpu_{gpu_id}.log"), "w", buffering=1)
    
    sys.stdout = log_file
    sys.stderr = log_file

    print(f"[{datetime.now()}] Worker-GPU{gpu_id} started.")

    # --- EVERYTHING BELOW THIS IS INSIDE THE LOOP ---
    while True:
        config = task_queue.get()
        if config is None:
            task_queue.task_done()
            break
        
        # Now 'config' is defined, so we can use it
        timestamp = datetime.now().strftime('%H%M%S')
        config.launch_id = f"{sweep_name}-{timestamp}"
        
        print(f"\n{'='*40}\nSTARTING: {config.run_id}\n{'='*40}")
        
        try:
            train(config=config)
            print(f"FINISHED: {config.run_id}")
        except Exception as e:
            # Use repr(e) here to see the actual error type and message
            print(f"FAILED: {config.run_id}\nError details: {repr(e)}")
        
        task_queue.task_done()
    
    log_file.close()

@click.command()
@click.argument("python_file", type=click.Path(exists=True))
@click.option("--gpus", default="0", type=str)
@click.option("--name", type=str, default="mqar_sweep")
def main(python_file, gpus: str, name: str):
    # 1. Load Configs
    spec = importlib.util.spec_from_file_location("config_module", python_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    configs = config_module.configs

    gpu_list = [int(g) for g in gpus.split(",")]
    num_workers = len(gpu_list)
    task_queue = mp.JoinableQueue()

    print(f"🚀 Launching sweep '{name}' on GPUs: {gpus}")
    print(f"📝 Logs will be saved to: ./logs/{name}/")

    for config in configs:
        task_queue.put(config)
    for _ in range(num_workers):
        task_queue.put(None)

    processes = []
    for i in range(num_workers):
        p = mp.Process(target=worker_fn, args=(gpu_list[i], task_queue, name))
        p.start()
        processes.append(p)

    # Simple terminal progress monitor
    try:
        task_queue.join()
    except KeyboardInterrupt:
        print("\nStopping sweep...")
        for p in processes:
            p.terminate()
    
    for p in processes:
        p.join()
    print("✅ All tasks complete.")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()