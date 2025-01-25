import os
import argparse
from pathlib import Path

# NOTE: Not in use at the moment

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", "-d", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Parameters
    base_path = Path('/home/iony/DTU/f24/thesis/code/lgvit/lgvit_repo')
    checkpoint_path = Path("/home/iony/DTU/f24/thesis/code/lgvit/LGViT-ViT-Cifar100")
    ignore_mismatched_sizes = False
    model_path = base_path / "models/deit_highway"

    # Update PYTHONPATH
    os.environ["PYTHONPATH"] = f"{base_path}:{os.environ.get('PYTHONPATH', '')}"
    os.environ["PYTHONPATH"] = f"{os.environ['PYTHONPATH']}:{model_path}"
    
    # Model configs
    backbone = "ViT"
    model_type = f"{backbone}-base"
    model_name = "facebook/deit-base-distilled-patch16-224"
    
    # Dataset config
    dataset = "uoft-cs/cifar100"
    dataname = "tiny-imagenet" if dataset == "Maysee/tiny-imagenet" else dataset
    
    # Training configs
    exit_strategy = "confidence"
    highway_type = "LGViT"
    paper_name = "LGViT"
    
    # CUDA and logging
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    report_to = "none"
    
    # Build command arguments
    run_args = {
        "run_name": f"{backbone}_{exit_strategy}_{highway_type}_{paper_name}",
        "image_processor_name": model_name,
        "config_name": model_name,
        "model_name_or_path": str(checkpoint_path),
        "dataset_name": dataset,
        "output_dir": f"../outputs/{model_type}/{dataset}/{paper_name}/{exit_strategy}/",
        "remove_unused_columns": False,
        "backbone": backbone,
        "exit_strategy": exit_strategy,
        "do_train": False,
        "do_eval": True,
        "per_device_eval_batch_size": 1,
        "seed": 777,
        "report_to": report_to,
        "use_auth_token": False,
        "ignore_mismatched_sizes": ignore_mismatched_sizes
    }

    cmd_args = " ".join([f"--{k} {v}" for k, v in run_args.items()])

    if args.verbose:
        print(f"Arguments:\n{cmd_args}")
    
    if args.dry_run:
        print("Dry run, exiting...")
        return

    script_path = base_path / "examples/run_highway_deit.py"
    os.system(f"python {script_path} {cmd_args}")

if __name__ == "__main__":
    main()