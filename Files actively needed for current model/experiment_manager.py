"""
Professional Experiment Manager for LaBraM + DEAP

Run:
    python experiment_manager.py

Edit ONLY the CONFIG section.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import torch

def get_next_experiment_id(output_root):
    """
    Returns the next experiment ID.
    Example:
        Exp001
        Exp002
        Exp003
    """

    counter_file = output_root / "experiment_counter.json"

    if counter_file.exists():
        with open(counter_file, "r") as f:
            data = json.load(f)
            exp_id = data["last_id"] + 1
    else:
        exp_id = 1

    with open(counter_file, "w") as f:
        json.dump({"last_id": exp_id}, f, indent=4)

    return exp_id

# ===========================
# CONFIG
# ===========================

DEAP_ROOT = r"G:\Reseacch2\LaBraM-main\datasets\deap-dataset\data_preprocessed_python"
CHECKPOINT = r"G:\Reseacch2\LaBraM-main\checkpoints\labram-base.pth"

# Automatically use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_WORKERS = 4
SEED = 119

# ===========================
# Dataset
# ===========================
NUM_CLASSES = 2      # Binary classification
# NUM_CLASSES = 4    # Uncomment if using 4-class DEAP

# Choose one or more
TASKS = [
    "valence",
    # "arousal",
    # "dominance",
    # "liking",
]

# Choose subjects
SUBJECTS = list(range(1, 33))
MAX_SUBJECTS_PER_RUN = 1

OUTPUT_ROOT = Path("experiments")
OUTPUT_ROOT.mkdir(exist_ok=True)

# ===========================

# Adapter defaults (exposed to experiments)
ADAPTER_REDUCTION = 6
ADAPTER_DROPOUT = 0.1
ADAPTER_ALPHA = 0.1
# New experimental adapter: preserves DEAP's channel-patch structure and
# uses a fixed 10-20 nearest-neighbour graph plus temporal patch filtering.
# Use 'bottleneck' to reproduce the previous adapter exactly.
ADAPTER_TYPE = "topology_temporal"
GRAPH_NEIGHBORS = 4
SELECTION_METRIC = "balanced_accuracy"
RESULTS_FILE = "results/topology_temporal_results.csv"

print("=" * 70)
print("LaBraM Experiment Manager")
print("=" * 70)

if DEVICE == "cuda":
    print(f"Using GPU : {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Using CPU.")

print("=" * 70)

subject_batches = [
    SUBJECTS[i:i + MAX_SUBJECTS_PER_RUN]
    for i in range(0, len(SUBJECTS), MAX_SUBJECTS_PER_RUN)
]

for task in TASKS:
    for batch_id, batch in enumerate(subject_batches, start=1):
        print("=" * 70)
        print(f"Starting Batch {batch_id}/{len(subject_batches)}")
        print(f"Subjects : {batch}")
        print("=" * 70)

        for subject in batch:
            exp_id = get_next_experiment_id(OUTPUT_ROOT)
            run_name = f"Exp{exp_id:03d}_{task.capitalize()}_Sub{subject:02d}"
            output_dir = OUTPUT_ROOT / run_name
            output_dir.mkdir(parents=True, exist_ok=True)

            config = {
                "experiment_id": f"Exp{exp_id:03d}",
                "run_name": run_name,
                "task": task,
                "subject": subject,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "device": DEVICE,
                "num_workers": NUM_WORKERS,
                "seed": SEED,
                "checkpoint": CHECKPOINT,
                "deap_root": DEAP_ROOT,
                "adapter_reduction": ADAPTER_REDUCTION,
                "adapter_dropout": ADAPTER_DROPOUT,
                "adapter_alpha": ADAPTER_ALPHA,
                "adapter_type": ADAPTER_TYPE,
                "graph_neighbors": GRAPH_NEIGHBORS,
                "selection_metric": SELECTION_METRIC,
                "results_file": RESULTS_FILE,
            }

            with open(output_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            print("="*70)
            print(f"Experiment : {run_name}")
            print("="*70)

            cmd = [
                sys.executable,
                "run_class_finetuning.py",
                "--dataset","DEAP",
                "--deap_root",DEAP_ROOT,
                "--task",task,
                "--leave_out_subject",str(subject),
                "--finetune",CHECKPOINT,
                "--output_dir",str(output_dir),
                "--epochs",str(EPOCHS),
                "--lr",str(LEARNING_RATE),
                "--batch_size",str(BATCH_SIZE),
                "--device",DEVICE,
                "--num_workers",str(NUM_WORKERS),
                "--seed",str(SEED),
                "--nb_classes", str(NUM_CLASSES),
                "--selection_metric", SELECTION_METRIC,
                "--results_file", RESULTS_FILE,
            ]

            # Pass adapter args to the training script
            cmd += [
                "--adapter_reduction", str(ADAPTER_REDUCTION),
                "--adapter_dropout", str(ADAPTER_DROPOUT),
                "--adapter_alpha", str(ADAPTER_ALPHA),
                "--adapter_type", ADAPTER_TYPE,
                "--graph_neighbors", str(GRAPH_NEIGHBORS),
            ]

            print(" ".join(cmd))
            start=datetime.now()
            result=subprocess.run(cmd)
            end=datetime.now()

            with open(output_dir/"run_log.txt","a",encoding="utf-8") as f:
                f.write(f"Run: {run_name}\n")
                f.write(f"Start : {start}\n")
                f.write(f"End   : {end}\n")
                f.write(f"Exit Code : {result.returncode}\n")
                f.write("-"*60+"\n")

            if result.returncode != 0:
                print(f"\nExperiment failed: {run_name}")
                print("Fix the error and rerun.")
                break

print("\nAll requested experiments finished.")
