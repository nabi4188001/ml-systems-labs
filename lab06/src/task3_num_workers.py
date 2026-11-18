import os
import time
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import train_pipeline as tp

# ==========================================
# Task 3: Vary num_workers and find the bottleneck
# ==========================================
WORKERS = [0, 1, 2, 4, 8]
config = tp.load_config()
tp.set_seed(config['seed'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = config['data']['batch_size']
train_set, _, _ = tp.get_datasets(config)
criterion = nn.CrossEntropyLoss()

def loading_only(workers):
    """Iterates over one epoch of data without a model: the speed of the input pipeline alone."""
    loader = tp.make_loader(train_set, batch_size, workers, shuffle=True, seed=config['seed'])
    start = time.perf_counter()
    first_batch = None
    n = 0
    for images, labels in loader:
        if first_batch is None:
            first_batch = time.perf_counter() - start
        n += len(labels)
    return n / (time.perf_counter() - start), first_batch

def full_training(workers):
    """One full training epoch (load + forward + backward + step)."""
    tp.set_seed(config['seed'])
    model = tp.build_model(config).to(device)
    optimizer = tp.build_optimizer(model, config)
    loader = tp.make_loader(train_set, batch_size, workers, shuffle=True, seed=config['seed'])
    start = time.perf_counter()
    _, _, seen = tp.train_one_epoch(model, loader, optimizer, criterion, device)
    return seen / (time.perf_counter() - start)

# Compute-only speed: training on batches that are already in memory (no loader at all)
cached = [batch for _, batch in zip(range(50), tp.make_loader(train_set, batch_size, 0, True))]
model = tp.build_model(config).to(device)
optimizer = tp.build_optimizer(model, config)
for images, labels in cached[:5]:   # Warm-up
    optimizer.zero_grad(); criterion(model(images), labels).backward(); optimizer.step()
start = time.perf_counter()
for images, labels in cached:
    optimizer.zero_grad(); criterion(model(images), labels).backward(); optimizer.step()
compute_only = sum(len(l) for _, l in cached) / (time.perf_counter() - start)

loading_only(0)  # Warm-up (file cache)

mlflow.set_tracking_uri('file:' + tp.repo_path(config['mlflow']['tracking_uri']))
mlflow.set_experiment('lab06_num_workers')

rows = []
for workers in WORKERS:
    load_tp, first_batch = loading_only(workers)
    train_tp = full_training(workers)
    rows.append((workers, load_tp, first_batch, train_tp))
    print(f"num_workers={workers}: loading {load_tp:.0f} samples/s | training {train_tp:.0f} samples/s")
    with mlflow.start_run(run_name=f"num_workers_{workers}"):
        mlflow.log_params({'num_workers': workers, 'batch_size': batch_size})
        mlflow.log_metrics({'loading_samples_per_s': load_tp, 'training_samples_per_s': train_tp,
                            'first_batch_s': first_batch, 'compute_only_samples_per_s': compute_only})

lines = []
lines.append("=== TASK 3: num_workers STUDY (one epoch, batch 128) ===")
lines.append(f"Machine: {os.cpu_count()} CPU cores | torch threads {torch.get_num_threads()} | device {device}")
lines.append(f"Compute-only throughput (batches already in memory): {compute_only:.0f} samples/s")
lines.append("")
lines.append(f"{'num_workers':<11} | {'Loading only (samples/s)':<24} | {'First batch (s)':<15} | {'Full training (samples/s)':<25} | Speed-up vs 0")
lines.append("-" * 100)
base = rows[0][3]
for workers, load_tp, first_batch, train_tp in rows:
    lines.append(f"{workers:<11} | {load_tp:<24.0f} | {first_batch:<15.3f} | {train_tp:<25.0f} | {train_tp / base:.2f}x")
lines.append("-" * 100)
best = max(rows, key=lambda r: r[3])
lines.append(f"Best full-training throughput: num_workers={best[0]} ({best[3]:.0f} samples/s)")

report = "\n".join(lines)
print("\n" + report)

results_dir = os.path.join(tp.REPO_ROOT, 'lab06', 'results')
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, 'num_workers_study.txt'), 'w') as f:
    f.write(report + "\n")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(WORKERS, [r[1] for r in rows], 'o-', label='Data loading only')
plt.plot(WORKERS, [r[3] for r in rows], 's-', label='Full training step')
plt.axhline(compute_only, color='gray', linestyle='--', label='Compute only (data in memory)')
plt.xlabel('num_workers')
plt.ylabel('Throughput (samples/s)')
plt.title(f'Throughput vs num_workers ({os.cpu_count()} CPU cores, batch {batch_size})')
plt.xticks(WORKERS)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'num_workers_throughput.png'), dpi=300, bbox_inches='tight')

print("\n✅ Results saved to ../results/num_workers_study.txt and ../results/num_workers_throughput.png")
