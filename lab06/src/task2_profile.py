import os
import time
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import mlflow
import train_pipeline as tp

# ==========================================
# Task 2: Profile one epoch with torch.profiler
# ==========================================
config = tp.load_config()
tp.set_seed(config['seed'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set, _, _ = tp.get_datasets(config)
loader = tp.make_loader(train_set, config['data']['batch_size'], config['data']['num_workers'], shuffle=True,
                        seed=config['seed'])
model = tp.build_model(config).to(device)
optimizer = tp.build_optimizer(model, config)
criterion = nn.CrossEntropyLoss()
model.train()

PHASES = ["data_loading", "forward", "backward", "optimizer_step"]
wall = {phase: 0.0 for phase in PHASES}

# Warm-up: a few steps outside the profiler so one-time costs are not counted
warm = iter(loader)
for _ in range(5):
    images, labels = next(warm)
    optimizer.zero_grad()
    criterion(model(images), labels).backward()
    optimizer.step()

# Profiling a whole epoch in one go records over a million events (every per-image transform)
# and runs out of memory. The epoch is therefore profiled in consecutive chunks of CHUNK steps;
# after each chunk the totals are added up and the raw events are freed.
CHUNK = 43
prof_us = {phase: 0.0 for phase in PHASES}
op_self_us, op_calls = {}, {}

def collect(prof):
    for e in prof.key_averages():
        if e.key in prof_us:
            prof_us[e.key] += e.cpu_time_total
        else:
            op_self_us[e.key] = op_self_us.get(e.key, 0.0) + e.self_cpu_time_total
            op_calls[e.key] = op_calls.get(e.key, 0) + e.count

n_steps = len(loader)
with profile(activities=[ProfilerActivity.CPU],
             schedule=torch.profiler.schedule(wait=0, warmup=0, active=CHUNK, repeat=-(-n_steps // CHUNK)),
             on_trace_ready=collect) as prof:
    epoch_start = time.perf_counter()
    iterator = iter(loader)
    n_batches = 0
    while True:
        t0 = time.perf_counter()
        with record_function("data_loading"):
            try:
                images, labels = next(iterator)
            except StopIteration:
                break
            images, labels = images.to(device), labels.to(device)
        t1 = time.perf_counter()
        with record_function("forward"):
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
        t2 = time.perf_counter()
        with record_function("backward"):
            loss.backward()
        t3 = time.perf_counter()
        with record_function("optimizer_step"):
            optimizer.step()
        t4 = time.perf_counter()
        wall["data_loading"] += t1 - t0
        wall["forward"] += t2 - t1
        wall["backward"] += t3 - t2
        wall["optimizer_step"] += t4 - t3
        n_batches += 1
        prof.step()
    epoch_time = time.perf_counter() - epoch_start

# Profiler totals for each labelled region
prof_ms = {phase: prof_us[phase] / 1000 for phase in PHASES}
prof_total = sum(prof_ms.values())
wall_total = sum(wall.values())

lines = []
lines.append("=== TASK 2: PROFILE OF ONE TRAINING EPOCH (torch.profiler) ===")
lines.append(f"Model: SimpleCNN | batch {config['data']['batch_size']} | num_workers {config['data']['num_workers']} | "
             f"{n_batches} batches | device {device} | torch threads {torch.get_num_threads()}")
lines.append(f"Elapsed time incl. the profiler post-processing each chunk: {epoch_time:.2f} s (training phases alone: {sum(wall.values()):.2f} s)")
lines.append("")
lines.append(f"{'Phase':<16} | {'Profiler (ms)':<13} | {'Share':<7} | {'Wall clock (ms)':<15} | Share")
lines.append("-" * 70)
for phase in PHASES:
    lines.append(f"{phase:<16} | {prof_ms[phase]:<13.1f} | {100 * prof_ms[phase] / prof_total:<6.1f}% | "
                 f"{wall[phase] * 1000:<15.1f} | {100 * wall[phase] / wall_total:.1f}%")
lines.append("-" * 70)
lines.append(f"{'Total':<16} | {prof_total:<13.1f} | {'100.0%':<7} | {wall_total * 1000:<15.1f} | 100.0%")
lines.append("")
lines.append("Top 15 operators by self CPU time (whole epoch):")
lines.append(f"{'Operator':<40} | {'Self CPU (ms)':<13} | {'Share':<7} | Calls")
lines.append("-" * 75)
all_self = sum(op_self_us.values())
for name, us in sorted(op_self_us.items(), key=lambda kv: -kv[1])[:15]:
    lines.append(f"{name[:40]:<40} | {us / 1000:<13.1f} | {100 * us / all_self:<6.1f}% | {op_calls[name]}")

report = "\n".join(lines)
print(report)

results_dir = os.path.join(tp.REPO_ROOT, 'lab06', 'results')
os.makedirs(results_dir, exist_ok=True)
out_file = os.path.join(results_dir, 'profile_breakdown.txt')
with open(out_file, 'w') as f:
    f.write(report + "\n")

# Log the breakdown to MLflow
mlflow.set_tracking_uri('file:' + tp.repo_path(config['mlflow']['tracking_uri']))
mlflow.set_experiment('lab06_profiling')
with mlflow.start_run(run_name='profile_one_epoch'):
    mlflow.log_params({'batch_size': config['data']['batch_size'], 'num_workers': config['data']['num_workers']})
    mlflow.log_metrics({f"share_{p}": 100 * prof_ms[p] / prof_total for p in PHASES})
    mlflow.log_metrics({f"ms_{p}": prof_ms[p] for p in PHASES})
    mlflow.log_artifact(out_file)

print("\n✅ Results saved to ../results/profile_breakdown.txt")
