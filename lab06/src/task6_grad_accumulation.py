import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_pipeline as tp

# ==========================================
# Task 6: Gradient accumulation (batch 64 x 4 steps = effective batch 256)
# ==========================================
EPOCHS = 5

# ------------------------------------------
# 1. One-step check: accumulated gradient vs real batch-256 gradient on the same 256 samples
# ------------------------------------------
config = tp.load_config()
tp.set_seed(config['seed'])
train_set, _, _ = tp.get_datasets(config)
images, labels = next(iter(tp.make_loader(train_set, 256, 0, shuffle=True, seed=config['seed'])))
criterion = nn.CrossEntropyLoss()

model = tp.build_model(config)
model.zero_grad()
criterion(model(images), labels).backward()
grad_big = [p.grad.clone() for p in model.parameters()]

model.zero_grad()
for i in range(4):
    x, y = images[i * 64:(i + 1) * 64], labels[i * 64:(i + 1) * 64]
    (criterion(model(x), y) / 4).backward()      # Each micro-batch loss scaled by 1/4, gradients add up
grad_acc = [p.grad.clone() for p in model.parameters()]
max_grad_diff = max((a - b).abs().max().item() for a, b in zip(grad_big, grad_acc))
print(f"Max abs gradient difference (batch 256 vs 4 x 64 accumulated): {max_grad_diff:.2e}")

# ------------------------------------------
# 2. Training runs with the same data order (the shuffle depends only on the seed)
# ------------------------------------------
RUNS = {
    'batch256': {'batch_size': 256, 'grad_accum_steps': 1, 'label': 'Batch 256'},
    'batch64_accum4': {'batch_size': 64, 'grad_accum_steps': 4, 'label': 'Batch 64 x accumulation 4'},
    'batch64': {'batch_size': 64, 'grad_accum_steps': 1, 'label': 'Batch 64 (no accumulation, reference)'},
}
results = {}
for name, r in RUNS.items():
    print(f"Training {name}...")
    config = tp.load_config(overrides={
        'data': {'batch_size': r['batch_size']},
        'training': {'epochs': EPOCHS, 'grad_accum_steps': r['grad_accum_steps']},
        'checkpoint': {'dir': '/tmp/lab06_checkpoints', 'name': name, 'log_to_mlflow': False},
        'mlflow': {'experiment': 'lab06_grad_accumulation', 'run_name': name},
    })
    results[name] = tp.run(config, verbose=False)

h256, hacc = results['batch256'][0], results['batch64_accum4'][0]
max_loss_diff = max(abs(a['train_loss'] - b['train_loss']) for a, b in zip(h256, hacc))
max_val_diff = max(abs(a['val_acc'] - b['val_acc']) for a, b in zip(h256, hacc))

lines = []
lines.append(f"=== TASK 6: GRADIENT ACCUMULATION (SimpleCNN, Adam lr=0.001, {EPOCHS} epochs) ===")
lines.append("Effective batch 256 emulated with micro-batch 64 and 4 accumulation steps.")
lines.append("Each micro-batch loss is weighted by its share of samples, then one optimizer step per group.")
lines.append("")
lines.append(f"One-step gradient check on the same 256 samples: max abs difference = {max_grad_diff:.2e}")
lines.append("")
lines.append(f"{'Run':<40} | {'Optimizer steps/epoch':<21} | {'Final train loss':<16} | {'Final val acc':<13} | {'Test acc':<8} | Samples/s")
lines.append("-" * 120)
for name, r in RUNS.items():
    history, final = results[name]
    steps = -(-55000 // (r['batch_size'] * r['grad_accum_steps']))
    tput = sum(h['throughput'] for h in history) / len(history)
    lines.append(f"{r['label']:<40} | {steps:<21} | {history[-1]['train_loss']:<16.4f} | {history[-1]['val_acc']:<13.4f} | "
                 f"{final['test_acc']:<8.4f} | {tput:.0f}")
lines.append("-" * 120)
lines.append("")
lines.append("Per-epoch train loss / val acc:")
lines.append(f"{'Epoch':<6} | {'Batch 256':<20} | {'Batch 64 x accum 4':<20} | {'Batch 64':<20}")
for e in range(EPOCHS):
    cells = [f"{results[n][0][e]['train_loss']:.4f} / {results[n][0][e]['val_acc']:.4f}" for n in RUNS]
    lines.append(f"{e + 1:<6} | {cells[0]:<20} | {cells[1]:<20} | {cells[2]:<20}")
lines.append("")
lines.append(f"Batch 256 vs batch 64 x accum 4: max train-loss difference {max_loss_diff:.2e}, "
             f"max val-acc difference {max_val_diff:.4f}")
report = "\n".join(lines)
print("\n" + report)

results_dir = os.path.join(tp.REPO_ROOT, 'lab06', 'results')
with open(os.path.join(results_dir, 'grad_accumulation.txt'), 'w') as f:
    f.write(report + "\n")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
epochs = list(range(1, EPOCHS + 1))
styles = {'batch256': 'o-', 'batch64_accum4': 'x--', 'batch64': 's:'}
for name, r in RUNS.items():
    history = results[name][0]
    axes[0].plot(epochs, [h['train_loss'] for h in history], styles[name], label=r['label'])
    axes[1].plot(epochs, [100 * h['val_acc'] for h in history], styles[name], label=r['label'])
axes[0].set_ylabel('Training loss')
axes[1].set_ylabel('Validation accuracy (%)')
for ax in axes:
    ax.set_xlabel('Epoch')
    ax.grid(True)
    ax.legend()
fig.suptitle('Gradient accumulation: batch 64 x 4 vs batch 256')
fig.tight_layout()
fig.savefig(os.path.join(results_dir, 'grad_accumulation.png'), dpi=300, bbox_inches='tight')
print("\n✅ Results saved to ../results/grad_accumulation.txt and ../results/grad_accumulation.png")
