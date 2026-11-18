import os
import sys
import subprocess
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_pipeline as tp

# ==========================================
# Task 7: Interrupt training at epoch 3, resume from the checkpoint
# ==========================================
# Same config as Task 1, so the uninterrupted Task 1 run is the reference curve.
# Run train_pipeline.py (Task 1) first.
NAME = 'cnn_resume'
INTERRUPT_EPOCH = 3
src_dir = os.path.dirname(os.path.abspath(__file__))
config = tp.load_config()
ckpt_dir = tp.repo_path(config['checkpoint']['dir'])

reference_ckpt = os.path.join(ckpt_dir, f"{config['checkpoint']['name']}_last.pt")
if not os.path.exists(reference_ckpt):
    sys.exit("Run train_pipeline.py (Task 1) first: its run is the uninterrupted reference.")
reference = torch.load(reference_ckpt, weights_only=False)['history']

# Start fresh
resume_ckpt = os.path.join(ckpt_dir, f"{NAME}_last.pt")
if os.path.exists(resume_ckpt):
    os.remove(resume_ckpt)

# 1. First process: trains and stops after epoch 3 (the process exits)
print(f"Process 1: training, interrupted after epoch {INTERRUPT_EPOCH}...")
subprocess.run([sys.executable, 'train_pipeline.py', '--run-name', NAME, '--stop-after-epoch', str(INTERRUPT_EPOCH)],
               cwd=src_dir, check=True)

# 2. Second, new process: resumes from the checkpoint and finishes the run
print("\nProcess 2: resuming from the checkpoint...")
subprocess.run([sys.executable, 'train_pipeline.py', '--run-name', NAME, '--resume'], cwd=src_dir, check=True)

resumed = torch.load(resume_ckpt, weights_only=False)['history']

max_loss_diff = max(abs(a['train_loss'] - b['train_loss']) for a, b in zip(reference, resumed))
max_val_diff = max(abs(a['val_acc'] - b['val_acc']) for a, b in zip(reference, resumed))

lines = []
lines.append("=== TASK 7: INTERRUPT AT EPOCH 3 AND RESUME FROM CHECKPOINT ===")
lines.append(f"Process 1 trained epochs 1-{INTERRUPT_EPOCH} and exited; process 2 loaded {NAME}_last.pt "
             f"(model, optimizer, RNG states, history, MLflow run id) and trained epochs {INTERRUPT_EPOCH + 1}-{len(resumed)}.")
lines.append("Reference: the uninterrupted Task 1 run with the same config and seed.")
lines.append("")
lines.append(f"{'Epoch':<6} | {'Ref train loss':<14} | {'Resumed train loss':<18} | {'Ref val acc':<11} | {'Resumed val acc':<15} | Segment")
lines.append("-" * 90)
for a, b in zip(reference, resumed):
    segment = 'before interruption' if b['epoch'] <= INTERRUPT_EPOCH else 'after resume'
    lines.append(f"{a['epoch']:<6} | {a['train_loss']:<14.6f} | {b['train_loss']:<18.6f} | {a['val_acc']:<11.4f} | "
                 f"{b['val_acc']:<15.4f} | {segment}")
lines.append("-" * 90)
lines.append(f"Max abs train-loss difference vs uninterrupted run: {max_loss_diff:.2e}")
lines.append(f"Max abs val-acc difference vs uninterrupted run:    {max_val_diff:.4f}")
report = "\n".join(lines)
print("\n" + report)

results_dir = os.path.join(tp.REPO_ROOT, 'lab06', 'results')
with open(os.path.join(results_dir, 'resume_from_checkpoint.txt'), 'w') as f:
    f.write(report + "\n")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ref_epochs = [r['epoch'] for r in reference]
before = [r for r in resumed if r['epoch'] <= INTERRUPT_EPOCH]
after = [r for r in resumed if r['epoch'] >= INTERRUPT_EPOCH]
for ax, key, ylabel in [(axes[0], 'train_loss', 'Training loss'), (axes[1], 'val_acc', 'Validation accuracy')]:
    ax.plot(ref_epochs, [r[key] for r in reference], '-', color='lightgray', linewidth=6, label='Uninterrupted (Task 1)')
    ax.plot([r['epoch'] for r in before], [r[key] for r in before], 'o-', color='tab:blue', label='Process 1 (epochs 1-3)')
    ax.plot([r['epoch'] for r in after], [r[key] for r in after], 's--', color='tab:orange', label='Process 2 (resumed)')
    ax.axvline(INTERRUPT_EPOCH, color='red', linestyle=':', label='Interruption')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
fig.suptitle('Training interrupted after epoch 3 and resumed from checkpoint')
fig.tight_layout()
fig.savefig(os.path.join(results_dir, 'resume_from_checkpoint.png'), dpi=300, bbox_inches='tight')
print("\n✅ Results saved to ../results/resume_from_checkpoint.txt and ../results/resume_from_checkpoint.png")
