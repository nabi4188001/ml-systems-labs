import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_pipeline as tp

# ==========================================
# Task 5: LR schedules (StepLR, CosineAnnealingLR, ReduceLROnPlateau)
#         x optimizers (SGD+momentum, Adam), equal 10-epoch budgets
# ==========================================
EPOCHS = 10
OPTIMIZERS = {'sgd': {'optimizer': 'sgd', 'lr': 0.05, 'momentum': 0.9},
              'adam': {'optimizer': 'adam', 'lr': 0.001}}
SCHEDULERS = {'step': 'StepLR(step_size=4, gamma=0.1)',
              'cosine': 'CosineAnnealingLR(T_max=10)',
              'plateau': 'ReduceLROnPlateau(val_acc, factor=0.1, patience=1)'}

results = {}
for opt_name, opt_cfg in OPTIMIZERS.items():
    for sched in SCHEDULERS:
        name = f"{opt_name}_{sched}"
        print(f"Training {name}...")
        config = tp.load_config(overrides={
            'training': {**opt_cfg, 'scheduler': sched, 'epochs': EPOCHS},
            'checkpoint': {'dir': '/tmp/lab06_checkpoints', 'name': name, 'log_to_mlflow': False},
            'mlflow': {'experiment': 'lab06_schedulers_optimizers', 'run_name': name},
        })
        history, final = tp.run(config, verbose=False)
        results[name] = (history, final)
        print(f"  -> val acc {history[-1]['val_acc']:.4f} | test acc {final['test_acc']:.4f}")

lines = []
lines.append(f"=== TASK 5: LR SCHEDULES x OPTIMIZERS (SimpleCNN, batch 128, {EPOCHS} epochs each) ===")
lines.append("SGD+momentum: lr=0.05, momentum=0.9 | Adam: lr=0.001")
for sched, desc in SCHEDULERS.items():
    lines.append(f"  {sched:<8}: {desc}")
lines.append("")
lines.append(f"{'Optimizer':<13} | {'Schedule':<9} | {'Final train loss':<16} | {'Best val acc':<12} | {'Final val acc':<13} | "
             f"{'Test acc':<8} | {'Final LR':<9} | Time (s)")
lines.append("-" * 110)
for name, (history, final) in results.items():
    opt_name, sched = name.split('_')
    label = 'SGD+momentum' if opt_name == 'sgd' else 'Adam'
    lines.append(f"{label:<13} | {sched:<9} | {history[-1]['train_loss']:<16.4f} | {max(r['val_acc'] for r in history):<12.4f} | "
                 f"{history[-1]['val_acc']:<13.4f} | {final['test_acc']:<8.4f} | {history[-1]['lr']:<9.2e} | "
                 f"{sum(r['epoch_time_s'] for r in history):.1f}")
lines.append("-" * 110)
lines.append("Final LR = learning rate used in the last epoch.")
lines.append("")
lines.append("Validation accuracy per epoch:")
lines.append(f"{'Run':<14} | " + " | ".join(f"E{e + 1:<5}" for e in range(EPOCHS)))
for name, (history, _) in results.items():
    lines.append(f"{name:<14} | " + " | ".join(f"{r['val_acc']:.4f}" for r in history))
report = "\n".join(lines)
print("\n" + report)

results_dir = os.path.join(tp.REPO_ROOT, 'lab06', 'results')
with open(os.path.join(results_dir, 'schedulers_optimizers.txt'), 'w') as f:
    f.write(report + "\n")

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
epochs = list(range(1, EPOCHS + 1))
for name, (history, _) in results.items():
    opt_name = name.split('_')[0]
    ax = axes[0] if opt_name == 'sgd' else axes[1]
    ax.plot(epochs, [100 * r['val_acc'] for r in history], 'o-', label=name.split('_')[1])
    # SGD solid, Adam dashed: both use the same relative schedule, so the lines would otherwise hide each other
    axes[2].plot(epochs, [r['lr'] / OPTIMIZERS[opt_name]['lr'] for r in history],
                 'o-' if opt_name == 'sgd' else 'x--', linewidth=3 if opt_name == 'sgd' else 1.5,
                 color={'step': 'tab:blue', 'cosine': 'tab:orange', 'plateau': 'tab:green'}[name.split('_')[1]],
                 label=name)
axes[0].set_title('SGD+momentum (lr=0.05): validation accuracy')
axes[1].set_title('Adam (lr=0.001): validation accuracy')
axes[2].set_title('Learning rate relative to initial LR')
for ax in axes[:2]:
    ax.set_ylabel('Validation accuracy (%)')
axes[2].set_ylabel('LR / initial LR')
for ax in axes:
    ax.set_xlabel('Epoch')
    ax.grid(True)
    ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(results_dir, 'schedulers_optimizers.png'), dpi=300, bbox_inches='tight')
print("\n✅ Results saved to ../results/schedulers_optimizers.txt and ../results/schedulers_optimizers.png")
