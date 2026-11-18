import os
import sys
import json
import resource
import subprocess
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_pipeline as tp

# ==========================================
# Task 4: Batch-size study
# ==========================================
BATCH_SIZES = [32, 64, 128, 256, 512]
EPOCHS = 5

def peak_rss_mib():
    """Peak resident memory of this process (ru_maxrss is KiB on Linux, bytes on macOS)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == 'darwin' else peak / 1024

def single_run(batch_size):
    """Trains one configuration. Runs in its own process so the peak memory belongs to this batch size only."""
    config = tp.load_config(overrides={
        'data': {'batch_size': batch_size},
        'training': {'epochs': EPOCHS},
        'checkpoint': {'dir': '/tmp/lab06_checkpoints', 'name': f'bs_{batch_size}', 'log_to_mlflow': False},
        'mlflow': {'experiment': 'lab06_batch_size', 'run_name': f'batch_{batch_size}'},
    })
    tp.get_datasets(config)          # Load the data first, so the baseline includes it
    baseline = peak_rss_mib()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    history, final = tp.run(config, verbose=False)
    result = {
        'batch_size': batch_size,
        'throughput': float(np.mean([r['throughput'] for r in history])),
        'epoch_time': float(np.mean([r['epoch_time_s'] for r in history])),
        'peak_rss_mib': peak_rss_mib(),
        'baseline_rss_mib': baseline,
        'peak_gpu_mib': torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
        'final_val_acc': history[-1]['val_acc'],
        'test_acc': final['test_acc'],
        'steps_per_epoch': -(-55000 // batch_size),
        'val_curve': [r['val_acc'] for r in history],
    }
    import mlflow
    with mlflow.start_run(run_id=final['run_id']):
        mlflow.log_metrics({'peak_rss_mib': result['peak_rss_mib'], 'mean_throughput': result['throughput']})
    print("RESULT " + json.dumps(result))

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == '--single':
        single_run(int(sys.argv[2]))
        sys.exit(0)

    results = []
    for bs in BATCH_SIZES:
        out = subprocess.run([sys.executable, __file__, '--single', str(bs)], capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.abspath(__file__)))
        line = [l for l in out.stdout.splitlines() if l.startswith('RESULT ')]
        if not line:
            print(out.stdout[-2000:], out.stderr[-2000:])
            sys.exit(1)
        r = json.loads(line[0][7:])
        results.append(r)
        print(f"batch {bs}: {r['throughput']:.0f} samples/s | peak RSS {r['peak_rss_mib']:.0f} MiB | test acc {r['test_acc']:.4f}")

    lines = []
    lines.append(f"=== TASK 4: BATCH-SIZE STUDY (SimpleCNN, Adam lr=0.001, {EPOCHS} epochs, num_workers 0) ===")
    lines.append(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'} | {os.cpu_count()} CPU cores")
    lines.append("Peak memory = peak resident memory (RSS) of the training process; each batch size runs in its own process.")
    lines.append("")
    lines.append(f"{'Batch':<6} | {'Steps/epoch':<11} | {'Throughput (samples/s)':<22} | {'Epoch time (s)':<14} | "
                 f"{'Peak RSS (MiB)':<14} | {'Training adds (MiB)':<19} | {'Val acc':<7} | Test acc")
    lines.append("-" * 125)
    for r in results:
        lines.append(f"{r['batch_size']:<6} | {r['steps_per_epoch']:<11} | {r['throughput']:<22.0f} | {r['epoch_time']:<14.2f} | "
                     f"{r['peak_rss_mib']:<14.1f} | {r['peak_rss_mib'] - r['baseline_rss_mib']:<19.1f} | "
                     f"{r['final_val_acc']:<7.4f} | {r['test_acc']:.4f}")
    lines.append("-" * 125)
    report = "\n".join(lines)
    print("\n" + report)

    results_dir = os.path.join(tp.REPO_ROOT, 'lab06', 'results')
    with open(os.path.join(results_dir, 'batch_size_study.txt'), 'w') as f:
        f.write(report + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    bs = [r['batch_size'] for r in results]
    axes[0].plot(bs, [r['throughput'] for r in results], 'o-')
    axes[0].set_ylabel('Throughput (samples/s)')
    axes[1].plot(bs, [r['peak_rss_mib'] for r in results], 'o-', color='orange')
    axes[1].set_ylabel('Peak RSS (MiB)')
    axes[2].plot(bs, [100 * r['test_acc'] for r in results], 'o-', color='green')
    axes[2].set_ylabel(f'Test accuracy after {EPOCHS} epochs (%)')
    for ax in axes:
        ax.set_xscale('log', base=2)
        ax.set_xticks(bs)
        ax.set_xticklabels(bs)
        ax.set_xlabel('Batch size')
        ax.grid(True)
    fig.suptitle('Batch-size study')
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'batch_size_study.png'), dpi=300, bbox_inches='tight')
    print("\n✅ Results saved to ../results/batch_size_study.txt and ../results/batch_size_study.png")
