import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import common as c

# ==========================================
# Task 5: One summary table organised by the Ch. 9 efficiency pillars (data, model, compute, energy)
# ==========================================
# Energy cannot be read from a power meter here (macOS powermetrics needs root), so two clearly
# labelled estimates are used:
#  (1) Measured CPU time x assumed package power. P_CPU is an assumption for an Apple M1 running
#      all performance cores, not a measurement.
#  (2) Arithmetic-only energy: MACs x 4.6 pJ (Horowitz, ISSCC 2014, 45 nm: fp32 multiply 3.7 pJ + add 0.9 pJ).
#      It ignores memory traffic, which usually dominates, so it is a lower bound that only shows the ratio.
P_CPU_W = 15.0
PJ_PER_MAC = 4.6
TRAIN_SAMPLES = 55000

if __name__ == "__main__":
    t1 = c.load_json('task1_baseline.json')
    t2 = c.load_json('task2_pruning.json')
    t3 = c.load_json('task3_latency_storage.json')
    t4 = c.load_json('task4_distillation.json')
    lat = {r['model']: r for r in t3['models']}
    layer90 = next(r for r in t2['unstructured'] if r['scope'] == 'layer' and r['target'] == 0.9)
    base_epochs = t1['epochs']
    ft = 3

    def row(name, acc, params_nz, size_bytes, size_fmt, macs, macs_nz, b1, b256, epochs_seen, labels):
        return {'name': name, 'acc': acc, 'params': params_nz, 'size': size_bytes, 'size_fmt': size_fmt,
                'macs': macs, 'macs_nz': macs_nz, 'b1': b1, 'b256': b256, 'samples': epochs_seen * TRAIN_SAMPLES,
                'labels': labels,
                'energy_cpu_mj': b256 / 256 * P_CPU_W,               # ms x W = mJ per image
                'energy_mac_uj': macs * PJ_PER_MAC * 1e-6}           # pJ -> uJ per image (dense MACs executed)

    u0, u90 = lat['Unstructured global 0%'], lat['Unstructured global 90%']
    s25, s50 = lat['Structured 25% filters'], lat['Structured 50% filters']
    stu = t4['student']
    rows = [
        row('Baseline BigCNN', t1['test_acc'], t1['params'], t1['size_bytes'], 'dense', t1['macs'], t1['macs'],
            u0['b1_ms'], u0['b256_ms'], base_epochs, 'hard'),
        row('Global unstructured 90%', u90['acc'], u90['nonzero_params'], u90['csr_bytes'], 'CSR', u90['macs'],
            u90['effective_macs'], u90['b1_ms'], u90['b256_ms'], base_epochs + ft, 'hard'),
        # Layer-wise 90% has the same shapes and the same number of zeros as global 90%, so it runs at the same
        # dense speed and its CSR file has the same size; those measurements are reused.
        row('Layer-wise unstructured 90%', layer90['acc_after_ft'], layer90['nonzero_params'], u90['csr_bytes'], 'CSR',
            t1['macs'], layer90['effective_macs'], u90['b1_ms'], u90['b256_ms'], base_epochs + ft, 'hard'),
        row('Structured 25% filters', s25['acc'], s25['nonzero_params'], s25['dense_bytes'], 'dense', s25['macs'],
            s25['macs'], s25['b1_ms'], s25['b256_ms'], base_epochs + ft, 'hard'),
        row('Structured 50% filters', s50['acc'], s50['nonzero_params'], s50['dense_bytes'], 'dense', s50['macs'],
            s50['macs'], s50['b1_ms'], s50['b256_ms'], base_epochs + ft, 'hard'),
        row('Plain student', t4['summary']['plain']['mean'], stu['params'], stu['size_bytes'], 'dense', stu['macs'],
            stu['macs'], stu['latency_b1_ms'], stu['latency_b256_ms'], t4['epochs'], 'hard'),
        row('Distilled student', t4['summary']['distilled']['mean'], stu['params'], stu['size_bytes'], 'dense', stu['macs'],
            stu['macs'], stu['latency_b1_ms'], stu['latency_b256_ms'], base_epochs + t4['epochs'], 'hard + teacher soft'),
    ]
    c.save_json(rows, 'task5_summary.json')

    b = rows[0]
    lines = ["=== TASK 5: SUMMARY BY EFFICIENCY PILLAR (Reddi Ch. 9: data, model, compute, energy) ===",
             "Accuracy = test set; student accuracy = mean of 3 seeds. Latency = CPU median, 8 threads.",
             "Data: training samples processed to produce the model (pruned models include the baseline they start from;",
             "      the distilled student includes the teacher's training).",
             f"Energy (estimates, not measured): CPU = (batch-256 latency / 256) x {P_CPU_W:g} W assumed package power;",
             f"      MAC = dense MACs executed x {PJ_PER_MAC} pJ (arithmetic only, 45 nm fp32).", "",
             f"{'':<28} | {'DATA':<34} | {'MODEL':<40} | {'COMPUTE':<44} | ENERGY",
             f"{'Model':<28} | {'Samples seen':<12} | {'Labels':<19} | {'Test acc':<8} | {'Non-zero params':<15} | {'Size':<13} | "
             f"{'Dense MACs':<10} | {'Non-zero MACs':<13} | {'b1 (ms)':<7} | {'b256 (ms)':<9} | {'CPU mJ/img':<10} | MAC uJ/img",
             "-" * 190]
    for r in rows:
        lines.append(f"{r['name']:<28} | {r['samples'] / 1e3:<10,.0f}k | {r['labels']:<19} | {r['acc']:<8.4f} | "
                     f"{r['params']:<15,} | {r['size'] / 2**20:<5.2f} MiB {r['size_fmt']:<5} | {r['macs'] / 1e6:<8.1f}M | "
                     f"{r['macs_nz'] / 1e6:<11.1f}M | {r['b1']:<7.3f} | {r['b256']:<9.2f} | {r['energy_cpu_mj']:<10.3f} | "
                     f"{r['energy_mac_uj']:.1f}")
    lines += ["-" * 190, "", "Ratios vs baseline:",
              f"{'Model':<28} | {'Acc change':<10} | {'Params':<7} | {'Size':<7} | {'MACs':<7} | {'b256 speed-up':<13} | CPU energy"]
    for r in rows[1:]:
        lines.append(f"{r['name']:<28} | {100 * (r['acc'] - b['acc']):+.2f} pts  | {b['params'] / r['params']:<6.1f}x | "
                     f"{b['size'] / r['size']:<6.1f}x | {b['macs'] / r['macs']:<6.1f}x | {b['b256'] / r['b256']:<12.2f}x | "
                     f"{b['energy_cpu_mj'] / r['energy_cpu_mj']:.2f}x less")
    c.write_txt(lines, 'task5_summary.txt')

    # ---- Accuracy vs cost frontier ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = ['black', 'tab:blue', 'tab:cyan', 'tab:olive', 'tab:green', 'tab:orange', 'tab:red']
    for ax, key, xlabel in [(axes[0], 'b256', 'CPU latency per batch of 256 (ms)'),
                            (axes[1], 'size', 'Size on disk (MiB, best format)')]:
        for r, col in zip(rows, colors):
            x = r[key] / 2**20 if key == 'size' else r[key]
            ax.scatter(x, 100 * r['acc'], s=80, color=col, label=r['name'], zorder=3)
        ax.set_xscale('log')
        ax.set_xlabel(xlabel + ' (log scale)')
        ax.set_ylabel('Test accuracy (%)')
        ax.grid(alpha=0.3, which='both')
    axes[0].set_title('Accuracy vs latency: only structural changes move left')
    axes[1].set_title('Accuracy vs storage')
    axes[1].legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(c.RESULTS_DIR, 'efficiency_summary.png'), dpi=150)
    print("Saved efficiency_summary.png")
