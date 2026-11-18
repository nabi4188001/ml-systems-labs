import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import common as c

# ==========================================
# Task 4: Knowledge distillation (teacher = BigCNN baseline, student = SmallCNN ~100k params)
# ==========================================
T = 4.0
ALPHA = 0.5
EPOCHS = 10
LR = 1e-3
SEEDS = [0, 1, 2]

def kd_loss(student_logits, labels, teacher_logits):
    """alpha * CE(student, labels) + (1 - alpha) * T^2 * KL between the temperature-softened distributions.
    F.kl_div(log q_student, p_teacher) computes KL(p_teacher || q_student), the standard Hinton et al. direction:
    the student is pushed to put probability wherever the teacher does."""
    ce = F.cross_entropy(student_logits, labels)
    kl = F.kl_div(F.log_softmax(student_logits / T, dim=1), F.softmax(teacher_logits / T, dim=1),
                  reduction='batchmean')
    return ALPHA * ce + (1 - ALPHA) * T * T * kl

def teacher_logits(teacher, x, batch_size=1000):
    """The teacher is frozen and there is no data augmentation, so its logits are computed once
    (eval mode, dropout off) instead of in every student step."""
    teacher.eval().to(c.TRAIN_DEVICE)
    out = []
    with torch.inference_mode():
        for i in range(0, len(x), batch_size):
            out.append(teacher(x[i:i + batch_size].to(c.TRAIN_DEVICE)).cpu())
    return torch.cat(out)

if __name__ == "__main__":
    data = c.load_data()
    base = c.load_json('task1_baseline.json')
    teacher = c.BigCNN()
    teacher.load_state_dict(torch.load(os.path.join(c.CKPT_DIR, 'baseline.pt'), map_location='cpu'))
    t_logits = teacher_logits(teacher, data['train'][0])
    t_soft = F.softmax(t_logits / T, dim=1)
    t_hard = F.softmax(t_logits, dim=1)
    teacher_train_acc = (t_logits.argmax(1) == data['train'][1]).float().mean().item()

    runs = {'plain': [], 'distilled': []}
    for seed in SEEDS:
        for mode in ['plain', 'distilled']:
            c.set_seed(seed)
            student = c.SmallCNN()       # same initial weights for both modes at the same seed
            history, train_time = c.train(student, data, EPOCHS, LR, seed=seed, tag=f'{mode} seed {seed}',
                                          loss_fn=kd_loss if mode == 'distilled' else None,
                                          extra=t_logits if mode == 'distilled' else None, verbose=False)
            test_acc, test_loss = c.evaluate(student, *data['test'])
            runs[mode].append({'seed': seed, 'test_acc': test_acc, 'test_loss': test_loss, 'train_time_s': train_time,
                               'val_curve': [h['val_acc'] for h in history]})
            if seed == SEEDS[0]:
                torch.save(student.state_dict(), os.path.join(c.CKPT_DIR, f'student_{mode}.pt'))
            print(f"{mode:<9} seed {seed} | test acc {test_acc:.4f} | {train_time:.0f}s")

    student = c.SmallCNN()
    s_params = c.count_params(student)
    s_size = c.serialized_size_bytes(c.dense_state_dict(student))
    s_macs = c.count_macs(student)
    s_lat1 = c.model_latency(student, batch_size=1, repeats=500)
    s_lat1_t1 = c.model_latency(student, batch_size=1, threads=1, repeats=300)
    s_lat256 = c.model_latency(student, batch_size=256, warmup=5, repeats=40)
    teacher_lat1 = c.model_latency(teacher, batch_size=1, repeats=500)
    teacher_lat1_t1 = c.model_latency(teacher, batch_size=1, threads=1, repeats=300)
    teacher_lat256 = c.model_latency(teacher, batch_size=256, warmup=5, repeats=40)

    summary = {}
    for mode in runs:
        accs = np.array([r['test_acc'] for r in runs[mode]])
        summary[mode] = {'mean': float(accs.mean()), 'std': float(accs.std(ddof=1)), 'all': accs.tolist(),
                         'train_time_s': float(np.mean([r['train_time_s'] for r in runs[mode]]))}
    diffs = [d - p for d, p in zip(summary['distilled']['all'], summary['plain']['all'])]
    result = {'T': T, 'alpha': ALPHA, 'epochs': EPOCHS, 'runs': runs, 'summary': summary, 'paired_diffs': diffs,
              'student': {'params': s_params, 'size_bytes': s_size, 'macs': s_macs, 'latency_b1_ms': s_lat1[0],
                          'latency_b1_1thread_ms': s_lat1_t1[0], 'latency_b256_ms': s_lat256[0]},
              'teacher': {'params': base['params'], 'size_bytes': base['size_bytes'], 'macs': base['macs'],
                          'test_acc': base['test_acc'], 'train_acc': teacher_train_acc, 'latency_b1_ms': teacher_lat1[0],
                          'latency_b1_1thread_ms': teacher_lat1_t1[0], 'latency_b256_ms': teacher_lat256[0],
                          'mean_max_prob_T1': float(t_hard.max(1).values.mean()),
                          'mean_max_prob_T4': float(t_soft.max(1).values.mean())}}
    c.save_json(result, 'task4_distillation.json')

    tch = result['teacher']
    lines = [f"=== TASK 4: KNOWLEDGE DISTILLATION (T={T}, alpha={ALPHA}, {EPOCHS} epochs, Adam lr={LR}, seeds {SEEDS}) ===",
             "Loss = alpha * CE(student, labels) + (1 - alpha) * T^2 * KL(p_teacher^T || q_student^T)",
             "Plain student = same architecture, same seeds and initial weights, same data order, loss = CE only.",
             f"Teacher soft targets: mean max-probability {tch['mean_max_prob_T1']:.3f} at T=1 vs "
             f"{tch['mean_max_prob_T4']:.3f} at T=4 (T=4 exposes the 'dark knowledge' in the non-max classes).",
             f"Teacher accuracy on the training split: {teacher_train_acc:.4f}", "",
             "Student architecture: conv(1->16) conv(16->32) pool conv(32->64) pool fc(3136->24) fc(24->10)", "",
             f"{'Seed':<5} | {'Plain test acc':<14} | {'Distilled test acc':<18} | Difference", "-" * 60]
    for p, d in zip(runs['plain'], runs['distilled']):
        lines.append(f"{p['seed']:<5} | {p['test_acc']:<14.4f} | {d['test_acc']:<18.4f} | {100 * (d['test_acc'] - p['test_acc']):+.2f} pts")
    lines += ["-" * 60,
              f"{'Mean':<5} | {summary['plain']['mean']:<14.4f} | {summary['distilled']['mean']:<18.4f} | "
              f"{100 * np.mean(diffs):+.2f} pts", f"{'Std':<5} | {summary['plain']['std']:<14.4f} | {summary['distilled']['std']:<18.4f} |", "",
              f"{'Model':<20} | {'Params':<9} | {'Size (MiB)':<10} | {'MACs (M)':<8} | {'b1 (ms)':<7} | {'b1 1-thr (ms)':<13} | "
              f"{'b256 (ms)':<9} | Test acc", "-" * 112,
              f"{'Teacher (BigCNN)':<20} | {tch['params']:<9,} | {tch['size_bytes'] / 2**20:<10.3f} | {tch['macs'] / 1e6:<8.1f} | "
              f"{tch['latency_b1_ms']:<7.3f} | {tch['latency_b1_1thread_ms']:<13.3f} | {tch['latency_b256_ms']:<9.2f} | {tch['test_acc']:.4f}"]
    for mode in ['plain', 'distilled']:
        lines.append(f"{'Student ' + mode:<20} | {s_params:<9,} | {s_size / 2**20:<10.3f} | {s_macs / 1e6:<8.1f} | "
                     f"{s_lat1[0]:<7.3f} | {s_lat1_t1[0]:<13.3f} | {s_lat256[0]:<9.2f} | {summary[mode]['mean']:.4f} (mean of {len(SEEDS)})")
    lines += ["-" * 112,
              f"Student vs teacher: {tch['params'] / s_params:.1f}x fewer parameters, {tch['macs'] / s_macs:.1f}x fewer MACs, "
              f"{tch['latency_b256_ms'] / s_lat256[0]:.1f}x faster at batch 256, {tch['latency_b1_ms'] / s_lat1[0]:.1f}x faster at batch 1."]
    c.write_txt(lines, 'task4_distillation.txt')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ep = np.arange(1, EPOCHS + 1)
    for mode, col in [('plain', 'tab:orange'), ('distilled', 'tab:blue')]:
        curves = np.array([r['val_curve'] for r in runs[mode]]) * 100
        ax.plot(ep, curves.mean(0), 'o-', color=col, label=f'Student {mode} (mean of {len(SEEDS)} seeds)')
        ax.fill_between(ep, curves.min(0), curves.max(0), color=col, alpha=0.2)
    ax.axhline(100 * base['val_acc'], color='grey', ls=':', label=f"Teacher val acc {100 * base['val_acc']:.2f}%")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation accuracy (%)')
    ax.set_title(f'Student training curves (T={T:g}, alpha={ALPHA})')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    names = ['Teacher\n(1.04M)', 'Plain student\n(~99k)', 'Distilled student\n(~99k)']
    means = [100 * tch['test_acc'], 100 * summary['plain']['mean'], 100 * summary['distilled']['mean']]
    errs = [0, 100 * summary['plain']['std'], 100 * summary['distilled']['std']]
    bars = ax.bar(names, means, yerr=errs, capsize=6, color=['grey', 'tab:orange', 'tab:blue'])
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.15, f'{m:.2f}%', ha='center', fontsize=9)
    ax.set_ylim(min(means) - 2, max(means) + 1)
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title('Test accuracy (error bars = std over seeds)')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(c.RESULTS_DIR, 'distillation.png'), dpi=150)
    print("Saved distillation.png")
