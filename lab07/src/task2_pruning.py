import os
import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import common as c

# ==========================================
# Task 2: Magnitude pruning (global, layer-wise, structured)
# ==========================================
SPARSITIES = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95]
FILTER_FRACTIONS = [0.10, 0.25, 0.50]
FT_EPOCHS = 3
FT_LR = 5e-4          # half the baseline LR: fine-tuning, not retraining

def load_baseline():
    model = c.BigCNN()
    model.load_state_dict(torch.load(os.path.join(c.CKPT_DIR, 'baseline.pt'), map_location='cpu'))
    return model

def apply_unstructured(model, sparsity, scope):
    """Rank |w| and mask the smallest. 'global': one threshold over all conv/linear weights.
    'layer': every layer gets the same sparsity. Biases are not pruned (0.2% of the parameters)."""
    layers = [(m, 'weight') for _, m in c.prunable_layers(model)]
    if sparsity == 0:
        for m, n in layers:
            prune.identity(m, n)       # all-ones mask, so the s=0 control is fine-tuned through the same code path
    elif scope == 'global':
        prune.global_unstructured(layers, pruning_method=prune.L1Unstructured, amount=sparsity)
    else:
        for m, n in layers:
            prune.l1_unstructured(m, n, amount=sparsity)

def make_permanent(model):
    for _, m in c.prunable_layers(model):
        if prune.is_pruned(m):
            prune.remove(m, 'weight')
    return model

def per_layer_sparsity(model):
    return {n: (m.weight == 0).float().mean().item() for n, m in c.prunable_layers(model)}

def structured_prune(model, fraction):
    """Physically removes the `fraction` of output filters with the smallest L2 norm from every conv layer,
    the matching input channels of the next conv, and the matching input columns of fc1.
    Returns a genuinely smaller dense BigCNN (no masks)."""
    convs = [model.conv1, model.conv2, model.conv3, model.conv4]
    keep = []
    for conv in convs:
        n_out = conv.out_channels
        n_keep = n_out - int(round(fraction * n_out))
        norms = conv.weight.detach().flatten(1).norm(p=2, dim=1)
        keep.append(torch.sort(torch.topk(norms, n_keep).indices).values)

    new = c.BigCNN(*[len(k) for k in keep], hidden=model.fc1.out_features)
    prev = torch.arange(1)                             # input image has one channel
    for conv, new_conv, k in zip(convs, [new.conv1, new.conv2, new.conv3, new.conv4], keep):
        new_conv.weight.data = conv.weight.data[k][:, prev].clone()
        new_conv.bias.data = conv.bias.data[k].clone()
        prev = k
    # fc1 input is flatten(C x 7 x 7): feature index = channel * 49 + position
    cols = (keep[3].unsqueeze(1) * 49 + torch.arange(49).unsqueeze(0)).flatten()
    new.fc1.weight.data = model.fc1.weight.data[:, cols].clone()
    new.fc1.bias.data = model.fc1.bias.data.clone()
    new.fc2.load_state_dict(model.fc2.state_dict())
    return new

if __name__ == "__main__":
    data = c.load_data()
    base = load_baseline()
    base_params = c.count_params(base)
    base_acc, _ = c.evaluate(copy.deepcopy(base), *data['test'])
    print(f"Baseline test acc {base_acc:.4f} | {base_params:,} params")

    results = {'unstructured': [], 'structured': [], 'baseline_acc': base_acc, 'baseline_params': base_params}

    # ---- Unstructured: global and layer-wise ----
    for scope in ['global', 'layer']:
        for s in SPARSITIES:
            model = copy.deepcopy(base)
            apply_unstructured(model, s, scope)
            acc_before, _ = c.evaluate(model, *data['test'])
            history, ft_time = c.train(model, data, FT_EPOCHS, FT_LR, verbose=False)   # masks stay fixed:
            make_permanent(model)                                                       # weight = orig * mask
            acc_after, _ = c.evaluate(model, *data['test'])
            model.cpu()
            r = {'scope': scope, 'target': s, 'sparsity': c.weight_sparsity(model),
                 'nonzero_params': c.count_params(model, nonzero=True), 'acc_before_ft': acc_before,
                 'acc_after_ft': acc_after, 'ft_time_s': ft_time, 'per_layer': per_layer_sparsity(model),
                 'effective_macs': c.effective_macs(model)}
            results['unstructured'].append(r)
            torch.save(model.state_dict(), os.path.join(c.CKPT_DIR, f'unstructured_{scope}_{s}.pt'))
            print(f"{scope:<6} s={s:<4} | acc before FT {acc_before:.4f} | after FT {acc_after:.4f} "
                  f"| measured sparsity {r['sparsity']:.4f}")

    # ---- Structured: remove whole filters by L2 norm ----
    for f in FILTER_FRACTIONS:
        model = structured_prune(copy.deepcopy(base), f)
        acc_before, _ = c.evaluate(model, *data['test'])
        history, ft_time = c.train(model, data, FT_EPOCHS, FT_LR, verbose=False)
        acc_after, _ = c.evaluate(model, *data['test'])
        model.cpu()
        params = c.count_params(model)
        r = {'filter_fraction': f, 'channels': list(model.channels), 'params': params,
             'param_reduction': 1 - params / base_params, 'macs': c.count_macs(model),
             'acc_before_ft': acc_before, 'acc_after_ft': acc_after, 'ft_time_s': ft_time}
        results['structured'].append(r)
        torch.save({'channels': list(model.channels), 'state_dict': model.state_dict()},
                   os.path.join(c.CKPT_DIR, f'structured_{f}.pt'))
        print(f"structured {f:.0%} | channels {model.channels} | params {params:,} "
              f"| acc before FT {acc_before:.4f} | after FT {acc_after:.4f}")

    c.save_json(results, 'task2_pruning.json')

    # ---- Text report ----
    lines = [f"=== TASK 2: MAGNITUDE PRUNING of BigCNN (baseline test acc {base_acc:.4f}, {base_params:,} params) ===",
             f"Fine-tuning: {FT_EPOCHS} epochs, Adam lr={FT_LR}, masks fixed (torch.nn.utils.prune, weight = weight_orig * mask).",
             "Sparsity = fraction of zero conv/linear weights (biases not pruned). Accuracy = test set (10,000 images).", "",
             "--- Unstructured magnitude pruning ---",
             f"{'Scope':<7} | {'Target':<6} | {'Measured':<8} | {'Non-zero params':<15} | {'Acc before FT':<13} | {'Acc after FT':<12} | Change vs baseline",
             "-" * 100]
    for r in results['unstructured']:
        lines.append(f"{r['scope']:<7} | {r['target']:<6} | {r['sparsity']:<8.4f} | {r['nonzero_params']:<15,} | "
                     f"{r['acc_before_ft']:<13.4f} | {r['acc_after_ft']:<12.4f} | {100 * (r['acc_after_ft'] - base_acc):+.2f} pts")
    lines += ["-" * 100, "", "Per-layer sparsity chosen by GLOBAL pruning (layer-wise gives every layer the target):",
              f"{'Target':<6} | " + " | ".join(f"{n:<6}" for n, _ in c.prunable_layers(base)), "-" * 64]
    for r in results['unstructured']:
        if r['scope'] == 'global':
            lines.append(f"{r['target']:<6} | " + " | ".join(f"{v:<6.3f}" for v in r['per_layer'].values()))
    lines += ["", "--- Structured pruning: whole conv filters removed by L2 norm (model physically rebuilt smaller) ---",
              f"{'Filters removed':<15} | {'Channels (c1,c2,c3,c4)':<22} | {'Params':<9} | {'Param reduction':<15} | "
              f"{'MACs (M)':<8} | {'Acc before FT':<13} | {'Acc after FT':<12} | Change", "-" * 120]
    for r in results['structured']:
        lines.append(f"{r['filter_fraction']:<15.0%} | {str(tuple(r['channels'])):<22} | {r['params']:<9,} | "
                     f"{r['param_reduction']:<15.1%} | {r['macs'] / 1e6:<8.1f} | {r['acc_before_ft']:<13.4f} | "
                     f"{r['acc_after_ft']:<12.4f} | {100 * (r['acc_after_ft'] - base_acc):+.2f} pts")
    c.write_txt(lines, 'task2_pruning.txt')

    # ---- Sparsity curves ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    styles = {'global': ('tab:blue', 'Global'), 'layer': ('tab:orange', 'Layer-wise')}
    for scope, (col, label) in styles.items():
        rs = [r for r in results['unstructured'] if r['scope'] == scope]
        xs = [r['sparsity'] for r in rs]
        ax.plot(xs, [100 * r['acc_after_ft'] for r in rs], 'o-', color=col, label=f'{label}, after 3-epoch fine-tune')
        ax.plot(xs, [100 * r['acc_before_ft'] for r in rs], 'o--', color=col, alpha=0.5, label=f'{label}, before fine-tune')
    st = results['structured']
    ax.plot([r['param_reduction'] for r in st], [100 * r['acc_after_ft'] for r in st], 's-', color='tab:green',
            label='Structured (filters), after fine-tune')
    ax.plot([r['param_reduction'] for r in st], [100 * r['acc_before_ft'] for r in st], 's--', color='tab:green',
            alpha=0.5, label='Structured (filters), before fine-tune')
    ax.axhline(100 * base_acc, color='grey', ls=':', label=f'Baseline {100 * base_acc:.2f}%')
    ax.set_xlabel('Sparsity (fraction of weights removed)')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title('Accuracy vs sparsity (full range)')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(alpha=0.3)

    ax = axes[1]
    for scope, (col, label) in styles.items():
        rs = [r for r in results['unstructured'] if r['scope'] == scope]
        ax.plot([r['sparsity'] for r in rs], [100 * r['acc_after_ft'] for r in rs], 'o-', color=col, label=label)
    ax.plot([r['param_reduction'] for r in st], [100 * r['acc_after_ft'] for r in st], 's-', color='tab:green',
            label='Structured (x = parameter reduction)')
    for r in st:
        ax.annotate(f"{r['filter_fraction']:.0%} filters", (r['param_reduction'], 100 * r['acc_after_ft']),
                    textcoords='offset points', xytext=(4, -12), fontsize=8, color='tab:green')
    ax.axhline(100 * base_acc, color='grey', ls=':')
    ax.set_xlabel('Sparsity (fraction of weights removed)')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title('After fine-tuning (zoomed)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(c.RESULTS_DIR, 'sparsity_curves.png'), dpi=150)
    print("Saved sparsity_curves.png")
