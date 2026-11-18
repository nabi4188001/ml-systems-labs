import os
import io
import zlib
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import common as c

# ==========================================
# Task 3: Latency and on-disk size, dense vs sparse
# ==========================================
SPARSITIES = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95]
FILTER_FRACTIONS = [0.10, 0.25, 0.50]
KERNEL_SPARSITIES = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99]

def load_unstructured(s):
    m = c.BigCNN()
    m.load_state_dict(torch.load(os.path.join(c.CKPT_DIR, f'unstructured_global_{s}.pt'), map_location='cpu'))
    return m.eval()

def load_structured(f):
    ck = torch.load(os.path.join(c.CKPT_DIR, f'structured_{f}.pt'), map_location='cpu')
    m = c.BigCNN(*ck['channels'])
    m.load_state_dict(ck['state_dict'])
    return m.eval()

def gzip_size(state_dict):
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return len(zlib.compress(buf.getvalue(), 9))

def latencies(model):
    b1 = c.model_latency(model, batch_size=1, repeats=500)
    b1_t1 = c.model_latency(model, batch_size=1, threads=1, repeats=300)
    b256 = c.model_latency(model, batch_size=256, warmup=5, repeats=40)
    return {'b1_ms': b1[0], 'b1_p90_ms': b1[1], 'b1_1thread_ms': b1_t1[0], 'b256_ms': b256[0]}

def kernel_benchmark(sparsity, batch, repeats):
    """fc1-shaped product y = W x with W of shape 128 x 6272 (77% of the baseline's parameters):
    dense torch.mm vs CSR torch.sparse.mm on the same matrix."""
    torch.manual_seed(0)
    w = torch.randn(128, 6272)
    if sparsity > 0:
        thr = w.abs().flatten().kthvalue(int(sparsity * w.numel())).values
        w[w.abs() <= thr] = 0
    w_csr = w.to_sparse_csr()
    x = torch.randn(6272, batch)
    dense = c.measure_latency(lambda: torch.mm(w, x), warmup=20, repeats=repeats)[0]
    sparse = c.measure_latency(lambda: torch.sparse.mm(w_csr, x), warmup=20, repeats=repeats)[0]
    return dense, sparse

if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count())
    base = c.load_json('task1_baseline.json')
    prune_res = c.load_json('task2_pruning.json')
    acc_unstr = {r['target']: r['acc_after_ft'] for r in prune_res['unstructured'] if r['scope'] == 'global'}
    acc_str = {r['filter_fraction']: r['acc_after_ft'] for r in prune_res['structured']}

    # ---- 1. Model latency and storage ----
    rows = []
    for s in SPARSITIES:
        m = load_unstructured(s)
        dense_sd = c.dense_state_dict(m)
        row = {'model': f'Unstructured global {int(s * 100)}%', 'kind': 'unstructured', 'sparsity': c.weight_sparsity(m),
               'acc': acc_unstr[s], 'nonzero_params': c.count_params(m, nonzero=True),
               'dense_bytes': c.serialized_size_bytes(dense_sd),
               'coo_bytes': c.serialized_size_bytes(c.sparse_state_dict(m, 'coo')),
               'csr_bytes': c.serialized_size_bytes(c.sparse_state_dict(m, 'csr')),
               'gzip_bytes': gzip_size(dense_sd), 'macs': c.count_macs(m), 'effective_macs': c.effective_macs(m)}
        row.update(latencies(m))
        rows.append(row)
        print(f"{row['model']:<28} | b1 {row['b1_ms']:.3f} ms | b256 {row['b256_ms']:.1f} ms | "
              f"dense {row['dense_bytes'] / 2**20:.2f} MiB | CSR {row['csr_bytes'] / 2**20:.2f} MiB")
    for f in FILTER_FRACTIONS:
        m = load_structured(f)
        dense_sd = c.dense_state_dict(m)
        row = {'model': f'Structured {int(f * 100)}% filters', 'kind': 'structured', 'sparsity': 1 - c.count_params(m) / base['params'],
               'acc': acc_str[f], 'nonzero_params': c.count_params(m), 'dense_bytes': c.serialized_size_bytes(dense_sd),
               'coo_bytes': None, 'csr_bytes': None, 'gzip_bytes': gzip_size(dense_sd),
               'macs': c.count_macs(m), 'effective_macs': c.count_macs(m)}
        row.update(latencies(m))
        rows.append(row)
        print(f"{row['model']:<28} | b1 {row['b1_ms']:.3f} ms | b256 {row['b256_ms']:.1f} ms | dense {row['dense_bytes'] / 2**20:.2f} MiB")

    # ---- 2. Sparse kernel micro-benchmark (fc1 shape) ----
    kernel = []
    for s in KERNEL_SPARSITIES:
        d1, s1 = kernel_benchmark(s, batch=1, repeats=500)
        d256, s256 = kernel_benchmark(s, batch=256, repeats=100)
        kernel.append({'sparsity': s, 'dense_b1_ms': d1, 'csr_b1_ms': s1, 'dense_b256_ms': d256, 'csr_b256_ms': s256})
        print(f"kernel s={s:<4} | b1 dense {d1 * 1000:.1f} us, CSR {s1 * 1000:.1f} us | "
              f"b256 dense {d256:.3f} ms, CSR {s256:.3f} ms")

    c.save_json({'models': rows, 'kernel': kernel, 'threads': torch.get_num_threads()}, 'task3_latency_storage.json')

    # ---- Text report ----
    MiB = 2**20
    ref = rows[0]
    lines = ["=== TASK 3: LATENCY AND ON-DISK SIZE ===",
             f"CPU: {os.cpu_count()} cores, {torch.get_num_threads()} PyTorch threads, PyTorch {torch.__version__}. "
             "Latency = median of 500 (batch 1) / 40 (batch 256) timed forward passes after warm-up.",
             "Unstructured models are executed as ordinary dense PyTorch models (zeros are stored and multiplied).", "",
             "--- Latency ---",
             f"{'Model':<28} | {'Sparsity':<8} | {'Test acc':<8} | {'Dense MACs (M)':<14} | {'Non-zero MACs (M)':<17} | "
             f"{'b1 (ms)':<7} | {'b1 1-thr (ms)':<13} | {'b256 (ms)':<9} | Speed-up b256", "-" * 140]
    for r in rows:
        lines.append(f"{r['model']:<28} | {r['sparsity']:<8.3f} | {r['acc']:<8.4f} | {r['macs'] / 1e6:<14.1f} | "
                     f"{r['effective_macs'] / 1e6:<17.1f} | {r['b1_ms']:<7.3f} | {r['b1_1thread_ms']:<13.3f} | "
                     f"{r['b256_ms']:<9.2f} | {ref['b256_ms'] / r['b256_ms']:.2f}x")
    lines += ["", "--- On-disk size (torch.save of the state_dict) ---",
              "Dense = fp32 tensors with zeros stored. COO = int64 row+col index per non-zero + fp32 value. "
              "CSR = int64 col index per non-zero + fp32 value + row pointers. gzip = zlib level 9 of the dense file.",
              f"{'Model':<28} | {'Non-zero params':<15} | {'Dense (MiB)':<11} | {'COO (MiB)':<9} | {'CSR (MiB)':<9} | "
              f"{'gzip dense (MiB)':<16} | Best vs dense baseline", "-" * 130]
    for r in rows:
        sizes = [x for x in [r['dense_bytes'], r['coo_bytes'], r['csr_bytes'], r['gzip_bytes']] if x is not None]
        coo = f"{r['coo_bytes'] / MiB:.2f}" if r['coo_bytes'] else '-'
        csr = f"{r['csr_bytes'] / MiB:.2f}" if r['csr_bytes'] else '-'
        lines.append(f"{r['model']:<28} | {r['nonzero_params']:<15,} | {r['dense_bytes'] / MiB:<11.2f} | {coo:<9} | "
                     f"{csr:<9} | {r['gzip_bytes'] / MiB:<16.2f} | {ref['dense_bytes'] / min(sizes):.2f}x smaller")
    lines += ["", "--- Kernel micro-benchmark: fc1-shaped W (128 x 6272) times X (6272 x batch) ---",
              f"{'Sparsity':<8} | {'Dense b1 (us)':<13} | {'CSR b1 (us)':<11} | {'CSR/dense b1':<12} | "
              f"{'Dense b256 (ms)':<15} | {'CSR b256 (ms)':<13} | CSR/dense b256", "-" * 105]
    for k in kernel:
        lines.append(f"{k['sparsity']:<8} | {k['dense_b1_ms'] * 1000:<13.1f} | {k['csr_b1_ms'] * 1000:<11.1f} | "
                     f"{k['csr_b1_ms'] / k['dense_b1_ms']:<12.2f} | {k['dense_b256_ms']:<15.3f} | "
                     f"{k['csr_b256_ms']:<13.3f} | {k['csr_b256_ms'] / k['dense_b256_ms']:.2f}")
    c.write_txt(lines, 'task3_latency_storage.txt')

    # ---- Plots ----
    un = [r for r in rows if r['kind'] == 'unstructured']
    st = [r for r in rows if r['kind'] == 'structured']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    ax.plot([r['sparsity'] for r in un], [r['b256_ms'] for r in un], 'o-', label='Unstructured (dense kernels), batch 256')
    ax.plot([r['sparsity'] for r in st], [r['b256_ms'] for r in st], 's-', color='tab:green', label='Structured (smaller dense model), batch 256')
    ax.plot([0] + [r['sparsity'] for r in un[1:]], [ref['b256_ms'] * (1 - r['sparsity']) for r in un], ':', color='grey',
            label='Ideal: time proportional to non-zeros')
    ax.set_xlabel('Sparsity (fraction of weights removed)')
    ax.set_ylabel('CPU latency per batch of 256 (ms)')
    ax.set_title('Model latency vs sparsity')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    xs = [k['sparsity'] for k in kernel]
    ax.plot(xs, [k['csr_b256_ms'] / k['dense_b256_ms'] for k in kernel], 'o-', label='batch 256')
    ax.plot(xs, [k['csr_b1_ms'] / k['dense_b1_ms'] for k in kernel], 's-', label='batch 1')
    ax.axhline(1.0, color='grey', ls=':', label='same speed as dense')
    ax.set_yscale('log')
    ax.set_xlabel('Sparsity of the fc1 weight matrix')
    ax.set_ylabel('CSR time / dense time (log scale, <1 = sparse faster)')
    ax.set_title('Sparse (CSR) vs dense matmul, fc1 shape 128x6272')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    ax = axes[2]
    xs = [r['sparsity'] for r in un]
    ax.plot(xs, [r['dense_bytes'] / MiB for r in un], 'o-', label='Dense fp32 (zeros stored)')
    ax.plot(xs, [r['coo_bytes'] / MiB for r in un], 'o-', label='Sparse COO (int64 row+col indices)')
    ax.plot(xs, [r['csr_bytes'] / MiB for r in un], 'o-', label='Sparse CSR (int64 col indices)')
    ax.plot(xs, [r['gzip_bytes'] / MiB for r in un], 'o-', label='Dense + gzip')
    ax.plot([r['sparsity'] for r in st], [r['dense_bytes'] / MiB for r in st], 's-', color='tab:green', label='Structured (dense, smaller)')
    ax.set_xlabel('Sparsity (fraction of weights removed)')
    ax.set_ylabel('Size on disk (MiB)')
    ax.set_title('Storage format vs sparsity')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(c.RESULTS_DIR, 'latency_storage.png'), dpi=150)
    print("Saved latency_storage.png")
