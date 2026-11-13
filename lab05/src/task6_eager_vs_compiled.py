import os
import re
import time
import random
import numpy as np
import torch
import torch.nn as nn

# ==========================================
# Global Rules: Set seeds for reproducibility
# ==========================================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(1)  # Single thread for stable timings

lines = []
def log(text=""):
    print(text)
    lines.append(text)

def median_time_ms(fn, runs, warmup=10):
    """Median wall time of fn() in milliseconds, measured after warm-up runs."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return np.median(times) * 1000

# ==========================================
# 1. Inspect the Graph of a Small Model
# ==========================================
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)
x_small = torch.randn(4, 2)
y_small = torch.randint(0, 2, (4, 1)).float()

log("=== 1. GRAPH INSPECTION (model: 2 -> 8 -> 1, sigmoid, BCE loss) ===")
log(f"PyTorch version: {torch.__version__} | CPU threads: {torch.get_num_threads()}")
log()

# 1a. Dynamic autograd graph, built during the eager forward pass
loss = nn.functional.binary_cross_entropy(model(x_small), y_small)

def walk_graph(fn, depth=0, seen=None):
    seen = set() if seen is None else seen
    if fn is None or fn in seen:
        return
    seen.add(fn)
    log("  " * depth + type(fn).__name__)
    for next_fn, _ in fn.next_functions:
        walk_graph(next_fn, depth + 1, seen)

log("Autograd graph (walking loss.grad_fn backwards):")
walk_graph(loss.grad_fn)
log()

# 1b. Static graph recorded by torch.jit.trace
traced_small = torch.jit.trace(model, x_small)
log("TorchScript graph recorded by torch.jit.trace:")
# Drop the trailing "# file:line" source comments to keep the printout readable
log(re.sub(r" # \S+", "", str(traced_small.inlined_graph)).strip())
log()

# ==========================================
# 2. Eager vs Traced vs Compiled (after warm-up)
# ==========================================
log("=== 2. EAGER vs torch.jit.trace vs torch.compile (inference, median of 200 runs after warm-up) ===")

def make_mlp(n_in, hidden, n_out):
    return nn.Sequential(
        nn.Linear(n_in, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, n_out),
    ).eval()

configs = [
    ("Small MLP 2->8->8->1, batch 1", make_mlp(2, 8, 1), torch.randn(1, 2)),
    ("Small MLP 2->8->8->1, batch 256", make_mlp(2, 8, 1), torch.randn(256, 2)),
    ("Large MLP 256->1024->1024->10, batch 256", make_mlp(256, 1024, 10), torch.randn(256, 256)),
]

log(f"{'Model / input':<42} | {'Eager (ms)':<10} | {'Traced (ms)':<11} | {'Compiled (ms)':<13} | {'Eager/Traced':<12} | {'Eager/Compiled'}")
log("-" * 115)

def interleaved_median_ms(fns, runs=200, warmup=20):
    """Times several callables in alternating order so machine noise affects all of them equally."""
    for _ in range(warmup):
        for fn in fns:
            fn()
    times = [[] for _ in fns]
    for _ in range(runs):
        for i, fn in enumerate(fns):
            start = time.perf_counter()
            fn()
            times[i].append(time.perf_counter() - start)
    return [np.median(t) * 1000 for t in times]

compile_status = "torch.compile used (default backend: inductor)"
with torch.no_grad():
    for name, net, inp in configs:
        traced = torch.jit.freeze(torch.jit.trace(net, inp))
        variants = [lambda: net(inp), lambda: traced(inp)]

        # torch.compile needs a working compiler toolchain; fall back to tracing only if unsupported
        try:
            compiled = torch.compile(net)
            compiled(inp)  # First call triggers compilation
            variants.append(lambda: compiled(inp))
        except Exception as e:
            compile_status = f"torch.compile unsupported here ({type(e).__name__}); torch.jit.trace used instead"

        results = interleaved_median_ms(variants)
        eager_ms, traced_ms = results[0], results[1]
        if len(results) == 3:
            compiled_str = f"{results[2]:<13.4f}"
            ratio_compiled = f"{eager_ms / results[2]:.2f}x"
        else:
            compiled_str = f"{'n/a':<13}"
            ratio_compiled = "n/a"

        log(f"{name:<42} | {eager_ms:<10.4f} | {traced_ms:<11.4f} | {compiled_str} | {f'{eager_ms / traced_ms:.2f}x':<12} | {ratio_compiled}")

log("-" * 115)
log(f"Note: {compile_status}")
log()

# ==========================================
# 3. Tensor Shape, Dtype, Contiguity
# ==========================================
log("=== 3. TENSOR INSPECTION ===")
base = torch.randn(2000, 2000)
view = base.t()  # Transposed view: same memory, swapped strides -> non-contiguous
copy = view.contiguous()  # Same values as the view, laid out contiguously

log(f"{'Tensor':<26} | {'shape':<12} | {'dtype':<13} | {'strides':<12} | {'contiguous':<10} | shares memory with base")
log("-" * 105)
for name, t in [("base = randn(2000, 2000)", base), ("view = base.t()", view), ("copy = view.contiguous()", copy)]:
    shares = t.data_ptr() == base.data_ptr()
    log(f"{name:<26} | {str(tuple(t.shape)):<12} | {str(t.dtype):<13} | {str(t.stride()):<12} | {str(t.is_contiguous()):<10} | {shares}")
log()

# ==========================================
# 4. Cost of a Non-Contiguous View
# ==========================================
log("=== 4. NON-CONTIGUOUS VIEW vs CONTIGUOUS COPY (same values, median of 50 runs after warm-up) ===")
operations = [
    ("x.sum()", lambda t: t.sum()),
    ("x * 2.0", lambda t: t * 2.0),
    ("torch.exp(x)", lambda t: torch.exp(t)),
    ("x.cumsum(dim=1)", lambda t: t.cumsum(dim=1)),
    ("x.reshape(-1)", lambda t: t.reshape(-1)),
]

log(f"{'Operation':<18} | {'Contiguous (ms)':<15} | {'Non-contiguous (ms)':<19} | Slowdown")
log("-" * 70)
for name, op in operations:
    contiguous_ms = median_time_ms(lambda: op(copy), runs=50, warmup=5)
    view_ms = median_time_ms(lambda: op(view), runs=50, warmup=5)
    log(f"{name:<18} | {contiguous_ms:<15.4f} | {view_ms:<19.4f} | {view_ms / contiguous_ms:.2f}x")
log("-" * 70)
log("Note: reshape(-1) is a free view on the contiguous tensor but must copy all 4M elements on the non-contiguous one.")

# ==========================================
# Save Results
# ==========================================
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, 'eager_vs_compiled.txt'), 'w') as f:
    f.write("\n".join(lines) + "\n")

print("\n✅ Results saved to ../results/eager_vs_compiled.txt")
