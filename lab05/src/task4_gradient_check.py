import os
import random
import numpy as np
import torch
from autograd_engine import Value

# ==========================================
# Global Rules: Set seeds for reproducibility
# ==========================================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ==========================================
# 1. Expression under test: ((a*b + c)**2).relu().sum()
# ==========================================
N = 10
a_np = np.random.randn(N)
b_np = np.random.randn(N)
c_np = np.random.randn(N)

def expression_engine(a, b, c):
    """Builds the expression with Value nodes (a, b, c are lists of Value)."""
    return sum(((ai * bi + ci) ** 2).relu() for ai, bi, ci in zip(a, b, c))

def expression_numpy(a, b, c):
    """Plain forward evaluation, used for the numerical gradients."""
    return np.sum(np.maximum(0, (a * b + c) ** 2))

# ==========================================
# 2. Gradients from our engine
# ==========================================
a_val = [Value(v) for v in a_np]
b_val = [Value(v) for v in b_np]
c_val = [Value(v) for v in c_np]
out = expression_engine(a_val, b_val, c_val)
out.backward()
engine_grads = np.array([[v.grad for v in vals] for vals in (a_val, b_val, c_val)])

# ==========================================
# 3. Gradients from PyTorch autograd
# ==========================================
a_t = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
b_t = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)
c_t = torch.tensor(c_np, dtype=torch.float64, requires_grad=True)
out_t = ((a_t * b_t + c_t) ** 2).relu().sum()
out_t.backward()
torch_grads = np.array([a_t.grad.numpy(), b_t.grad.numpy(), c_t.grad.numpy()])

# ==========================================
# 4. Numerical gradients (central differences)
# ==========================================
def numerical_gradients(inputs, epsilon=1e-6):
    grads = np.zeros((3, N))
    for k in range(3):
        for i in range(N):
            original = inputs[k][i]
            inputs[k][i] = original + epsilon
            f_plus = expression_numpy(*inputs)
            inputs[k][i] = original - epsilon
            f_minus = expression_numpy(*inputs)
            inputs[k][i] = original
            grads[k, i] = (f_plus - f_minus) / (2 * epsilon)
    return grads

numeric_grads = numerical_gradients([a_np.copy(), b_np.copy(), c_np.copy()])

# ==========================================
# 5. Compare
# ==========================================
diff_torch = np.max(np.abs(engine_grads - torch_grads))
diff_numeric = np.max(np.abs(engine_grads - numeric_grads))
passed = diff_torch < 1e-6 and diff_numeric < 1e-6

lines = []
lines.append("=== GRADIENT CHECK: ((a*b + c)**2).relu().sum() ===")
lines.append(f"Inputs: a, b, c are vectors of {N} random values (seed 42), float64")
lines.append("")
lines.append(f"Forward value (engine):  {out.data:.10f}")
lines.append(f"Forward value (PyTorch): {out_t.item():.10f}")
lines.append("")
lines.append(f"{'Input':<6} | {'Engine':<15} | {'PyTorch':<15} | {'Numerical':<15}")
lines.append("-" * 60)
for k, name in enumerate(['a', 'b', 'c']):
    for i in range(N):
        lines.append(f"{name}[{i}]   | {engine_grads[k, i]:<15.8f} | {torch_grads[k, i]:<15.8f} | {numeric_grads[k, i]:<15.8f}")
lines.append("-" * 60)
lines.append(f"Max abs difference (engine vs PyTorch autograd): {diff_torch:.2e}")
lines.append(f"Max abs difference (engine vs numerical):        {diff_numeric:.2e}")
lines.append("Requirement: max abs difference < 1e-6")
lines.append(f"Result: {'PASSED' if passed else 'FAILED'}")

report = "\n".join(lines)
print(report)

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, 'gradient_check.txt'), 'w') as f:
    f.write(report + "\n")

if passed:
    print("\n✅ Gradient check passed! Results saved to ../results/gradient_check.txt")
else:
    print("\n❌ Gradient check failed.")
