# Lab 5: Frameworks and Automatic Differentiation

## 1. Goal
The goal of this lab is to build a scalar automatic-differentiation engine from scratch (graph construction during the forward pass and topological backpropagation with gradient accumulation), validate its gradients against PyTorch autograd and numerical estimates, train a tiny model with it, and then use PyTorch to assess eager versus traced/compiled execution and the cost of non-contiguous tensors.

## 2. Method
- **Value node** (`src/autograd_engine.py`): Each `Value` stores `data`, `grad`, its parent nodes (`_prev`) and a local derivative rule (`_backward`) that pushes its gradient to the parents.
- **Operations**: Addition, multiplication, constant powers and ReLU each create a new `Value` linked to its inputs, so the computation graph is built during the forward pass. Sigmoid and log were added for the BCE model; negation, subtraction and division are expressed through the primitives.
- **Backward pass**: The graph is topologically sorted (iterative DFS), the output gradient is set to 1, nodes are visited in reverse order, and every local rule uses `+=` so gradients accumulate instead of being overwritten.
- **Gradient check** (`src/task4_gradient_check.py`): `((a*b + c)**2).relu().sum()` was evaluated on vectors of 10 random float64 values (seed 42). Engine gradients were compared with PyTorch autograd and with central-difference numerical gradients ($\epsilon = 10^{-6}$).
- **Tiny model** (`src/task5_train_moons.py`): `make_moons(n_samples=100, noise=0.1, random_state=42)`, architecture 2 → 8 (ReLU) → 1 (sigmoid), BCE loss, full-batch gradient descent with lr = 1.0 for 100 epochs. The engine and PyTorch (float64) start from the same initial weights. Training time is the median of 5 runs after 1 warm-up run.
- **Eager vs compiled** (`src/task6_eager_vs_compiled.py`): The autograd graph of the 2 → 8 → 1 model was inspected through `loss.grad_fn`, and its static graph through `torch.jit.trace`. Inference of eager, `torch.jit.trace` (frozen) and `torch.compile` versions was timed on one thread after warm-up (median of 200 interleaved runs) for a small and a large MLP.
- **Tensor contiguity**: A 2000×2000 float32 tensor, its transposed view (`.t()`) and a contiguous copy of that view were inspected (shape, dtype, strides, contiguity), and five operations were timed on the view and on the copy (median of 50 runs).

## 3. Results

**Gradient Check** (`results/gradient_check.txt`):
| Comparison | Max abs difference |
| :--- | :---: |
| Engine vs PyTorch autograd | 0.00e+00 |
| Engine vs numerical gradients | 3.66e-09 |

Both are below the required $10^{-6}$.

**Tiny Model on `make_moons`** (`results/moons_training.txt`):
| Implementation | Final Loss | Train Accuracy | Median Train Time (s) |
| :--- | :---: | :---: | :---: |
| Autograd engine | 0.234912 | 0.8700 | 7.5998 |
| PyTorch | 0.234912 | 0.8700 | 0.0302 |

The loss curves of the two implementations are identical (max abs difference 2.78e-16 over all epochs). The engine is about 252x slower than PyTorch.

**Eager vs Traced vs Compiled** (`results/eager_vs_compiled.txt`, median of 200 runs):
| Model / input | Eager (ms) | Traced (ms) | Compiled (ms) | Eager / Traced | Eager / Compiled |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Small MLP 2→8→8→1, batch 1 | 0.0239 | 0.0132 | 0.0235 | 1.81x | 1.02x |
| Small MLP 2→8→8→1, batch 256 | 0.0483 | 0.0322 | 0.0766 | 1.50x | 0.63x |
| Large MLP 256→1024→1024→10, batch 256 | 6.1840 | 6.2053 | 6.3213 | 1.00x | 0.98x |

**Tensor Inspection**:
| Tensor | Shape | Dtype | Strides | Contiguous |
| :--- | :---: | :---: | :---: | :---: |
| `base = randn(2000, 2000)` | (2000, 2000) | float32 | (2000, 1) | True |
| `view = base.t()` | (2000, 2000) | float32 | (1, 2000) | False |
| `copy = view.contiguous()` | (2000, 2000) | float32 | (2000, 1) | True |

**Non-Contiguous View vs Contiguous Copy** (median of 50 runs):
| Operation | Contiguous (ms) | Non-contiguous (ms) | Slowdown |
| :--- | :---: | :---: | :---: |
| `x.sum()` | 0.7258 | 0.7271 | 1.00x |
| `x * 2.0` | 1.4170 | 1.4048 | 0.99x |
| `torch.exp(x)` | 1.7571 | 1.7594 | 1.00x |
| `x.cumsum(dim=1)` | 3.3284 | 12.9560 | 3.89x |
| `x.reshape(-1)` | 0.0017 | 10.4505 | 6120.35x |

## 4. Conclusions
1. **The engine is correct but slow**: Its gradients match PyTorch autograd exactly and numerical gradients to 3.66e-09, and training from the same weights reproduces PyTorch's loss curve to 2.78e-16. It is still about 252x slower, because every scalar operation creates a Python object and a closure, while PyTorch runs the same math as a few vectorized tensor kernels. This is the framework overhead of a per-scalar graph.
2. **When compilation helps**: Tracing helps when the model is small and the run time is dominated by framework overhead (Python dispatch and module calls): the small MLP was 1.5–1.8x faster after `torch.jit.trace`. For the large MLP all three versions took the same time (1.00x and 0.98x), because almost all of it is spent inside the matrix-multiplication kernels, which compilation does not change. `torch.compile` gave no gain on these CPU MLPs and was slower on the small model at batch 256 (0.63x), so its guard checks and generated code are not free. Compilation therefore pays off for many small operations that can be fused or stripped of Python overhead and that run many times after warm-up; it does not pay off when a few large kernels dominate, or when the one-time compilation cost cannot be amortized.
3. **Non-contiguous views are free to create but can be expensive to use**: The transposed view shares memory with the base tensor and only swaps the strides. Elementwise operations and `sum()` were unaffected (≈1.00x), since PyTorch can iterate in memory order regardless of the logical layout. Operations that must follow the logical row order pay for it: `cumsum(dim=1)` was 3.89x slower because of strided, cache-unfriendly access, and `reshape(-1)` went from a free view (0.0017 ms) to a full copy of 4M elements (10.45 ms). Calling `.contiguous()` once before such operations avoids paying this cost repeatedly.
