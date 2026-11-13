import os
import time
import random
import numpy as np
import torch
from sklearn.datasets import make_moons
from autograd_engine import Value

# ==========================================
# Global Rules: Set seeds for reproducibility
# ==========================================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ==========================================
# 1. Data and Hyperparameters
# ==========================================
N_SAMPLES = 100
HIDDEN = 8
EPOCHS = 100
LR = 1.0
N_RUNS = 5

X, y = make_moons(n_samples=N_SAMPLES, noise=0.1, random_state=42)

# Shared initial weights so both implementations start from the same point
# Architecture: 2 -> 8 (ReLU) -> 1 (sigmoid), BCE loss
init = {
    'W1': np.random.randn(HIDDEN, 2) * 0.5,
    'b1': np.zeros(HIDDEN),
    'W2': np.random.randn(1, HIDDEN) * 0.5,
    'b2': np.zeros(1),
}

# ==========================================
# 2. Training with our scalar autograd engine
# ==========================================
def train_engine():
    W1 = [[Value(w) for w in row] for row in init['W1']]
    b1 = [Value(b) for b in init['b1']]
    W2 = [Value(w) for w in init['W2'][0]]
    b2 = Value(init['b2'][0])
    params = [w for row in W1 for w in row] + b1 + W2 + [b2]

    def forward(x):
        hidden = [(W1[j][0] * x[0] + W1[j][1] * x[1] + b1[j]).relu() for j in range(HIDDEN)]
        return (sum(W2[j] * hidden[j] for j in range(HIDDEN)) + b2).sigmoid()

    losses = []
    for epoch in range(EPOCHS):
        # Forward: full-batch BCE loss (builds a fresh graph every epoch)
        loss = Value(0.0)
        for xi, yi in zip(X, y):
            p = forward(xi)
            loss = loss + (p.log() if yi == 1 else (1 - p).log())
        loss = loss * (-1.0 / N_SAMPLES)

        # Backward
        for p in params:
            p.grad = 0.0
        loss.backward()

        # SGD update
        for p in params:
            p.data -= LR * p.grad
        losses.append(loss.data)

    preds = np.array([forward(xi).data > 0.5 for xi in X])
    return losses, np.mean(preds == y)

# ==========================================
# 3. Training with PyTorch (same init, same loop)
# ==========================================
X_t = torch.tensor(X, dtype=torch.float64)
y_t = torch.tensor(y, dtype=torch.float64).unsqueeze(1)

def train_pytorch():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, 1),
        torch.nn.Sigmoid(),
    ).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(init['W1']))
        model[0].bias.copy_(torch.tensor(init['b1']))
        model[2].weight.copy_(torch.tensor(init['W2']))
        model[2].bias.copy_(torch.tensor(init['b2']))

    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    loss_fn = torch.nn.BCELoss()

    losses = []
    for epoch in range(EPOCHS):
        loss = loss_fn(model(X_t), y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    with torch.no_grad():
        preds = (model(X_t) > 0.5).double()
    return losses, (preds == y_t).double().mean().item()

# ==========================================
# 4. Timing: 1 warm-up run, then median of 5 runs
# ==========================================
def time_training(train_fn):
    train_fn()  # Warm-up
    times = []
    for _ in range(N_RUNS):
        start = time.perf_counter()
        losses, acc = train_fn()
        times.append(time.perf_counter() - start)
    return times, losses, acc

if __name__ == "__main__":
    print("Training 2 -> 8 -> 1 model on make_moons...\n")

    engine_times, engine_losses, engine_acc = time_training(train_engine)
    torch_times, torch_losses, torch_acc = time_training(train_pytorch)

    engine_median = np.median(engine_times)
    torch_median = np.median(torch_times)
    loss_diff = np.max(np.abs(np.array(engine_losses) - np.array(torch_losses)))

    lines = []
    lines.append("=== TINY MODEL ON make_moons: ENGINE vs PYTORCH ===")
    lines.append(f"Data: make_moons(n_samples={N_SAMPLES}, noise=0.1, random_state=42)")
    lines.append(f"Architecture: 2 -> {HIDDEN} (ReLU) -> 1 (sigmoid), BCE loss")
    lines.append(f"Training: full-batch gradient descent, lr={LR}, {EPOCHS} epochs, float64, same initial weights")
    lines.append("")
    lines.append(f"{'Epoch':<6} | {'Engine loss':<12} | {'PyTorch loss':<12}")
    lines.append("-" * 38)
    for epoch in [0, 9, 24, 49, 74, 99]:
        lines.append(f"{epoch + 1:<6} | {engine_losses[epoch]:<12.6f} | {torch_losses[epoch]:<12.6f}")
    lines.append("-" * 38)
    lines.append(f"Max abs loss difference over all epochs: {loss_diff:.2e}")
    lines.append(f"Final train accuracy (engine):  {engine_acc:.4f}")
    lines.append(f"Final train accuracy (PyTorch): {torch_acc:.4f}")
    lines.append("")
    lines.append(f"Training time, {N_RUNS} runs after 1 warm-up (s):")
    lines.append(f"  Engine:  {', '.join(f'{t:.4f}' for t in engine_times)}")
    lines.append(f"  PyTorch: {', '.join(f'{t:.4f}' for t in torch_times)}")
    lines.append(f"Median training time (engine):  {engine_median:.4f} s")
    lines.append(f"Median training time (PyTorch): {torch_median:.4f} s")
    lines.append(f"Ratio (Engine / PyTorch): {engine_median / torch_median:.1f}x")

    report = "\n".join(lines)
    print(report)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'moons_training.txt'), 'w') as f:
        f.write(report + "\n")

    print("\n✅ Results saved to ../results/moons_training.txt")
