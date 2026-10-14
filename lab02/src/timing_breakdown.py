import numpy as np
import random
import torch
import time
import os
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ==========================================
# Global Rules: Set seeds for reproducibility
# ==========================================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ==========================================
# 1. Data Loading (from Part 1)
# ==========================================
X, y = load_digits(return_X_y=True)
X = X / 16.0
num_classes = 10
y_onehot = np.zeros((y.shape[0], num_classes))
y_onehot[np.arange(y.shape[0]), y] = 1
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.30, random_state=42, stratify=y)

# ==========================================
# 2. Network Functions (from Parts 2-5)
# ==========================================
def initialize_network(W):
    W1 = np.random.normal(0, 0.01, (64, W))
    b1 = np.random.normal(0, 0.01, (W,))
    W2 = np.random.normal(0, 0.01, (W, W))
    b2 = np.random.normal(0, 0.01, (W,))
    W3 = np.random.normal(0, 0.01, (W, 10))
    b3 = np.random.normal(0, 0.01, (10,))
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}

def relu(Z): return np.maximum(0, Z)
def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def forward_pass(X, params):
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    W3, b3 = params['W3'], params['b3']
    Z1 = np.dot(X, W1) + b1; A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2; A2 = relu(Z2)
    Z3 = np.dot(A2, W3) + b3; A3 = softmax(Z3)
    return A3, {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3}

def cross_entropy_loss(Y_hat, Y):
    m = Y.shape[0]
    Y_hat_clipped = np.clip(Y_hat, 1e-12, 1.0 - 1e-12)
    return -np.sum(Y * np.log(Y_hat_clipped)) / m

def backward_pass(X, Y, params, cache):
    m = X.shape[0]
    W1, W2, W3 = params['W1'], params['W2'], params['W3']
    Z1, A1, Z2, A2, A3 = cache['Z1'], cache['A1'], cache['Z2'], cache['A2'], cache['A3']
    dZ3 = A3 - Y
    dW3 = (1/m) * np.dot(A2.T, dZ3); db3 = (1/m) * np.sum(dZ3, axis=0)
    dA2 = np.dot(dZ3, W3.T); dZ2 = dA2 * (Z2 > 0)
    dW2 = (1/m) * np.dot(A1.T, dZ2); db2 = (1/m) * np.sum(dZ2, axis=0)
    dA1 = np.dot(dZ2, W2.T); dZ1 = dA1 * (Z1 > 0)
    dW1 = (1/m) * np.dot(X.T, dZ1); db1 = (1/m) * np.sum(dZ1, axis=0)
    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}

# ==========================================
# 3. Timing Breakdown Implementation
# ==========================================
if __name__ == "__main__":
    print("Starting Timing Breakdown (Median of 100 runs)...\n")

    # Use a single mini-batch of 32 for timing
    W = 64
    params = initialize_network(W)
    X_batch = X_train[:32]
    y_batch = y_train[:32]

    # 1. Warm-up run (required by Global Rules)
    _, cache = forward_pass(X_batch, params)
    _ = backward_pass(X_batch, y_batch, params, cache)

    # 2. Time Forward Pass (100 runs)
    forward_times = []
    for _ in range(100):
        start = time.perf_counter()
        _, _ = forward_pass(X_batch, params)
        end = time.perf_counter()
        forward_times.append(end - start)

    # 3. Time Backward Pass (100 runs)
    backward_times = []
    for _ in range(100):
        # Generate cache outside the timer so we ONLY time the backward math
        _, cache = forward_pass(X_batch, params)

        start = time.perf_counter()
        _ = backward_pass(X_batch, y_batch, params, cache)
        end = time.perf_counter()
        backward_times.append(end - start)

    # 4. Calculate Medians
    median_forward = np.median(forward_times) * 1000  # Convert to milliseconds
    median_backward = np.median(backward_times) * 1000

    print(f"Median Forward Pass Time:  {median_forward:.4f} ms")
    print(f"Median Backward Pass Time: {median_backward:.4f} ms")
    print(f"Ratio (Backward / Forward): {median_backward / median_forward:.2f}x")

    # Save results to a text file for easy copy-pasting into the report
    os.makedirs('../results', exist_ok=True)
    with open('../results/timing_breakdown.txt', 'w') as f:
        f.write(f"Median Forward Pass Time:  {median_forward:.4f} ms\n")
        f.write(f"Median Backward Pass Time: {median_backward:.4f} ms\n")
        f.write(f"Ratio (Backward / Forward): {median_backward / median_forward:.2f}x\n")

    print("\n✅ Timing results saved to ../results/timing_breakdown.txt")