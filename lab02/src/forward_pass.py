import numpy as np
import random
import torch

# ==========================================
# Global Rules: Set seeds for reproducibility
# ==========================================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ==========================================
# 1. Network Initialization (from Part 2)
# ==========================================
def initialize_network(W):
    W1 = np.random.normal(0, 0.01, (64, W))
    b1 = np.random.normal(0, 0.01, (W,))
    W2 = np.random.normal(0, 0.01, (W, W))
    b2 = np.random.normal(0, 0.01, (W,))
    W3 = np.random.normal(0, 0.01, (W, 10))
    b3 = np.random.normal(0, 0.01, (10,))
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}

# ==========================================
# 2. Activation Functions
# ==========================================
def relu(Z):
    """ReLU activation: max(0, Z)"""
    return np.maximum(0, Z)

def softmax(Z):
    """Softmax activation with numerical stability."""
    # Subtract the max value in each row to prevent exponential overflow
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# ==========================================
# 3. Forward Pass Implementation
# ==========================================
def forward_pass(X, params):
    """
    Computes the forward pass layer by layer.

    Args:
        X: Input data of shape (batch_size, 64)
        params: Dictionary of weights and biases

    Returns:
        A3: Output probabilities of shape (batch_size, 10)
        cache: Dictionary of intermediate values for backpropagation
    """
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    W3, b3 = params['W3'], params['b3']

    # Layer 1: Input -> Hidden 1
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    # Layer 2: Hidden 1 -> Hidden 2
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)

    # Layer 3: Hidden 2 -> Output
    Z3 = np.dot(A2, W3) + b3
    A3 = softmax(Z3)

    # Cache intermediate values for backpropagation (Step 5)
    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3}

    return A3, cache

# ==========================================
# Sanity Checks
# ==========================================
if __name__ == "__main__":
    print("Testing Forward Pass Sanity Checks\n")

    # Create dummy input data: batch of 5 samples, 64 features each
    batch_size = 5
    X_dummy = np.random.rand(batch_size, 64)

    # Initialize network with W=16
    params = initialize_network(W=16)

    # Run forward pass
    output, cache = forward_pass(X_dummy, params)

    # Check 1: Output shape is (batch, 10)
    print(f"1. Output shape: {output.shape} (Expected: ({batch_size}, 10))")
    assert output.shape == (batch_size, 10), "Shape check failed!"

    # Check 2: Each row sums to 1 within 1e-6
    row_sums = np.sum(output, axis=1)
    print(f"2. Row sums: {row_sums}")

    # Check if all sums are within 1e-6 of 1.0
    is_valid = np.all(np.abs(row_sums - 1.0) < 1e-6)
    print(f"3. All rows sum to 1 (within 1e-6)? {is_valid}")
    assert is_valid, "Row sum check failed!"

    print("\n✅ All forward pass sanity checks passed!")