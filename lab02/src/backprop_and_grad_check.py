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
# 2. Activation Functions (from Part 3)
# ==========================================
def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# ==========================================
# 3. Forward Pass (from Part 3)
# ==========================================
def forward_pass(X, params):
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    W3, b3 = params['W3'], params['b3']

    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = softmax(Z3)

    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3}
    return A3, cache

# ==========================================
# 4. Cross-Entropy Loss (from Part 4)
# ==========================================
def cross_entropy_loss(Y_hat, Y):
    m = Y.shape[0]
    Y_hat_clipped = np.clip(Y_hat, 1e-12, 1.0 - 1e-12)
    loss = -np.sum(Y * np.log(Y_hat_clipped)) / m
    return loss

# ==========================================
# 5. Backward Pass Implementation (Analytical Gradients)
# ==========================================
def backward_pass(X, Y, params, cache):
    """
    Computes the gradients of the loss with respect to the parameters
    using the chain rule.
    """
    m = X.shape[0]
    W1, W2, W3 = params['W1'], params['W2'], params['W3']
    Z1, A1, Z2, A2, A3 = cache['Z1'], cache['A1'], cache['Z2'], cache['A2'], cache['A3']

    # Output layer (Softmax + Cross-Entropy combined derivative: dL/dZ3 = A3 - Y)
    dZ3 = A3 - Y
    dW3 = (1/m) * np.dot(A2.T, dZ3)
    db3 = (1/m) * np.sum(dZ3, axis=0)

    # Hidden layer 2 (ReLU derivative: 1 if Z2 > 0 else 0)
    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * (Z2 > 0)
    dW2 = (1/m) * np.dot(A1.T, dZ2)
    db2 = (1/m) * np.sum(dZ2, axis=0)

    # Hidden layer 1 (ReLU derivative)
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = (1/m) * np.dot(X.T, dZ1)
    db1 = (1/m) * np.sum(dZ1, axis=0)

    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}
    return grads

# ==========================================
# 6. Numerical Gradient Checking
# ==========================================
def gradient_check(X, Y, params, epsilon=1e-5):
    """
    Nudges 10 random weights by ±epsilon and compares analytic vs numerical slopes.
    """
    # 1. Get analytic gradients
    _, cache = forward_pass(X, params)
    grads = backward_pass(X, Y, params, cache)

    # 2. Select 10 random indices from W1 to check
    np.random.seed(42) # For reproducible random indices
    rows = np.random.randint(0, params['W1'].shape[0], 10)
    cols = np.random.randint(0, params['W1'].shape[1], 10)

    print(f"{'Index':<10} | {'Analytic':<15} | {'Numeric':<15} | {'Rel Error':<15} | {'Pass?'}")
    print("-" * 75)

    passes = 0
    for i in range(10):
        r, c = rows[i], cols[i]

        # Analytic gradient
        analytic_val = grads['dW1'][r, c]

        # Numerical gradient (+epsilon)
        params['W1'][r, c] += epsilon
        loss_plus = cross_entropy_loss(forward_pass(X, params)[0], Y)

        # Numerical gradient (-epsilon)
        params['W1'][r, c] -= 2 * epsilon
        loss_minus = cross_entropy_loss(forward_pass(X, params)[0], Y)

        # Reset weight to original value
        params['W1'][r, c] += epsilon

        numeric_val = (loss_plus - loss_minus) / (2 * epsilon)

        # Relative error calculation
        rel_error = np.abs(analytic_val - numeric_val) / max(np.abs(analytic_val) + np.abs(numeric_val), 1e-8)

        passed = rel_error < 1e-6
        if passed:
            passes += 1

        print(f"({r},{c})      | {analytic_val:<15.8f} | {numeric_val:<15.8f} | {rel_error:<15.2e} | {'✅' if passed else '❌'}")

    print("-" * 75)
    print(f"Passed {passes}/10 checks. (Requirement: >= 9/10)")

    if passes >= 9:
        print("✅ Gradient check passed!")
    else:
        print("❌ Gradient check failed. Check your backward pass derivations.")

    return passes >= 9

# ==========================================
# Sanity Checks
# ==========================================
if __name__ == "__main__":
    print("Testing Backpropagation and Gradient Checking\n")

    # Create dummy input data
    batch_size = 32
    X_dummy = np.random.rand(batch_size, 64)

    # Create random one-hot labels
    Y_dummy = np.zeros((batch_size, 10))
    random_labels = np.random.randint(0, 10, batch_size)
    Y_dummy[np.arange(batch_size), random_labels] = 1

    # Initialize network with W=16
    params = initialize_network(W=16)

    # Run gradient check
    gradient_check(X_dummy, Y_dummy, params)