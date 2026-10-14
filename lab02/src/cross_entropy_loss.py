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
# 4. Cross-Entropy Loss Implementation
# ==========================================
def cross_entropy_loss(Y_hat, Y):
    """
    Computes the cross-entropy loss.

    Args:
        Y_hat: Predicted probabilities from softmax, shape (batch_size, 10)
        Y: True one-hot encoded labels, shape (batch_size, 10)

    Returns:
        loss: Scalar value of the average cross-entropy loss
    """
    m = Y.shape[0]

    # IMPORTANT: Clip predictions to prevent log(0) which results in -inf
    Y_hat_clipped = np.clip(Y_hat, 1e-12, 1.0 - 1e-12)

    # Compute cross-entropy loss: -1/m * sum(Y * log(Y_hat))
    loss = -np.sum(Y * np.log(Y_hat_clipped)) / m

    return loss

# ==========================================
# Sanity Checks
# ==========================================
if __name__ == "__main__":
    print("Testing Cross-Entropy Loss Sanity Check\n")

    # Create dummy input data
    batch_size = 100
    X_dummy = np.random.rand(batch_size, 64)

    # Create random one-hot labels to simulate a balanced dataset
    Y_dummy = np.zeros((batch_size, 10))
    random_labels = np.random.randint(0, 10, batch_size)
    Y_dummy[np.arange(batch_size), random_labels] = 1

    # Initialize network with W=16 (untrained)
    params = initialize_network(W=16)

    # Run forward pass
    Y_hat, _ = forward_pass(X_dummy, params)

    # Calculate loss
    loss = cross_entropy_loss(Y_hat, Y_dummy)
    expected_loss = np.log(10)

    print(f"Untrained network loss: {loss:.4f}")
    print(f"Expected loss (ln(10)): {expected_loss:.4f}")

    # Check if it's close to ln(10) (allowing a tiny variance due to random weights)
    is_valid = np.abs(loss - expected_loss) < 0.05

    if is_valid:
        print("\n✅ Cross-entropy loss sanity check passed!")
    else:
        print("\n❌ Sanity check failed. Check your loss implementation.")