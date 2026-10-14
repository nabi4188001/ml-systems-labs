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
# Network Initialization Function
# ==========================================
def initialize_network(W):
    """
    Initializes weights and biases for a 2-hidden-layer neural network.
    Architecture: Input(64) -> Hidden(W) -> ReLU -> Hidden(W) -> ReLU -> Output(10) -> Softmax

    Args:
        W (int): The width of the hidden layers (e.g., 16 or 64)

    Returns:
        dict: Dictionary containing initialized weights and biases
    """
    # Weight shapes: (64, W), (W, W), (W, 10)
    # Bias shapes: (W,), (W,), (10,)
    # Initialization: Normal distribution N(0, 0.01)

    W1 = np.random.normal(0, 0.01, (64, W))
    b1 = np.random.normal(0, 0.01, (W,))

    W2 = np.random.normal(0, 0.01, (W, W))
    b2 = np.random.normal(0, 0.01, (W,))

    W3 = np.random.normal(0, 0.01, (W, 10))
    b3 = np.random.normal(0, 0.01, (10,))

    params = {
        'W1': W1, 'b1': b1,
        'W2': W2, 'b2': b2,
        'W3': W3, 'b3': b3
    }
    return params

# ==========================================
# Sanity Checks
# ==========================================
if __name__ == "__main__":
    print("Testing Network Initialization for W = 16 and W = 64\n")

    for W in [16, 64]:
        print(f"--- Hidden Width W = {W} ---")
        params = initialize_network(W)

        # 1. Shape Checks
        print(f"W1 shape: {params['W1'].shape} (Expected: (64, {W}))")
        print(f"b1 shape: {params['b1'].shape} (Expected: ({W},))")
        print(f"W2 shape: {params['W2'].shape} (Expected: ({W}, {W}))")
        print(f"b2 shape: {params['b2'].shape} (Expected: ({W},))")
        print(f"W3 shape: {params['W3'].shape} (Expected: ({W}, 10))")
        print(f"b3 shape: {params['b3'].shape} (Expected: (10,))")

        # 2. Distribution Checks (Mean ~0, Std ~0.01)
        print(f"W1 mean: {params['W1'].mean():.5f} (Expected ~0.0)")
        print(f"W1 std:  {params['W1'].std():.5f} (Expected ~0.01)")
        print("-" * 45)