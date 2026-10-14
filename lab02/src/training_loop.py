import numpy as np
import random
import torch
import matplotlib.pyplot as plt
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

def update_params(params, grads, lr):
    for key in params:
        params[key] -= lr * grads['d' + key]
    return params

# ==========================================
# 3. Training Loop Implementation
# ==========================================
def train_network(X_train, y_train, X_test, y_test, W, lr, epochs=50, batch_size=32):
    params = initialize_network(W)
    train_losses = []
    test_accs = []
    m = X_train.shape[0]

    for epoch in range(epochs):
        # 1. Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        # 2. Mini-batch training
        epoch_loss = 0
        num_batches = 0

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Forward
            Y_hat, cache = forward_pass(X_batch, params)
            loss = cross_entropy_loss(Y_hat, y_batch)
            epoch_loss += loss
            num_batches += 1

            # Backward & Update
            grads = backward_pass(X_batch, y_batch, params, cache)
            params = update_params(params, grads, lr)

        # 3. Per-epoch logging
        train_losses.append(epoch_loss / num_batches)

        # 4. Evaluate on test set
        Y_hat_test, _ = forward_pass(X_test, params)
        preds = np.argmax(Y_hat_test, axis=1)
        true_labels = np.argmax(y_test, axis=1)
        acc = np.mean(preds == true_labels)
        test_accs.append(acc)

    return train_losses, test_accs

# ==========================================
# 4. Run Experiments & Plot
# ==========================================
if __name__ == "__main__":
    learning_rates = [0.001, 0.01, 0.1]
    widths = [16, 64]
    results = {}

    print("Starting Hyperparameter Tuning...\n")

    for W in widths:
        for lr in learning_rates:
            print(f"Training W={W}, lr={lr}...")
            train_losses, test_accs = train_network(X_train, y_train, X_test, y_test, W, lr)
            results[(W, lr)] = {
                'train_losses': train_losses,
                'final_acc': test_accs[-1]
            }
            print(f"  -> Final Test Accuracy: {results[(W, lr)]['final_acc']:.4f}")

    # Plot loss curves
    plt.figure(figsize=(10, 6))
    for W in widths:
        for lr in learning_rates:
            plt.plot(results[(W, lr)]['train_losses'], label=f'W={W}, lr={lr}')

    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curves for Different Hyperparameters')
    plt.legend()
    plt.grid(True)

    # Save to results folder
    os.makedirs('../results', exist_ok=True)
    plt.savefig('../results/loss_curves.png', dpi=300, bbox_inches='tight')
    print("\n✅ Plot saved to ../results/loss_curves.png")

    # Print final accuracies table for the report
    print("\n--- Final Test Accuracies (Copy to report.md) ---")
    print(f"{'Width (W)':<10} | {'LR':<10} | {'Accuracy'}")
    print("-" * 35)
    for W in widths:
        for lr in learning_rates:
            print(f"{W:<10} | {lr:<10} | {results[(W, lr)]['final_acc']:.4f}")