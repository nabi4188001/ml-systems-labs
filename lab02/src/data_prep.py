import numpy as np
import random
import torch
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
# 1. Load the Dataset
# ==========================================
# load_digits returns 8x8 images (64 features) and labels (0-9)
X, y = load_digits(return_X_y=True)
print(f"Original X shape: {X.shape} | Original y shape: {y.shape}")

# ==========================================
# 2. Scale Inputs
# ==========================================
# Pixel values are integers from 0 to 16. Dividing by 16 scales them to [0.0, 1.0]
X = X / 16.0
print(f"Scaled X min: {X.min():.2f} | Scaled X max: {X.max():.2f}")

# ==========================================
# 3. One-Hot Encode Labels
# ==========================================
num_classes = 10
# Create an array of zeros with shape (num_samples, 10)
y_onehot = np.zeros((y.shape[0], num_classes))
# Set the appropriate index to 1 for each sample
y_onehot[np.arange(y.shape[0]), y] = 1

print(f"One-hot encoded y shape: {y_onehot.shape}")
print(f"Sample one-hot vector (first row): {y_onehot[0]}")
print(f"Sum of sample vector (should be 1.0): {y_onehot[0].sum()}")

# ==========================================
# 4. Stratified 70/30 Train/Test Split
# ==========================================
# We stratify by the original 1D 'y' array to ensure class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("-" * 30)
print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
print(f"X_test shape:  {X_test.shape}  | y_test shape:  {y_test.shape}")