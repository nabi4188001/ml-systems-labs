import torch
import torch.nn as nn
from torchinfo import summary
import random
import numpy as np

# ==========================================
# Global Rules: Set seeds for reproducibility
# ==========================================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ==========================================
# 1. MLP Architecture
# ==========================================
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# ==========================================
# 2. CNN Architecture
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        # 28x28 -> MaxPool(2) -> 14x14 -> MaxPool(2) -> 7x7. Channels = 32.
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

# ==========================================
# 3. Attention (Transformer) Architecture
# ==========================================
class PatchAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # Split 28x28 into 16 patches of 7x7 using a Conv2d with stride=7
        # Output will be (batch, 64, 4, 4) -> 16 patches, 64 features each
        self.patchify = nn.Conv2d(1, 64, kernel_size=7, stride=7)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True # Allows input shape (batch, seq_len, features)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Final classification head
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        # x shape: (batch, 1, 28, 28)
        x = self.patchify(x)             # (batch, 64, 4, 4)
        x = x.flatten(2)                 # (batch, 64, 16)
        x = x.transpose(1, 2)            # (batch, 16, 64) -> (batch, seq_len, d_model)

        x = self.transformer(x)          # (batch, 16, 64)

        # Mean pool over the sequence dimension (16 patches)
        x = x.mean(dim=1)                # (batch, 64)

        x = self.classifier(x)           # (batch, 10)
        return x

# ==========================================
# Sanity Checks & Parameter Counting
# ==========================================
if __name__ == "__main__":
    # Dummy input shape: (batch_size, channels, height, width)
    dummy_input = (1, 1, 28, 28)

    print("="*70)
    print("1. MLP Architecture")
    print("="*70)
    mlp = SimpleMLP()
    summary(mlp, input_size=dummy_input, col_names=["input_size", "output_size", "num_params"], row_settings=["var_names"])

    print("\n" + "="*70)
    print("2. CNN Architecture")
    print("="*70)
    cnn = SimpleCNN()
    summary(cnn, input_size=dummy_input, col_names=["input_size", "output_size", "num_params"], row_settings=["var_names"])

    print("\n" + "="*70)
    print("3. Attention (Transformer) Architecture")
    print("="*70)
    attn = PatchAttention()
    summary(attn, input_size=dummy_input, col_names=["input_size", "output_size", "num_params"], row_settings=["var_names"])

    print("\n✅ Task 2 Model Definitions Complete!")