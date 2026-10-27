import torch
import torch.nn as nn
import thop
import time
import numpy as np
import random

# ==========================================
# Global Rules: Set seeds for reproducibility
# ==========================================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ==========================================
# 1. Model Definitions (from Task 2)
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
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

class PatchAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.patchify = nn.Conv2d(1, 64, kernel_size=7, stride=7)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = self.patchify(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

# ==========================================
# 2. Hand-Calculating FLOPs (One Layer Each)
# ==========================================
def calculate_hand_flops():
    print("="*60)
    print("HAND-CALCULATED FLOPs (For One Representative Layer)")
    print("="*60)

    # MLP: First Linear Layer (784 -> 256)
    # Formula: 2 * in * out
    mlp_flops = 2 * 784 * 256
    print(f"1. MLP (Linear 784 -> 256):")
    print(f"   Formula: 2 * 784 * 256 = {mlp_flops:,} FLOPs\n")

    # CNN: First Conv2d Layer (1 -> 16, 3x3, 28x28 -> 28x28)
    # Formula: 2 * K * K * C_in * C_out * H_out * W_out
    cnn_flops = 2 * 3 * 3 * 1 * 16 * 28 * 28
    print(f"2. CNN (Conv2d 1->16, 3x3, 28x28 out):")
    print(f"   Formula: 2 * 3 * 3 * 1 * 16 * 28 * 28 = {cnn_flops:,} FLOPs\n")

    # Attention: One Transformer Encoder Layer (seq=16, d_model=64, ff=128)
    # QKV Projections: 2 * seq * d_model * 3 * d_model
    qkv = 2 * 16 * 64 * 3 * 64
    # Attention Scores (Q * K^T): 2 * seq^2 * d_model
    attn_scores = 2 * (16**2) * 64
    # Attention * V: 2 * seq^2 * d_model
    attn_v = 2 * (16**2) * 64
    # Feedforward (64 -> 128 -> 64): (2 * seq * 64 * 128) + (2 * seq * 128 * 64)
    ff = (2 * 16 * 64 * 128) + (2 * 16 * 128 * 64)

    attn_total = qkv + attn_scores + attn_v + ff
    print(f"3. Attention (One Transformer Layer, seq=16, d=64):")
    print(f"   QKV Projections: 2 * 16 * 64 * 3 * 64 = {qkv:,}")
    print(f"   Attn Scores:     2 * 16^2 * 64        = {attn_scores:,}")
    print(f"   Attn * V:        2 * 16^2 * 64        = {attn_v:,}")
    print(f"   Feedforward:     (2*16*64*128)*2      = {ff:,}")
    print(f"   TOTAL for 1 layer: {attn_total:,} FLOPs\n")

# ==========================================
# 3. thop.profile Comparison
# ==========================================
def profile_models():
    print("="*60)
    print("MEASURED FLOPs (using thop.profile)")
    print("="*60)

    models = {
        "MLP": SimpleMLP(),
        "CNN": SimpleCNN(),
        "Attention": PatchAttention()
    }

    dummy_input = torch.randn(1, 1, 28, 28)

    for name, model in models.items():
        model.eval()
        # thop returns (flops, params)
        flops, params = thop.profile(model, inputs=(dummy_input,), verbose=False)
        print(f"{name:<10} | Total FLOPs: {flops:>12,.0f} | Total Params: {params:>10,.0f}")
    print()

# ==========================================
# 4. Inference Latency Measurement
# ==========================================
def measure_latency(model, input_tensor, num_runs, warmup=1):
    model.eval()
    with torch.no_grad():
        # Warm-up run (Global Rule)
        for _ in range(warmup):
            _ = model(input_tensor)

        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            end = time.perf_counter()
            times.append(end - start)

    return np.median(times) * 1000 # Return in milliseconds

def test_latency():
    print("="*60)
    print("INFERENCE LATENCY (Median in ms)")
    print("="*60)

    models = {
        "MLP": SimpleMLP(),
        "CNN": SimpleCNN(),
        "Attention": PatchAttention()
    }

    input_b1 = torch.randn(1, 1, 28, 28)
    input_b128 = torch.randn(128, 1, 28, 28)

    print(f"{'Model':<10} | {'Batch 1 (100 runs)':<20} | {'Batch 128 (20 runs)':<20}")
    print("-" * 60)

    for name, model in models.items():
        lat_b1 = measure_latency(model, input_b1, num_runs=100)
        lat_b128 = measure_latency(model, input_b128, num_runs=20)
        print(f"{name:<10} | {lat_b1:<20.4f} ms | {lat_b128:<20.4f} ms")

    print("\n✅ Task 3 Compute Cost Complete!")

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    calculate_hand_flops()
    profile_models()
    test_latency()