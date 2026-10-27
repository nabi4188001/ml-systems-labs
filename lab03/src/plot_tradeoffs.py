import torch
import torch.nn as nn
import torchinfo
import time
import numpy as np
import matplotlib.pyplot as plt
import os
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
# PRE-FILLED WITH YOUR EXACT TASK 4 RESULTS
# ==========================================
ACCURACIES = {
    "MLP": 88.18,
    "CNN": 88.61,
    "Attention": 83.68
}

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
        return self.fc3(x)

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
        return self.fc(x)

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
        return self.classifier(x)

# ==========================================
# 2. Measurement Functions
# ==========================================
def get_params(model):
    info = torchinfo.summary(model, input_size=(1, 1, 28, 28), verbose=0)
    return info.total_params

def measure_latency(model, input_tensor, num_runs=100, warmup=1):
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            end = time.perf_counter()
            times.append(end - start)

    return np.median(times) * 1000

# ==========================================
# 3. Main Execution & Plotting
# ==========================================
if __name__ == "__main__":
    print("Gathering metrics for trade-off plots...")

    models = {
        "MLP": SimpleMLP(),
        "CNN": SimpleCNN(),
        "Attention": PatchAttention()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, 1, 28, 28).to(device)

    params_list = []
    latency_list = []
    acc_list = []
    names = []

    for name, model in models.items():
        model = model.to(device)
        p = get_params(model)
        lat = measure_latency(model, dummy_input, num_runs=100)
        acc = ACCURACIES[name]

        params_list.append(p)
        latency_list.append(lat)
        acc_list.append(acc)
        names.append(name)

        print(f"{name:<10} | Params: {p:>7,} | Latency (Batch 1): {lat:.4f} ms | Acc: {acc:.2f}%")

    os.makedirs('../results', exist_ok=True)

    # Plot 1: Accuracy vs. Parameters
    plt.figure(figsize=(8, 5))
    plt.scatter(params_list, acc_list, s=150, alpha=0.8)
    for i, name in enumerate(names):
        plt.annotate(name, (params_list[i], acc_list[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.xscale('log')
    plt.xlabel('Number of Parameters (log scale)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Accuracy vs. Model Size (Parameters)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('../results/acc_vs_params.png', dpi=300)
    print("\n✅ Saved: ../results/acc_vs_params.png")

    # Plot 2: Accuracy vs. Latency
    plt.figure(figsize=(8, 5))
    plt.scatter(latency_list, acc_list, s=150, alpha=0.8, color='orange')
    for i, name in enumerate(names):
        plt.annotate(name, (latency_list[i], acc_list[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.xscale('log')
    plt.xlabel('Inference Latency per sample, Batch 1 (ms, log scale)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Accuracy vs. Inference Latency')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('../results/acc_vs_latency.png', dpi=300)
    print("✅ Saved: ../results/acc_vs_latency.png")

    print("\n✅ Task 5 Plotting Complete!")