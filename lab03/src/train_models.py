import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# ==========================================
# Global Rules: Set seeds for reproducibility
# ==========================================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ==========================================
# 1. Data Loading (from Task 1)
# ==========================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
os.makedirs(data_dir, exist_ok=True)

full_train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

train_dataset, val_dataset = random_split(full_train_dataset, [55000, 5000], generator=torch.Generator().manual_seed(42))

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==========================================
# 2. Model Definitions (from Task 2)
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
# 3. Training and Evaluation Functions
# ==========================================
def train_and_evaluate(model, model_name, train_loader, test_loader, epochs=5, lr=1e-3):
    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n--- Training {model_name} on {device} for {epochs} epochs ---")

    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()       # Clear old gradients
            outputs = model(images)     # Forward pass
            loss = criterion(outputs, labels) # Compute loss
            loss.backward()             # Backward pass (compute gradients)
            optimizer.step()            # Update weights

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # Evaluation on Test Set
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad(): # Disable gradient tracking for inference
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print(f"✅ {model_name} Final Test Accuracy: {test_acc:.2f}%")
    return test_acc

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    models = {
        "MLP": SimpleMLP(),
        "CNN": SimpleCNN(),
        "Attention": PatchAttention()
    }

    results = {}
    for name, model in models.items():
        acc = train_and_evaluate(model, name, train_loader, test_loader, epochs=5, lr=1e-3)
        results[name] = acc

    print("\n" + "="*40)
    print("FINAL TEST ACCURACIES (5 Epochs)")
    print("="*40)
    for name, acc in results.items():
        print(f"{name:<10} : {acc:.2f}%")