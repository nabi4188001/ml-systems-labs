import os
import io
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

# ==========================================
# Paths, seeds and devices
# ==========================================
LAB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPO_ROOT = os.path.abspath(os.path.join(LAB_DIR, '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
RESULTS_DIR = os.path.join(LAB_DIR, 'results')
CKPT_DIR = os.path.join(LAB_DIR, 'checkpoints')      # not committed, recreated by task1
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# Training runs on the Apple GPU (MPS) when available, only to save wall-clock time.
# Every latency number in this lab is measured on the CPU, where the question
# "does sparsity speed up dense kernels?" is asked.
TRAIN_DEVICE = torch.device('mps' if torch.backends.mps.is_available() else
                            'cuda' if torch.cuda.is_available() else 'cpu')
LATENCY_DEVICE = torch.device('cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_json(obj, name):
    with open(os.path.join(RESULTS_DIR, name), 'w') as f:
        json.dump(obj, f, indent=2)

def load_json(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)

def write_txt(lines, name):
    text = "\n".join(lines) + "\n"
    with open(os.path.join(RESULTS_DIR, name), 'w') as f:
        f.write(text)
    print(text)

# ==========================================
# Data: same normalisation and 55k/5k split as Labs 3 and 6,
# but the whole dataset is pre-processed once into tensors in memory
# (the fix suggested by the Lab 6 profile, where per-image transforms were 45% of an epoch).
# ==========================================
MEAN, STD = 0.2860, 0.3530

def load_data():
    train_raw = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True)
    test_raw = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True)

    def to_tensor(ds):
        x = (ds.data.float().div(255.0).sub(MEAN).div(STD)).unsqueeze(1)
        return x, ds.targets.clone()

    x_full, y_full = to_tensor(train_raw)
    x_test, y_test = to_tensor(test_raw)
    perm = torch.randperm(len(x_full), generator=torch.Generator().manual_seed(42))
    val_idx, train_idx = perm[:5000], perm[5000:]
    return {
        'train': (x_full[train_idx], y_full[train_idx]),
        'val': (x_full[val_idx], y_full[val_idx]),
        'test': (x_test, y_test),
    }

def batches(x, y, batch_size, shuffle, seed=0, extra=None):
    """Yields mini-batches from in-memory tensors. `extra` (e.g. teacher logits) is sliced with the same indices."""
    n = len(x)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        b = idx[i:i + batch_size]
        if extra is None:
            yield x[b], y[b]
        else:
            yield x[b], y[b], extra[b]

# ==========================================
# Models
# ==========================================
class BigCNN(nn.Module):
    """Deliberately oversized baseline / teacher (~1.04M parameters)."""
    def __init__(self, c1=32, c2=64, c3=128, c4=128, hidden=128, dropout=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1)
        self.conv4 = nn.Conv2d(c3, c4, 3, padding=1)
        self.fc1 = nn.Linear(c4 * 7 * 7, hidden)
        self.fc2 = nn.Linear(hidden, 10)
        self.dropout = nn.Dropout(dropout)
        self.channels = (c1, c2, c3, c4)

    def forward(self, x):
        x = F.relu(self.conv1(x))                    # 28x28
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 14x14
        x = F.relu(self.conv3(x))                    # 14x14
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)   # 7x7
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

class SmallCNN(nn.Module):
    """Compact student (~99k parameters): same layout idea, far fewer channels and a narrow FC layer."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 24)
        self.fc2 = nn.Linear(24, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))                    # 28x28
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 14x14
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)   # 7x7
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def prunable_layers(model):
    return [(name, m) for name, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]

# ==========================================
# Training and evaluation
# ==========================================
def evaluate(model, x, y, batch_size=1000, device=TRAIN_DEVICE):
    model.eval().to(device)
    correct, loss_sum = 0, 0.0
    with torch.inference_mode():
        for xb, yb in batches(x, y, batch_size, shuffle=False):
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss_sum += F.cross_entropy(out, yb, reduction='sum').item()
            correct += (out.argmax(1) == yb).sum().item()
    return correct / len(x), loss_sum / len(x)

def train(model, data, epochs, lr, batch_size=128, seed=42, loss_fn=None, extra=None,
          after_step=None, verbose=True, tag=''):
    """Generic Adam training loop. `loss_fn(out, yb, extra_b)` overrides plain cross-entropy (used for distillation).
    `after_step(model)` runs after every optimizer step (not needed for torch.nn.utils.prune masks, which are
    re-applied in every forward pass, but kept for clarity)."""
    set_seed(seed)
    model.to(TRAIN_DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_tr, y_tr = data['train']
    history = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        loss_sum, n = 0.0, 0
        for batch in batches(x_tr, y_tr, batch_size, shuffle=True, seed=seed * 1000 + epoch, extra=extra):
            batch = [b.to(TRAIN_DEVICE) for b in batch]
            xb, yb = batch[0], batch[1]
            out = model(xb)
            loss = F.cross_entropy(out, yb) if loss_fn is None else loss_fn(out, yb, batch[2])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if after_step is not None:
                after_step(model)
            loss_sum += loss.item() * len(xb)
            n += len(xb)
        val_acc, _ = evaluate(model, *data['val'])
        history.append({'epoch': epoch, 'train_loss': loss_sum / n, 'val_acc': val_acc,
                        'epoch_time_s': time.perf_counter() - t0})
        if verbose:
            print(f"  {tag} epoch {epoch}/{epochs} | loss {loss_sum / n:.4f} | val acc {val_acc:.4f} "
                  f"| {time.perf_counter() - t0:.1f}s")
    return history, time.perf_counter() - start

# ==========================================
# Size, sparsity, compute and latency
# ==========================================
def count_params(model, nonzero=False):
    if nonzero:
        return int(sum((p != 0).sum().item() for p in model.parameters()))
    return int(sum(p.numel() for p in model.parameters()))

def weight_sparsity(model):
    """Fraction of zeros among the conv/linear weights (biases excluded), i.e. what pruning targets."""
    total = zeros = 0
    for _, m in prunable_layers(model):
        total += m.weight.numel()
        zeros += (m.weight == 0).sum().item()
    return zeros / total

def serialized_size_bytes(state_dict):
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return buf.getbuffer().nbytes

def dense_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def sparse_state_dict(model, layout='csr'):
    """Stores every conv/linear weight in a sparse format (conv weights are reshaped to 2-D first), biases dense."""
    sd = {}
    for k, v in model.state_dict().items():
        v = v.detach().cpu()
        if k.endswith('weight') and v.dim() >= 2:
            w2d = v.reshape(v.shape[0], -1)
            sd[k] = w2d.to_sparse_csr() if layout == 'csr' else w2d.to_sparse_coo()
        else:
            sd[k] = v.clone()
    return sd

def count_macs(model, input_shape=(1, 1, 28, 28)):
    """Dense multiply-accumulates per image, counted with forward hooks (conv: Cout*Cin*k*k*H*W, linear: in*out)."""
    macs = {'total': 0}
    hooks = []

    def hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size[0] * m.kernel_size[1]
            macs['total'] += out.shape[1] * (m.in_channels // m.groups) * k * out.shape[2] * out.shape[3]
        elif isinstance(m, nn.Linear):
            macs['total'] += m.in_features * m.out_features

    for _, m in prunable_layers(model):
        hooks.append(m.register_forward_hook(hook))
    model.eval().cpu()
    with torch.inference_mode():
        model(torch.zeros(input_shape))
    for h in hooks:
        h.remove()
    return macs['total']

def effective_macs(model, input_shape=(1, 1, 28, 28)):
    """MACs that touch a non-zero weight: what an ideal sparse kernel would have to do."""
    total = {'v': 0}
    hooks = []

    def hook(m, inp, out):
        nz_frac = (m.weight != 0).float().mean().item()
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size[0] * m.kernel_size[1]
            dense = out.shape[1] * (m.in_channels // m.groups) * k * out.shape[2] * out.shape[3]
        else:
            dense = m.in_features * m.out_features
        total['v'] += dense * nz_frac

    for _, m in prunable_layers(model):
        hooks.append(m.register_forward_hook(hook))
    model.eval().cpu()
    with torch.inference_mode():
        model(torch.zeros(input_shape))
    for h in hooks:
        h.remove()
    return int(total['v'])

def measure_latency(fn, warmup=30, repeats=300):
    """Median and p90 wall-clock time (ms) of fn() on the CPU."""
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return float(np.median(times)), float(np.percentile(times, 90))

def model_latency(model, batch_size=1, threads=None, warmup=30, repeats=300):
    old = torch.get_num_threads()
    if threads is not None:
        torch.set_num_threads(threads)
    model = model.eval().to(LATENCY_DEVICE)
    x = torch.randn(batch_size, 1, 28, 28)
    try:
        return measure_latency(lambda: model(x), warmup, repeats)
    finally:
        torch.set_num_threads(old)
