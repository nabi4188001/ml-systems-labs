import os
import sys
import copy
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import yaml
import mlflow
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')

# ==========================================
# 1. Config and Seeds
# ==========================================
def load_config(path=DEFAULT_CONFIG, overrides=None):
    """Loads the YAML config into one dict and applies nested overrides like {'training': {'lr': 0.01}}."""
    with open(path) as f:
        config = yaml.safe_load(f)
    for section, values in (overrides or {}).items():
        if isinstance(values, dict):
            config.setdefault(section, {}).update(values)
        else:
            config[section] = values
    return config

def repo_path(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    # Each DataLoader worker gets a seed derived from the main process seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ==========================================
# 2. Data (same transforms and split as Lab 3)
# ==========================================
def get_datasets(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    data_dir = repo_path(config['data']['dir'])
    full_train = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    val_size = config['data']['val_size']
    train, val = random_split(full_train, [len(full_train) - val_size, val_size],
                              generator=torch.Generator().manual_seed(42))
    return train, val, test

def make_loader(dataset, batch_size, num_workers, shuffle, seed=42):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      worker_init_fn=seed_worker, generator=generator,
                      persistent_workers=False)

# ==========================================
# 3. Model: best model from Lab 3
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
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

def build_model(config):
    if config['model']['name'] != 'SimpleCNN':
        raise ValueError(f"Unknown model: {config['model']['name']}")
    return SimpleCNN()

# ==========================================
# 4. Optimizer and Scheduler
# ==========================================
def build_optimizer(model, config):
    t = config['training']
    if t['optimizer'] == 'adam':
        return torch.optim.Adam(model.parameters(), lr=t['lr'], weight_decay=t['weight_decay'])
    if t['optimizer'] == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=t['lr'], momentum=t['momentum'], weight_decay=t['weight_decay'])
    raise ValueError(f"Unknown optimizer: {t['optimizer']}")

def build_scheduler(optimizer, config):
    t = config['training']
    if t['scheduler'] == 'none':
        return None
    if t['scheduler'] == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=t['step_size'], gamma=t['gamma'])
    if t['scheduler'] == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t['epochs'])
    if t['scheduler'] == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=t['plateau_factor'],
                                                          patience=t['plateau_patience'])
    raise ValueError(f"Unknown scheduler: {t['scheduler']}")

# ==========================================
# 5. Train / Evaluate One Epoch
# ==========================================
def train_one_epoch(model, loader, optimizer, criterion, device, grad_accum_steps=1):
    """
    One training epoch. With grad_accum_steps > 1, gradients of several micro-batches are
    summed before one optimizer step. Each micro-batch loss is weighted by its share of the
    samples in the group, so the summed gradient equals the gradient of the big batch.
    """
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    iterator = iter(loader)
    while True:
        # Collect one accumulation group of micro-batches
        group = []
        for _ in range(grad_accum_steps):
            try:
                group.append(next(iterator))
            except StopIteration:
                break
        if not group:
            break
        group_size = sum(len(labels) for _, labels in group)

        optimizer.zero_grad()
        for images, labels in group:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            (loss * len(labels) / group_size).backward()   # Gradients accumulate in .grad
            total_loss += loss.item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
            seen += len(labels)
        optimizer.step()
    return total_loss / seen, correct / seen, seen

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * len(labels)
        correct += (outputs.argmax(1) == labels).sum().item()
        seen += len(labels)
    return total_loss / seen, correct / seen

# ==========================================
# 6. Checkpointing
# ==========================================
def checkpoint_path(config):
    return os.path.join(repo_path(config['checkpoint']['dir']), f"{config['checkpoint']['name']}_last.pt")

def save_checkpoint(path, model, optimizer, scheduler, epoch, history, run_id):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,                                   # Last finished epoch (1-based)
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
        'history': history,
        'mlflow_run_id': run_id,
        'rng_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        },
    }, path)

def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    if scheduler is not None and ckpt['scheduler_state'] is not None:
        scheduler.load_state_dict(ckpt['scheduler_state'])
    random.setstate(ckpt['rng_state']['python'])
    np.random.set_state(ckpt['rng_state']['numpy'])
    torch.set_rng_state(ckpt['rng_state']['torch'])
    return ckpt['epoch'], ckpt['history'], ckpt['mlflow_run_id']

# ==========================================
# 7. Full Pipeline Run (with MLflow logging)
# ==========================================
def flatten_config(config, prefix=''):
    flat = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_config(value, name + '.'))
        else:
            flat[name] = value
    return flat

def run(config, verbose=True):
    """Trains according to the config. Returns the per-epoch history and final metrics."""
    set_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = config['training']

    train_set, val_set, test_set = get_datasets(config)
    batch_size, workers = config['data']['batch_size'], config['data']['num_workers']
    val_loader = make_loader(val_set, 512, 0, shuffle=False)
    test_loader = make_loader(test_set, 512, 0, shuffle=False)

    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    criterion = nn.CrossEntropyLoss()

    # Resume from checkpoint if requested
    ckpt_file = checkpoint_path(config)
    start_epoch, history, run_id = 0, [], None
    if config['checkpoint']['resume'] and os.path.exists(ckpt_file):
        start_epoch, history, run_id = load_checkpoint(ckpt_file, model, optimizer, scheduler)
        if verbose:
            print(f"Resumed from {os.path.relpath(ckpt_file, REPO_ROOT)} after epoch {start_epoch}")

    mlflow.set_tracking_uri('file:' + repo_path(config['mlflow']['tracking_uri']))
    mlflow.set_experiment(config['mlflow']['experiment'])
    with mlflow.start_run(run_id=run_id, run_name=None if run_id else config['mlflow']['run_name']) as active:
        if run_id is None:
            mlflow.log_params(flatten_config(config))
        run_id = active.info.run_id

        for epoch in range(start_epoch, t['epochs']):
            # Shuffle order depends only on (seed, epoch), so a resumed run sees the same batches
            train_loader = make_loader(train_set, batch_size, workers, shuffle=True, seed=config['seed'] + epoch)

            start = time.perf_counter()
            train_loss, train_acc, seen = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                                          t['grad_accum_steps'])
            epoch_time = time.perf_counter() - start
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            lr = optimizer.param_groups[0]['lr']

            if scheduler is not None:
                if t['scheduler'] == 'plateau':
                    scheduler.step(val_acc)
                else:
                    scheduler.step()

            row = {'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss,
                   'val_acc': val_acc, 'lr': lr, 'epoch_time_s': epoch_time, 'throughput': seen / epoch_time}
            history.append(row)
            mlflow.log_metrics({k: v for k, v in row.items() if k != 'epoch'}, step=epoch + 1)
            save_checkpoint(ckpt_file, model, optimizer, scheduler, epoch + 1, history, run_id)

            if verbose:
                print(f"Epoch [{epoch + 1}/{t['epochs']}] train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                      f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | lr {lr:.5f} | {epoch_time:.1f}s")

            if t.get('stop_after_epoch') is not None and epoch + 1 == t['stop_after_epoch']:
                if verbose:
                    print(f"⚠️ Training interrupted after epoch {epoch + 1} (checkpoint saved)")
                return history, None

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        mlflow.log_metrics({'test_loss': test_loss, 'test_acc': test_acc})

        # Artifacts: config, checkpoint, per-epoch history
        results_dir = repo_path(os.path.join('lab06', 'results'))
        os.makedirs(results_dir, exist_ok=True)
        config_file = os.path.join(results_dir, f"{config['mlflow']['run_name']}_config.yaml")
        history_file = os.path.join(results_dir, f"{config['mlflow']['run_name']}_history.json")
        with open(config_file, 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False)
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        mlflow.log_artifact(config_file)
        mlflow.log_artifact(history_file)
        if config['checkpoint'].get('log_to_mlflow', True):
            mlflow.log_artifact(ckpt_file, artifact_path='checkpoints')
        os.remove(config_file)
        os.remove(history_file)

        if verbose:
            print(f"✅ Test accuracy: {test_acc:.4f} | MLflow run: {run_id}")
        return history, {'test_loss': test_loss, 'test_acc': test_acc, 'run_id': run_id}

# ==========================================
# Main: train from the YAML config
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YAML-driven training pipeline (Lab 6)")
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--resume', action='store_true', help="Resume from the last checkpoint")
    parser.add_argument('--stop-after-epoch', type=int, default=None, help="Simulate an interruption after this epoch")
    parser.add_argument('--run-name', default=None, help="Override MLflow run name and checkpoint name")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.resume:
        config['checkpoint']['resume'] = True
    if args.stop_after_epoch is not None:
        config['training']['stop_after_epoch'] = args.stop_after_epoch
    if args.run_name is not None:
        config['mlflow']['run_name'] = args.run_name
        config['checkpoint']['name'] = args.run_name
    history, final = run(config)

    # Save a per-epoch summary for the report (default run only)
    if final is not None and args.run_name is None:
        lines = ["=== TASK 1: YAML-DRIVEN TRAINING PIPELINE (SimpleCNN, Fashion-MNIST) ==="]
        lines.append(f"Config: {os.path.relpath(args.config, REPO_ROOT)} | seed {config['seed']} | "
                     f"{config['training']['optimizer']} lr={config['training']['lr']} | batch {config['data']['batch_size']} | "
                     f"{config['training']['epochs']} epochs")
        lines.append(f"MLflow experiment: {config['mlflow']['experiment']} | run id: {final['run_id']}")
        lines.append(f"Checkpoint: {os.path.relpath(checkpoint_path(config), REPO_ROOT)}")
        lines.append("")
        lines.append(f"{'Epoch':<6} | {'Train loss':<10} | {'Train acc':<9} | {'Val loss':<8} | {'Val acc':<7} | {'Time (s)':<8} | Samples/s")
        lines.append("-" * 75)
        for r in history:
            lines.append(f"{r['epoch']:<6} | {r['train_loss']:<10.4f} | {r['train_acc']:<9.4f} | {r['val_loss']:<8.4f} | "
                         f"{r['val_acc']:<7.4f} | {r['epoch_time_s']:<8.2f} | {r['throughput']:.0f}")
        lines.append("-" * 75)
        lines.append(f"Final test accuracy: {final['test_acc']:.4f} (Lab 3 CNN, 5 epochs: 0.8861)")
        report = "\n".join(lines)
        print("\n" + report)
        with open(os.path.join(REPO_ROOT, 'lab06', 'results', 'pipeline_run.txt'), 'w') as f:
            f.write(report + "\n")
        print("\n✅ Results saved to ../results/pipeline_run.txt")
