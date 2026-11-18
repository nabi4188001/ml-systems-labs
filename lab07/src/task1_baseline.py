import os
import torch
import common as c

# ==========================================
# Task 1: Oversized baseline CNN (~1M parameters)
# ==========================================
EPOCHS = 8
LR = 1e-3

if __name__ == "__main__":
    data = c.load_data()
    c.set_seed(42)
    model = c.BigCNN()
    print(f"BigCNN parameters: {c.count_params(model):,} | training on {c.TRAIN_DEVICE}")

    history, train_time = c.train(model, data, EPOCHS, LR, tag='baseline')
    test_acc, test_loss = c.evaluate(model, *data['test'])
    torch.save(model.state_dict(), os.path.join(c.CKPT_DIR, 'baseline.pt'))

    model.cpu()
    sd = c.dense_state_dict(model)
    size_bytes = c.serialized_size_bytes(sd)
    macs = c.count_macs(model)
    lat1 = c.model_latency(model, batch_size=1)
    lat256 = c.model_latency(model, batch_size=256, repeats=50)
    lat1_t1 = c.model_latency(model, batch_size=1, threads=1)

    per_layer = [(n, m.weight.numel() + (m.bias.numel() if m.bias is not None else 0))
                 for n, m in c.prunable_layers(model)]
    result = {
        'params': c.count_params(model), 'size_bytes': size_bytes, 'macs': macs,
        'test_acc': test_acc, 'test_loss': test_loss, 'val_acc': history[-1]['val_acc'],
        'latency_b1_ms': lat1[0], 'latency_b1_p90_ms': lat1[1], 'latency_b1_1thread_ms': lat1_t1[0],
        'latency_b256_ms': lat256[0], 'train_time_s': train_time, 'epochs': EPOCHS,
        'history': history, 'per_layer_params': per_layer,
    }
    c.save_json(result, 'task1_baseline.json')

    lines = [f"=== TASK 1: OVERSIZED BASELINE (BigCNN, Adam lr={LR}, batch 128, {EPOCHS} epochs) ===",
             f"Training device: {c.TRAIN_DEVICE} | latency device: CPU ({torch.get_num_threads()} threads), PyTorch {torch.__version__}",
             "", "Architecture: conv(1->32) conv(32->64) pool conv(64->128) conv(128->128) pool fc(6272->128) dropout fc(128->10)",
             "", f"{'Layer':<8} | Parameters", "-" * 24]
    lines += [f"{n:<8} | {p:,}" for n, p in per_layer]
    lines += ["-" * 24, f"{'Total':<8} | {result['params']:,}", "",
              f"{'Epoch':<6} | {'Train loss':<10} | {'Val acc':<7} | Time (s)", "-" * 42]
    lines += [f"{h['epoch']:<6} | {h['train_loss']:<10.4f} | {h['val_acc']:<7.4f} | {h['epoch_time_s']:.1f}" for h in history]
    lines += ["-" * 42, "",
              f"Test accuracy:            {test_acc:.4f}",
              f"Parameters:               {result['params']:,}",
              f"Size on disk (fp32, .pt): {size_bytes / 2**20:.2f} MiB ({size_bytes:,} bytes)",
              f"MACs per image:           {macs / 1e6:.1f} M",
              f"CPU latency, batch 1:     {lat1[0]:.3f} ms median (p90 {lat1[1]:.3f} ms), 1 thread: {lat1_t1[0]:.3f} ms",
              f"CPU latency, batch 256:   {lat256[0]:.2f} ms ({lat256[0] / 256 * 1000:.1f} us/image)",
              f"Training time:            {train_time:.0f} s"]
    c.write_txt(lines, 'task1_baseline.txt')
