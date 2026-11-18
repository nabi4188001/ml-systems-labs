# Individual Project Proposal: Real-Time Traffic Sign Recognition on a CPU Edge Device

**Student:** Maist Iqbal | **Student ID:** 72865 | **Course:** Machine Learning (CII5SE039CI), Group 02

## 1. Problem
Driver-assistance and low-cost dash-cam systems need to recognise traffic signs on the device itself, without a network connection and without a GPU. The task is to classify a cropped traffic-sign image into one of 43 sign classes fast enough for live video and small enough to ship on an embedded CPU board or a phone.

## 2. Dataset
**German Traffic Sign Recognition Benchmark (GTSRB)**, available through `torchvision.datasets.GTSRB`.
- 43 classes, about 26,600 training images and 12,630 official test images, cropped sign regions of varying size (resized to 32×32 RGB).
- Real-world variation: lighting, blur, partial occlusion, and strong class imbalance (the rarest classes have about 10x fewer images than the most common ones).
- Split: stratified 90/10 train/validation split of the training set (seed 42); the official test set is used once, for final reporting.

## 3. Success Metrics
| Metric | Target |
| :--- | :--- |
| Test accuracy (official GTSRB test set) | ≥ 95% |
| Macro-F1 (accounts for class imbalance) | ≥ 0.93 |
| Accuracy drop after optimization vs FP32 model | ≤ 1 percentage point |

## 4. Deployment Constraints
| Constraint | Limit | How it is measured |
| :--- | :--- | :--- |
| Latency | ≤ 50 ms per image | Batch size 1, single CPU thread, median of 100 runs after warm-up (same protocol as Labs 1–5) |
| Model size | ≤ 5 MB | Size of the saved model file on disk |
| Hardware | CPU only | Laptop CPU limited to 1 thread, as a stand-in for an embedded board |

## 5. Planned Approach
1. **Baseline:** a compact CNN (3 conv blocks with batch norm, global average pooling, linear head), trained with the YAML + MLflow pipeline from Lab 6 (seeded, checkpointed, every run logged).
2. **Teacher (optional upper bound):** a ResNet-18 fine-tuned on GTSRB, used only to show the accuracy ceiling. It is expected to exceed the 5 MB limit (about 45 MB in FP32).
3. **Planned optimization: post-training INT8 quantization** of the baseline CNN (`torch.ao.quantization`, static quantization with a calibration set from the training data). It cuts the model size by about 4x and speeds up CPU inference. If accuracy drops by more than 1 point, quantization-aware training is used instead.
4. **Evaluation:** accuracy, macro-F1, per-class confusion matrix, latency and file size for the FP32 and INT8 models, and the trade-off plot of accuracy against latency and size.

## 6. Risks and Mitigation
- **Class imbalance:** use a class-weighted loss and report macro-F1, not just accuracy.
- **Quantization accuracy loss:** fall back to quantization-aware training (QAT) or keep the first and last layers in FP32.
- **Unsupported operations in INT8:** design the baseline with quantization-friendly layers only (Conv, BatchNorm, ReLU, Linear).

## 7. Timeline
| Week | Milestone |
| :--- | :--- |
| 1 | Data pipeline, baseline CNN, MLflow tracking |
| 2 | Hyperparameter tuning, optional teacher model |
| 3 | INT8 quantization (PTQ, then QAT if needed), latency and size measurements |
| 4 | Final evaluation, report and presentation |
