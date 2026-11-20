# Individual Project Proposal: Pocket Anime Tagger – Compressing a 95M-Parameter Anime Tagger into a 5 MB On-Device Model

**Student:** Maist Iqbal | **Student ID:** 72865 | **Course:** Machine Learning (CII5SE039CI), Group 02

## 1. Problem
Anime fans collect thousands of screenshots and illustrations, but finding "every screenshot of character X" or "all night-time rooftop scenes" means tagging them by hand. Open-source anime taggers do this well, but they are about 95M-parameter vision transformers (about 380 MB in FP32) that need a GPU or a server. The goal is an **offline, on-device tagger**: a model under 5 MB that runs on a phone or laptop CPU in under 50 ms per image and predicts **multiple tags per image** (characters, clothing, scene, objects and a content rating).

This is harder than single-label classification: it is **multi-label** (an image has about 30 tags on average), the tags follow a **long-tailed distribution** (a few tags are very common, most are rare), and the labels are **crowdsourced and noisy**.

## 2. Data and Teacher Model
- **Dataset:** a subset of **Danbooru2023** (Hugging Face `nyanko7/danbooru2023`), a large crowd-tagged anime illustration dataset with metadata for every image.
  - **Only images rated "general" (safe) are used**; all other ratings are filtered out from the metadata before any image is downloaded.
  - About **150,000 images**, resized to 256×256. Images stay local and are not redistributed (Danbooru images keep their artists' copyrights; research use only).
  - **Label space:** the 600 most frequent general tags + 100 most frequent characters + the rating.
  - **Split:** Danbooru ID modulo 1000. Train on buckets 0000–0899, validate on 0900–0949, test on **0950–0999**. These test buckets are the ones the teacher was *validated* on, never trained on, so the teacher's score on them is fair.
- **Teacher:** `SmilingWolf/wd-swinv2-tagger-v3` (SwinV2, about 98M parameters, Apache-2.0), trained on Danbooru and reporting a macro-F1 of 0.4541 on its full tag set. Its predicted probabilities for all 150k training images are computed once (on a free Kaggle/Colab GPU) and stored as **soft labels**.

## 3. Success Metrics (measured on the test buckets)
| Metric | Target |
| :--- | :--- |
| Macro-F1 over the 700 tags (per-tag thresholds tuned on validation) | ≥ 85% of the teacher's macro-F1 on the same tags |
| Character tags: mean average precision (mAP) | ≥ 0.70 |
| Distillation gain: distilled student vs the same student trained on labels only | ≥ +2 macro-F1 points |
| Accuracy cost of pruning + INT8 quantization vs the FP32 distilled student | ≤ 1 macro-F1 point |

## 4. Deployment Constraints
| Constraint | Limit | How it is measured |
| :--- | :--- | :--- |
| Latency | ≤ 50 ms per image | Batch 1, 256×256, single CPU thread, ONNX Runtime, median of 100 runs after warm-up |
| Model size | ≤ 5 MB | Size of the final ONNX file on disk |
| Hardware | CPU only, offline | Laptop CPU limited to 1 thread as a stand-in for a phone |

## 5. Planned Optimization (three stages, each measured separately)
1. **Knowledge distillation (main technique):** student = MobileNetV3-Small-style CNN (about 1.5–2.5M parameters) with a 701-output sigmoid head. Loss = α · BCE(student, Danbooru labels) + (1 − α) · BCE(student, teacher soft labels at temperature T). For multi-label tasks, distillation uses a per-tag sigmoid with temperature instead of the softmax + KL form used for single-label classes. α and T are tuned on validation, and every result is compared against an identical student trained without the teacher.
2. **Structured pruning:** remove 30–50% of the student's channels by L1 norm, then fine-tune with distillation, so the network becomes physically smaller and faster (Lab 7 showed that unstructured sparsity does not speed up CPU kernels).
3. **INT8 quantization:** post-training static quantization with a 1,000-image calibration set. If the drop exceeds 1 macro-F1 point, quantization-aware training is used instead. About 2M parameters × 1 byte ≈ 2 MB, well under 5 MB.
