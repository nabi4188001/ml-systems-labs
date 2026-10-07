# Lab 1: Environment and First System Measurements

## 1. Goal
The objective of this lab was to set up a reproducible Python 3.11.8 environment, train baseline machine learning models on the Breast Cancer Wisconsin dataset, and rigorously measure their system costs (training time, inference latency, peak memory, and model size) to evaluate their deployment suitability across different hardware budgets.

## 2. Method
- **Environment**: Created an isolated Python 3.11.8 virtual environment with pinned dependencies. Versions were verified using `src/print_versions.py`.
- **Data**: Loaded the Breast Cancer Wisconsin dataset via `sklearn.datasets.load_breast_cancer`. Applied a 70/30 stratified train-test split with `random_state=42`.
- **Models**: Trained two baseline models: `LogisticRegression(max_iter=1000, random_state=42)` and `RandomForestClassifier(n_estimators=100, random_state=42)`. Global seeds were set for `random`, `numpy`, and `torch` to ensure reproducibility.
- **Measurements**: 
  - Training time: 1 warm-up run followed by 5 runs, reporting the median.
  - Inference latency: 1 warm-up run followed by 100 single-sample predictions, reporting the median.
  - Peak memory: Measured in MiB using `memory_profiler`.
  - Model size: Saved via `joblib.dump` and measured in bytes and KB.

## 3. Results

| Model | Median Train Time (s) | Median Inf. Latency (ms) | Peak Train Mem (MiB) | Model Size (KB) |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.3440 | 0.0355 | 245.84 | 1.08 |
| Random Forest | 0.1174 | 1.7493 | 108.16 | 284.09 |

*(Note: Test accuracies were 0.9708 for LR and 0.9649 for RF, saved in `results/baseline_accuracy.csv`. Full detailed metrics are in `results/system_costs.csv`)*

## 4. Conclusions
1. **Reproducibility is critical**: By strictly setting random seeds across all libraries and using median values over multiple runs (5 for training, 100 for inference), we eliminated variance and ensured that the system measurements are reliable and reproducible.
2. **Trade-offs between model complexity and cost**: While both models achieved high accuracy, the Random Forest model had a significantly larger file size (284 KB vs 1 KB) and different memory profiling characteristics compared to the lightweight Logistic Regression model, demonstrating the classic accuracy-vs-efficiency trade-off.
3. **Deployment-fit evaluation**: Both models easily satisfy Cloud, Edge, and Mobile deployment budgets (e.g., inference latency is well under the 20ms Mobile threshold, and model sizes are under 10 MB). However, neither model is suitable for TinyML out-of-the-box, as their memory usage (~100–245 MB) vastly exceeds the strict TinyML limit of ≤ 256 KB. Further optimization (e.g., quantization or pruning) would be required for TinyML deployment.
