import time
import random
import numpy as np
import torch
import os
import pandas as pd
import joblib
from memory_profiler import memory_usage
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# --- GLOBAL RULE: Set Seeds ---
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

def main():
    os.makedirs("lab01/results", exist_ok=True)

    # 1. Load and split data
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = []
    X_single = X_test[0:1]  # Single sample for inference latency test

    for name, model in models.items():
        print(f"--- Measuring {name} ---")

        # 2. Training Time: warm up once, then 5 runs, report median
        model.fit(X_train, y_train)  # Warm-up run
        train_times = []
        for _ in range(5):
            start = time.perf_counter()
            model.fit(X_train, y_train)
            end = time.perf_counter()
            train_times.append(end - start)
        median_train_time = np.median(train_times)

        # 3. Peak Memory during training (returns peak usage in MiB)
        peak_train_mem = memory_usage((model.fit, (X_train, y_train)), max_usage=True)

        # 4. Inference Latency: repeat 100 times, report median
        model.predict(X_single)  # Warm-up
        inf_times = []
        for _ in range(100):
            start = time.perf_counter()
            model.predict(X_single)
            end = time.perf_counter()
            inf_times.append(end - start)
        median_inf_time = np.median(inf_times)

        # 5. Peak Memory during inference
        peak_inf_mem = memory_usage((model.predict, (X_single,)), max_usage=True)

        # 6. Model Size: save and measure file size
        filename = f"lab01/results/{name}.joblib"
        joblib.dump(model, filename)
        size_bytes = os.path.getsize(filename)
        size_kb = size_bytes / 1024.0

        results.append({
            "model": name,
            "train_time_s": f"{median_train_time:.4f}",
            "train_mem_MiB": f"{peak_train_mem:.2f}",
            "inf_time_ms": f"{median_inf_time * 1000:.4f}",
            "inf_mem_MiB": f"{peak_inf_mem:.2f}",
            "size_bytes": size_bytes,
            "size_kb": f"{size_kb:.2f}"
        })

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("lab01/results/system_costs.csv", index=False)
    print("\nMeasurements complete! Saved to lab01/results/system_costs.csv")
    print(df)

if __name__ == "__main__":
    main()
