import random
import numpy as np
import torch
import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- GLOBAL RULE: Set Seeds for Reproducibility ---
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

def main():
    # Ensure results directory exists
    os.makedirs("lab01/results", exist_ok=True)

    # 1. Load Dataset
    X, y = load_breast_cancer(return_X_y=True)

    # 2. Split 70/30 stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    # 3. Train Linear Model
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr_model.predict(X_test))

    # 4. Train Tree Ensemble
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

    # 5. Record test accuracy to 4 decimals in CSV
    results = {
        "model": ["LogisticRegression", "RandomForestClassifier"],
        "test_accuracy": [f"{lr_acc:.4f}", f"{rf_acc:.4f}"]
    }
    df = pd.DataFrame(results)
    df.to_csv("lab01/results/baseline_accuracy.csv", index=False)

    print("Training complete!")
    print(df)
    print("Saved to lab01/results/baseline_accuracy.csv")

if __name__ == "__main__":
    main()
