import sys
import importlib.metadata

def main():
    # Print Python version
    print(f"Python version: {sys.version.split()[0]}")
    print("-" * 40)

    # Exact packages from the Lab 1 requirements
    packages = [
        "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "seaborn",
        "torch", "torchvision", "torchinfo", "thop", "onnx", "onnxruntime",
        "mlflow", "memory-profiler", "psutil", "codecarbon", "fastapi",
        "uvicorn", "pytest", "httpx", "locust", "requests", "pyarrow",
        "joblib", "tqdm"
    ]

    # Print version for each package
    for pkg in packages:
        try:
            version = importlib.metadata.version(pkg)
            print(f"{pkg}=={version}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{pkg}: NOT INSTALLED")

if __name__ == "__main__":
    main()
