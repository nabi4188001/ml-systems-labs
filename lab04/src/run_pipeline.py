import subprocess
import sys
import os

scripts = [
    "task1_2_ingestion.py",
    "task3_validation.py",
    "task4_cleaning.py",
    "task5_feature_engineering.py",
    "task6_label_quality.py",
    "task7_hash_config.py"
]

def main():
    print("🚀 Starting Lab 4 Full Pipeline...\n")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for script in scripts:
        print(f"▶️ Running {script}...")
        result = subprocess.run([sys.executable, script], cwd=script_dir)
        if result.returncode != 0:
            print(f"❌ Pipeline failed at {script}!")
            sys.exit(1)
        print(f"✅ {script} completed successfully.\n")
    print("🎉 Full pipeline completed successfully!")

if __name__ == "__main__":
    main()