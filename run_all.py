"""
Runs the analysis scripts in order and stops at the first failure.

    python run_all.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PIPELINE = [
    "claiming_sample.py",
    "mortality.py",
    "claiming.py",
    "subjective_survival.py",
    "mortality_gap.py",
    "benchmark.py",
    "retrospective.py",
    "sensitivities.py",
    "interactions.py",
    "timing.py",
    "claims_at_70.py",
]


def main():
    for script in PIPELINE:
        print(f"\n--- {script} ---", flush=True)
        if subprocess.run([sys.executable, script], cwd=ROOT).returncode != 0:
            sys.exit(f"{script} failed.")
    print("\nAll scripts completed.")


if __name__ == "__main__":
    main()
