"""
End-to-end pipeline runner.

Usage:
  python run_pipeline.py --demo           # instant test with synthetic data
  python run_pipeline.py                  # full run (downloads ~4GB MS MARCO)
  python run_pipeline.py --skip-download  # skip download, use existing data
  python run_pipeline.py --skip-features  # skip feature extraction
  python run_pipeline.py --skip-train     # skip training, only evaluate
"""

import argparse
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))


def run(script: str) -> None:
    print(f"\n{'='*60}")
    print(f"STEP: {os.path.relpath(script, BASE)}")
    print("=" * 60)
    subprocess.run([sys.executable, script], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",          action="store_true", help="Run on synthetic data (no download)")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-train",    action="store_true")
    args = parser.parse_args()

    if args.demo:
        print("\n*** DEMO MODE — synthetic data (10 queries, 20 passages) ***\n")
        run(os.path.join(BASE, "data", "demo_data.py"))
        run(os.path.join(BASE, "features", "build_dataset.py"))
        run(os.path.join(BASE, "models", "lambdamart.py"))
        run(os.path.join(BASE, "eval", "metrics.py"))
        return

    if not args.skip_download:
        run(os.path.join(BASE, "data", "download_msmarco.py"))

    if not args.skip_features:
        run(os.path.join(BASE, "features", "build_dataset.py"))

    if not args.skip_train:
        run(os.path.join(BASE, "models", "lambdamart.py"))

    run(os.path.join(BASE, "eval", "metrics.py"))


if __name__ == "__main__":
    main()
