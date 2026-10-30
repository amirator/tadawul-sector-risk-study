"""Single entry point: python run.py [--refresh]  (download -> analysis -> tables -> figures)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from tadawul_risk.pipeline import run

if __name__ == "__main__":
    run(refresh="--refresh" in sys.argv)
    print("done: see reports/tables, reports/figures")
