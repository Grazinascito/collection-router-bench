"""Compute summary/metrics from a run's raw records and write all report outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from crb.report import write_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--data-dir", default=str(REPO / "data"))
    args = ap.parse_args()
    run_dir = REPO / "results" / args.run_id
    if not (run_dir / "raw").exists():
        raise SystemExit(f"no raw records under {run_dir}")
    summary = write_all(run_dir, Path(args.data_dir))
    print(f"wrote summary/report/facts/parquet/figures to {run_dir}")
    print(
        f"micro-F1 A={summary['arms'].get('A', {}).get('quality', {}).get('micro_f1')} "
        f"B={summary['arms'].get('B', {}).get('quality', {}).get('micro_f1')}"
    )


if __name__ == "__main__":
    main()
