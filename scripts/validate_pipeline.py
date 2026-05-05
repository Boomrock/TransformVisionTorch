#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation import collect_artifact_report, format_artifact_report


def main():
    parser = argparse.ArgumentParser(description="Validate generated dataset artifacts.")
    parser.add_argument(
        "--data-dir",
        default=PROJECT_ROOT / "data",
        type=Path,
        help="Directory with sports captions, parquet splits, and embeddings.npz.",
    )
    parser.add_argument(
        "--captions-filename",
        default="sports_captions.json",
        help="JSONL captions file name inside data-dir.",
    )
    args = parser.parse_args()

    report = collect_artifact_report(args.data_dir, args.captions_filename)
    print(format_artifact_report(report))

    if not report.embeddings_match_parquets:
        print(
            "\nERROR: embeddings.npz does not match train/val/test parquet row counts. "
            "Re-run notebooks/02_embedding_extraction.ipynb after data preparation.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
