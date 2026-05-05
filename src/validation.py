from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ArtifactReport:
    captions_count: int = 0
    captions_unique_ids: int = 0
    captions_duplicates: int = 0
    captions_bad_lines: int = 0
    parquet_rows: dict[str, int] = field(default_factory=dict)
    embedding_rows: dict[str, int] = field(default_factory=dict)

    @property
    def parquet_total(self):
        return sum(self.parquet_rows.values())

    @property
    def embedding_total(self):
        return sum(self.embedding_rows.values())

    @property
    def embeddings_match_parquets(self):
        if not self.parquet_rows or not self.embedding_rows:
            return False
        return all(
            self.parquet_rows.get(split, 0) == self.embedding_rows.get(split, 0)
            for split in ("train", "val", "test")
        )


def _count_jsonl(path):
    count = 0
    bad_lines = 0
    ids = set()
    duplicate_ids = 0

    if not path.exists():
        return count, len(ids), duplicate_ids, bad_lines

    with path.open(encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue

            video_id = str(record.get("video_id", ""))
            if not video_id:
                continue
            if video_id in ids:
                duplicate_ids += 1
            ids.add(video_id)

    return count, len(ids), duplicate_ids, bad_lines


def collect_artifact_report(data_dir, captions_filename="sports_captions.json"):
    data_dir = Path(data_dir)
    report = ArtifactReport()

    (
        report.captions_count,
        report.captions_unique_ids,
        report.captions_duplicates,
        report.captions_bad_lines,
    ) = _count_jsonl(data_dir / captions_filename)

    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.parquet"
        if path.exists():
            report.parquet_rows[split] = len(pd.read_parquet(path))

    embeddings_path = data_dir / "embeddings.npz"
    if embeddings_path.exists():
        data = np.load(embeddings_path)
        for split in ("train", "val", "test"):
            key = f"{split}_emb"
            if key in data:
                report.embedding_rows[split] = int(data[key].shape[0])

    return report


def format_artifact_report(report):
    lines = [
        "Dataset artifact report",
        f"captions: {report.captions_count:,} lines, "
        f"{report.captions_unique_ids:,} unique ids, "
        f"{report.captions_duplicates:,} duplicates, "
        f"{report.captions_bad_lines:,} bad lines",
        f"parquets: {report.parquet_rows} total={report.parquet_total:,}",
        f"embeddings: {report.embedding_rows} total={report.embedding_total:,}",
        f"embeddings match parquets: {report.embeddings_match_parquets}",
    ]
    return "\n".join(lines)
