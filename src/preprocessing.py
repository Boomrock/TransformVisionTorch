import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd


def _clean_text_items(items):
    clean = []
    for item in items:
        if item is None:
            continue
        try:
            if pd.isna(item):
                continue
        except (TypeError, ValueError):
            pass

        text = str(item).strip()
        if text:
            clean.append(text)
    return clean


def _caption_text_items(caption_data):
    if caption_data is None:
        return []

    if isinstance(caption_data, dict):
        items = caption_data.get("text", [])
    elif isinstance(caption_data, (list, tuple, np.ndarray, pd.Series)):
        items = caption_data
    else:
        return []

    if isinstance(items, np.ndarray):
        items = items.tolist()
    elif isinstance(items, pd.Series):
        items = items.tolist()

    return _clean_text_items(items)


def extract_caption_text(caption_data):
    """Return one normalized text string from a caption dict/list."""
    return " ".join(_caption_text_items(caption_data))


def chunk_caption(caption_data, chunk_size=5):
    """Split caption text fragments into fixed-size text chunks."""
    texts = _caption_text_items(caption_data)
    chunks = []
    for start in range(0, len(texts), chunk_size):
        chunk = " ".join(texts[start:start + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def build_label_maps(labels):
    """Build stable class-name mappings for JSON serialization and training."""
    class_names = sorted({str(label) for label in labels if str(label).strip()})
    label2id = {label: idx for idx, label in enumerate(class_names)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def _json_default(value):
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def filter_captions_by_ids(input_path, output_path, video_ids, progress_every=1000):
    """Stream HowTo100M caption JSON dict and write selected videos as JSONL."""
    try:
        import ijson
    except ImportError as exc:
        raise RuntimeError(
            "ijson is required to stream the full HowTo100M caption JSON. "
            "Install project requirements before running this step."
        ) from exc

    input_path = Path(input_path)
    output_path = Path(output_path)
    ids = {str(video_id) for video_id in video_ids}
    matched_count = 0
    total_count = 0

    with input_path.open("rb") as fin, output_path.open("w", encoding="utf-8") as fout:
        for video_id, captions_data in ijson.kvitems(fin, ""):
            total_count += 1
            if str(video_id) not in ids:
                continue

            record = {
                "video_id": str(video_id),
                "start": captions_data.get("start", []),
                "end": captions_data.get("end", []),
                "text": captions_data.get("text", []),
            }
            fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
            matched_count += 1

            if progress_every and matched_count % progress_every == 0:
                print(
                    f"Matched {matched_count:,} captions after scanning "
                    f"{total_count:,} videos"
                )

    return {"total": total_count, "matched": matched_count}
