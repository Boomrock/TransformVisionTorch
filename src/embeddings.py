import numpy as np
import pandas as pd


def coerce_chunks(chunks, fallback_text, max_seq_len):
    """Normalize a parquet chunks cell to a non-empty list of strings."""
    if chunks is None:
        values = []
    elif isinstance(chunks, np.ndarray):
        values = chunks.tolist()
    elif isinstance(chunks, pd.Series):
        values = chunks.tolist()
    elif isinstance(chunks, (list, tuple)):
        values = list(chunks)
    elif isinstance(chunks, str):
        values = [chunks]
    else:
        try:
            values = [] if pd.isna(chunks) else list(chunks)
        except (TypeError, ValueError):
            values = []

    clean = []
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass

        text = str(value).strip()
        if text:
            clean.append(text)

    if not clean:
        fallback = str(fallback_text or "").strip()[:512]
        clean = [fallback]

    return clean[:max_seq_len]


def extract_embeddings(df, model, batch_size=256, max_seq_len=20, show_progress_bar=True):
    """Encode caption chunks and return padded video-level embedding sequences."""
    all_chunks = []
    video_indices = []

    for _, row in df.iterrows():
        chunks = coerce_chunks(row["chunks"], row["full_text"], max_seq_len)
        start = len(all_chunks)
        all_chunks.extend(chunks)
        end = len(all_chunks)
        video_indices.append((int(row["label"]), start, end, len(chunks)))

    print(f"Total chunks to encode: {len(all_chunks):,}")

    chunk_embeddings = model.encode(
        all_chunks,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=True,
    )
    chunk_embeddings = np.asarray(chunk_embeddings, dtype=np.float32)
    emb_dim = chunk_embeddings.shape[1]

    embeddings = []
    labels = []
    seq_lens = []

    for label, start, end, seq_len in video_indices:
        video_emb = np.zeros((max_seq_len, emb_dim), dtype=np.float32)
        video_emb[:seq_len] = chunk_embeddings[start:end]
        embeddings.append(video_emb)
        labels.append(label)
        seq_lens.append(seq_len)

    return (
        np.asarray(embeddings, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(seq_lens, dtype=np.int64),
    )
