import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.embeddings import coerce_chunks, extract_embeddings
from src.preprocessing import build_label_maps, chunk_caption, extract_caption_text
from src.validation import collect_artifact_report


def test_caption_helpers_filter_empty_values_and_chunk_text():
    caption = {
        "text": [
            " first ",
            "",
            np.nan,
            "second",
            None,
            "third",
            " fourth ",
        ]
    }

    assert extract_caption_text(caption) == "first second third fourth"
    assert chunk_caption(caption, chunk_size=2) == ["first second", "third fourth"]


def test_label_maps_keep_human_readable_class_names():
    label2id, id2label = build_label_maps(
        ["Team Sports", "Outdoor Recreation", "Team Sports"]
    )

    assert label2id == {"Outdoor Recreation": 0, "Team Sports": 1}
    assert id2label == {0: "Outdoor Recreation", 1: "Team Sports"}


def test_coerce_chunks_preserves_numpy_arrays_instead_of_falling_back():
    chunks = np.array(["chunk one", "chunk two", "chunk three"], dtype=object)

    assert coerce_chunks(chunks, fallback_text="fallback", max_seq_len=2) == [
        "chunk one",
        "chunk two",
    ]


def test_extract_embeddings_uses_all_rows_and_real_sequence_lengths():
    class FakeModel:
        def __init__(self):
            self.seen_texts = None

        def encode(self, texts, batch_size, show_progress_bar, normalize_embeddings):
            self.seen_texts = list(texts)
            return np.array(
                [[float(i), float(len(text))] for i, text in enumerate(texts)],
                dtype=np.float32,
            )

    df = pd.DataFrame(
        [
            {
                "chunks": np.array(["a", "bb", "ccc"], dtype=object),
                "full_text": "fallback should not be used",
                "label": 2,
            },
            {"chunks": [], "full_text": "fallback text", "label": 1},
        ]
    )
    model = FakeModel()

    embeddings, labels, seq_lens = extract_embeddings(
        df, model, batch_size=8, max_seq_len=2, show_progress_bar=False
    )

    assert model.seen_texts == ["a", "bb", "fallback text"]
    assert embeddings.shape == (2, 2, 2)
    assert labels.tolist() == [2, 1]
    assert seq_lens.tolist() == [2, 1]
    assert embeddings[1, 1].tolist() == [0.0, 0.0]


def test_collect_artifact_report_detects_stale_embedding_file(tmp_path):
    pd.DataFrame({"label": [0, 1], "category_2": ["A", "B"]}).to_parquet(
        tmp_path / "train.parquet", index=False
    )
    pd.DataFrame({"label": [0], "category_2": ["A"]}).to_parquet(
        tmp_path / "val.parquet", index=False
    )
    pd.DataFrame({"label": [1], "category_2": ["B"]}).to_parquet(
        tmp_path / "test.parquet", index=False
    )
    np.savez_compressed(
        tmp_path / "embeddings.npz",
        train_emb=np.zeros((1, 2, 3), dtype=np.float32),
        train_labels=np.array([0]),
        val_emb=np.zeros((1, 2, 3), dtype=np.float32),
        val_labels=np.array([0]),
        test_emb=np.zeros((1, 2, 3), dtype=np.float32),
        test_labels=np.array([1]),
    )
    (tmp_path / "sports_captions.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"video_id": "v1", "text": ["hello"]}),
                json.dumps({"video_id": "v2", "text": ["world"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = collect_artifact_report(tmp_path, captions_filename="sports_captions.jsonl")

    assert report.parquet_total == 4
    assert report.embedding_total == 3
    assert not report.embeddings_match_parquets


def test_dataset_mask_marks_only_padding_after_truncation():
    pytest.importorskip("torch")
    from src.dataset import SportsEmbeddingDataset

    embeddings = np.ones((1, 5, 3), dtype=np.float32)
    dataset = SportsEmbeddingDataset(embeddings, np.array([0]), max_seq_len=3)

    item = dataset[0]

    assert item["embeddings"].shape == (3, 3)
    assert item["mask"].tolist() == [False, False, False]


def test_dataset_uses_explicit_sequence_lengths_for_padded_embeddings():
    pytest.importorskip("torch")
    from src.dataset import SportsEmbeddingDataset

    embeddings = np.ones((1, 5, 3), dtype=np.float32)
    dataset = SportsEmbeddingDataset(
        embeddings,
        np.array([0]),
        seq_lens=np.array([2]),
        max_seq_len=5,
    )

    item = dataset[0]

    assert item["mask"].tolist() == [False, False, True, True, True]


def test_transformer_classifier_accepts_padding_mask_on_mps_when_available():
    torch = pytest.importorskip("torch")
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS is not available")

    from src.model import TransformerClassifier

    model = TransformerClassifier(
        num_classes=2,
        d_model=8,
        nhead=2,
        num_encoder_layers=1,
        dim_feedforward=16,
        max_seq_len=4,
    ).to("mps")
    model.eval()
    src = torch.randn(2, 4, 8, device="mps")
    mask = torch.tensor(
        [[False, False, True, True], [False, False, False, True]],
        device="mps",
    )

    with torch.no_grad():
        logits = model(src, src_key_padding_mask=mask)

    assert tuple(logits.shape) == (2, 2)


def test_positional_encoding_scales_input_before_adding_positions():
    torch = pytest.importorskip("torch")
    from src.model import PositionalEncoding

    pe = PositionalEncoding(d_model=4, max_len=2, dropout=0.0)
    x = torch.ones(1, 2, 4)

    out = pe(x)

    expected = x * 2.0 + pe.pe[:, :2]
    assert torch.allclose(out, expected)
