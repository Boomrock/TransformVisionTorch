import torch
from torch.utils.data import Dataset
import numpy as np


class SportsEmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels, seq_lens=None, max_seq_len=20):
        self.embeddings = embeddings
        self.labels = labels
        self.seq_lens = seq_lens
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emb = self.embeddings[idx]

        if isinstance(emb, np.ndarray) and emb.ndim == 2:
            original_len = int(self.seq_lens[idx]) if self.seq_lens is not None else emb.shape[0]
            seq_len = min(original_len, self.max_seq_len)
            if emb.shape[0] > self.max_seq_len:
                emb = emb[:self.max_seq_len]
            elif emb.shape[0] < self.max_seq_len:
                pad = np.zeros((self.max_seq_len - emb.shape[0], emb.shape[1]))
                emb = np.concatenate([emb, pad], axis=0)
            mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
            mask[seq_len:] = True
        else:
            emb = emb.reshape(1, -1)
            emb = np.concatenate([emb, np.zeros((self.max_seq_len - 1, emb.shape[1]))], axis=0)
            mask = torch.ones(self.max_seq_len, dtype=torch.bool)
            mask[0] = False

        return {
            "embeddings": torch.tensor(emb, dtype=torch.float32),
            "mask": mask,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
