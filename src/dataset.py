import torch
from torch.utils.data import Dataset
import numpy as np


class SportsEmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels, max_seq_len=20):
        self.embeddings = embeddings
        self.labels = labels
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emb = self.embeddings[idx]

        if isinstance(emb, np.ndarray) and emb.ndim == 2:
            seq_len = emb.shape[0]
            if seq_len > self.max_seq_len:
                emb = emb[:self.max_seq_len]
            elif seq_len < self.max_seq_len:
                pad = np.zeros((self.max_seq_len - seq_len, emb.shape[1]))
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
