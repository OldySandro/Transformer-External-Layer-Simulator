from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Optional
import torch
BUILTIN_SAMPLE_TEXT = "\nThe Transformer is a neural network architecture that processes sequences\nof data using an attention mechanism, rather than recurrence like RNNs or\nLSTMs. Every token in the sequence can attend to any other token directly,\nregardless of their distance within the sequence.\n\nThe self-attention mechanism computes a relevance score between every pair\nof tokens using query, key, and value vectors projected from the input\nembedding. This score is then normalized with softmax to form an attention\nprobability distribution.\n\nGrouped Query Attention (GQA) is an efficient variant of multi-head\nattention, where several query heads share the same key and value heads.\nThis reduces the memory required to store the key-value cache during\ninference, especially for models with many heads and long sequences.\n\nRotary Position Embedding (RoPE) encodes positional information by\nrotating the query and key vectors within an even-dimensional space,\ninstead of adding a position vector directly to the token embedding. This\napproach keeps the relative relationship between token positions\nconsistent.\n\nExternal modules such as the Knowledge Access Controller (KAC) are\ndesigned to be inserted into the Transformer stack, either in parallel\n(added to the residual stream) or sequentially (replacing the hidden state\nthat flows into the next layer). The effect of this insertion on the\nmodel's representation can be measured through drift, both in the hidden\nstate and in the attention distribution at every layer.\n\nSmall experiments with limited data cannot replace full-scale\npretraining, but they can still reveal whether a module causes numerical\ninstability, representation collapse, or a significant throughput\nslowdown, long before the real cost of pretraining is incurred.\n".strip()

@dataclass
class CharTokenizer:
    chars: List[str]

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def __post_init__(self):
        self._stoi = {ch: i for i, ch in enumerate(self.chars)}
        self._itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def fit(cls, text: str) -> 'CharTokenizer':
        unique_chars = sorted(set(text))
        return cls(chars=unique_chars)

    def encode(self, text: str) -> List[int]:
        unk = self._stoi.get(' ', 0)
        return [self._stoi.get(ch, unk) for ch in text]

    def decode(self, ids: List[int]) -> str:
        return ''.join((self._itos.get(i, '?') for i in ids))

    def to_dict(self) -> dict:
        return {'chars': self.chars}

    @classmethod
    def from_dict(cls, d: dict) -> 'CharTokenizer':
        return cls(chars=d['chars'])

class TextDataset:

    def __init__(self, text: str, tokenizer: CharTokenizer, seed: int=42):
        self.text = text
        self.tokenizer = tokenizer
        self.token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self._generator = torch.Generator(device='cpu').manual_seed(seed)
        if len(self.token_ids) < 2:
            raise ValueError(f'Dataset is too short after tokenization ({len(self.token_ids)} tokens). Provide a longer text.')

    def get_batch(self, batch_size: int, seq_len: int) -> torch.Tensor:
        n = len(self.token_ids)
        max_start = n - (seq_len + 1)
        if max_start < 0:
            raise ValueError(f'seq_len+1 ({seq_len + 1}) is larger than the dataset length ({n} tokens). Reduce seq_len or grow the dataset.')
        if max_start == 0:
            starts = torch.zeros(batch_size, dtype=torch.long)
        else:
            starts = torch.randint(0, max_start + 1, (batch_size,), generator=self._generator)
        batch = torch.stack([self.token_ids[s:s + seq_len + 1] for s in starts])
        return batch

def load_dataset(data_path: Optional[str], seed: int=42, max_chars: Optional[int]=None) -> TextDataset:
    if data_path is None:
        text = BUILTIN_SAMPLE_TEXT
    else:
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f'Dataset file not found: {data_path}')
        with open(data_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        if not text.strip():
            raise ValueError(f"Dataset file '{data_path}' is empty after reading.")
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars]
    tokenizer = CharTokenizer.fit(text)
    return TextDataset(text, tokenizer, seed=seed)