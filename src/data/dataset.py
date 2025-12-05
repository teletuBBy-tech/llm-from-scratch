# src/data/dataset.py
"""
Robust sliding-window dataset and dataloader factory for autoregressive LM training.
"""

from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
from src.data.tokenizer import TiktokenWrapper  # use absolute import for reliability


class GPTDatasetV1(Dataset):
    def __init__(self, text: str, tokenizer: TiktokenWrapper, max_length: int = 256, stride: int = 128):
        """
        Builds input/target sliding windows for next-token prediction.
        - text: raw text string
        - tokenizer: TiktokenWrapper instance
        - max_length: sequence length (context length)
        - stride: sliding step between windows
        """
        token_ids: List[int] = tokenizer.encode(text)

        # If the text is extremely short, duplicate it so we have at least one sample
        if len(token_ids) < 2:
            token_ids = token_ids + token_ids

        self.inputs: List[torch.LongTensor] = []
        self.targets: List[torch.LongTensor] = []

        # Ensure at least one window even if len(token_ids) < max_length
        if len(token_ids) <= max_length:
            inp = token_ids[:max_length]
            tgt = token_ids[1:max_length + 1]
            # pad if necessary
            if len(inp) < max_length:
                pad_len = max_length - len(inp)
                inp = inp + [0] * pad_len
                tgt = tgt + [0] * pad_len
            self.inputs.append(torch.tensor(inp, dtype=torch.long))
            self.targets.append(torch.tensor(tgt, dtype=torch.long))
        else:
            # normal sliding windows
            end_limit = len(token_ids) - max_length
            for i in range(0, end_limit + 1, stride):
                inp = token_ids[i: i + max_length]
                tgt = token_ids[i + 1: i + max_length + 1]

                # if last window shorter than max_length (unlikely here), pad
                if len(inp) < max_length:
                    pad_len = max_length - len(inp)
                    inp = inp + [0] * pad_len
                    tgt = tgt + [0] * pad_len

                self.inputs.append(torch.tensor(inp, dtype=torch.long))
                self.targets.append(torch.tensor(tgt, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        return self.inputs[idx], self.targets[idx]


def create_dataloader_v1(text: str, batch_size: int, max_length: int, stride: int,
                         shuffle: bool = True) -> Tuple[DataLoader, TiktokenWrapper]:
    """
    Convenience factory: returns (dataloader, tokenizer).
    Dataloader yields tuples (input_ids, target_ids) as LongTensors.
    """
    tok = TiktokenWrapper("gpt2")
    ds = GPTDatasetV1(text, tok, max_length=max_length, stride=stride)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return dl, tok

