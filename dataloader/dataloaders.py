import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import Vocab
from torchtext.data import Iterator
from torch.utils.data import DataLoader
from typing import Dict, Tuple

from .constants import PAD_IDX, BATCH_SIZE, SRC_LANGUAGE, TGT_LANGUAGE


def get_dataloader(
    iter: Iterator,
    token_transform: Dict[str, any],
    vocab_transform: Dict[str, Vocab],
) -> Tuple[DataLoader, DataLoader]:
    # Define a collate function to process your batches
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_item, tgt_item in batch:
            src_batch.append(torch.tensor([vocab_transform[SRC_LANGUAGE][token] for token in token_transform[SRC_LANGUAGE](src_item)], dtype=torch.long))
            tgt_batch.append(torch.tensor([vocab_transform[TGT_LANGUAGE][token] for token in token_transform[TGT_LANGUAGE](tgt_item)], dtype=torch.long))

        # pad sequences and transpose batch from seq_len x batch_size to batch_size x seq_len
        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX).transpose(0, 1)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX).transpose(0, 1)

        return src_batch, tgt_batch

    # Convert your dataset into a map-style dataset (allows for easier batching and shuffling)
    dataset = to_map_style_dataset(iter)

    # Create a DataLoader to handle batching of your dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                             shuffle=True, collate_fn=collate_fn)

    # Sanity check data loaders
    small_dataloader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, range(min(1000, len(dataset) // 2))),
                                                   batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    return dataloader, small_dataloader