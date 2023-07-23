from torchtext.datasets import IWSLT2017
from torch.utils.data import DataLoader
from typing import Tuple

from .constants import SRC_LANGUAGE, TGT_LANGUAGE

def get_iters(path_to_data: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Get the training, validation and test data
    train_iter, val_iter, test_iter = IWSLT2017(
        path_to_data,
        split=('train', 'valid', 'test'),
        language_pair=(SRC_LANGUAGE, TGT_LANGUAGE)
    )

    return train_iter, val_iter, test_iter