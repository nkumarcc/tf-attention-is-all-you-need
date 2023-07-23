from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torch.utils.data import DataLoader
from typing import Dict, Tuple

from .constants import SRC_LANGUAGE, TGT_LANGUAGE, UNK_IDX, special_symbols

def get_vocab_transform(train_iter: DataLoader) -> Tuple[Dict[str, any], Dict[str, Vocab]]:
    # Define the tokenizers
    token_transform = {}
    token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
    token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

    # Define function to yield list of tokens
    def yield_tokens(data_iter, language):
        language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

        for data_sample in data_iter:
            yield token_transform[language](data_sample[language_index[language]])

    # Building vocab
    vocab_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)

    # Set UNK_IDX as the default index. This index is returned when the token is not found.
    # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab_transform[ln].set_default_index(UNK_IDX)
    
    return token_transform, vocab_transform