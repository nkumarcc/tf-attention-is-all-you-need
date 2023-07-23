# Set the source and target language
SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

BATCH_SIZE = 32

# Define the special tokens
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
UNK = '<unk>'
PAD = '<pad>'
BOS = '<bos>'
EOS = '<eos>'
special_symbols = [UNK, PAD, BOS, EOS]
