import torch
from torch.utils.data import Dataset
import re


# Dataset for character-level name generation
class NameDataset(Dataset):

    def __init__(self, txt_file):
        # load names from file (one per line)
        with open(txt_file, 'r', encoding='utf-8') as f:
            self.names = [line.strip() for line in f if line.strip()]

        # build character vocabulary
        # collect all unique chars across dataset
        chars = sorted(list(set(''.join(self.names))))

        # special tokens
        self.PAD = 0   # padding
        self.SOS = 1   # start of sequence
        self.EOS = 2   # end of sequence

        # char → index mapping
        self.char2idx = {
            '<PAD>': self.PAD,
            '<SOS>': self.SOS,
            '<EOS>': self.EOS
        }

        # assign indices to actual characters (start from 3)
        for idx, char in enumerate(chars, start=3):
            self.char2idx[char] = idx

        # reverse mapping (for decoding later)
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

        self.vocab_size = len(self.char2idx)

    def __len__(self):
        # number of training examples
        return len(self.names)

    def __getitem__(self, idx):
        # get a single name string
        name = self.names[idx]

        # convert characters → indices
        indices = [self.char2idx[c] for c in name]

        # input sequence:
        # <SOS> + characters
        x = [self.SOS] + indices

        # target sequence:
        # characters + <EOS>
        # (this is basically "next character" prediction)
        y = indices + [self.EOS]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )


# Custom collate function for batching variable-length sequences
def collate_fn(batch):
    # batch is a list of (x, y) pairs
    xs, ys = zip(*batch)

    # store original lengths (useful if you want masking later)
    lengths = torch.tensor([len(x) for x in xs])

    # pad sequences to same length (important for batching)
    # shape after padding: (B, T_max)
    xs_padded = torch.nn.utils.rnn.pad_sequence(
        xs, batch_first=True, padding_value=0
    )

    ys_padded = torch.nn.utils.rnn.pad_sequence(
        ys, batch_first=True, padding_value=0
    )

    return xs_padded, ys_padded, lengths