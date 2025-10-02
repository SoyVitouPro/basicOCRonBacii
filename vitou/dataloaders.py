import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


# ----- Vocabulary for CTC -----
class OCRVocab:
    """
    For CTC: use PAD=0 as the blank index.
    """
    PAD, UNK = 0, 1

    def __init__(self, charset: Optional[str] = None):
        if charset is None:
            # For Khmer, you should include full Unicode charset from your data.
            # This auto-builds charset from dataset if not given.
            charset = ""
        self.chars = charset
        self.c2i = {c: i + 2 for i, c in enumerate(self.chars)}  # start from 2
        self.i2c = {i + 2: c for i, c in enumerate(self.chars)}

    def build_from_labels(self, labels: List[str]):
        charset = sorted(set("".join(labels)))
        self.chars = "".join(charset)
        self.c2i = {c: i + 2 for i, c in enumerate(self.chars)}
        self.i2c = {i + 2: c for i, c in enumerate(self.chars)}

    @property
    def num_classes(self) -> int:
        """CTC: include PAD=0 as blank, plus UNK=1, plus all chars."""
        return 2 + len(self.chars)

    def encode(self, text: str) -> List[int]:
        return [self.c2i.get(ch, self.UNK) for ch in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.i2c.get(i, "�") for i in ids if i > 1)


# ----- Dataset -----
class OCRJsonDataset(Dataset):
    def __init__(self, json_path: str, vocab: OCRVocab, transform: Optional[T.Compose] = None):
        self.samples = json.load(open(json_path, encoding="utf-8"))
        self.vocab = vocab

        # Default transforms
        self.transform = transform or T.Compose([
            T.Grayscale(),
            T.Resize((32, 128)),  # adjust H,W for your model
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path, text = sample["image_path"], sample["label"]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label_ids = self.vocab.encode(text)
        return img, torch.tensor(label_ids, dtype=torch.long)


# ----- Collate for CTC -----
def ctc_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_value: int = OCRVocab.PAD):
    imgs, labels = zip(*batch)
    images = torch.stack(imgs, dim=0)

    lengths = torch.tensor([len(t) for t in labels], dtype=torch.long)
    max_len = int(lengths.max())
    padded = torch.full((len(labels), max_len), pad_value, dtype=torch.long)

    for i, t in enumerate(labels):
        padded[i, : len(t)] = t

    return {
        "images": images,            # [B, C, H, W]
        "labels": padded,            # [B, Lmax]
        "label_lengths": lengths,    # [B]
    }
