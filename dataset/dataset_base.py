import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd
import json
from dataset.utils import pre_question
Image.MAX_IMAGE_PIXELS = None

class baseDataset(Dataset):
    def __init__(self, file, transform=None, img_dir=None, max_words=30):
        self.data = json.load(open(file,'r'))
        self.key = list(self.data.keys())
        self.transform = transform
        self.img_dir = img_dir
        self.max_words = max_words

    def __len__(self):
        return len(self.key)

    def __getitem__(self, idx):
        id = self.key[idx]
        value = self.data[id]
        if self.img_dir:
            path = os.path.join(self.img_dir, value)
            value = Image.open(path)
        else:
            value = pre_question(value, self.max_words)
        if self.transform:
           value = self.transform(value)
        return int(id), value

if __name__ == "__main__":
    data = baseDataset('')
    print(data.__len__(), data.__getitem__(0))