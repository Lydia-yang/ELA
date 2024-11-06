import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd
import json
from dataset.utils import pre_question
Image.MAX_IMAGE_PIXELS = None

class testDataset(Dataset):
    def __init__(self, file, transform=None, max_words=30):
        with open(file, 'r') as f:
            self.data = f.readlines()
        self.transform = transform
        self.max_words = max_words

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        line = line.strip('\n')
        query = line
        query = pre_question(query, self.max_words)
        return query, query, query

if __name__ == "__main__":
    data = baseDataset('')
    print(data.__len__(), data.__getitem__(0))