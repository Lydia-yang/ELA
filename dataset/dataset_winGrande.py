import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd
import json
from dataset.utils import pre_question
import random
Image.MAX_IMAGE_PIXELS = None

class winGDataset(Dataset):
    def __init__(self, file, image_file=None, image_dir=None, transform=None, max_words=30):
        with open(file, 'r') as f:
            self.data = f.readlines()
        self.transform = transform
        self.max_words = max_words
        if image_file:
            image_file = json.load(open(image_file, 'r'))
            self.image_file = image_file['images']
            self.query = image_file['querys']
        else:
            self.image_file = None
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        line = line.strip('\n')
        value = json.loads(line)
        query = value['sentence']
        query = pre_question(query, self.max_words)
        label = int(value['answer'])-1
        answers = [value['option1'], value['option2']]
        if self.image_file:
            if query!=self.query[idx]:
                print("wrong:"+query)
            num = random.randint(0,49)
            image_path = os.path.join(self.image_dir, self.image_file[idx][num])        
            image = Image.open(image_path).convert('RGB')   
            image = self.transform(image)
            return image, query, answers, label
        return query, answers, label
