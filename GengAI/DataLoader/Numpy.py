import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import random

class Dataset(Dataset):
    def __init__(self, class_paths, transform=None):
        """
        Args:
            image_dir (str): Path to the folder containing images.
            annotations (dict): A dictionary where keys are image filenames, 
                                 and values are dicts containing 'boxes' and 'labels'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = []
        for idx, classes in enumerate(class_paths):
            for _, _, files in os.walk(classes):
                for file in files:
                    self.data.append( (idx, file) )
        random.shuffle( self.data )
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data     = torch.from_numpy(np.load(self.data[idx][1]))
        category = self.data[idx][0]
        return self.transform(data), category