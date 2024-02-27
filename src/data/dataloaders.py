from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision.io import read_image
import scipy.io as sio
import os
import numpy as np

import cv2

class DarkTestDataset(Dataset):
    def __init__(self, data_dir, transforms = None):
        self.data_dir = data_dir
        self.data = list(set([file.split('.')[0] for file in os.listdir(data_dir) if file.endswith(".jpg")]))
        self.transforms = transforms

    def __getitem__(self, idx):
        image = read_image(os.path.join(self.data_dir, self.data[idx]+".jpg"))
        if os.path.exists(os.path.join(self.data_dir, self.data[idx]+"_ann.mat")):
            points = sio.loadmat(os.path.join(self.data_dir, self.data[idx]+"_ann.mat"))['annPoints']
        else:
            points = np.array([])

        if self.transforms != None:
            image = self.transforms(image)
            
        return image, points

    def __len__(self):
        return len(self.data)

    