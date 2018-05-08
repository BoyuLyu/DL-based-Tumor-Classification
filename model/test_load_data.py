import numpy as np
from skimage import io
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim
import os
from skimage import io
import pandas as pd
import numpy as np
from skimage import img_as_float
from PIL import Image

test_csv_path ='/home/boyu93/VT_coursework/Computation_Life_Sci/project/Project/preprocessed_results_1/img_fold9/test/labels_test.csv'
test_root_path ='/home/boyu93/VT_coursework/Computation_Life_Sci/project/Project/preprocessed_results_1/img_fold9/test'

"""
test_root_path = '/Users/apple/Documents/Study/VT coursework/Computation for Life Sciences II/Project/Editing_file/test_results/test1_e50_b500_lr01/test'
test_csv_path = '/Users/apple/Documents/Study/VT coursework/Computation for Life Sciences II/Project/Editing_file/test_results/test1_e50_b500_lr01/test/labels_test.csv'
"""



class TumorDatasetTest(Dataset):

    def __init__(self, csv_file , root_dir, transform=None):
        self.labels_frame = np.array(pd.read_csv(csv_file, skiprows= 1, sep=',', header= None)).astype(np.float32)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = str(idx) + '.png'
        img_path = os.path.join(self.root_dir, img_name)
        """
        !!!Pay attention!!!
        The image size is set here
        """
        img = np.empty(shape=(1, 66, 66))
        img[0, :, :] = (img_as_float(io.imread(img_path)) - 0.5)/0.5
        label = np.array([self.labels_frame[idx, 1]-1])
        test_sample = {'image': img, 'label': label}

        if self.transform:
            test_sample = self.transform(test_sample)
        return test_sample


class ToTensor(object):

    def __call__(self, sample):
        image, labels = sample['image'], sample['label']
        return {'image': torch.from_numpy(image), 'label': torch.torch.LongTensor(labels)}

