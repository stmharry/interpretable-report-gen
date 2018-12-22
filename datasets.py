import torch
from torch.utils.data import Dataset
import h5py
import json
import os

class MimicDataset(Dataset):
    """
    Mimic-CXR dataset.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """

        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Load encoded reports (completely into memory)
        with open(os.path.join(data_folder, self.split + '_REPORT_' + data_name + '.json'), 'r') as j:
            self.reports = json.load(j)

        # Load sentences lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_SENLENS_' + data_name + '.json'), 'r') as j:
            self.senlens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.reports)

    def __getitem__(self, i):

        img = torch.FloatTensor(self.imgs[i] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        # List of sentences in a report
        report = torch.LongTensor(self.reports[i])

        # List of sentences' length in a report
        senlen = torch.LongTensor([self.senlens[i]])

        return img, report, senlen

    def __len__(self):
        return self.dataset_size