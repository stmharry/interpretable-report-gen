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

        # Reports per image
        self.rpi = self.h.attrs['reports_per_image']

        # Load encoded reports (completely into memory)
        with open(os.path.join(data_folder, self.split + '_REPORT_' + data_name + '.json'), 'r') as j:
            self.reports = json.load(j)

        # Load report lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_REPLENS_' + data_name + '.json'), 'r') as j:
            self.replens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.reports)

    def __getitem__(self, i):
        # Remember, the Nth report corresponds to the (N // reports_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.rpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        report = torch.LongTensor(self.reports[i])

        replen = torch.LongTensor([self.replens[i]])

        if self.split is 'TRAIN':
            return img, report, replen
        else:
            # For validation of testing, also return all 'reports_per_image' reports to find BLEU-4 score
            all_reports = torch.LongTensor(
                self.reports[((i // self.rpi) * self.rpi):(((i // self.rpi) * self.rpi) + self.rpi)])
            return img, report, replen, all_reports

    def __len__(self):
        return self.dataset_size