"""Implement custom dataset for FER."""

import os
import pickle
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image


class FerDataset(Dataset):

    def __init__(self, base_path, data='fer', mode='train',
                 label='fer_emotion', transform=None):
        """Define FER-specific dataset.

        Args:
            base_path (string): path to base folder containing output of create_dataset.py
            data (string): one of {'fer', 'ferplus'}
            mode (string): one of {'train', 'eval', 'test'}
            label (string): one of {'fer_emotion', 'ferplus_max', 'ferplus_votes'}
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Sanity checks
        if data not in ['fer', 'ferplus']:
            raise ValueError('Choose one of {"fer", "ferplus"} as data.')
        if mode not in ['train', 'eval', 'test']:
            raise ValueError('Choose one of {"train", "eval", "test"} as mode.')
        if label not in ['fer_emotion', 'ferplus_max', 'ferplus_votes']:
            raise ValueError('Choose one of {"fer_emotion", "ferplus_max", "ferplus_votes"} as label.')

        self.base = base_path
        if transform is not None:
            raise NotImplementedError('Additional transforms beside normalization unsupported as of yet.')

        # Load mean and std for normalization
        self.norm_mean = self._load_pickle(os.path.join(self.base, 'mean_Training.pkl'))
        self.norm_std = self._load_pickle(os.path.join(self.base, 'std_Training.pkl'))

        # Read the label info
        self.image_names = []
        self.labels = None
        self.n_classes = 7 if label == 'fer_emotion' else 10
        self._prepare_data(data, mode, label)
        self.data_len = len(self.labels)

    @staticmethod
    def _load_pickle(path):
        with open(path, 'rb') as fp:
            content = pickle.load(fp)
        return content

    def _prepare_data(self, data, mode, label):

        # Load labels from pickle
        modes = {'train': 'Training', 'eval': 'PublicTest', 'test': 'PrivateTest'}
        mode = modes[mode]
        info_list = self._load_pickle(
            os.path.join(self.base, '{}_{}.pkl'.format(data, mode)))
        print('{}_{}.pkl'.format(data, mode))

        # Prepare labels
        emotions = []
        votes_all = []
        votes_max = []
        for image_name, emotion, votes in info_list:
            self.image_names.append(os.path.join(self.base, mode, image_name))
            emotions.append(emotion)
            votes_all.append(votes)
            votes_max.append(np.argmax(votes))

        self.image_names = np.array(self.image_names)
        self.data_len = len(info_list)

        onehot = np.zeros((self.data_len, self.n_classes))
        indices = emotions if label == 'fer_emotion' else votes_max
        onehot[np.arange(self.data_len), indices] = 1.

        votes_all = np.array(votes_all, dtype=np.float) / 10.

        if label == 'ferplus_votes':
            self.labels = torch.tensor(votes_all)
        else:
            self.labels = torch.tensor(onehot)

    def __getitem__(self, index):
        # Load and preprocess image
        image_name = self.image_names[index]
        image = Image.open(image_name)
        image = (image - self.norm_mean) / self.norm_std
        image = torch.tensor(image).expand(1, -1, -1)

        # Get label
        label = self.labels[index]

        return (image, label)

    def __len__(self):
        return self.data_len
