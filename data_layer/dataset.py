import os

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from util.files_operations import load_csv

DEFAULT_BASE_PATH = 'C:/Covid-Screening2/data_layer/raw_data'
DEFAULT_METADATA_BASE_PATH = os.path.join(DEFAULT_BASE_PATH, 'metadata')
DEFAULT_IMAGES_BASE_PATH = os.path.join(DEFAULT_BASE_PATH, 'images')
# DEFAULT_IMAGE_FRAME_PATH = os.path.join(DEFAULT_IMAGES_BASE_PATH, 'image_frame.csv')
DEFAULT_CHANNELS = (1, 2, 3, 4, 5)

RGB_MAP = {
    1: {
        'rgb': np.array([19, 0, 249]),
        'range': [0, 51]
    },
    2: {
        'rgb': np.array([42, 255, 31]),
        'range': [0, 107]
    },
    3: {
        'rgb': np.array([255, 0, 25]),
        'range': [0, 64]
    },
    4: {
        'rgb': np.array([45, 255, 252]),
        'range': [0, 191]
    },
    5: {
        'rgb': np.array([250, 0, 253]),
        'range': [0, 89]
    },
    6: {
        'rgb': np.array([254, 255, 40]),
        'range': [0, 191]
    }
}


class CovidMetadata(Dataset):

    def __init__(self, path):

        self.metadata = pd.read_csv("C:\\Covid-Screening\\data_layer\\raw_data\\metadata.csv")
        # self.metadata = load_csv(path)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.metadata, self.image_frame.ix[id, 0])
        image = cv2.imread(img_name)
        return image

    def get_ids_by_type(self, field, value):

        return self.metadata[self.metadata[field] == value]['site_id']


class CovidDataset(Dataset):

    def __init__(self, inds, target_channel, root_dir=DEFAULT_BASE_PATH, csv_file='metadata.csv'):

        self.csv_file = pd.read_csv(os.path.join(self.root_dir,csv_file)).iloc[inds]
        self.root_dir = root_dir
        self.target_channel = target_channel

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, id):
        # rec =
        rec = self.csv_file.iloc[id]
        input, target = self.load_site(rec['experiment'], rec['plate'], rec['well'], rec['site'],self.target_channel)
        # image = cv2.imread(img_name)

        return input, target

    def load_images_as_tensor(self, image_paths, dtype=np.uint8):

        n_channels = len(image_paths)
        data = np.ndarray(shape=(1024, 1024, n_channels), dtype=dtype)

        for ix, img_path in enumerate(image_paths):
            data[:, :, ix] = cv2.imread(img_path)

        return data

    def image_path(self,
                   dataset,
                   experiment,
                   plate,
                   address,
                   site,
                   channel,
                   base_path=DEFAULT_IMAGES_BASE_PATH):
        """
        Returns the path of a channel image.
        Parameters
        ----------
        dataset : str
            what subset of the data: train, test
        experiment : str
            experiment name
        plate : int
            plate number
        address : str
            plate address
        site : int
            site number
        channel : int
            channel number
        base_path : str
            the base path of the raw images
        Returns
        -------
        str the path of image
        """
        return os.path.join(base_path, dataset, experiment, "Plate{}".format(plate),
                            "{}_s{}_w{}.png".format(address, site, channel))

    def load_site(self,
                  dataset,
                  experiment,
                  plate,
                  well,
                  site,
                  target_channel,
                  channels=DEFAULT_CHANNELS,
                  base_path=DEFAULT_IMAGES_BASE_PATH):
        """
        Returns the image data of a site
        Parameters
        ----------
        dataset : str
            what subset of the data: train, test
        experiment : str
            experiment name
        plate : int
            plate number
        address : str
            plate address
        site : int
            site number
        channels : list of int
            channels to include
        base_path : str
            the base path of the raw images
        Returns
        -------
        np.ndarray the image data of the site
        """

        input_channels = list(DEFAULT_CHANNELS)
        input_channels.remove(target_channel)

        target_path = self.image_path(dataset, experiment, plate, well, site, target_channel, base_path=base_path)
        input_paths = [
            self.image_path(dataset, experiment, plate, well, site, c, base_path=base_path)
            for c in channels
        ]
        return self.load_images_as_tensor(input_paths), self.load_images_as_tensor(target_path)

    def convert_tensor_to_rgb(t, channels=DEFAULT_CHANNELS, vmax=255, rgb_map=RGB_MAP):
        """
        Converts and returns the image data as RGB image
        Parameters
        ----------
        t : np.ndarray
            original image data
        channels : list of int
            channels to include
        vmax : int
            the max value used for scaling
        rgb_map : dict
            the color mapping for each channel
            See rxrx.io.RGB_MAP to see what the defaults are.
        Returns
        -------
        np.ndarray the image data of the site as RGB channels
        """
        colored_channels = []
        for i, channel in enumerate(channels):
            x = (t[:, :, i] / vmax) / \
                ((rgb_map[channel]['range'][1] - rgb_map[channel]['range'][0]) / 255) + \
                rgb_map[channel]['range'][0] / 255
            x = np.where(x > 1., 1., x)
            x_rgb = np.array(
                np.outer(x, rgb_map[channel]['rgb']).reshape(512, 512, 3),
                dtype=int)
            colored_channels.append(x_rgb)
        im = np.array(np.array(colored_channels).sum(axis=0), dtype=int)
        im = np.where(im > 255, 255, im)
        return im

