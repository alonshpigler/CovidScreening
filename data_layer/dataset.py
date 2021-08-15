import os

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
# from configuration import config

from visuals.visualize import show_input_and_target

use_cuda = torch.cuda.is_available()
print("USE CUDA=" + str(use_cuda))
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


# DATA_DIR, METADATA_PATH, _ ,IMAGES_PATH, _ = config.get_paths()

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

    def __init__(self,
                 root_dir,
                 csv_file='metadata.csv',):

        self.metadata = pd.read_csv(os.path.join(root_dir,csv_file))
        # self.metadata = load_csv(path)

    def get_inds_by(self, plates:list, field:str, value:str, experiment ='HRCE-1'):
        """
            :param plates: list containing the plate numbers
            :param field: The field the inds are gathered by. options:{'disease_condition','treatment'
            :return: Value: The desired value in the field the inds to be gathered by
            """
        return list(self.metadata[(self.metadata['plate'].isin(plates)) & (self.metadata[field] == value)& (self.metadata['experiment'] == experiment)].index)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # img_name = os.path.join(self.metadata, self.image_frame.ix[id, 0])
        # image = cv2.imread(img_name)
        return self.metadata[idx]

    def get_ids_by_type(self, field, value):

        return self.metadata[self.metadata[field] == value]['site_id']


class CovidDataset(Dataset):

    def __init__(self,
                 inds,
                 target_channel,
                 root_dir,
                 input_channels = 4,
                 transform = None,
                 csv_file='metadata.csv',
                 for_data_statistics_calc = False,
                 is_test = False):

        super(Dataset).__init__()
        self.csv_file = pd.read_csv(os.path.join(root_dir,csv_file)).iloc[inds]
        self.root_dir = root_dir
        self.images_path = os.path.join(root_dir,'images')
        self.target_channel = target_channel
        self.input_channels = input_channels
        self.transform = transform

        rec = self.csv_file.iloc[0]
        sample = self.load_site(rec['experiment'], rec['plate'], rec['well'], rec['site'])
        self.im_shape = sample.shape
        self.for_data_statistics_calc = for_data_statistics_calc
        self.is_test = is_test

    def __len__(self):
        return len(self.csv_file)


    def load_images_as_tensor(self, image_paths, dtype=np.uint8):

        n_channels = len(image_paths)
        data = np.ndarray(shape=(1024, 1024, n_channels), dtype=dtype)

        for ix, img_path in enumerate(image_paths):
            data[:, :, ix] = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)

        return data

    def image_path(self,
                   experiment,
                   plate,
                   address,
                   site,
                   channel):
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
        return os.path.join(self.images_path, experiment, "Plate{}".format(plate),
                            "{}_s{}_w{}.png".format(address, site, channel))

    def load_site(self,
                  experiment,
                  plate,
                  well,
                  site
                  # target_channel=None,
                  # channels=DEFAULT_CHANNELS,
                  ):
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

        # if target_channel is not None:
        #
        #     input_channels.remove(target_channel)
        #
        #     target_path = [self.image_path(experiment, plate, well, site, target_channel, base_path=base_path)]
        #     input_paths = [
        #         self.image_path(experiment, plate, well, site, c, base_path=base_path)
        #         for c in input_channels
        #     ]
        #     return self.load_images_as_tensor(input_paths), self.load_images_as_tensor(target_path)
        # else:
        input_paths = [
            self.image_path(experiment, plate, well, site, c)
            for c in input_channels
        ]
        return self.load_images_as_tensor(input_paths)


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


    def __getitem__(self, id, show_sample = False):
        # rec =
        rec = self.csv_file.iloc[id]

        # if self.target_channel is not None:
        #     input, target = self.load_site(rec['experiment'], rec['plate'], rec['well'], rec['site'],self.target_channel)
        #     if self.transform:
        #         self.transform(input), target
        #     return input, target
        # else:
        input = self.load_site(rec['experiment'], rec['plate'], rec['well'], rec['site'])
        if not self.for_data_statistics_calc:
            if show_sample:
                trans_input = np.zeros((input.shape[2],input.shape[0],input.shape[1]))
                for i in range(5):
                    trans_input[i,:,:] = input[:,:,i]
                show_input_and_target(trans_input, title='before transforms')
            if self.transform:
                input = self.transform(input)
                if show_sample:
                    show_input_and_target(input, title='after transforms')
            if self.input_channels == 4:
                input, target = self.split_target_from_tensor(input, show_sample)
            elif self.input_channels == 1:
                input, target = input[self.target_channel-1:self.target_channel,:,:], input[self.target_channel-1:self.target_channel,:,:]
            elif self.input_channels == 5:
                target = input
            else:
                raise ValueError('Number of input channels is not supported')
            if self.is_test:
                # rec = dict_fields_to_str(rec.to_frame().to_dict()[rec.name])
                return input, target, id
            else:
                return input, target
        else:
            return input

        # image = cv2.imread(img_name)

    def split_target_from_tensor(self, input, show_sample=False):

        num_channels = input.shape[0]

        if self.target_channel == 1:

            target, input = torch.split(input, [1, num_channels - 1])
        elif self.target_channel == num_channels:
            input, target = torch.split(input, [num_channels - 1, 1])
        else:
            after = num_channels - self.target_channel
            before = num_channels - after - 1
            a, target, c = torch.split(input, [before, 1, after])
            input = torch.cat((a, c), dim=0)

        if show_sample:
            show_input_and_target(input.detach().numpy(), target.detach().numpy(),
                                  title='after split to train and target')

        return input, target

# class CovidUnsupervisedDataset(CovidDataset):
#
#     def __init__(self,
#                  *args, **kwargs):
#
#         # super(CovidDataset).__init__()
#         super(CovidUnsupervisedDataset, self).__init__(*args, **kwargs)
#


class CovidSupervisedDataset(CovidDataset):

    def __init__(self,
                 *args, **kwargs):

        # super(CovidDataset).__init__()
        super(CovidSupervisedDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, id, show_sample=False):
        # rec =
        rec = self.csv_file.iloc[id]

        # if self.target_channel is not None:
        #     input, target = self.load_site(rec['experiment'], rec['plate'], rec['well'], rec['site'],self.target_channel)
        #     if self.transform:
        #         self.transform(input), target
        #     return input, target
        # else:
        input = self.load_site(rec['experiment'], rec['plate'], rec['well'], rec['site'])
        target = self._get_target(rec['disease_condition'])
        if not self.for_data_statistics_calc:
            if self.transform:
                input = self.transform(input)
                if show_sample:
                    show_input_and_target(input, title='after transforms')
            # if self.is_test:
            #     rec = dict_fields_to_str(rec.to_frame().to_dict()[rec.name])
            #     return input, target

            return input, target, rec['plate']
        else:
            return input

    def _get_target(self, condition: str) -> int:

        if condition == 'Active SARS-CoV-2':
            target = 0
        elif condition == 'Mock' or condition == 'UV Inactivated SARS-CoV-2':
            target = 1
        else:
            target = 2

        return target
        # image = cv2.imread(img_name)
