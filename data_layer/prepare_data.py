import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from data_layer.dataset import CovidDataset
from util.files_operations import *


def load_data(args):
    """

    :param args:
        metadata_path: path to image filenames
        plates_split: dict containing:
            train: plates numbers used for training
            test: plates numbers used for test
        split_ratio (float in [0,1]): train-val split param
        target_channel (int): channel to predict

    :return:
    """
    plates = [1, 25]
    train_plates = [25]
    test_plates = [1]
    # TODO - implement drawing plates random split
    # modes = ['train', 'test']

    # mock_train =
    df = pd.read_csv(args.metadata_path)

    partitions = split_by_plates(df, args.plates_split['train'],args.plates_split['test'])
    partitions['train'], partitions['val'] = train_test_split(np.asarray(partitions['train']), train_size=args.split_ratio,
                                                            shuffle=True)

    datasets = create_datasets(partitions, args.target_channel)
    dataloaders = create_dataloaders(datasets, partitions,args.batch_size)

    return dataloaders


def split_by_plates(df, train_plates, test_plates) -> dict:
    partitions = {
        'train': list(df[(df['plate'].isin(train_plates)) & (df['disease_condition'] == 'Mock')].index),
        # 'mock_test': list(df[(df['plate'].isin(test_plates)) & (df['disease_condition'] == 'Mock')].index),
        'test': {
            'mock': {},
            'irradiated_from_train_plates': list(
                df[(df['plate'].isin(train_plates)) & (df['disease_condition'] == 'UV Inactivated SARS-CoV-2')].index),
            # 'irradiated_test_plate': list(df[(df['plate'].isin(test_plates)) & (df['disease_condition'] == 'UV Inactivated SARS-CoV-2')].index),
            'irradiated_from_test_plates': {},
            'active_from_train_plates': list(
                df[(df['plate'].isin(train_plates)) & (df['disease_condition'] == 'Active SARS-CoV-2')].index),
            # 'active_test_plate': list(df[(df['plate'].isin(train_plates)) & (df['disease_condition'] == 'Active SARS-CoV-2')].index)
            'active_from_test_plates': {}
        }
    }
    # divide test data into plates (irradiated and active from train plates)
    for plate in train_plates:
        partitions['test']['irradiated_from_train_plates'][str(plate)] = list(
            df[(df['plate'] == plate) & (df['disease_condition'] == 'UV Inactivated SARS-CoV-2')].index),
        partitions['test']['active_from_train_plates'][str(plate)] = list(
            df[(df['plate'] == plate) & (df['disease_condition'] == 'Active SARS-CoV-2')].index)
    # divide test data into plates (mock, irradiated and active from test plates)
    for plate in test_plates:
        partitions['test']['mock'][str(plate)] = list(
            df[(df['plate'] == plate) & (df['disease_condition'] == 'Mock')].index)
        partitions['test']['irradiated_from_test_plates'][str(plate)] = list(
            df[(df['plate'] == plate) & (df['disease_condition'] == 'UV Inactivated SARS-CoV-2')].index),
        partitions['test']['active_from_test_plates'][str(plate)] = list(
            df[(df['plate'] == plate) & (df['disease_condition'] == 'Active SARS-CoV-2')].index)
    return partitions


def create_datasets(partitions, target_channel):
    datasets = {
        'train': CovidDataset(partitions['train'], target_channel),
        'val': CovidDataset(partitions['val'], target_channel),
        'test': {}
    }
    for key in list(partitions['test'].keys()):
        for plate in partitions['test'][key].keys():
            datasets['test'][key][plate] = \
                CovidDataset(partitions['test'][key][plate], target_channel)
    return datasets


def create_dataloaders(datasets, partitions, batch_size) -> dict:
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size,
                            shuffle=True),
        'val': DataLoader(datasets['val'], batch_size=batch_size,
                          shuffle=False),
        'test': {}
    }
    for key in list(partitions['test'].keys()):
        for plate in partitions['test'][key].keys():
            dataloaders['test'][key][plate] = \
                DataLoader(datasets[key][plate], batch_size=batch_size,
                                                         shuffle=False)
    return dataloaders
