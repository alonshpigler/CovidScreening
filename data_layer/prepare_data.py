import logging
import pandas as pd

from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_layer import transforms, dataset

from data_layer.dataset import CovidDataset

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


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
    experiment = 1
    # TODO - implement drawing plates random split
    # modes = ['train', 'test']

    # mock_train =
    df = pd.read_csv(args.metadata_path)

    partitions = split_by_plates(df, experiment, args.plates_split[0],args.plates_split[1],args.test_samples_per_plate)
    partitions['train'], partitions['val'] = train_test_split(np.asarray(partitions['train']), train_size=args.split_ratio,
                                                            shuffle=True)

    datasets = create_datasets(args.plates_split[0], partitions, args.target_channel, args.input_size,args.device, args.num_input_channels)
    print_data_statistics(datasets)
    dataloaders = create_dataloaders(datasets, partitions,args.batch_size)

    return dataloaders


def split_by_plates(df, experiment, train_plates, test_plates, test_samples_per_plate=None,test_on_train_plates=False) -> dict:
    partitions = {
        'train': list(df[(df['plate'].isin(train_plates)) & (df['disease_condition'] == 'Mock') & (df['experiment']==('HRCE-'+str(experiment)))].index),
        # 'mock_test': list(df[(df['plate'].isin(test_plates)) & (df['disease_condition'] == 'Mock')].index),
        'test': {
            # 'mock': {},
            # 'irradiated_from_train_plates': list(
            #     df[(df['plate'].isin(train_plates)) & (df['disease_condition'] == 'UV Inactivated SARS-CoV-2')].index),
            # 'irradiated_from_train_plates':{},
            # 'irradiated_test_plate': list(df[(df['plate'].isin(test_plates)) & (df['disease_condition'] == 'UV Inactivated SARS-CoV-2')].index),
            # 'irradiated_from_test_plates': {},
            # 'active_from_train_plates':{},
            # 'active_from_train_plates': list(
            #     df[(df['plate'].isin(train_plates)) & (df['disease_condition'] == 'Active SARS-CoV-2')].index),
            # 'active_test_plate': list(df[(df['plate'].isin(train_plates)) & (df['disease_condition'] == 'Active SARS-CoV-2')].index)
            # 'active_from_test_plates': {}
        }
    }

    # divide test data into plates (irradiated and active from train plates)
    if test_on_train_plates:
        for plate in train_plates:
            partitions['test'][str(plate)] = {}
            partitions['test'][str(plate)]['irradiated_from_train_plates'] = list(
                df[(df['plate'] == plate) & (df['disease_condition'] == 'UV Inactivated SARS-CoV-2') & (df['experiment']==('HRCE-'+str(experiment)))].index)[:test_samples_per_plate]
            partitions['test'][str(plate)]['active_from_train_plates'] = list(
                df[(df['plate'] == plate) & (df['disease_condition'] == 'Active SARS-CoV-2') & (df['experiment']==('HRCE-'+str(experiment)))].index)[:test_samples_per_plate]
            # partitions['test_all_sets']

    # divide test data into plates (mock, irradiated and active from test plates)
    for plate in test_plates:
        partitions['test'][str(plate)] = {}
        partitions['test'][str(plate)]['mock'] = list(
            df[(df['plate'] == plate) & (df['disease_condition'] == 'Mock') & (
                        df['experiment'] == ('HRCE-' + str(experiment)))].index)[:test_samples_per_plate]
        partitions['test'][str(plate)]['irradiated_from_test_plates'] = list(
            df[(df['plate'] == plate) & (df['disease_condition'] == 'UV Inactivated SARS-CoV-2') & (
                        df['experiment'] == ('HRCE-' + str(experiment)))].index)[:test_samples_per_plate]
        partitions['test'][str(plate)]['active_from_test_plates'] = list(
            df[(df['plate'] == plate) & (df['disease_condition'] == 'Active SARS-CoV-2') & (
                        df['experiment'] == ('HRCE-' + str(experiment)))].index)[:test_samples_per_plate]

    return partitions


# def get_inds_by(df, plates:list,Type:str, Value:str, experiment = 'HRCE-1'):
#     """
#
#     :param plates: list containing the plate numbers
#     :param Type: The field the inds are gathered by. options:{'disease_condition','treatment'
#     :return: Value: The desired value in the field the inds to be gathered by
#     """
#     df[(df['plate'].isin(plates)) & (df['disease_condition'] == 'Active SARS-CoV-2')].index)

def create_datasets(train_plates, partitions, target_channel, input_size, device, num_input_channels):

    # mean, std = [0.5,0.4,0.3,0.2], [0.1,0.1,0.1,0.1]
    # mean, std = calc_mean_and_std(partitions['train'])
    mean, std = get_data_stats(partitions['train'], train_plates, device)
    # Y_mean, Y_std = mean[target_channel], std[target_channel]
    # X_mean, X_std = mean.remove(target_channel), std.remove(target_channel)

    train_transforms = transforms.Compose([
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)


    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ]
    )

    datasets = {
        'train': CovidDataset(partitions['train'], target_channel, transform=train_transforms, input_channels=num_input_channels),
        'val': CovidDataset(partitions['val'], target_channel, transform=train_transforms, input_channels=num_input_channels),
        'val_for_test':CovidDataset(partitions['val'], target_channel, transform=test_transforms, is_test=True, input_channels=num_input_channels),
        'test': {}
    }

    for plate in list(partitions['test'].keys()):
        datasets['test'][plate] = {}
        for key in partitions['test'][plate].keys():
            datasets['test'][plate][key] = \
                CovidDataset(partitions['test'][plate][key], target_channel, transform = test_transforms, is_test=True, input_channels=num_input_channels)

    print_data_statistics(datasets)

    return datasets


def print_data_statistics(datasets):

    print('train set contains ' + str(len(datasets['train'])) + ' images')
    print('val set contains ' + str(len(datasets['val'])) + ' images')

    for plate in list(datasets['test'].keys()):
        for key in datasets['test'][plate].keys():
            print(' test set from plate ' + plate + ' of ' + key + ' contains ' + str(len(datasets['test'][plate][key])) + ' images')


def get_data_stats(train_inds, train_plates, device):

    if train_plates==[1]:
        mean = [0.0168, 0.0667, 0.0783, 0.0395, 0.0480]
        std = [0.0114, 0.0946, 0.1034, 0.0492, 0.0811]
    elif train_plates == [1,2,3,4,5]:
        mean = [0.01589026488363743, 0.062156736850738525, 0.07818971574306488, 0.03667591139674187, 0.04590002819895744]
        std = [0.011114954948425293, 0.08782727271318436, 0.10349440574645996, 0.04710451513528824, 0.07981055229902267]
    else:
        logging.info('calculating mean and std...')
        mean, std = calc_mean_and_std(train_inds, len(train_plates), device)

    print('mean of mocks in train plates ' + str(train_plates) + 'is ' + str(mean))
    print('std of mocks in train plates ' + str(train_plates) + 'is ' + str(std))

    return mean, std


def calc_mean_and_std(inds, num_batches,device):

    # calculate_mean

    # num_samples = len(inds)

    train_data = dataset.CovidDataset(inds, target_channel=None, for_data_statistics_calc=True)
    batch_size = int(len(train_data)/num_batches)
    train_loader = DataLoader(train_data, batch_size=batch_size)
    num_channels = train_data.im_shape[2]

    mean = torch.zeros(num_channels).to(device)
    std = torch.zeros(num_channels).to(device)

    for images in train_loader:

        images = images.to(device)
        batch_mean, batch_std = torch.std_mean(images.float().div(255), dim = (0,1,2))

        mean += batch_mean
        std += batch_std

    mean /= num_batches
    std /= num_batches
    print('mean of train data is ' + str(mean.tolist()))
    print('std of train data is ' + str(std.tolist()))

    return mean.tolist(), std.tolist()


def create_dataloaders(datasets, partitions, batch_size) -> dict:
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size,
                            shuffle=True),
        'val': DataLoader(datasets['val'], batch_size=batch_size,
                          shuffle=False),
        'val_for_test':DataLoader(datasets['val_for_test'], batch_size=1,
                          shuffle=False),
        'test': {}
    }
    for plate in list(partitions['test'].keys()):
        dataloaders['test'][plate] = {}
        for key in partitions['test'][plate].keys():
            dataloaders['test'][plate][key] = \
                DataLoader(datasets['test'][plate][key], batch_size=1,
                                                         shuffle=False)
    return dataloaders
