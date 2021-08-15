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

    partitions = split_by_plates(df, experiment, args.plates_split[0],args.plates_split[1],args.test_samples_per_plate,supervised=args.supervised)
    partitions['train'], partitions['val'] = train_test_split(np.asarray(partitions['train']), train_size=args.split_ratio,
                                                            shuffle=True)

    datasets = create_datasets(args.plates_split[0], partitions, args.dataset, args.data_path, args.target_channel, args.input_size,args.device, args.num_input_channels,supervised=args.supervised)
    print_data_statistics(datasets,args.supervised)
    dataloaders = create_dataloaders(datasets, partitions,args.batch_size,supervised=args.supervised)

    return dataloaders


def split_by_plates(df, experiment, train_plates, test_plates, test_samples_per_plate=None,test_on_train_plates=False,supervised=False) -> dict:

    if supervised:
        partitions = {
            'train': list(df[(df['plate'].isin(train_plates)) & (df['disease_condition'].isin(['Mock','UV Inactivated SARS-CoV-2','Active SARS-CoV-2'])) & (
                        df['experiment'] == ('HRCE-' + str(experiment))) & (df['treatment'].isna())].index),
            'test': list(df[(df['plate'].isin(test_plates)) & (df['disease_condition'].isin(['Mock','UV Inactivated SARS-CoV-2','Active SARS-CoV-2'])) & (
                        df['experiment'] == ('HRCE-' + str(experiment)))& (df['treatment'].isna())].index),
        }
    else:
        partitions = {
            'train': list(df[(df['plate'].isin(train_plates)) & (df['disease_condition'] == 'Mock') & (df['experiment']==('HRCE-'+str(experiment)))].index),
            # 'mock_test': list(df[(df['plate'].isin(test_plates)) & (df['disease_condition'] == 'Mock')].index),
            'test': {

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


def create_datasets(train_plates, partitions, Dataset, data_dir, target_channel, input_size, device, num_input_channels,supervised=False):

    # mean, std = [0.5,0.4,0.3,0.2], [0.1,0.1,0.1,0.1]
    # mean, std = calc_mean_and_std(partitions['train'])

    # Y_mean, Y_std = mean[target_channel], std[target_channel]
    # X_mean, X_std = mean.remove(target_channel), std.remove(target_channel)
    mean, std = get_data_stats(partitions['train'], train_plates, data_dir, device, supervised)
    if not supervised:

        train_transforms = transforms.Compose([
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
    ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:

        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(input_size),
            transforms.Normalize(mean, std)
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(input_size),
            transforms.Normalize(mean, std)
        ])

    datasets = {
        'train': Dataset(partitions['train'], target_channel, root_dir = data_dir, transform=train_transforms, input_channels=num_input_channels),
        'val': Dataset(partitions['val'], target_channel, root_dir = data_dir, transform=train_transforms, input_channels=num_input_channels),
        'val_for_test': Dataset(partitions['val'], target_channel, root_dir = data_dir, transform=test_transforms, is_test=True, input_channels=num_input_channels),
        'test': {}
    }
    if not supervised:
        for plate in list(partitions['test'].keys()):
            datasets['test'][plate] = {}
            for key in partitions['test'][plate].keys():
                datasets['test'][plate][key] = \
                    Dataset(partitions['test'][plate][key], target_channel, root_dir = data_dir, transform = test_transforms, is_test=True, input_channels=num_input_channels)
    else:
        datasets['test'] = Dataset(partitions['test'], target_channel, root_dir = data_dir, transform = test_transforms, is_test=True, input_channels=num_input_channels)

    return datasets


def print_data_statistics(datasets,supervised=False):

    print('train set contains ' + str(len(datasets['train'])) + ' images')
    print('val set contains ' + str(len(datasets['val'])) + ' images')
    if not supervised:
        for plate in list(datasets['test'].keys()):
            for key in datasets['test'][plate].keys():
                print(' test set from plate ' + plate + ' of ' + key + ' contains ' + str(len(datasets['test'][plate][key])) + ' images')
    else:
        print(' test set contains ' + str(len(datasets['test'])) + ' images')


def get_data_stats(train_inds, train_plates, data_dir, device,supervised=False):
    if supervised:
        if train_plates == [1]:
            mean = [0.018005717545747757, 0.07998745888471603, 0.08620914071798325, 0.04662375524640083, 0.05453427508473396]
            std = [0.012143701314926147, 0.10490397363901138, 0.11406225711107254, 0.05428086221218109, 0.09004668891429901]
        elif train_plates == [1, 2]:
            mean = [0.015809599310159683, 0.0705508440732956, 0.07977999001741409, 0.04070518910884857, 0.04949751868844032]
            std = [0.01072636153548956, 0.09140551835298538, 0.10539335012435913, 0.047243546694517136, 0.08209426701068878]
        elif train_plates == [1, 2, 3, 4, 5]:
            mean = [0.016160082072019577, 0.06823156774044037, 0.07900794595479965, 0.040070559829473495, 0.048638951033353806]
            std = [0.011033792980015278, 0.08838692307472229, 0.10397928953170776, 0.04720102623105049, 0.08068393170833588]
        elif train_plates == list(np.arange(1,26)):
            mean = [0.016741905361413956, 0.06599768251180649, 0.08367346227169037, 0.04055846482515335, 0.050428975373506546]
            std =[0.011758865788578987, 0.08694437891244888, 0.10990524291992188, 0.04767781123518944, 0.08285659551620483]
        elif train_plates == [8,6,17,14]:
            mean = [0.017729641869664192, 0.07067170739173889, 0.08315683901309967, 0.042248740792274475, 0.05103209614753723]
            std = [0.011853627860546112, 0.09128410369157791, 0.10873471945524216, 0.04914622753858566, 0.08351156115531921]
        elif train_plates == [11, 21, 2, 4]:
            mean = [0.01602199673652649, 0.07256611436605453, 0.08944295346736908, 0.04201145097613335, 0.05323413759469986]
            std = [0.011012655682861805, 0.09150492399930954, 0.11658839136362076, 0.04829978570342064, 0.0870295837521553]
        else:
            logging.info('calculating mean and std...')
            mean, std = calc_mean_and_std(train_inds, data_dir, len(train_plates)*8, device)

    else:
        if train_plates==[1]:
            mean = [0.0168, 0.0667, 0.0783, 0.0395, 0.0480]
            std = [0.0114, 0.0946, 0.1034, 0.0492, 0.0811]
        elif train_plates == [1,2,3,4,5]:
            mean = [0.01589026488363743, 0.062156736850738525, 0.07818971574306488, 0.03667591139674187, 0.04590002819895744]
            std = [0.011114954948425293, 0.08782727271318436, 0.10349440574645996, 0.04710451513528824, 0.07981055229902267]
        else:
            logging.info('calculating mean and std...')
            mean, std = calc_mean_and_std(train_inds,data_dir, len(train_plates), device)

    return mean, std


def calc_mean_and_std(inds, data_dir, num_batches,device):

    # calculate_mean

    # num_samples = len(inds)

    train_data = dataset.CovidDataset(inds, root_dir=data_dir, target_channel=None, for_data_statistics_calc=True)
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


def create_dataloaders(datasets, partitions, batch_size, supervised=False,num_workers=32) -> dict:


    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size,
                            shuffle=True,num_workers=num_workers),
        'val': DataLoader(datasets['val'], batch_size=batch_size,
                          shuffle=False,num_workers=num_workers),
        'val_for_test':DataLoader(datasets['val_for_test'], batch_size=1,
                          shuffle=False,num_workers=num_workers),
        'test': {}
    }
    if not supervised:
        for plate in list(partitions['test'].keys()):
            dataloaders['test'][plate] = {}
            for key in partitions['test'][plate].keys():
                dataloaders['test'][plate][key] = \
                    DataLoader(datasets['test'][plate][key], batch_size=1,
                                                  shuffle=False,num_workers=num_workers)
    else:
        dataloaders['test'] = DataLoader(datasets['test'], batch_size=4,
                                              shuffle=False,num_workers=num_workers)
        dataloaders['pca_all'] = DataLoader(datasets['val'], batch_size=len(datasets['val']),
                                        shuffle=False,num_workers=num_workers)
        dataloaders['pca_batches'] = DataLoader(datasets['val'], batch_size=15,
                                              shuffle=False,num_workers=num_workers)
        dataloaders['test_pca_batches'] = DataLoader(datasets['test'], batch_size=15,
                                              shuffle=False,num_workers=num_workers)
    return dataloaders
