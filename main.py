import logging
import pickle
import random
import sys
import time
import scipy
import pandas as pd
from argparse import ArgumentParser
import numpy as np
import os
import time
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import config
from data_layer.dataset import CovidDataset
from data_layer.prepare_data import load_data
from model_layer.DummyAutoEncoder import LitAutoEncoder
from model_layer.UNET import Unet
from tqdm import tqdm
import matplotlib.pyplot as plt

from process_images import process_image
from util.files_operations import write_dict_to_csv_with_pandas, save_to_pickle, load_pickle
from visuals.util import show_input_and_target


def test_by_partition(model, test_dataloaders,input_size,exp_dir=None):

    res = {}
    res = pd.DataFrame()
    for plate in list(test_dataloaders):
        # res[plate] = {}
        for key in test_dataloaders[plate].keys():
            # res[plate][key] = []
            set_name = 'plate ' + plate + ', population ' + key
            plate_res = test(model, test_dataloaders[plate][key], input_size, set_name,exp_dir)
            # res[plate][key].append(results)
            res = pd.concat([res,plate_res])
    return res


def test(model, data_loader, input_size, title ='', save_dir=''):
    start=0
    results = pd.DataFrame(columns=['experiment','plate','well','site','disease_condition', 'treatment_conc','pcc'])
    # results = {}
    pccs=[]
    for i, (input, target,ind) in tqdm(enumerate(data_loader),total=len(data_loader)):

        # deformation to patches and reconstruction based on https://discuss.pytorch.org/t/creating-nonoverlapping-patches-from-3d-data-and-reshape-them-back-to-the-image/51210/6
        rec = data_loader.dataset.csv_file.iloc[ind]
        pred = process_image(model, input, input_size)
        pcc, p_value = scipy.stats.pearsonr(pred.flatten(), target.cpu().detach().numpy().flatten())
        results = results.append(rec, ignore_index=True)
        results.pcc[start] = pcc
        # pccs.append(pcc)

        if start == 0:
            show_input_and_target(input.cpu().detach().numpy().squeeze(), target.cpu().detach().numpy().squeeze(), pred, title, save_dir)
        start += 1

    # results['pcc'] = pccs

    return results


def analyze_results(res):
    pass
    # inter_plate_zfactor(res)
    # for plate in res:
    # z_factor()


def main(args):
    logging.info('Preparing data...')
    dataloaders = load_data(args)
    logging.info('Preparing data finished.')

    model = Unet(**args.model_args)
    args.checkpoint = config.get_checkpoint(args.log_dir, args.target_channel)
    if args.mode == 'predict' and args.checkpoint is not None:
        logging.info('loading model from file...')
        model = model.load_from_checkpoint(args.checkpoint)
        model.to(args.device)
        logging.info('loading model from file finished')

    else:
        logging.info('training model...')
        model = Unet(**args.model_args)
        # model = Unet(args)
        model.to(args.device)
        logger = TensorBoardLogger(args.log_dir, name="UNET on channel" + str(args.target_channel))
        if args.DEBUG:
            trainer = pl.Trainer(max_epochs=args.epochs, progress_bar_refresh_rate=1, logger=logger,
                                gpus=1)
        else:
            trainer = pl.Trainer(max_epochs=args.epochs, progress_bar_refresh_rate=1, logger=logger,gpus=1)
        trainer.fit(model, dataloaders['train'], dataloaders['val'])
        logging.info('training model finished.')

    if args.mode == 'analyze':
        logging.info('loading predictions from file...')
        res = load_pickle(os.path.join(args.exp_dir, 'results.pkl'))
        logging.info('loading predictions from file finished')

    else:
        logging.info('testing model...')
        if not args.DEBUG:
            res_on_val = test(model, dataloaders['val_for_test'], args.input_size, 'validation_set',args.exp_dir)
        res = test_by_partition(model, dataloaders['test'], args.input_size,args.exp_dir)
        # res.update(res_on_val)
        logging.info('testing model finished...')
        save_to_pickle(res, os.path.join(args.exp_dir, 'results.pkl'))
        save_to_pickle(args, os.path.join(args.exp_dir, 'args.pkl'))
        res.to_csv(os.path.join(args.exp_dir, 'results.csv'))

    # summerized_res = analyze_results(res)
    # write_dict_to_csv_with_pandas(summerized_res, os.path.join(args.exp_dir, 'results.csv'))

    # analyze_results(res)
    # res = {}
    # for key in list(dataloaders['test']):
    #     res[key] = {}
    #     for plate in dataloaders['test'][key].keys():
    #         res[key][plate] = trainer.test(autoencoder, dataloaders['test'][key][plate])


if __name__ == '__main__':
    exp_num = 1  # if None, new experiment directory is created with the next avaialible number
    channels_to_predict = [1,2,3,4,5]
    for target_channel in channels_to_predict:

        # torch.cuda.empty_cache()
        args = config.parse_args(exp_num,target_channel)

        args.mode = 'train'
        args.plates_split = [[1, 2, 3, 4, 5], [25]]
        args.DEBUG = False

        if args.DEBUG:
            args.test_samples_per_plate = 5
            # args.save_dir = True
            # args.epochs = 5
        args.test_samples_per_plate = 50
        args.batch_size = 24
        args.input_size = 256

        main(args)

