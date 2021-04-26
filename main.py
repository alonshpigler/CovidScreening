import logging
import pickle
import random
import sys
import time
from argparse import ArgumentParser
import numpy as np
import os
from pathlib import Path

import torch
import pytorch_lightning as pl
from data_layer.dataset import CovidDataset
from data_layer.prepare_data import load_data
from model_layer.DummyAutoEncoder import LitAutoEncoder

print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDA VERSION')
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')
# call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
# print('Active CUDA Device: GPU', torch.cuda.current_device())
# print('Available devices ', torch.cuda.device_count())
# print('Current cuda device ', torch.cuda.current_device())
use_cuda = torch.cuda.is_available()
print("USE CUDA=" + str(use_cuda))
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

def parse_args():

    parser = ArgumentParser()
    parser.add_argument('-m','--mode', type = str,  choices=('train', 'val', 'predict'))
    ROOT_DIR = Path(__file__).parent
    data_dir = f"{ROOT_DIR}\\data_layer\\raw_data\\"
    parser.add_argument('--data_path', type=Path, default=os.path.join(data_dir,'images\\'),
                        help='path to the data root. It assumes format like in Kaggle with unpacked archives')
    parser.add_argument('--metadata_path', type=Path, default=os.path.join(data_dir,'metadata.csv'),
            help='path to the data root. It assumes format like in Kaggle with unpacked archives')
    parser.add_argument('--plates_split', type=list, default=[[1],[25]],
                        help='plates split between train and test. left is train and right is test')
    parser.add_argument('--split-ratio', type=int, default=0.8,
                        help='split ratio between train and validation. value in [0,1]')

    parser.add_argument('--data-split-seed', type=int, default=0,
            help='seed for splitting experiments for folds')
    parser.add_argument('--num-data-workers', type=int, default=4,
            help='number of data loader workers')
    parser.add_argument('--device',type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help='device for running code')
    parser.add_argument('--seed', type=int,
            help='global seed (for weight initialization, data sampling, etc.). '
                 'If not specified it will be randomized (and printed on the log)')

    parser.add_argument('--target_channel', type=int, default=5, choices=(1,2,3,4,5),
                        help='the channel predicted by the network')
    parser.add_argument('-b', '--batch_size', type=int, default=24)
    parser.add_argument('--gradient-accumulation', type=int, default=2,
            help='number of iterations for gradient accumulation')
    parser.add_argument('-e', '--epochs', type=int, default=90)
    parser.add_argument('-l', '--lr', type=float, default=1.5e-4)

    args = parser.parse_args()

    # if args.mode == 'train':
    #     assert args.save is not None
    # if args.mode == 'val':
    #     assert args.save is None
    # if args.mode == 'predict':
    #     assert args.load is not None
    #     assert args.save is None

    if args.seed is None:
        args.seed = random.randint(0, 10 ** 9)

    return args


def setup_logging(args):
    head = '{asctime}:{levelname}: {message}'
    handlers = [logging.StreamHandler(sys.stderr)]
    # if args.mode == 'train':
    #     handlers.append(logging.FileHandler(args.save + '.log', mode='w'))
    # if args.mode == 'predict':
    #     handlers.append(logging.FileHandler(args.load + '.output.log', mode='w'))
    logging.basicConfig(level=logging.DEBUG, format=head, style='{', handlers=handlers)
    logging.info('Start with arguments {}'.format(args))


def setup_determinism(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def main(args):

    dataloaders = load_data(args)
    autoencoder = LitAutoEncoder()
    autoencoder.to(args.device)
    trainer = pl.Trainer(max_epochs=3,progress_bar_refresh_rate=20)
    trainer.fit(autoencoder, dataloaders['train'], dataloaders['val'])


    trainer.test(autoencoder, dataloaders['test'])
    # sets =


    # model = ModelAndLoss(args)
    # logging.info('Model:\n{}'.format(str(model)))

    # if args.load is not None:
    #     logging.info('Loading model from {}'.format(args.load))
    #     model.load_state_dict(torch.load(str(args.load)))
    #
    # if args.mode in ['train', 'val']:
    #     train(args, model)
    # elif args.mode == 'predict':
    #     predict(args, model)
    # else:
    #     assert 0

if __name__ == '__main__':

    args = parse_args()
    setup_logging(args)
    setup_determinism(args)
    main(args)