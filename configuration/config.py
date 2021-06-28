import logging
import os
import sys
from argparse import ArgumentParser
import random
import torch
import numpy as np
from pathlib import Path

from util.files_operations import is_folder_exist, make_folder

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


def parse_args(exp_num=None,target_channel=1, model_type='UNET4TO1', description='Learning 4to1 mappings'):

    parser = ArgumentParser(description=description)
    parser.add_argument('-m','--mode', type = str,  choices=('train', 'val', 'predict'))
    DATA_DIR, METADATA_PATH, LOG_DIR, IMAGES_PATH, EXP_DIR = get_paths(exp_num, model_type, target_channel)

    parser.add_argument('--data_path', type=Path, default=os.path.join(DATA_DIR,'images\\'),
                        help='path to the data root. It assumes format like in Kaggle with unpacked archives')
    parser.add_argument('--metadata_path', type=Path, default=METADATA_PATH,
            help='path to the data root. It assumes format like in Kaggle with unpacked archives')
    parser.add_argument('--log_dir', type=Path, default=LOG_DIR,
                        help='path to experiment logs.')
    parser.add_argument('--exp_dir', type=Path, default=EXP_DIR,
                        help='path to experiment results.')
    parser.add_argument('--plates_split', type=list, default=[[1,2,3],[25]],
                        help='plates split between train and test. left is train and right is test')
    parser.add_argument('--test_samples_per_plate', type=int, default=None,
                        help='number of test samples for each plate. if None, all plates are taken')

    parser.add_argument('--split-ratio', type=int, default=0.8,
                        help='split ratio between train and validation. value in [0,1]')

    parser.add_argument('--data-split-seed', type=int, default=0,
            help='seed for splitting experiments for folds')
    parser.add_argument('--num-data-workers', type=int, default=4,
            help='number of data loader workers')
    parser.add_argument('--device',type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        help='device for running code')
    parser.add_argument('--seed', type=int,
            help='global seed (for weight initialization, data sampling, etc.). '
                 'If not specified it will be randomized (and printed on the log)')

    parser.add_argument('-c','--target_channel', type=int, default=target_channel, choices=(1,2,3,4,5),
                        help='the channel predicted by the network')
    parser.add_argument('--num_input_channels', type=int, default=4, choices = (1,4,5),
                        help='defines what autoencoder is trained (4to1, 1to1, 5to5)')
    parser.add_argument('-i', '--input_size', type=int, default=256,
                        help='width and hight input into the network')

    parser.add_argument('-b', '--batch_size', type=int, default=12)
    parser.add_argument('--gradient-accumulation', type=int, default=2,
            help='number of iterations for gradient accumulation')
    parser.add_argument('-e', '--epochs', type=int, default=20)
    parser.add_argument('-l', '--lr', type=float, default=1.5e-4)
    parser.add_argument('-minimize_net_factor', type=int, default=4,
                        help='reduces the network number of convolution maps by a factor')

    parser.add_argument('--checkpoint', type=str,
                        default='lightning_logs/version_66/checkpoints/epoch=18-step=113.ckpt',
                        help='path to load existing model from')

    args = parser.parse_args()

    # args.model_args = {'lr': args.lr, 'n_classes': n_classes, 'input_size': args.input_size,'minimize_net_factor': args.minimize_net_factor}
    # if args.mode == 'train':
    #     assert args.save is not None
    # if args.mode == 'val':
    #     assert args.save is None
    # if args.mode == 'predict':
    #     assert args.load is not None
    #     assert args.save is None

    if args.seed is None:
        args.seed = random.randint(0, 10 ** 9)

    setup_logging(args)
    setup_determinism(args)

    return args


def get_paths(exp_num=None, model_type = 'UNET4TO1', target_channel=1):

    if use_cuda:
        ROOT_DIR = f"{Path(__file__).parent.parent.parent.parent}home/alonshp"
    else:
        ROOT_DIR = Path(__file__).parent.parent.parent

    DATA_DIR = f"{ROOT_DIR}/Data"
    LOG_DIR = f"{ROOT_DIR}/log_dir"
    EXP_DIR = f"{ROOT_DIR}/exp_dir"
    # if exp_num is None:
    #     exp_num = get_exp_num(EXP_DIR)
    EXP_DIR = os.path.join(EXP_DIR, str(exp_num),model_type, "channel " + str(target_channel))
    # exp_num = get_exp_num(EXP_DIR)
    if exp_num is not None:
        make_folder(EXP_DIR)
    METADATA_PATH = os.path.join(DATA_DIR, 'metadata.csv')
    IMAGES_PATH = os.path.join(DATA_DIR, 'images')

    return DATA_DIR, METADATA_PATH, LOG_DIR, IMAGES_PATH,EXP_DIR


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
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def get_checkpoint(LOG_DIR, model_name, target_channel):
    if model_name == 'UNET4TO1':
        if target_channel == 1:
            checkpoint = f"{LOG_DIR}/UNET4TO1 on channel1/version_1/checkpoints/epoch=18-step=341.ckpt"
        elif target_channel == 2:
            checkpoint = f"{LOG_DIR}/UNET4TO1 on channel2/version_0/checkpoints/epoch=17-step=323.ckpt"
        elif  target_channel == 3:
            checkpoint = f"{LOG_DIR}/UNET4TO1 on channel3/version_0/checkpoints/epoch=19-step=359.ckpt"
        elif target_channel == 4:
            checkpoint =f"{LOG_DIR}/UNET4TO1 on channel4/version_0/checkpoints/epoch=16-step=305.ckpt"

    elif model_name == 'UNET5TO5':
            checkpoint = f"{LOG_DIR}/UNET on channel1/version_16/checkpoints/epoch=23-step=431.ckpt"

    elif model_name == 'UNET1TO1':
        if target_channel == 1:
            checkpoint = f"{LOG_DIR}/UNET on channel1/version_24/checkpoints/epoch=19-step=359.ckpt"
        if target_channel == 2:
            checkpoint = f"{LOG_DIR}/UNET1TO1 on channel2/version_2/checkpoints/epoch=15-step=287.ckpt"
        if target_channel == 3:
            checkpoint = f"{LOG_DIR}/UNET1TO1 on channel3/version_0/checkpoints/epoch=19-step=359.ckpt"
        elif target_channel==4:
            checkpoint = f"{LOG_DIR}/UNET1TO1 on channel4/version_1/checkpoints/epoch=16-step=305.ckpt"
        elif target_channel == 5:
            checkpoint = f"{LOG_DIR}/UNET1TO1 on channel5/version_0/checkpoints/epoch=19-step=359.ckpt"

    if 'checkpoint' not in locals():
        checkpoint = None

    return checkpoint


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def get_exp_num(EXP_DIR):
#     num=0
#     exp_name = os.path.join(EXP_DIR,str(num))
#     while is_folder_exist(exp_name):
#         num+=1
#         exp_name = os.path.join(EXP_DIR, str(num))
#     return num
