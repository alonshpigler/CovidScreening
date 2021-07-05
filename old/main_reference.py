import logging
import pickle
import random
import sys
import time
from argparse import ArgumentParser

import os
import torch
from pandas.tests.extension.numpy_.test_numpy_nested import np


def parse_args():

    parser = ArgumentParser()

    parser.add_argument("--gpu_id", default=0, type=int, help="GPU to use.")
    parser.add_argument("--n_imgs", default=40, type=int, help="Number of images to use.")
    parser.add_argument(
        "--n_iterations", default=50000, type=int, help="Number of training iterations."
    )
    parser.add_argument(
        "--interval_checkpoint",
        default=10000,
        type=int,
        help="Number of training iterations between checkpoints.",
    )

    def lr_type(x):
        x = x.split(',')
        return x[0], list(map(float, x[1:]))

    def bool_type(x):
        if x.lower() in ['1', 'true']:
            return True
        if x.lower() in ['0', 'false']:
            return False
        raise ValueError()

    parser = ArgumentParser()
    parser.add_argument('-m', '--mode', default='train', choices=('train', 'val', 'predict'))
    parser.add_argument('--backbone', default='unet',
            help='backbone for the architecture. Supported backbones: UNET')
    parser.add_argument('--save',
            help='path for the checkpoint with best accuracy. '
                 'Checkpoint for each epoch will be saved with suffix .<number of epoch>')
    parser.add_argument('--load',
            help='path to the checkpoint which will be loaded for inference or fine-tuning')
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--pred-suffix', default='',
            help='suffix for prediction output. '
                 'Predictions output will be stored in <loaded checkpoint path>.output<pred suffix>')
    parser.add_argument('--all-controls-train', type=bool_type, default=True,
            help='train using all control images (also these from the test set)')
    parser.add_argument('--data-normalization', choices=('global', 'experiment', 'sample'), default='sample',
            help='image normalization type: '
                 'global -- use statistics from entire dataset, '
                 'experiment -- use statistics from experiment, '
                 'sample -- use mean and std calculated on given example (after normalization)')
    parser.add_argument('--data', type=Path, default=Path('../data'),
            help='path to the data root. It assumes format like in Kaggle with unpacked archives')
    parser.add_argument('--data-split-seed', type=int, default=0,
            help='seed for splitting experiments for folds')
    parser.add_argument('--num-data-workers', type=int, default=4,
            help='number of data loader workers')
    parser.add_argument('--seed', type=int,
            help='global seed (for weight initialization, data sampling, etc.). '
                 'If not specified it will be randomized (and printed on the log)')
    parser.add_argument('-b', '--batch_size', type=int, default=24)
    parser.add_argument('--gradient-accumulation', type=int, default=2,
            help='number of iterations for gradient accumulation')
    parser.add_argument('-e', '--epochs', type=int, default=90)
    # parser.add_argument('-l', '--lr', type=lr_type, default=('cosine', [1.5e-4]),
    #         help='learning rate values and schedule given in format: schedule,value1,epoch1,value2,epoch2,...,value{n}. '
    #              'in epoch range [0, epoch1) initial_lr=value1, in [epoch1, epoch2) initial_lr=value2, ..., '
    #              'in [epoch{n-1}, total_epochs) initial_lr=value{n}, '
    #              'in every range the same learning schedule is used. Possible schedules: cosine, const')
    parser.add_argument('-l', '--lr', type=float, default=1.5e-4)

    args = parser.parse_args()

    if args.mode == 'train':
        assert args.save is not None
    if args.mode == 'val':
        assert args.save is None
    if args.mode == 'predict':
        assert args.load is not None
        assert args.save is None

    if args.seed is None:
        args.seed = random.randint(0, 10 ** 9)

    return args


def setup_logging(args):
    head = '{asctime}:{levelname}: {message}'
    handlers = [logging.StreamHandler(sys.stderr)]
    if args.mode == 'train':
        handlers.append(logging.FileHandler(args.save + '.log', mode='w'))
    if args.mode == 'predict':
        handlers.append(logging.FileHandler(args.load + '.output.log', mode='w'))
    logging.basicConfig(level=logging.DEBUG, format=head, style='{', handlers=handlers)
    logging.info('Start with arguments {}'.format(args))


def setup_determinism(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def main(args):

    gpu_id = args.gpu_id
    n_images_to_download = args.n_imgs  # more images the better
    train_fraction = 0.5

    image_save_dir = "{}/".format(os.getcwd())
    model_save_dir = "{}/model/".format(os.getcwd())
    prefs_save_path = "{}/prefs.json".format(model_save_dir)

    data_save_path_train = "{}/image_list_train.csv".format(image_save_dir)
    data_save_path_test = "{}/image_list_test.csv".format(image_save_dir)

    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)


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