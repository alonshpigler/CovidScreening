import logging
import scipy
import pandas as pd
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from configuration import config
from configuration.model_config import Model_Config
from data_layer.prepare_data import load_data
from tqdm import tqdm

from process_images import process_image
from util.files_operations import save_to_pickle, load_pickle, is_file_exist
from visuals.visualize import show_input_and_target


def test_by_partition(model, test_dataloaders,input_size, input_channels, exp_dir=None):

    res = {}
    res = pd.DataFrame()
    for plate in list(test_dataloaders):
        # res[plate] = {}
        for key in test_dataloaders[plate].keys():
            # res[plate][key] = []
            set_name = 'plate ' + plate + ', population ' + key
            plate_res = test(model, test_dataloaders[plate][key], input_size, input_channels, set_name,exp_dir)
            # res[plate][key].append(results)
            res = pd.concat([res,plate_res])
    return res


def test(model, data_loader, input_size, input_channels=4, title ='', save_dir='',show_images=True):
    start=0
    results = pd.DataFrame(columns=['experiment','plate','well','site','disease_condition', 'treatment_conc','pcc'])
    # results = {}
    pccs=[]
    for i, (input, target,ind) in tqdm(enumerate(data_loader),total=len(data_loader)):

        # deformation to patches and reconstruction based on https://discuss.pytorch.org/t/creating-nonoverlapping-patches-from-3d-data-and-reshape-them-back-to-the-image/51210/6
        rec = data_loader.dataset.csv_file.iloc[ind]
        pred = process_image(model, input, input_size, input_channels)
        pcc, p_value = scipy.stats.pearsonr(pred.flatten(), target.cpu().detach().numpy().flatten())
        results = results.append(rec, ignore_index=True)
        results.pcc[start] = pcc
        # pccs.append(pcc)

        if show_images and start == 0:
            if input_channels == 5:
                show_input_and_target(input.cpu().detach().numpy()[0, :, :, :],
                                      pred=pred, title=title, save_dir=save_dir)
            else:
                show_input_and_target(input.cpu().detach().numpy()[0,:,:,:], target.cpu().detach().numpy()[0,:,:,:], pred, title, save_dir)
        start += 1

    # results['pcc'] = pccs

    return results

def save_results(res, args,kwargs):
    for arg in kwargs:
        res[arg] = kwargs[arg]

    res_dir = os.path.join(args.exp_dir, 'results.csv')
    if is_file_exist(res_dir):
        prev_res = pd.read_csv(res_dir)
        res = pd.concat([prev_res, res])

    save_to_pickle(res, os.path.join(args.exp_dir, 'results' + kwargs.__str__() + '.pkl'))
    save_to_pickle(args, os.path.join(args.exp_dir, 'args' + kwargs.__str__() + '.pkl'))
    res.to_csv(res_dir)


def main(Model, args, kwargs={}):

    print_exp_description(Model, args, kwargs)

    logging.info('Preparing data...')
    dataloaders = load_data(args)
    logging.info('Preparing data finished.')

    model = Model.model_class(**Model.params)
    args.checkpoint = config.get_checkpoint(args.log_dir, Model.name, args.target_channel)
    if args.mode == 'predict' and args.checkpoint is not None:
        logging.info('loading model from file...')
        model = model.load_from_checkpoint(args.checkpoint)
        model.to(args.device)
        logging.info('loading model from file finished')

    else:
        logging.info('training model...')
        # model = Unet(**args.model_args)
        # model = Unet(args)
        model.to(args.device)
        logger = TensorBoardLogger(args.log_dir, name=Model.name+ " on channel" + str(args.target_channel))
        trainer = pl.Trainer(max_epochs=args.epochs, progress_bar_refresh_rate=1, logger=logger,gpus=1,auto_scale_batch_size='binsearch',weights_summary='full')
        trainer.fit(model, dataloaders['train'], dataloaders['val'])
        logging.info('training model finished.')

    # if args.mode == 'analyze':
    #     logging.info('loading predictions from file...')
    #     res = load_pickle(os.path.join(args.exp_dir, 'results.pkl'))
    #     logging.info('loading predictions from file finished')

    # else:
    logging.info('testing model...')
    # if not args.DEBUG:
    #     res_on_val = test(model, dataloaders['val_for_test'], args.input_size, 'validation_set',args.exp_dir)
    res = test_by_partition(model, dataloaders['test'], args.input_size, args.num_input_channels, args.exp_dir)
    # res.update(res_on_val)
    logging.info('testing model finished...')

    save_results(res, args, kwargs)



    # summerized_res = analyze_results(res)
    # write_dict_to_csv_with_pandas(summerized_res, os.path.join(args.exp_dir, 'results.csv'))

    # analyze_results(res)
    # res = {}
    # for key in list(dataloaders['test']):
    #     res[key] = {}
    #     for plate in dataloaders['test'][key].keys():
    #         res[key][plate] = trainer.test(autoencoder, dataloaders['test'][key][plate])


def print_exp_description(Model, args, kwargs):
    description = 'Training Model ' + Model.name + ' with target ' + str(args.target_channel)
    for arg in kwargs:
        description += ', ' + arg + ': ' + str(kwargs[arg])
    print(description)


if __name__ == '__main__':
    exp_num = 3  # if None, new experiment directory is created with the next avaialible number
    description = 'Checking 1to1 prediction on 5'
    DEBUG = False

    models = [
        # Model_Config.UNET1TO1,
        Model_Config.UNET4TO1,
        # Model_Config.UNET5TO5
        ]

    print(description)
    channels_to_predict = [3]

    for model in models:
        for target_channel in channels_to_predict:

            print('Training Model '+ model.name +' with target ' + str(target_channel))
            # torch.cuda.empty_cache()
            args = config.parse_args(exp_num, target_channel, model.name)
            args.num_input_channels = model.value[2]['n_input_channels']

            args.mode = 'predict'
            args.plates_split = [[1, 2, 3, 4, 5], [25]]

            args.test_samples_per_plate = None
            args.batch_size = 36
            args.input_size = 128

            if DEBUG:
                args.test_samples_per_plate = 5
                args.epochs = 3

            main(args, model)

