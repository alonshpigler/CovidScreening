import csv
import sys
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import tqdm
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel, update_bn
import main_pipe
from model_layer import TripletLoss
from configuration import config
from data_layer.prepare_data import load_data
from model_layer.ArcFace import ArcFace
from model_layer.ResnetClassifier import LitResnetwithTripletLoss, LitResnet
from process_images import process_image
import matplotlib.pyplot as plt
import seaborn as sns

seed_everything(7)

PATH_DATASETS = os.environ.get('PATH_DATASETS', '.')
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 128 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)


def test(model, data_loader, title ='', save_dir=''):
    df = pd.read_csv(args.metadata_path)
    results = pd.DataFrame(columns=['experiment','plate','well','site','disease_condition', 'treatment_conc','pcc'])
    # results = {}
    pccs=[]
    for batch in data_loader:
        x, y, plates = batch
        pred, features = model(x)
        # deformation to patches and reconstruction based on https://discuss.pytorch.org/t/creating-nonoverlapping-patches-from-3d-data-and-reshape-them-back-to-the-image/51210/6

        # rec = data_loader.dataset.csv_file.iloc[ind]
        # pred_patches = model.forward(input_patches.to(model.device))  # inference
        # pcc, p_value = scipy.stats.pearsonr(pred.flatten(), target.cpu().detach().numpy().flatten())
        # results = results.append(rec, ignore_index=True)
        # results.pcc[start] = pcc
        # pccs.append(pcc)

    return results


def run_pca(dataloaders, model, alpha,plates,loss_type, args):

    # torch.cuda.empty_cache()

    # for batch in dataloaders['pca_all']:
    #     x, y, plate_label = batch
    #     x = x.to(args.device)
    #     plate_label = plate_label.to(args.device)
    #
    # # x = x.flatten(1).cpu().numpy()
    # model.to(args.device)
    # logits, features = model.forward(x)
    #
    # del x

    # x = torch.tensor(dtype=args.device)
    # y = torch.tensor(dtype=args.device)
    # plate_label = torch.tensor(dtype=args.device)
    # features = torch.tensor(dtype=args.device)
    val_plate_losses=[]
    torch.cuda.empty_cache()

    start=True
    for batch in dataloaders['pca_batches']:
        x_batch, y_batch, plate_label_batch = batch
        x_batch = x_batch.to(args.device)
        # plate_label_batch = plate_label_batch.to(args.device)
        batch_logits, batch_features = model.forward(x_batch)

        if start:
            x = x_batch
            y = y_batch
            plate_label = plate_label_batch
            features = batch_features
            start = False
        else:
            x = torch.cat((x, x_batch), dim=0)
            y = torch.cat((y, y_batch), dim=0)
            plate_label = torch.cat((plate_label, plate_label_batch), dim=0)
            features = torch.cat((features, batch_features), dim=0)

        batch_val_plate_loss, _ = TripletLoss.batch_all_triplet_loss(plate_label_batch.to(args.device), batch_features.squeeze(), 1.0, squared=False,loss_type=loss_type)
        val_plate_losses.append(batch_val_plate_loss.detach().cpu().numpy())

    val_plate_loss = np.mean(val_plate_losses)
    # x = x.flatten(1).cpu().numpy()

    test_features = features.detach().clone()
    test_plate_labels = plate_label.detach().clone()
    test_y = y.detach().clone()

    features.detach_()
    plate_label.detach_()
    y.detach_()
    # test_features = torch.tensor(dtype=args.device)

    test_plate_losses = []

    for batch in dataloaders['test_pca_batches']:
        x_batch, y_batch, plate_label_batch = batch
        x_batch = x_batch.to(args.device)
        batch_logits, batch_features = model.forward(x_batch)
        # plate_label_batch = plate_label_batch.to(args.device)

        test_features = torch.cat((test_features,batch_features.detach()), dim=0)
        test_plate_labels = torch.cat((test_plate_labels,plate_label_batch.detach()), dim=0)
        test_y = torch.cat((test_y,y_batch.detach()), dim=0)
        test_plate_loss, _ = TripletLoss.batch_all_triplet_loss(plate_label_batch.to(args.device), batch_features.squeeze(), 1.0, squared=False,loss_type=loss_type)
        test_plate_losses.append(test_plate_loss.detach().cpu().numpy())

    test_plate_loss = np.mean(test_plate_losses)

    # features_test_and_val = torch.cat((features, test_features), dim=0)

    y = y.numpy()
    test_plates = plates.copy()
    test_plates.append(25)
    plate_label = plate_label.numpy()
    test_plate_labels = test_plate_labels.numpy()
    test_y = test_y.numpy()

    del model
    torch.cuda.empty_cache()

    feautres_names = ['val', 'val+test']
    val_pca = pca_from_features(alpha, args, features, feautres_names[0], plate_label, plates, y)
    test_pca = pca_from_features(alpha, args, test_features, feautres_names[1], test_plate_labels, test_plates, test_y)

    val_dists = compute_cluster_distances(val_pca, plate_label, plates)
    test_dists = compute_cluster_distances(test_pca, test_plate_labels, test_plates)

    sum_distances_val = np.triu(val_dists).sum() - np.trace(val_dists)
    test_plate_sum_distances = np.sum(test_dists,0)[-1]

    del plate_label, test_plate_labels, features, test_features
    torch.cuda.empty_cache()

    return val_plate_loss, test_plate_loss, sum_distances_val, test_plate_sum_distances


def compute_cluster_distances(x_pca, plate_label, plates):
    plate_centroids = np.zeros((len(plates), 2))
    for i in range(len(plates)):
        # plate_samples = x_pca[plate_label[plate_label == plates[i-1]], :]
        plate_samples = x_pca[plate_label == plates[i]]
        plate_centroids[i, :] = plate_samples.mean(0)
    dists = euclidean_distances(plate_centroids)
    return dists


def pca_from_features(alpha, args, features, feautres_name, plate_label, plates, y):

    features = torch.flatten(features,1).cpu().numpy()

    name = 'plates ' + str(plates) + feautres_name + 'predictions with alpha=' + str(alpha)
    np.savez(os.path.join(args.exp_dir, name), features, y, plate_label)
    sc = StandardScaler()
    features_scaled = sc.fit_transform(features)
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(features_scaled)
    x_pca[x_pca > 80] = 80
    x_pca[x_pca < -80] = -80
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    sns.scatterplot(x_pca[:, 0], x_pca[:, 1], hue=y, palette='Set1', ax=ax[0])
    sns.scatterplot(x_pca[:, 0], x_pca[:, 1], hue=plate_label, palette='Set2', ax=ax[1])
    ax[0].set_title(name + " by cell state", fontsize=15, pad=15)
    ax[1].set_title(name + " by plate", fontsize=15, pad=15)
    # ax[1].set_title("PCA of IRIS dataset", fontsize=15, pad=15)
    ax[0].set_xlabel("PC1", fontsize=12)
    ax[0].set_ylabel("PC2", fontsize=12)
    ax[1].set_xlabel("PC1", fontsize=12)
    ax[1].set_ylabel("PC2", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.exp_dir, name + '.png'), dpi=80)

    return x_pca


exp_num = 45  # if None, new experiment directory is created with the next avaialible number
DEBUG = False
epochs = 15
# alpha = 0.99
load = True
# loss_types = ['triplet_plate','arcface_plate', 'arcface', 'softmax', 'contractive_plate]
# loss_types = ['softmax', 'triplet_plate']
loss_type = 'contractive_plate'
alphas = [0,0.01,0.05, 0.1, 0.2]
# train_plates_try = [[1,2],[1,2,3,4,5]]
train_plates = [1, 2]
test_plates = [25]
models = [
    # LitResnet,
    LitResnetwithTripletLoss,
]


if len(train_plates)<6:
    exp_name = 'plates '+str(train_plates)
else:
    exp_name = 'plates ' + str(train_plates[0])+'-'+str(train_plates[-1])


res = {'test_acc': [], 'test_loss': [], 'val_plate_loss': [],'test_plate_loss':[], 'sum_distances_val':[], 'test_plate_sum_distances':[], 'alpha':alphas}

for alpha in alphas:

    model = models[0](lr=10e-5,alpha=alpha, loss_type=loss_type)
    args = config.parse_args(exp_num, model_type=model._get_name(), num_input_channels=5,
                             supervised=True,
                             debug=DEBUG)
    args.input_size = 64
    model.to(args.device)

    torch.autograd.set_detect_anomaly(True)
    args.plates_split = [train_plates, test_plates]
    dataloaders = load_data(args)

    logger = TensorBoardLogger(args.log_dir, name='resnet' + str(exp_num))
    trainer = Trainer(
        progress_bar_refresh_rate=5,
        max_epochs=epochs,
        gpus=AVAIL_GPUS,
        logger=logger,
        auto_scale_batch_size='binsearch',
        callbacks=[LearningRateMonitor(logging_interval='step')]
    )

    if load:
        # if args.plates_split[0]==[11, 21, 2, 4] and alpha==0:
            # checkpoint = "/home/alonshp/log_dir/resnet0/version_6/checkpoints/epoch=6-step=335.ckpt"
            # checkpoint = "/home/alonshp/log_dir/resnet30/version_1/checkpoints/epoch=11-step=575.ckpt"

        if alpha==0:
            if args.plates_split[0] == [1,2]:
                checkpoint = "/home/alonshp/log_dir/resnet42/version_0/checkpoints/epoch=2-step=71.ckpt"
        elif loss_type=='triplet_plate':
            if args.plates_split[0]==[11, 21, 2, 4] and alpha==0.001:
                checkpoint = "/home/alonshp/log_dir/resnet30/version_3/checkpoints/epoch=0-step=47.ckpt"
            elif args.plates_split[0] == [11, 21, 2, 4] and alpha == 0.0015:
                checkpoint = "/home/alonshp/log_dir/resnet40/version_2/checkpoints/epoch=3-step=191.ckpt"
            elif args.plates_split[0] == [11, 21, 2, 4] and alpha == 0.002:
                checkpoint = "/home/alonshp/log_dir/resnet40/version_3/checkpoints/epoch=2-step=143.ckpt"
            elif args.plates_split[0] == [11, 21, 2, 4] and alpha == 0.005:
                checkpoint = "/home/alonshp/log_dir/resnet40/version_4/checkpoints/epoch=3-step=191.ckpt"
            elif args.plates_split[0] == [11, 21, 2, 4] and alpha == 0.01:
                checkpoint = "/home/alonshp/log_dir/resnet40/version_5/checkpoints/epoch=3-step=191.ckpt"

            elif args.plates_split[0] == [1, 2] and alpha == 0.01:
                checkpoint = "/home/alonshp/log_dir/resnet42/version_1/checkpoints/epoch=8-step=215.ckpt"
            elif args.plates_split[0] == [1,2] and alpha == 0.05:
                checkpoint = "/home/alonshp/log_dir/resnet42/version_2/checkpoints/epoch=8-step=215.ckpt"
        elif loss_type=='contractive_plate':
            if args.plates_split[0] == [1,2] and alpha == 0.01:
                checkpoint = "/home/alonshp/log_dir/resnet44/version_4/checkpoints/epoch=8-step=215.ckpt"
            elif args.plates_split[0] == [1,2] and alpha == 0.05:
                checkpoint = "/home/alonshp/log_dir/resnet44/version_5/checkpoints/epoch=8-step=215.ckpt"

        if 'checkpoint' in locals():
            trainer.fit(model, train_dataloader=dataloaders['train'], val_dataloaders=dataloaders['val'])
        else:
            model = model.load_from_checkpoint(checkpoint)
            model.to(args.device)
    else:
        trainer.fit(model, train_dataloader=dataloaders['train'], val_dataloaders=dataloaders['val'])


    # test(model,dataloaders['test'])


    plate_loss,test_plate_loss,sum_distances_val, test_plate_sum_distances = run_pca(dataloaders, model.to(args.device), alpha, train_plates,loss_type, args)
    exp_res = trainer.test(model, test_dataloaders=dataloaders['test'])
    del model
    torch.cuda.empty_cache()
    # main_pipe.save_results(res,args)
    print(exp_res)

    res['test_loss'].append(exp_res[0]['test_loss'])
    res['test_acc'].append(exp_res[0]['test_acc'])
    res['val_plate_loss'].append(plate_loss)
    res['test_plate_loss'].append(test_plate_loss)
    res['sum_distances_val'].append(sum_distances_val)
    res['test_plate_sum_distances'].append(test_plate_sum_distances)
    np.savez(os.path.join(args.exp_dir, 'results on ' + exp_name), res)

    with open(os.path.join(args.exp_dir, 'results on ' + exp_name+"test.csv"), "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(res.keys())
        writer.writerows(zip(*res.values()))

with open(os.path.join(args.exp_dir, 'final results on ' + exp_name + "test.csv"), "w") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(res.keys())
    writer.writerows(zip(*res.values()))

            # res = pd.DataFrame()
            # for plate in list(test_dataloaders):
            #     # res[plate] = {}
            #     for key in test_dataloaders[plate].keys():
            #         # res[plate][key] = []
            #         set_name = 'plate ' + plate + ', population ' + key
            #
            #         plate_res = test(model, test_dataloaders[plate][key], input_size, input_channels, set_name,exp_dir)
            #         # res[plate][key].append(results)
            #         res = pd.concat([res,plate_res])

        #
        # models = [
        #     # Model_Config.RESNET
        # ]
        #
        # for model in models:
        #     for target_channel in channels_to_predict:
        #         args = config.parse_args(exp_num, num_input_channels=model.value[2]['n_input_channels'],
        #                                  target_channel=target_channel,
        #                                  model_type=model.name,
        #                                  debug=DEBUG)
        #         main_pipe.main(model, args)
        #         if model.name == 'UNET5to5':
        #             break
        #
