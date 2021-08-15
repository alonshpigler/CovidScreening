import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pandas.io.common import file_exists
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import tqdm
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader
from sklearn.datasets import load_boston
import main_pipe
from data_layer import transforms
from model_layer import TripletLoss
from configuration import config
from data_layer.prepare_data import load_data, get_data_stats
from model_layer.ArcFace import ArcFace
from model_layer.ResnetClassifier import LitResnetwithTripletLoss, LitResnet
from process_images import process_image
from data_layer.dataset import CovidSupervisedDataset
from util.files_operations import is_file_exist

seed_everything(7)

PATH_DATASETS = os.environ.get('PATH_DATASETS', '.')
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 128 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)

exp_num=0
models = [
    # LitResnet,
    LitResnetwithTripletLoss,
]

model = models[0](lr=10e-4, loss_type='triplet_plate')
args = config.parse_args(exp_num, model_type=model._get_name(), num_input_channels=5,
                         supervised=True)
# args.input_size = 64
input_size = 64
# boston = load_boston()
# args.load_all_data = True
model.to(args.device)
# args.plates_split = [[1,2], [25]]
# plates = n.arange(1,25)
plates = list(np.arange(1,26))
# plates = [25,8,6,17,14]
use_model=True

if len(plates)<6:
    exp_name = 'plates '+str(plates)
else:
    exp_name = 'plates ' + str(plates[0])+'-'+str(plates[-1])

loaded_data_path = os.path.join(args.exp_dir, 'data of '+ exp_name +'.npz')
if is_file_exist(loaded_data_path):
    print('Loading data...')
    numpy_file = np.load(loaded_data_path)
    x, y, plate_label = numpy_file['arr_0'], numpy_file['arr_1'], numpy_file['arr_2']
else:
    df = pd.read_csv(args.metadata_path)
    data_inds= list(df[(df['plate'].isin(plates)) & (df['disease_condition'].isin(['Mock','UV Inactivated SARS-CoV-2','Active SARS-CoV-2'])) & (
                            df['experiment'] == ('HRCE-' + str(1))) & (df['treatment'].isna())].index),

    mean, std = get_data_stats(data_inds, plates, args.data_path, args.device, args.supervised)

    trfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(input_size),
        transforms.Normalize(mean, std)
    ])

    dataset = CovidSupervisedDataset(data_inds, target_channel=1, root_dir = args.data_path, transform=trfs, input_channels=5)
    dataloader = DataLoader(dataset, batch_size=len(dataset),shuffle=False)

    for batch in dataloader:
        x, y, plate_label = batch
    x = x.flatten(1).cpu().numpy()
    y = y.cpu().numpy()
    plate_label = plate_label.cpu().numpy()
    np.savez(loaded_data_path, x, y, plate_label)

print('Running PCA...')
sc = StandardScaler()
X_scaled = sc.fit_transform(x)
pca = PCA(n_components=2)
x_pca = pca.fit_transform(X_scaled)
# x_pca[x_pca > 50] = 50
# x_pca[x_pca < -50] = -50

plate_centroids = np.zeros((len(plates)+1,2))
for i in range(1,len(plates)+1):
    # plate_samples = x_pca[plate_label[plate_label == plates[i-1]], :]
    plate_samples = x_pca[plate_label==i]
    plate_centroids[i, :] = plate_samples.mean(0)
dists = euclidean_distances(plate_centroids)

# closest = [25,8,6,17,14]
# furthest = [25, 11, 21, 2, 4]
select_specific_plates = True
if select_specific_plates:
    plates = [25, 11, 21, 2, 4]
    selected_samples = np.zeros(plate_label.shape)
    for i in plates:
        selected_samples = np.logical_or(selected_samples, plate_label == i)
    x_pca = x_pca[selected_samples]
    y = y[selected_samples]
    plate_label = plate_label[selected_samples]
    exp_name = 'plates' + str(plates)
# selected_plates = np.logical_or(np.logical_or(np.logical_or(np.logical_or(plate_label == 24, plate_label == 18),plate_label == 10),plate_label == 8),plate_label == 5)

print('Cluster distances:')
print(dists)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(13.5, 4))
sns.scatterplot(x_pca[:, 0], x_pca[:, 1], hue=y, palette='Set1', ax=ax[0])
sns.scatterplot(x_pca[:, 0], x_pca[:, 1], hue=plate_label, palette='Set2', ax=ax[1])

ax[0].set_title("PCA of plates" + str(plates) +" by cell state", fontsize=15, pad=15)
ax[1].set_title("PCA of plates" + str(plates) +" by plate", fontsize=15, pad=15)
# ax[1].set_title("PCA of IRIS dataset", fontsize=15, pad=15)

ax[0].set_xlabel("PC1", fontsize=12)
ax[0].set_ylabel("PC2", fontsize=12)

ax[1].set_xlabel("PC1", fontsize=12)
ax[1].set_ylabel("PC2", fontsize=12)

plt.savefig(os.path.join(args.exp_dir,'PCA of '+ exp_name + '.png'), dpi=80)
plt.tight_layout()
plt.show()