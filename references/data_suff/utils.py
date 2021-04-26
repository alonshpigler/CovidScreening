import os

import numpy as np
# from skimage.io import imread
import pandas as pd

from util.files_operations import load_csv

import sys
from zipfile import ZipFile
from PIL import Image # $ pip install pillow
#
# filename = 'D:\\RxRx19a-images.zip'
# with ZipFile(filename) as archive:
#     for entry in archive.infolist():
#         with archive.open(entry) as file:
#             img = Image.open(file)
#             print(img.size, img.mode, len(img.getdata()))
#

DEFAULT_BASE_PATH = 'C:/Covid-Screening/data_layer/raw_data'
DEFAULT_METADATA_BASE_PATH = os.path.join(DEFAULT_BASE_PATH, 'metadata')
DEFAULT_IMAGES_BASE_PATH = os.path.join(DEFAULT_BASE_PATH, 'images')
DEFAULT_CHANNELS = (1, 2, 3, 4, 5)


def _load_dataset(base_path, dataset, include_controls=True):
    df = load_csv(os.path.join(base_path, dataset + '.csv'))
    if include_controls:
        controls = load_csv(
            os.path.join(base_path, dataset + '_controls.csv'))
        df['well_type'] = 'treatment'
        df = pd.concat([controls, df], sort=True)
    df['cell_type'] = df.experiment.str.split("-").apply(lambda a: a[0])
    df['dataset'] = dataset
    dfs = []
    for site in (1, 2):
        df = df.copy()
        df['site'] = site
        dfs.append(df)
    res = pd.concat(dfs).sort_values(
        by=['id_code', 'site']).set_index('id_code')
    return res

_load_dataset(DEFAULT_BASE_PATH,'metadata',False)