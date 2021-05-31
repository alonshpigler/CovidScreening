from zipfile import ZipFile

# from skimage.io import imread
import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

from data_layer.util import image_path
from references.configuration.config import Config
from util.files_operations import *
# from data_layer.dataset import CovidMetadata

# DEFAULT_BASE_PATH = 'C:/Covid-Screening/data_layer/raw_data'
DEFAULT_BASE_PATH = os.pathPath(__file__).parent

DEFAULT_METADATA_BASE_PATH = os.path.join(DEFAULT_BASE_PATH, 'metadata.csv')
# DEFAULT_IMAGES_BASE_PATH = os.path.join(DEFAULT_BASE_PATH, 'images')
# DEFAULT_CHANNELS = (1, 2, 3, 4, 5)


ROOT_DIR = Path(__file__).parent
CHANNELS = (1,2,3,4,5)
save_path = f"{ROOT_DIR}\\data_layer\\raw_data\\mocks\\"
# config = Config('trasfer_data')


# metadata = CovidMetadata(config.DEFAULT_METADATA_BASE_PATH)
# metadata = load_csv(DEFAULT_METADATA_BASE_PATH)
metadata2 = pd.read_csv(DEFAULT_METADATA_BASE_PATH)

img_filenames = []
channels = []
i=0

for rec in metadata:
    for c in CHANNELS:
        i+=1
        img_filename = image_path(rec['experiment'], rec['plate'], rec['well'], rec['site'],c)
        img_filenames.append(img_filename)
        channels.append(c)

metadata = pd.DataFrame(metadata)
reps = [5]*metadata.shape[0]
image_frame = metadata.loc[np.repeat(metadata.index.values,reps)]
image_frame['channel'] = channels
image_frame['img_filename'] = img_filenames
image_frame.to_csv(image_frame, os.path.join(DEFAULT_BASE_PATH, 'image_frame'), columns=None, index=None)
write_dict_to_csv_with_pandas(image_frame, os.path.join(DEFAULT_BASE_PATH, 'image_frame.csv'))

filename = "D:\\RxRx19a-images.zip"
with ZipFile(filename) as archive:
    for entry in archive.infolist():
        with archive.open(entry) as file:
            if file.name.__contains__( 'HRCE'):

            # last_sep = file.name.rindex('/')
                img = Image.open(file)
                print(img.size, img.mode, len(img.getdata()))

