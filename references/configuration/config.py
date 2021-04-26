from pathlib import Path
import os
import torch

ROOT_DIR = Path(__file__).parent.parent


class Config:

    RAW_DATA_PATH: str
    DEFAULT_BASE_PATH: str
    DEFAULT_METADATA_BASE_PATH: str
    DEFAULT_IMAGES_BASE_PATH: str
    DEFAULT_CHANNELS: tuple
    MODEL_PATH: str
    FAST_RUN: bool
    OUTPUT_PATH: str
    DEVICE: str

    # EARLY_STOPPING_EPSILON = 0.01 / 100  # For full fit, this is the epsilon in percentage (i.e. 0.01%)

    def __init__(self, exp_name, fast_run=True):
        self.__class__.DEFAULT_BASE_PATH = f"{ROOT_DIR}\\data_layer\\raw_data\\"
        self.__class__.DEFAULT_METADATA_BASE_PATH = os.path.join(self.DEFAULT_BASE_PATH, 'metadata.csv')
        self.__class__.DEFAULT_IMAGES_BASE_PATH = os.path.join(self.DEFAULT_BASE_PATH, 'images')
        self.__class__.DEFAULT_CHANNELS = (1, 2, 3, 4, 5)
        pipeline_name = f"{exp_name}_pipeline.pkl"
        if pipeline_name.startswith('_'):
            pipeline_name = pipeline_name[1:]
        if fast_run:
            pipeline_name = pipeline_name.replace('.pkl', '_fast_run.pkl')
        self.__class__.PIPELINE_PATH = f"{ROOT_DIR}\\pipeline\\{pipeline_name}"
        self.__class__.FAST_RUN = fast_run
        self.__class__.MODEL_PATH = f"{ROOT_DIR}\\models\\{exp_name}\\"
        self.__class__.OUTPUT_PATH = f"{ROOT_DIR}\\output\\{exp_name}\\"
        self.__class__.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @classmethod
    def get_raw_data_file_path(cls, file_name):
        return f"{cls.RAW_DATA_PATH}\\{file_name}"
