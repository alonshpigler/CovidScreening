import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DEFAULT_IMAGES_BASE_PATH = f"{ROOT_DIR}\\data_layer\\raw_data\\images\\"

def image_path(experiment,
               plate,
               address,
               site,
               channel,
               base_path=DEFAULT_IMAGES_BASE_PATH):
    """
    Returns the path of a channel image.
    Parameters
    ----------
    dataset : str
        what subset of the data: train, test
    experiment : str
        experiment name
    plate : int
        plate number
    address : str
        plate address
    site : int
        site number
    channel : int
        channel number
    base_path : str
        the base path of the raw images
    Returns
    -------
    str the path of image
    """
    return os.path.join(base_path, experiment, "Plate{}".format(plate),
                        "{}_s{}_w{}.png".format(address, site, channel))
