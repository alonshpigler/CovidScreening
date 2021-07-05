import sys

import main_pipe
from configuration import config
from configuration.model_config import Model_Config

if len(sys.argv)>1:
    channels_to_predict = [sys.argv[1]]
else:
    channels_to_predict = [1, 2, 3, 4, 5]

exp_num = 4  # if None, new experiment directory is created with the next avaialible number

DEBUG = False

models = [
    # Model_Config.UNET1TO1,
    Model_Config.UNET4TO1,
    # Model_Config.UNET5TO5
]

for model in models:
    for target_channel in channels_to_predict:
        args = config.parse_args(exp_num, num_input_channels=model.value[2]['n_input_channels'],
                                 target_channel=target_channel,
                                 model_type=model.name,
                                 debug=DEBUG)
        main_pipe.main(model, args)
        if model.name == 'UNET5to5':
            break

