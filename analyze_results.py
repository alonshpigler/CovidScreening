from pathlib import Path

import os
import pandas as pd
from configuration import config

from configuration.model_config import Model_Config
from util.files_operations import load_pickle
from visualize import violin_plot, treatment_plot


def analyze_results(res):
    pass
    # inter_plate_zfactor(res)
    # for plate in res:
    # z_factor()

# def load_populations_by(parameter, values, EXP_DIR, test_plate = 25, inspected_treatments=['Remdesivir (GS-5734)'], experiment='HRCE-1'):
#
#     args = load_pickle(os.path.join(EXP_DIR, 'args.pkl'))
#     res_path = os.path.join(EXP_DIR, 'results.csv')
#     df = pd.read_csv(res_path)
#     disease_conditions = ['Mock', 'Active SARS-CoV-2', 'UV Inactivated SARS-CoV-2', ]
#
#     for value in values:
#     for condition in disease_conditions:
#         populations[condition] = df[(df['plate'] == test_plate) & (df['disease_condition'] == condition) & (df['experiment'] == experiment)& (df['treatment'].isnull())& (df[parameter] == parameter_value)]
#     for treatment in inspected_treatments:
#         treatment_populations[treatment] = df[
#             (df['plate'] == test_plate) & (df['treatment'] == treatment) & (df[parameter] == parameter_value)]
#
#     return populations, treatment_populations


def load_populations(EXP_DIR, test_plate = 25, inspected_treatments=['Remdesivir (GS-5734)'], experiment='HRCE-1', parameter = 'cell_type', parameter_value='HRCE'):

    args = load_pickle(os.path.join(EXP_DIR, 'args.pkl'))
    res_path = os.path.join(EXP_DIR, 'results.csv')
    df = pd.read_csv(res_path)
    disease_conditions = ['Mock', 'Active SARS-CoV-2', 'UV Inactivated SARS-CoV-2', ]
    populations={}
    treatment_populations={}
    for condition in disease_conditions:
        populations[condition] = df[(df['plate'] == test_plate) & (df['disease_condition'] == condition) & (df['experiment'] == experiment)& (df['treatment'].isnull())& (df[parameter] == parameter_value)]
    for treatment in inspected_treatments:
        treatment_populations[treatment] = df[
            (df['plate'] == test_plate) & (df['treatment'] == treatment) & (df[parameter] == parameter_value)]

    # mocks = df[
    #     (df['plate'] == test_plate) & (df['disease_condition'] == 'Mock') & (df['experiment'] == ('HRCE-' + str(1)))]
    # active_no_treat = df[
    #     (df['plate'] == test_plate) & (df['disease_condition'] == 'Active SARS-CoV-2') & (df['treatment'].isnull())]
    # active_with_treat = df[
    #     (df['plate'] == test_plate) & (df['disease_condition'] == 'Active SARS-CoV-2') & (~df['treatment'].isnull())]
    # uv = df[(df['plate'] == test_plate) & (df['disease_condition'] == 'UV Inactivated SARS-CoV-2')]
    # all_active = df[
    #     (df['plate'] == test_plate) & (df['disease_condition'] == 'Active SARS-CoV-2')]
    # remdesivir = df[
    #     (df['plate'] == test_plate) & (df['disease_condition'] == 'Active SARS-CoV-2') & (df['treatment'] == inspected_treatment)]

    return populations, treatment_populations


if __name__ == '__main__':

    exp_num = 4
    channels_to_predict = [1, 2, 3, 4, 5]
    fig_title = 'checking net minimize'
    hyper_parameter = 'net_minimize'
    hyper_parameter_values = [1,2,4,8]

    inspected_treatments =[
        'Remdesivir (GS-5734)',  # Exp 1, Plate 25
        'GS-441524'  # Exp 1, Plates 25,26
        # 'Aloxistatin',  # 1-25,26 2-27, DEEM-D paper
        # 'Colchicine',  # 2-3,11,19, DEEM-D paper
        # 'oxibendazole',  # 2-3,11,19
        # 'Mebendazole'   #2-1,9,17
    ]

    models = [
        Model_Config.UNET4TO1,
        # Model_Config.UNET1TO1,
        # Model_Config.UNET5TO5
    ]
    for model in models:
        for target_channel in channels_to_predict:

            DATA_DIR, METADATA_PATH, _, IMAGES_PATH,EXP_DIR = config.get_paths(exp_num, model.name, target_channel)

            save_dir = Path(EXP_DIR).parent
            args = load_pickle(os.path.join(EXP_DIR,'args.pkl'))

            populations, treatment_populations = load_populations(EXP_DIR, parameter='minimize_net', value=value)
            treatment_plot(populations, treatment_populations, save_dir, target_channel, unique_title= fig_title)
            violin_plot(populations, treatment_populations, save_dir, target_channel, unique_title = fig_title)
            # TODO : make treatment graph normalized to zero and compared to active and mock
            # TODO : show treatment graph on exp 3 - compared to the other networks (1to1 and 5to5)
            # TODO : Combine same sites (process image)
            # TODO : show on exp 4 - different networks


