import numpy as np
import os
import pandas as pd
from configuration import config
import matplotlib.pyplot as plt

from configuration.model_config import Model_Config
from util.files_operations import load_pickle


def analyze_results(res):
    pass
    # inter_plate_zfactor(res)
    # for plate in res:
    # z_factor()


if __name__ == '__main__':

    exp_num = 3
    channels_to_predict = [1]
    models = [
        Model_Config.UNET4TO1,
        # Model_Config.UNET1TO1,
        # Model_Config.UNET5TO5
    ]
    for model in models:
        for target_channel in channels_to_predict:

            DATA_DIR, METADATA_PATH, _, IMAGES_PATH,EXP_DIR = config.get_paths(exp_num, model.name, target_channel)
            args = load_pickle(os.path.join(EXP_DIR,'args.pkl'))

            treatment = 'Remdesivir (GS-5734)'
            res_path = os.path.join(EXP_DIR, 'results.csv')
            df = pd.read_csv(res_path)
            mocks = df[(df['plate'] == 25) & (df['disease_condition'] == 'Mock') & (df['experiment'] == ('HRCE-' + str(1)))]
            active_no_treat = df[
                (df['plate'] == 25) & (df['disease_condition'] == 'Active SARS-CoV-2') & (df['treatment'].isnull())]
            active_with_treat = df[
                (df['plate'] == 25) & (df['disease_condition'] == 'Active SARS-CoV-2') & (~df['treatment'].isnull())]
            uv = df[(df['plate'] == 25) & (df['disease_condition'] == 'UV Inactivated SARS-CoV-2')]
            all_active = df[
                (df['plate'] == 25) & (df['disease_condition'] == 'Active SARS-CoV-2')]
            remdesivir = df[
                (df['plate'] == 25) & (df['disease_condition'] == 'Active SARS-CoV-2') & (df['treatment']==treatment)]
            # main(args)
            pccs = remdesivir.groupby('treatment_conc').mean()['pcc']


            lines = [
                pccs,
                [mocks['pcc'].mean()] * len(pccs.index),
                [uv['pcc'].mean()] * len(pccs.index),
                [active_no_treat['pcc'].mean()] * len(pccs.index),
                [active_with_treat['pcc'].mean()] * len(pccs.index),
                [all_active['pcc'].mean()] * len(pccs.index),
            ]

            labels = [
                treatment,
                'Mocks average',
                'uv avg',
                'Active w/o treat avg',
                'Active w treat avg',
                'Active all avg',
            ]


            fig = plt.figure()
            fig_name = treatment + ' by dose on channel ' + str(target_channel)
            for l in range(len(lines)):
                plt.plot(pccs.index, lines[l], label=labels[l], linewidth=1.2)


            plt.title(fig_name)
            plt.xlabel('Dosage')
            plt.ylabel('PCC')
            plt.xscale('log')
            plt.legend()
            plt.show()
            fig.savefig(os.path.join(EXP_DIR, fig_name+'.jpg'))
            plt.close(fig)


            violin_lines = [
                mocks['pcc'],
                uv['pcc'],
                active_no_treat['pcc'],
                active_with_treat['pcc'],
                all_active['pcc'],
                remdesivir['pcc']
            ]

            violin_labels = [
                'Mocks' +'\n',
                'UV' +'\n',
                'Active w/o t' +'\n',
                'Active w t' +'\n',
                'Active all' +'\n',
                'Remdesivir' + '\n'
            ]
            mocks_mean = mocks['pcc'].mean()
            mocks_std = mocks['pcc'].std()
            z_factors = [
                0,
                np.round(1 - 3 * (uv['pcc'].std() + mocks_std)/np.abs((uv['pcc'].mean() - mocks_mean)),3),
                np.round(1 - 3 * (active_no_treat['pcc'].std() + mocks_std)/ np.abs((active_no_treat['pcc'].mean() - mocks_mean)), 3),
                np.round(1 - 3 * (active_with_treat['pcc'].std() + mocks_std)/np.abs((active_with_treat['pcc'].mean() - mocks_mean)), 3),
                np.round(1 - 3 * (all_active['pcc'].std() + mocks_std)/ np.abs((all_active['pcc'].mean() - mocks_mean)) ,3),
                np.round(1 - 3 * (remdesivir['pcc'].std() + mocks_std)/ np.abs((remdesivir['pcc'].mean() - mocks_mean)),3)
            ]

            for l in range(len(violin_labels)):
                # df['cluster3'] = ['\n'.join(wrap(x, 12)) for x in df['cluster3']
                violin_labels[l] += 'z='+str(z_factors[l])

            # plt.plot(pccs.index, [mocks['pcc'].mean()]*len(pccs.index), label='Mocks average')
            # plt.plot(pccs.index, [active_no_treat.mean()['pcc']] * len(pccs.index), label='Active no treatment average')
            # unique_conc = remdesivir['treatment_conc'].unique()
            # pcc = remdesivir.groupby('treatment_conc').mean()['pcc']
            fig = plt.figure()
            title = 'PCC of populations on channel ' + str(target_channel)
            plt.title(title)
            plt.violinplot(violin_lines)
            plt.xlabel('Population')
            plt.ylim([0.75,1])
            plt.ylabel('PCC')
            plt.xticks([1, 2, 3, 4, 5, 6] ,labels = violin_labels)
            plt.show()
            fig.savefig(os.path.join(EXP_DIR, title + '.jpg'))
            plt.close(fig)
