import os

import numpy as np
from matplotlib import pyplot as plt


def violin_plot(populations, treatment_populations, exp_dir,target_channel=1, unique_title = ''):

    violin_lines = []
    violin_labels =[]

    for p in populations.keys():
        violin_lines.append(populations[p]['pcc'])
        violin_labels.append(p)

    for treatment in treatment_populations:
        violin_lines.append(treatment_populations[treatment]['pcc'])
        violin_labels.append(treatment)
    # violin_lines = [
    #     mocks['pcc'],
    #     uv['pcc'],
    #     active_no_treat['pcc'],
    #     active_with_treat['pcc'],
    #     all_active['pcc'],
    #     remdesivir['pcc']
    # ]

    # violin_labels = [
    #     'Mocks' + '\n',
    #     'UV' + '\n',
    #     'Active w/o t' + '\n',
    #     'Active w t' + '\n',
    #     'Active all' + '\n',
    #     'Remdesivir' + '\n'
    # ]

    # control population
    mocks_mean = populations['Mock']['pcc'].mean()
    mocks_std = populations['Mock']['pcc'].std()

    z_factors = [0]  # z_factor of control
    violin_labels[0] += '\n z=0'
    # z_factor comparing to control
    for l in range(len(violin_labels)-1):
        z_factor = np.round(1 - 3 * (violin_lines[l+1].std() + mocks_std) / np.abs((violin_lines[l+1].mean() - mocks_mean)), 3)
        z_factors.append(z_factor)
        violin_labels[l+1] += '\n z=' + str(z_factors[l+1])
    # z_factors = [
    #     0,
    #     np.round(1 - 3 * (uv['pcc'].std() + mocks_std) / np.abs((uv['pcc'].mean() - mocks_mean)), 3),
    #     np.round(
    #         1 - 3 * (active_no_treat['pcc'].std() + mocks_std) / np.abs((active_no_treat['pcc'].mean() - mocks_mean)),
    #         3),
    #     np.round(1 - 3 * (active_with_treat['pcc'].std() + mocks_std) / np.abs(
    #         (active_with_treat['pcc'].mean() - mocks_mean)), 3),
    #     np.round(1 - 3 * (all_active['pcc'].std() + mocks_std) / np.abs((all_active['pcc'].mean() - mocks_mean)), 3),
    #     np.round(1 - 3 * (remdesivir['pcc'].std() + mocks_std) / np.abs((remdesivir['pcc'].mean() - mocks_mean)), 3)
    # ]

    # for l in range(len(violin_labels)):


    fig = plt.figure()
    fig_name = 'PCC of populations on channel ' + str(target_channel)+ ' '
    plt.title(fig_name + unique_title)
    plt.violinplot(violin_lines)
    plt.xlabel('Population')
    plt.ylim([0.75, 1])
    plt.ylabel('PCC')
    plt.xticks([1, 2, 3, 4], labels=violin_labels)
    plt.show()
    fig.savefig(os.path.join(exp_dir, fig_name + unique_title + '.jpg'))
    plt.close(fig)


def treatment_plot(populations, treatment_populations, exp_dir, target_channel = 1, treatment ='Remdesivir (GS-5734)', test_plate = 25, unique_title =''):

    fig = plt.figure()
    fig_name = treatment + ' by dose on channel ' + str(target_channel)

    lines = []
    stds = []
    labels = []

    first_key = next(iter(treatment_populations))
    dosages = treatment_populations[first_key].groupby('treatment_conc').mean()['pcc'].index

    # treatment lines
    for treatment in treatment_populations:
        line = treatment_populations[treatment].groupby('treatment_conc').mean()['pcc']
        std = treatment_populations[treatment].groupby('treatment_conc').mean()['pcc']
        label = treatment
        # lines.append(treatment_populations[treatment].groupby('treatment_conc').mean()['pcc'])
        # stds.append(treatment_populations[treatment].groupby('treatment_conc').std()['pcc'])
        # labels.append(treatment)
        # plt.errorbar(dosages, lines[l], yerr=stds[l], label=labels[l], linewidth=1.2)
        plt.errorbar(dosages, line, yerr=std, uplims = True, lolims = True, label=label, linewidth=1.2)

    # reference lines
    colors = ['g','r','y']
    for i, p in enumerate(populations):
        line = [populations[p]['pcc'].mean()]* len(dosages)
        std = populations[p]['pcc'].std()
        err_up = [line[0]+std]* len(dosages)
        err_lo = [line[0]-std]* len(dosages)
        label = p + ' avg'
        plt.plot(dosages, line, label=label, linewidth=1.2, color =colors[i])
        plt.plot(dosages, err_up, linestyle = '--', linewidth=0.8, color =colors[i])
        plt.plot(dosages, err_lo, linestyle = '--', linewidth=0.8, color =colors[i])
        # lines.append([populations[p]['pcc'].mean()]* len(dosages))
        # stds.append([populations[p]['pcc'].std()] * len(dosages))
        # labels.append(p +' avg')
    # lines = [
    #     pccs,
    #     [mocks['pcc'].mean()] * len(pccs.index),
    #     [uv['pcc'].mean()] * len(pccs.index),
    #     [active_no_treat['pcc'].mean()] * len(pccs.index),
    #     [active_with_treat['pcc'].mean()] * len(pccs.index),
    #     [all_active['pcc'].mean()] * len(pccs.index),
    # ]
    #
    # labels = [
    #     treatment,
    #     'Mocks average',
    #     'uv avg',
    #     'Active w/o treat avg',
    #     'Active w treat avg',
    #     'Active all avg',
    # ]

    # plotting


    # for l in range(len(lines)):
        # plt.plot(dosages, lines[l], label=labels[l], linewidth=1.2)
        # plt.errorbar(dosages, lines[l], yerr=stds[l], label=labels[l], linewidth=1.2)

    plt.title(fig_name + unique_title)
    plt.xlabel('Dosage')
    plt.ylabel('PCC')
    plt.xscale('log')
    plt.legend()
    plt.show()
    fig.savefig(os.path.join(exp_dir, fig_name + unique_title + '.jpg'))
    plt.close(fig)