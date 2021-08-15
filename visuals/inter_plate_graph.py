import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

filename = "/home/alonshp/exp_dir/44/LitResnetwithTripletLoss/channel 1/"
# name = "results on plates [11, 21, 2, 4]test"
name = "results on plates [1, 2]test"

df = pd.read_csv(os.path.join(filename,name + ".csv"))
# df['new_a'] = 1-df['alpha']

num_plots=4
fig, ax = plt.subplots(nrows=1, ncols=num_plots, figsize=(12, 4))
a = df['val_plate_loss']
# fig = plt.figure()
# ax = fig.add_subplot(2, 1, 1)
ys = ['test_acc', 'val_plate_loss','sum_distances_val','test_plate_sum_distances']
y_names = ['Accuracy', 'Inter-plate loss', 'Inter-plate distances', 'Test plate distances']
colors=['blue','green','yellow','red']

for i in range(num_plots):
    # y = 'val_plate_loss'
    # y_name = 'inter-plate val loss'
    y= ys[i]
    y_name = y_names[i]
    ax[i].plot(df['alpha'],df[y], linewidth=1.4, marker ='p' ,color=colors[i])
    ax[i].set_xscale('log')
    ax[i].set_title(y_name, fontsize=14)
    ax[i].set_xlabel('alpha', fontsize=14)
    ax[i].set_ylabel(y, fontsize=14)
    ax[i].set_xticks([0.0001, 0.001, 0.01, 0.1])
    # ax[i].set_xticklabels([0.0001,0.001,0.01,0.1])
    # ax[i].set_xticklabels(['zero', 'two', 'four', 'six'])
    if y=='test_acc':
        # ax[i].set_xlim([0.0001,0.2])
        ax[i].set_ylim([0.2,1])
        # ax[i].xticks([0,0.001,0.01, 0.05, 0.1, 0.2])
    else:
        ax[i].set_ylim([0, np.max(df[y])+1])

fig.tight_layout()
fig.savefig(os.path.join(filename,"plot "+name + ".png"))

plt.show()