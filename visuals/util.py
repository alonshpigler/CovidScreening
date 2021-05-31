import os
from math import floor
import matplotlib.pyplot as plt


def show_input_and_target(input, target=None,pred = None, title='', save_dir=None):
    import matplotlib.pyplot as plt
    from math import floor
    num_rows, num_cols = 2,3
    num_channels = input.shape[0]
    fig, ax = plt.subplots(num_rows,num_cols)
    for c in range(num_channels):
        ax[floor(c/num_cols),c % num_cols].imshow(input[c, :, :])
        ax[floor(c / num_cols), c % num_cols].set_title('Input channel ' + str(c))
    if target is not None:
        pos1 =ax[1,num_channels % num_cols].imshow(target[:, :])
        ax[1, num_channels % num_cols].set_title('Target')
        # fig.colorbar(pos1,ax=ax[1, num_channels % num_cols])
    if pred is not None:
        pos2 = ax[1,(num_channels+1) % num_cols].imshow(pred[:, :])
        ax[ 1, (num_channels+1) % num_cols].set_title('Prediction')
        # fig.colorbar(pos2, ax=ax[1, (num_channels+1) % num_cols])

    fig.suptitle(title)
    plt.show()

    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, title+'.jpg'))