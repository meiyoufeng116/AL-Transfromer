import os

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')


def draw_lines(loss_list, metrics_list, eval_loss_list, save_path):
    num_epochs = len(loss_list)
    plt.subplot(3, 1, 1)
    x1 = range(0, num_epochs)
    plt.plot(x1, loss_list, 'o-')
    plt.ylabel('Train loss')
    plt.subplot(3, 1, 2)
    plt.plot(x1, metrics_list, '.-')
    plt.xlabel("Epochs")
    plt.ylabel("Geo mean")
    plt.subplot(3, 1, 3)
    plt.plot(x1, eval_loss_list)
    plt.savefig(os.path.join(save_path, "fig.png"))

