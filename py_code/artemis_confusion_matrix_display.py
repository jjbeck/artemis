import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


def build_heatmap(conf_matrix, boot_round, fig_title):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues, interpolation='nearest')
    fig.colorbar(cax)

    labels = ['drink', 'eat', 'groom', 'hang', 'sniff', 'rear', 'rest', 'walk', 'eathand']

    kwargs = {'color': 'w'}
    for i in range(9):
        for j in range(9):
            plt.text(j, i, '%.2f' % conf_matrix[i, j], va='center', ha='center', **kwargs)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticklabels([''] + labels, rotation=45)
    ax.set_xlabel('Predictions')
    ax.set_ylabel('Ground Truth')
    ax.set_yticklabels([''] + labels)
    ax.set_title(fig_title)
    plt.show()



