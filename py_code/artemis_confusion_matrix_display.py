import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


def build_heatmap(conf_matrix_both, con_matrix_old, conf_matrix_new):


    beh = ['drink', 'eat', 'groom',
           'hang', 'sniff', 'rear', 'rest',
           'walk', 'eathand']

    boot_numbers = []
    for boot in conf_matrix_both:
        boot_numbers.append(boot[boot.rfind("csv"):-1])

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15,15))
    fig.suptitle('Confusion Matrix for Bootstrap {}, {}, and {}'.format(boot_numbers[0],boot_numbers[1],boot_numbers[2]))

    for ax, data in zip(axes.flatten(), conf_matrix_both):
        sn.heatmap(conf_matrix_both[data],xticklabels=beh, yticklabels=beh,
                            annot=True, cmap="YlGnBu", ax=ax, )


    # Gets current figure, sets size (width x height, in inches, given constant dpi)
    # plt.gcf().set_size_inches(12, 12)
    plt.xticks(rotation=60)
    plt.show()