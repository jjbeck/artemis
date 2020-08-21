import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


def build_heatmap(conf_matrix_both, conf_matrix_old, conf_matrix_new):
    beh = ['drink', 'eat', 'groom',
           'hang', 'sniff', 'rear', 'rest',
           'walk', 'eathand']

    boot_numbers, number_of_bootstraps = get_bootstrap_rounds(conf_matrix_both), get_total_rounds(conf_matrix_both)

    columns = 3
    rows = number_of_bootstraps

    # The color maps at the top of the image will be 1/8th the height of each confusion matrix
    height_ratios = [1 / (rows * 8)] + ([1] * rows)
    # We add an extra row for the color maps.
    fig, axes = plt.subplots(nrows=rows + 1, ncols=3, figsize=(columns * 5, rows * 5),
                             gridspec_kw={'height_ratios': height_ratios})

    title_suffix = ", ".join([str(number) for number in boot_numbers])
    # This just replaces the last comma with '.' :)
    title_suffix = title_suffix + "."
    fig_title = "Confusion Matrices for Bootstraps: " + title_suffix

    # We assign axes for the color map to be above the confusion matrices: The first row of axes
    #  is reserved for colormaps.
    #
    #  They are preceded by newlines cause trying find a way to make
    #  fig.suptitle appear above column labels drove me to the brink of insanity.
    newlines = "".join(["\n"] * rows)

    cbar_both_ax = axes[0][0]
    cbar_both_ax.set_title(newlines + 'Both')

    cbar_old_ax = axes[0][1]
    cbar_old_ax.set_title(newlines + 'Old')

    cbar_new_ax = axes[0][2]
    cbar_new_ax.set_title(newlines + 'New')

    for row_number, path_key in zip(np.arange(1, number_of_bootstraps + 1), conf_matrix_both):
        both_axes = axes[row_number, 0]
        old_axes = axes[row_number, 1]
        new_axes = axes[row_number, 2]
        boot_number = boot_numbers[row_number - 1]
        # Build the 'both' confusion matrices
        both_heatmap = sn.heatmap(conf_matrix_both[path_key], xticklabels=beh, yticklabels=beh,
                                  annot=True, cmap="YlGnBu", ax=both_axes,
                                  cbar=(row_number == 1),
                                  cbar_ax=cbar_both_ax,
                                  cbar_kws={'orientation': 'horizontal'})
        both_heatmap.set_xticklabels(both_heatmap.get_xticklabels(), rotation=60)
        both_axes.set_ylabel(boot_number, rotation=45, fontsize=15)


        # Build the 'old' confusion matrices
        old_heatmap = sn.heatmap(conf_matrix_old[path_key], xticklabels=beh, yticklabels=beh,
                                 annot=True, cmap="YlOrRd", ax=old_axes,
                                 cbar=(row_number == 1),
                                 cbar_ax=cbar_old_ax,
                                 cbar_kws={'orientation': 'horizontal'})
        old_heatmap.set_xticklabels(old_heatmap.get_xticklabels(), rotation=60)

        # Build the 'new'
        new_heatmap = sn.heatmap(conf_matrix_new[path_key], xticklabels=beh, yticklabels=beh,
                                 annot=True, cmap="PuBuGn", ax=new_axes,
                                 cbar=(row_number == 1),
                                 cbar_ax=cbar_new_ax,
                                 cbar_kws={'orientation': 'horizontal'})
        new_heatmap.set_xticklabels(new_heatmap.get_xticklabels(), rotation=60)

    # Gets current figure, sets size (width x height, in inches, given constant dpi)
    # fig.suptitle(fig_title, y=1.05)
    # plt.gcf().set_size_inches(12, 12)
    fig.suptitle(fig_title)

    plt.show()


def get_bootstrap_rounds(conf_matrix_both):
    bootstrap_numbers = []
    for boot in conf_matrix_both:
        index_of_boot_char = boot.rfind("b")
        index_of_ending_slash = boot.rfind("/")
        number = int(boot[(index_of_boot_char + 1):index_of_ending_slash])
        bootstrap_numbers.append(number)

    return bootstrap_numbers


def get_total_rounds(conf_matrix_both):
    return len(get_bootstrap_rounds(conf_matrix_both))
