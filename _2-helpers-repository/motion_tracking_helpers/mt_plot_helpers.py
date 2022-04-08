#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:03:40 2021

@author: marlinberger

helper functions for plotting in the context of the developement of the neural
networks for motion classification
"""

# python packages
from adjustText import adjust_text
from enum import Enum
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)


class Color(Enum):
    """specify some color-rgb-enums
    """
    PINK_0 = [255, 0, 243]
    PINK_1 = [255, 71, 255]
    PINK_2 = [209, 42, 209]
    PINK_3 = [156, 0, 155]
    PINK_4 = [213, 0, 203]
    MATT_GREEN = [72, 170, 120]
    DARK_GREEN = [0, 127, 9]
    WHITE = [255, 255, 255]

    FLAT_RED = "#EC7063"
    STRONG_RED = "#A93226"
    FLAT_PURPLE = "#A569BD"
    FLAT_BLUE = "#5DADE2"
    FLAT_MINT = "#45B39D"
    FLAT_GREEN = "#58D68D"
    FLAT_YELLOW = "#F4D03F"
    FLAT_ORANGE = "#EB984E"
    FLAT_GREY = "#AAB7B8"
    FLAT_NIGHTBLUE = "#34495E"

    fixed_class_colors = {
        "0": FLAT_RED,
        "1": FLAT_PURPLE,
        "2": FLAT_BLUE,
        "3": FLAT_MINT,
        "4": STRONG_RED,
        "5": FLAT_GREEN,
        "6": FLAT_YELLOW,
        "7": FLAT_ORANGE,
        "8": FLAT_GREY,
        "9": FLAT_NIGHTBLUE,
    }


def create_multicolor_single_line_legend(ax, numpoints, text, cmap):
    """add a legend that contains one single line, but in multiple colors.
    This makes sense when having plotted multiple lines in different colors,
    that shall be referred to.
    Possibly pass a custom, discrete colormap, that has just as many colors as
    numpoints are given. And that number again shall match the plottet lines.
    This way, a legend will be created with one single line, but this line will
    be multicolored with all colors used for the individual lines

    NOTE: if this function is included, the figure will be unpickleble due to
          the custom class for the ColorLineColection-Handler

    Args:
        ax (mpl.axes): the axes to add the legend to
        numpoints (int): of how many color segments the legend-line-symbol will
                         be composed
        text (str): the text to put next to the line, in the legend
        cmap (mpl.colors.cmap): a colormap to use
    """
    # create a custom HandlerLineCollection-class to control the color of the
    # line in the legend and make it multicolored
    class HandlerColorLineCollection(mpl.legend_handler.HandlerLineCollection):
        # adjust the create artists method
        def create_artists(self, legend, artist ,xdescent, ydescent,
                            width, height, fontsize,trans):
            # initialise the x-spanning points
            x = np.linspace(0, width, self.get_numpoints(legend)+1)
            # adjust the associated y points
            y = np.zeros(self.get_numpoints(legend)+1)+height/2.-ydescent
            # create the points and segments for line collection call
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            # create the line collection
            lc = mpl.collections.LineCollection(segments, cmap=artist.cmap,
                        transform=trans)
            lc.set_array(x)
            lc.set_linewidth(artist.get_linewidth())
            return [lc]
    
    # initialise the segments of the dummy multicolor line, for which the
    # legend will be drawn
    t = np.array([int(segment) for segment in range(numpoints)])
    x = t
    y = t
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # initialise the dummy-multicolor line
    lc = mpl.collections.LineCollection(segments, cmap=cmap,
                        norm=plt.Normalize(0, 10), linewidth=3)
    lc.set_array(t)

    # plot the legend with the multicolor line
    ax.legend(
        [lc],
        [text],
        loc="upper left",
        handler_map={lc: HandlerColorLineCollection(numpoints=numpoints)},
        handlelength=int(numpoints/2),
        framealpha=1,
    )


def allign_annotation_texts(ax, value_text_pairs, pos_x, space_min=.06,
        color_lookup=False, standard_color="k", fontsize=10, ha="left"):
    """function to place texts at different y-values on a fixed x point in a
    way they dont overlap and are as close as possible to the value, they are
    inteded to be

    Args:
        ax (mpl.axes): the axes to add the legend to
        value_text_pairs (list(list)): a nested list that contains stacked
                                       pairs of the desired y-value and the
                                       a text to place there
        pos_x (float): at which x-position the texts are placed
        space_min (float): the minimal space between the texts
        color_lookup (False || dict): Wether false, or a dict that can be
                                      lookup'ed to reveal a color for every
                                      text. If false, the standard_color will
                                      be used
        standard_color (str): a color to use for the texts if no color_lookup
                              is given
        fontsize (float): the fontsize for the texts
        ha (str): horizantal allignmnet. in ["left", "center", "right"]
    """
    # adjust the text positions, ensure y-position and horizontal allignment
    # still fit
    sorted_value_text_pairs = sorted(
        value_text_pairs, key=lambda x: x[0], reverse=True
    )
    texts_sorted = np.array([str(pair[1]) for pair in sorted_value_text_pairs])
    n_texts = len(texts_sorted)
    positions_init = np.array([pair[0] for pair in sorted_value_text_pairs])
    strict_down = np.array([1 - i * space_min for i in range(n_texts)])
    strict_up = np.array([0 + i * space_min for i in range(n_texts)][::-1])
    good_pos_found = np.array([False for _ in range(n_texts)])
    positions_dyn_1 = np.array([0. for _ in range(n_texts)])
    positions_final = np.array([-1. for _ in range(n_texts)])
    # check if strict upwards placement needs to be done at some class
    # starting
    # save temporal evaluations in temp_sto
    temp_sto = []
    for n, (pos_orig, pos_down) in enumerate(
            zip(positions_init, strict_down)):
        # if the desired position is bigger than the strictly decending numbers,
        # all texts above the currently evaluated need to be corrected by the
        # initial spacing pattern
        if pos_orig >= pos_down:
            temp_sto.append(n)
    # perform it if necessary
    if len(temp_sto) > 0:
        lowest_upper_sort_bound = temp_sto[-1]
        for n in range(lowest_upper_sort_bound+1):
            positions_final[n] = strict_down[n]
            good_pos_found[n] = True
    # check if strict downwards placement needs to be done at some class
    # starting
    # save temporal evaluations in temp_sto
    temp_sto = []
    for n, (pos_orig, pos_up) in enumerate(
            zip(positions_init, strict_up)):
        # if the desired position is lower than the strictly ascending numbers,
        # all texts beneath the currently evaluated need to be corrected by the
        # initial spacing pattern
        if pos_orig <= pos_up:
            temp_sto.append(n)
    # perform it if necessary
    if len(temp_sto) > 0:
        highest_lower_sort_bound = temp_sto[0]
        for n in range(n_texts - highest_lower_sort_bound):
            positions_final[-(n+1)] = strict_up[-(n+1)]
            good_pos_found[-(n+1)] = True
    # initialise the dynamic array by substituting the init values for those
    # that are not yet found
    positions_dyn_1 = positions_final
    idx_left = np.where(positions_dyn_1 == -1)
    positions_dyn_1[idx_left] = positions_init[idx_left]

    # write them initially to the axes
    texts = []
    for plot_text, pos_y in zip(texts_sorted, positions_dyn_1):
        # assign the color for the current text
        if not color_lookup:
            color = standard_color
        else:
            color = color_lookup[str(plot_text)]
        # plot the text
        text = ax.text(
            pos_x,
            pos_y,
            plot_text,
            color=color,
            fontsize=fontsize,
            horizontalalignment=ha
        )
        # save the text to later adjust it's position to not overlap each other
        texts.append(text)
    
    # extract the texts that do not rely on the conditions above
    adjusts_left = [texts[i] for i in idx_left[0]]
    # pass them to the adjust_text-method
    adjust_text(
        adjusts_left,
        only_move={'points':'y', 'text':'xy', 'objects':'xy'},
        ha=ha
    )

    # finally ensure the texts are all set up at the right x value
    for text in texts:
        _, y = text.get_position()
        text.set_position((pos_x, y))
        text.set_horizontalalignment("left")


def colormap_from_to_rgb(from_rgb, to_rgb):
    """return a matplotlib colormap, fading from one given rbg_colorcode to
    another

    Args:
        from_rgb (list(int)): the start-rgb-color, ranging from 0 to 255
        to_rgb (list(int)): the end-rgb-color, ranging from 0 to 255
    
    Returns:
        cmap (mpl.cmap): a colormap object from matplotlib
    """
    # transform the inputs to a range from 0 to 1
    from_rgb = [color_num / 255. for color_num in from_rgb]
    to_rgb = [color_num / 255. for color_num in to_rgb]
    # seperate the rgb values
    r1, g1, b1 = from_rgb
    r2, g2, b2 = to_rgb
    # generate the color-dict
    cdict = {
        "red": ((0, r1, r1), (1, r2, r2)),
        "green": ((0, g1, g1), (1, g2, g2)),
        "blue": ((0, b1, b1), (1, b2, b2))
    }
    # generate linear colormap
    cmap = mpl.colors.LinearSegmentedColormap("lin_custom_cmap", cdict)
    return(cmap)


def add_single_value_on_y(ax, tick_label, secondary=False):
    """add a single value, centered, on the y-axis

    Args:
        ax (mpl.axes): the axes to process
        tick_label (str): the string to place on the axis
        secondary (bool): if the value shall be displayed on an additional axes
    """
    # get secondary axis if desired
    ax_relevant = ax.twinx() if secondary else ax
    # center the tick on the new axes by averaging the current y-limits of the
    # axis
    ax_relevant.set_yticks([np.average(ax_relevant.get_ylim())])
    # set the desired label
    ax_relevant.set_yticklabels([tick_label])
    # remove the ticks themselv to only keep the label. itteration is needed
    # as the twinx-func breaks previous deactivated ticks
    for axis in [ax, ax_relevant]:
        axis.tick_params(
            axis='y',
            right=False,
            left=False,
        )


def convert_to_sweet_k(value):
    """convert a given value to a sweet k-string if it is a big number, sothat
    it takes away less space but still contains the desired amount of
    imformation
    
    Args:
        value (float || int): the value to transform
    
    Returns:
        sweet_value (str): the transformed value representation
    """
    # if the value exceeds 4 digits, transform it
    if value > 999:
        value = np.round(value / 1000., 1)
        # drop the decimals for values over 9.9k
        if value >= 10:
            value = int(value)
        # make it a 'k'-str
        value = str(value) + "k"
    else:
        # different ranges for different value ranges
        if value > 9.9:
            value = np.round(value, 1)
        elif value < 1:
            # drop the 0 in front
            value = str(np.round(value, 3))[1:]
        else:
            value = np.round(value, 2)
        # make it a str
        value = str(value)
    return(value)


def custom_grid_0_n(ax, n=1., tick_decimals=1, ticks_only=False):
    """add a sweet custom grid, that ranges from 0 to n, where n defaults to 1.
    5 major and 20 minor lines are added. The major line tick values will be
    diaplayed by matplotlibs default method

    Args:
        ax (plt.axes): the axis to add the grid on
        n (float): the value for the upper end of the grid
        tick_decimals (int): how many decimals the ticks shall have
        ticks_only (bool): supress the grid lines, which makes sense for
                           secondary axes, to not plot the grid above other's
                           axes elements
    """
    # initialise the tick-arrays
    major_ticks = np.round(np.linspace(0, n, 6), tick_decimals)
    minor_ticks = np.linspace(0, n, 21)
    # set the ticks
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    # plot them
    if not ticks_only:
        ax.grid(axis='y', which='minor', alpha=0.2)
        ax.grid(axis='y', which='major', alpha=0.5)


def add_values_above_bars(axes, padding_top=.02, decimals=2, color="k",
        fontsize=9.5, fontweight="bold"):
    """given a list of axes, this function gets all the bars, their values, and
    adds the values as text on top

    Args:
        axes (list(axes)): a tuple with axes
        padding_top (float): the space above the bar
        decimals (int): the decimals displayed
        color (str): the color of the text
        fontsize (float): the fontsize
        fontweight (str): the fontweight
    """
    # loop through all given axes
    for ax in axes:
        # loop over all bar containers that are found at the given axis
        for bar_con in ax.containers:
            # loop through all plotted bars per container, get theire
            # coordinates (-> values) and print theire y values on top of each
            # bar
            for bar in bar_con:
                # extract each bar's value
                w,h = bar.get_width(), bar.get_height()
                # lower left vertex
                x0, y0 = bar.xy
                # top left vertex
                x2, y2 = x0,y0+h
                # top right vertex
                x3, _ = x0+w,y0+h
                # create the text on top of the bar
                ax.text(
                    ((x2+x3)/2), y2+padding_top,
                    str(round(y2,decimals)),
                    color=color,
                    ha='center',
                    fontsize=fontsize,
                    fontweight=fontweight
                )


def annotate_im_matrix(matrix, ax1):
    """display the values behind the numbers as text, in each field of the axes,
    an imshow()-object got plotted

    Args:
        matrix (np.array): the matrix that is displayed
        ax (plt.axes): the axis to add the grid on
    """
    # itterate over the dimensions and display the values
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # make the plotting of the numbers sweet, by not displaying zeros
            # before the colon and displaying round numbers without decimals
            if float(matrix[i, j]).is_integer():
                val = int(matrix[i, j])
            else:
                val = f"{round(matrix[i, j], 2)}"[1:]
            text = ax1.text(
                j, i, val, ha="center", va="center", color="k"
            )


def get_classes_from_Y(Y_arrays):
    """return the (unique) sorted labels that appear in the Y-data

    Args:
        Y_arrays (list(np.array)): any number of Y-data arrays (like Y_true,
                                   Y_pred,...) with the shape (n,), that will be
                                   reduced the the unique number of appearing
                                   classes
                            
    Return:
        classes (np.array): contains the unique classes
    """
    classes = np.unique(np.concatenate(Y_arrays))
    return(classes)


def models_vs_datasets_accuracy(Eval_Specs, enum_names=False):
    """plot the accuracys from all models and datasets that are in the present
    analysation run against each other

    Args:
        Eval_Specs (obj): object with all the information about the choosen
                          evaluation folder and the current run
        enum_names (Bool): if the names of the models shall get enumerated in
                           in the plot at the y-axis, to easier reference on
                           them in a paper e.g.
    
    Returns:
        f (matplotlib.figure): the created figure is returned, sothat it can
                               get saved or whatsoever
    """
    # create a simplyfied storage for easy plotting. This storrage conatins one
    # array for every desires axis: model_name, id, dataset_name and accuracy
    unique_model_names = sorted(
        list(set(Eval_Specs.eval_storrage.keys())),
        key=lambda x: x.split("--")[-1]
    )
    # sort the data names in order of the names without the created-date
    unique_data_names = sorted(
        list(set(Eval_Specs.eval_storrage[unique_model_names[0]].keys()))
    )

    # the storage for all accuracy values
    accuracys_matr = []

    # itterate over the eval storage, extract what is desired
    for model_name in unique_model_names:
        # temp storage for the accurcys per model
        accuracy_vector = []
        # ensure the order is always the same for the datapacks
        for data_name in unique_data_names:
            accuracy = (
                Eval_Specs.eval_storrage[model_name][data_name]["accuracy"]
            )
            accuracy_vector.append(accuracy)
        # save the accuracys to the matrix
        accuracys_matr.append(accuracy_vector)
    
    # get numpy representation
    acc_arr = np.array(accuracys_matr)

    # slice the names down to the wandb-name-part
    unique_model_names = [name.split("--")[-1] for name in unique_model_names]

    # if id's are desired insted of run names, they are available in the
    # Eval_Specs object and looked up here
    if Eval_Specs.fetched_run_data_available:
        # TODO: get run-id's from wandb-df and overwrite the unique_model_names
        #       list with the id's
        print("\nusing the id's for naming the runs is not yet implemented\n")
    
    # get dataframe representation
    accuracys_df = pd.DataFrame(
        acc_arr,
        index=unique_model_names,
        columns=unique_data_names
    )
    
    # initialise the plot
    f, ax1 = plt.subplots()
    # set the limits for the plotting color range and the limits for the data
    vmin_plot = .5
    vmax_plot = 1
    vmin_data = 0
    vmax_data = 1

    # create a custom colormap
    cmap1 = colormap_from_to_rgb(Color.WHITE.value, Color.DARK_GREEN.value)

    # plot the accuracy matrix, add the values in the cells
    im = ax1.imshow(
        acc_arr, cmap=cmap1, vmin=vmin_plot, vmax=vmax_plot,  aspect="auto"
    )
    annotate_im_matrix(acc_arr, ax1)

    # initialise the colorbar
    cbar = ax1.figure.colorbar(im, ax=ax1, location="right")
    cbar.ax.set_ylabel("Accuracy", rotation=-90, va="bottom")
    cbar.ax.set_ylim([vmin_data, vmax_data])
    # match the position with the axes from the matrix
    cbar_ax_pos = cbar.ax.get_position()
    ax1_pos = ax1.get_position()
    new_pos = [cbar_ax_pos.x0, ax1_pos.y0, cbar_ax_pos.x1, ax1_pos.y1]
    cbar.ax.set_position(new_pos)

    # add a number to each model's name, for easy referencing in text documents
    if enum_names:
        unique_model_names_mod = [
            f"{i}-{name}" for i, name in enumerate(unique_model_names)
        ]
    else:
        unique_model_names_mod = unique_model_names

    # set the ticks
    ax1.set_xticks(np.arange(len(unique_data_names)))
    ax1.set_xticklabels(unique_data_names, rotation = -30, ha="left")
    ax1.set_yticks(np.arange(len(unique_model_names_mod)))
    ax1.set_yticklabels(unique_model_names_mod, rotation = 25, va="top")
    # make the ticksize smaller as there are many
    ax1.tick_params(axis='both', which='major', labelsize=8.5)

    # set up final stuff
    ax1.set_title(f"Models vs. Datapacks\nAccuracy", pad=13)
    ax1.set_ylabel("Model")
    ax1.set_xlabel("Datapack")
    f.tight_layout()

    return(f)


def plot_acc_over_class_appearance(Y_max_prob, pass_args, buckets=4):
    """extract and plot the accuracy for a class, relative to the action
    appearance, to determine if actions are by a higher margin correctly
    identified in the middle of a class appearance in contrast to instances
    that appear close to an action transition

    NOTE: Importantly, the given Y_max_prob-values must not be shuffeled for
          this to work correctly.

    Args:
        Y_max_prob (tuple(np.array)): contains the true- and pred- labels
        pass_args (tuple): contains 3 arguments that describe how the given
                           predictions were derived:
                                run_name (str): the name of the run, in which
                                                the model has it's origin
                                datapack_name (str): the name of the datapack
                                                     that's currently analysed
                                config (dict): config dict from the run
        buckets (int): with how many seperating steps the accuracy is determined
                       over the actions appearance
    
    Returns:
        f (matplotlib.figure): the created figure is returned, sothat it can
                               get saved or whatsoever
    """
    # unpack the bundled inputs
    Y_true, Y_pred = Y_max_prob
    run_name, datapack_name, config = pass_args

    # get present classes only in the true values
    classes = get_classes_from_Y([Y_true])

    # get the points where the action class changes
    changing_idx = np.where(Y_true[:-1] != Y_true[1:])[0] + 1
    # add the start and end point
    changing_idx = np.insert(
        changing_idx, [0, len(changing_idx)], [0, len(Y_true)]
    )

    # create a per-class-pred-and-wheigths-buckets storage
    sto = {
        class_name: {
            "pred_eval": [[] for _ in range(buckets)],
            "weight": [[] for _ in range(buckets)],
            "bucketized_accuracy": [],
            "bucket_support": []
        } for class_name in classes
    }

    # itterate through each "appearance pack"
    for idx_0, idx_1 in zip(changing_idx[:-1], changing_idx[1:]):
        # extract the pack's labels and prediction
        pack_true = Y_true[idx_0:idx_1]
        pack_pred = Y_pred[idx_0:idx_1]
        pack_corr = (pack_true == pack_pred)
        pack_true_class = pack_true[0]
        # get the breaking points for the buckets
        step_len = len(pack_true) / float(buckets)
        seq_points = [i * step_len for i in range(buckets+1)]

        # sort the evaluation's of the packs into the right buckets
        for bucket_n, (edge_1, edge_2) in enumerate(
                zip(seq_points[:-1], seq_points[1:])):
            # shorten some calls
            curr_pred_list = sto[pack_true_class]["pred_eval"][bucket_n]
            curr_weight_list = sto[pack_true_class]["weight"][bucket_n]
            # first, get the full weighted instances
            full_weight_instances = pack_corr[
                int(np.ceil(edge_1)):int(np.floor(edge_2))
            ]
            # save the evaluations
            curr_pred_list.extend(list(full_weight_instances))
            # ...and save theire wheights, which is 1 for the full_weighted's
            curr_weight_list.extend(
                [1 for _ in range(len(full_weight_instances))]
            )
            # inspect the lower end for partially weighted instances
            if int(np.ceil(edge_1)) > edge_1:
                # get the instance that needs to be segmented on two bucketes
                # with a corresponding weight
                evaluation = pack_corr[int(np.floor(edge_1))]
                weight = 1 - (edge_1 - np.floor(edge_1))
                # save the evaluation
                curr_pred_list.append(evaluation)
                # ...and it's weight
                curr_weight_list.append(weight)
            # inspect the upper end for partially weighted instances
            if int(np.floor(edge_2)) < edge_2:
                # get the instance that needs to be segmented on two bucketes
                # with a corresponding weight
                evaluation = pack_corr[int(np.ceil(edge_2))-1]
                weight = edge_2 - np.floor(edge_2)
                # save the evaluation
                curr_pred_list.append(evaluation)
                # ...and it's weight
                curr_weight_list.append(weight)
    
    # calculate the accuracys per class for each bucket
    for class_n in classes:
        # itterate over the buckets
        for bucket_n in range(buckets):
            # assign the current class & bucket lists as arrays for ease
            temp_pred_eval = np.array(sto[class_n]["pred_eval"][bucket_n])
            temp_weights = np.array(sto[class_n]["weight"][bucket_n])
            # get the indexes where the model predicted correctly and where not
            idx_corr = np.where(temp_pred_eval == True)[0]
            idx_incorr = np.where(temp_pred_eval == False)[0]
            # get the weights that correspond to correct & incorrect predictions
            corr_pred_weights = temp_weights[idx_corr]
            incorr_pred_weights = temp_weights[idx_incorr]
            # calculate the associated supports
            support_corr = np.sum(corr_pred_weights)
            support_complete = (
                np.sum(corr_pred_weights) + np.sum(incorr_pred_weights)
            )
            # finally calculate the bucket accuracy
            bucket_accuracy = support_corr / support_complete
            # save the results
            sto[class_n]["bucketized_accuracy"].append(bucket_accuracy)
            sto[class_n]["bucket_support"].append(support_complete)

    # start with the plotting
    f, ax1 = plt.subplots()
    ax1.set_ylim([-.05, 1.14])

    # make a hard color assignment here, sothat the classes always have the
    # same colors
    colors = Color.fixed_class_colors.value
    colors_in_this_plot = (
        [colors[str(associated_color)] for associated_color in classes]
        )

    # initialise the x values: ranging from 0 to 100 %
    x_vals = np.round(np.linspace(0, 100, buckets), 2)
    class_marker_x = 101

    # plot the different classes and add theire name as text in the same color
    class_acc_sto = []
    for class_n in classes:
        # plot the lines and texts for each class
        accuracys = sto[class_n]["bucketized_accuracy"]
        ax1.plot(x_vals, accuracys, c=colors[str(class_n)], alpha=.95)
        # store the class and it's last accuracy for later making the
        # annotations of the classes
        class_acc_sto.append([accuracys[-1], class_n])
    
    # ensure the texts do not overlap, are ordered right and placed in a neat
    # way
    allign_annotation_texts(
        ax1, class_acc_sto, class_marker_x, color_lookup=colors
    )
    
    # create the a discrete colomap from the line colors,
    # for the multicolor-single-line-legend
    cmap_legend_line = mpl.colors.LinearSegmentedColormap.from_list(
        "Custom cmap",
        colors_in_this_plot,
        len(classes)
    )
    create_multicolor_single_line_legend(
        ax1, len(classes), "(action-) classes", cmap_legend_line
    )

    # plot labels, grid, axes...
    custom_grid_0_n(ax1)
    ax1.set_title(f"{run_name}\n{datapack_name}\n{config['Script']}")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Action appearance [%]")
    f.tight_layout()

    return(f)


def plot_acc_over_sample_time(Y_max_prob, pass_args, buckets=8,
        mask_zero_support=True):
    """plot the accuracy over the sample time, to evaluate if models performs
    differently while an operator learns (as all test persons performed the
    process for the first time).
    The results are harmonized, bucketized and weighted. This means, the
    accuracy is evaluated for each class own its own. This is performed
    bucketized for each class. If per_class=True, these lines are plotted,
    whereas for per_class=False, these per-class-buckets are using a weighted
    average.

    NOTE: Importantly, the given Y_max_prob-values must not be shuffeled for
          this to work correctly.
          The function also assumes that the instances per-class appear approx.
          equal distributed across the dataset. If this is not the case, the
          significance of the plot goes down. To be clear: This means skewness
          in total is fine, but if a class only apperas at the very start of the
          dataset, this is not fine.
    
    NOTE: This also makes only sense if only one consecutive video is contained
          in the dataset to analyse, otherwise the unshuffeled dataset still
          passes the instances for one video after another. If those should be
          equally distributed, this must be implemented in the DataBuilder class

    Args:
        Y_max_prob (tuple(np.array)): contains the true- and pred- labels
        pass_args (tuple): contains 3 arguments that describe how the given
                           predictions were derived:
                                run_name (str): the name of the run, in which
                                                the model has it's origin
                                datapack_name (str): the name of the datapack
                                                     that's currently analysed
                                config (dict): config dict from the run
        buckets (int): with how many seperating steps the accuracy is determined
                       over the sample data
        mask_zero_support (bool): wether areas, where no supporting instances
                                  exist, should be masked (in grey) or not
    
    Returns:
        f (matplotlib.figure): the created figure is returned, sothat it can
                               get saved or whatsoever
    """
    # unpack the bundled inputs
    Y_true, Y_pred = Y_max_prob
    run_name, datapack_name, config = pass_args

    # get present classes only in the true values
    classes = get_classes_from_Y([Y_true])

    # for which number of buckets*instances the bucketization is valid. If not
    # valid, the bucket number will be halfed until a valid value is found
    bucket_valid_factor = 10
    
    # combined storage for the labels and predictions per class
    results = {
        class_n: {"percent": [], "accuracy": [], "support": []
        } for class_n in classes
    }
    for true_class in classes:
        # get the indexes where the current class appears
        idx_true_class = np. where(Y_true == true_class)
        # slice out the true and predicted labels
        y_true_class_n = Y_true[idx_true_class]
        y_pred_class_n = Y_pred[idx_true_class]

        # get the number off instances backing this class
        support = np.shape(y_true_class_n)[0]
        # get a valid number of buckets for the current class
        buckets_i = buckets
        while buckets_i > 1:
            if support >= bucket_valid_factor*buckets_i:
                break
            else:
                buckets_i /= 2
        # determine the resulting bucket size and create the array with the
        # seperation point indexes for the buckets
        bucket_size = int(support/buckets_i)
        sep_points = [i*bucket_size for i in range(int(buckets_i))]
        # adjust the last entry to exacly match the data len, what is not
        # necicarially given due to numerical int-float-dividing issues
        sep_points[-1] = support-bucket_size-1

        # map the buckets to percent values which will be plotted on the x-axis
        percents = np.round(
            np.linspace(0, 100, int(buckets_i)+1, endpoint=False)[1:],
            1
        )

        # get the accuracys per bucket
        for percent, idx_s in zip(percents, sep_points):
            # assign the whole datapack if there is only one bucket
            if buckets_i == 1:
                label = Y_true[idx_true_class]
                pred = Y_pred[idx_true_class]
            else:
                label = y_true_class_n[idx_s:idx_s+bucket_size]
                pred = y_pred_class_n[idx_s:idx_s+bucket_size]
            # get the accuracy for the current bucket
            bucket_i_accuracy = accuracy_score(label, pred)
            # support equals bucket_size. calculate it dynamically if this
            # changes in future versions
            support = len(label)
            # save the results
            results[true_class]["percent"].append(percent)
            results[true_class]["accuracy"].append(bucket_i_accuracy)
            results[true_class]["support"].append(support)
    
    # spread the results weighted into the given desired number of buckets,
    # eventhough the calculation was performed for less buckets for a class
    for true_class in classes:
        # skip this processing part for classes that got calculated with
        # the desired bucket number imidiatly
        if len(results[true_class]["accuracy"]) == buckets:
            continue

        # get the mapping for the default bucket size
        percents = np.round( 
            np.linspace(0, 100, buckets+1, endpoint=False)[1:], 1
        )
        
        # use -1 as placeholder for later masking and not confusing empty
        # buckets with those that really have accuracy=0
        new_accuracys = [-1 for _ in range(buckets)]
        new_supports = [0 for _ in range(buckets)]
        
        # itterate over the results and distribute them into the desired initial
        # buckets
        r = results[true_class]
        for percent, accuracy, support in zip(
                r["percent"], r["accuracy"], r["support"]):
            # get the percent-bucket-values the current bucket value liew in
            # between
            idx_first_bigger_bucket = np.where(percents > percent)[0][0]
            lower_percent = percents[idx_first_bigger_bucket - 1]
            upper_percent = percents[idx_first_bigger_bucket]
            # calculate the support values with which the distributed accuracy
            # gets weighted in the new buckets
            lower_support_fac = (
                abs(upper_percent-percent) / (upper_percent - lower_percent)
            )
            upper_support_fac = (
                abs(lower_percent-percent) / (upper_percent - lower_percent)
            )
            lower_support = int(lower_support_fac * support)
            upper_support = int(upper_support_fac * support)
            # assign the new values. but only if there is any support left, else
            # just keep the -1 as marker for "no value" and later masking
            if lower_support > 0:
                new_accuracys[idx_first_bigger_bucket-1] = accuracy
                new_supports[idx_first_bigger_bucket-1] = lower_support
            if upper_support > 0:
                new_accuracys[idx_first_bigger_bucket] = accuracy
                new_supports[idx_first_bigger_bucket] = upper_support
        
        # overwrite the initial set buckets for the class with the newly
        # distributed arrays, that have the defaultly desired bucket size
        results[true_class]["percent"] = percents
        results[true_class]["accuracy"] = new_accuracys
        results[true_class]["support"] = new_supports


    # start with the visualization
    fig_scale = 3
    f = plt.figure(figsize=(2.2*fig_scale,1.4*fig_scale))
    
    # initialise and set height ratios for subplots
    height_ratios = np.ones(len(classes)+1)
    height_ratios[-1] = 1.5
    gs = mpl.gridspec.GridSpec(len(classes)+1, 1, height_ratios=height_ratios)

    # initialise axes container
    ax_save = []

    # open up buckets for the later calculated weighted average across all
    # classes
    accuracy_sto = []
    support_sto = []

    # initialise some general plot vars
    vmin = 0
    vmax = 1
    linecolor = "k"
    linewidth = 1

    # plot per-class-stuff
    for i, (ax_i, class_n) in enumerate(zip(gs, results.keys())):
        # extract the accuracys to plot
        accuracys = np.array(results[class_n]["accuracy"])
        accuracy_sto.append(accuracys)
        im_accuracys = np.expand_dims(accuracys, axis=0)

        # initialise the colormap
        cmap = mpl.colormaps["YlOrBr"]
        # mask -1 accuracys, which means: no value, by setting a different
        # color for values below the range defines in the imshow-call
        if mask_zero_support:
            cmap.set_under("#D3D3D3")
        
        # get the current axes and save it
        ax = plt.subplot(ax_i)
        ax_save.append(ax)

        # plot the accuracys as line
        ax.plot(.5-accuracys, c=linecolor, linewidth=linewidth)
        # display the colored pads
        ax.imshow(
            im_accuracys, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto"
        )

        # plot the class on the y-axis
        add_single_value_on_y(ax, class_n)

        # process the support values
        supports =  np.array(results[class_n]["support"])
        support_sto.append(supports)
        # make the formatting sweat for large values
        support_sum = convert_to_sweet_k(np.sum(supports))
        # set the support on the right side on the y axis
        add_single_value_on_y(ax, support_sum, secondary=True)

        # the last axes is the averaged plot, which will be handelsd seperatly
        if i == len(classes):
            break
    
    # calculate the weighted average over all classes per time step
    eps = 1e-12
    accuracy_matr = np.array(accuracy_sto)
    support_matr = np.array(support_sto)
    # get weights while avoiding zero division
    weights = support_matr / (np.sum(support_matr, axis=0) + eps)
    weighted_accuracys = np.sum(accuracy_matr * weights, axis=0)

    # plot averaged values
    ax = plt.subplot(gs[-1])
    ax_save.append(ax)
    ax.plot(.5-weighted_accuracys, c=linecolor, linewidth=linewidth)
    im = ax.imshow(
        np.expand_dims(weighted_accuracys, axis=0), cmap=cmap, vmin=vmin,
        vmax=vmax, aspect='auto'
    )
    add_single_value_on_y(ax, r"$\sum_n$")


    # plot the percents on the x axis
    percent_ticks_n = 5
    # initialise the ticks to display
    percent_ticks = np.round(np.linspace(0, 100, percent_ticks_n), 1)
    # set up the spacing as it depends on the buckets used
    ax.set_xticks(
        np.linspace(0, buckets-1, len(percent_ticks))
    )
    # set the labels, format them to have a percent sign with them
    ax.set_xticklabels(
        [str(i) + " %" for i in percent_ticks])
    # deactivate the x ticks
    ax.tick_params(
            axis='x',
            top=False,
            bottom=False,
        )

    # add the support
    support_sum = convert_to_sweet_k(np.sum(support_matr))
    add_single_value_on_y(ax, support_sum, secondary=True)

    # remove vertical gap between subplots, make space for the colorbar
    subplots_end = .79
    f.subplots_adjust(hspace=.0, left=.07, right=subplots_end, top=.84)
    
    # set positions for the colorbar
    x0 = subplots_end + .11
    y0 = ax_save[-1].get_position().y0
    width = 0.01
    height = 0.73
    # initialise and display the colorbar
    cbar_ax = f.add_axes([x0, y0, width, height])
    cbar = f.colorbar(im, cax=cbar_ax)
    #cbar.ax.set_ylabel("accuracy", rotation=-90, va="bottom")
    cbar.ax.set_ylim([0, 1])

    # make the plot sweet with last layout steps
    f.suptitle(
        f"{run_name}\n{datapack_name}\n{config['Script']}",
        fontsize=12
    )
    # get the middle position of the cureve subplots axes
    plot_middle = (
        ax_save[0].get_position().x0 + ax_save[0].get_position().width / 2
    )
    # put up all the labels for the commulated axes
    f.text(
        plot_middle, 0.028, "Dataset progress", ha="center", va="center",
        fontsize=10
    )
    f.text(
        0.024, 0.5, "Class", ha="center", va="center", rotation=90,
        fontsize=10)
    f.text(
        subplots_end+.19, 0.5, "Accuracy", ha="center", va="center",
        rotation=-90, fontsize=10)
    f.text(
        subplots_end+.08, 0.5, "Support", ha="center", va="center",
        rotation=-90, fontsize=10)
    
    return(f)


def plot_precision_and_recal_per_class(Y_max_prob, pass_args):
    """plot precision and recall for all classes represented in the given data.
    Note, that if the number of different classes is hugh, this plot will be
    overloaded and not the weapon of choice.
    Remember to include a "plt.show()" at the end of the script that uses this
    function, to keep the plot open.

    Args:
        Y_max_prob (tuple(np.array)): contains the true- and pred- labels
        pass_args (tuple): contains 3 arguments that describe how the given
                           predictions were derived:
                                run_name (str): the name of the run, in which
                                                the model has it's origin
                                datapack_name (str): the name of the datapack
                                                     that's currently analysed
                                config (dict): config dict from the run
    
    Returns:
        f (matplotlib.figure): the created figure is returned, sothat it can
                               get saved or whatsoever
    """
    # unpack the bundled inputs
    Y_true, Y_pred = Y_max_prob
    run_name, datapack_name, config = pass_args
    
    target_names = [str(i) for i in list(get_classes_from_Y((Y_true, Y_pred)))]
    class_rep = classification_report(
        Y_true, Y_pred, target_names=target_names, output_dict=True,
        zero_division=0)
    precisions = [class_rep[c]["precision"] for c in target_names]
    recalls = [class_rep[c]["recall"] for c in target_names]

    # initialise the plot
    f, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # how many groups of bars will exist
    bar_groups = len(target_names)
    # map those to x-values. the "span_till" value, combined with the width will
    # mainly dictate how packed or spacious the bars are
    span_till = bar_groups-1
    # initialise the spacing
    x = np.linspace(0, span_till, bar_groups)
    # define values for the bar's layout
    width = .27
    gap = .035
    alpha_val = 1
    # plot the bars with the macro values
    ax1.bar(
        x-width/2-gap/2 , precisions, width, color="#DFA4EB",
        alpha=alpha_val, zorder=10, edgecolor="k", linewidth=.5,
        label="precision"
    )
    # plot the bars with the weighted values
    ax2.bar(
        x+width/2+gap/2 , recalls, width, color="#90B2B4",
        alpha=alpha_val, zorder=10, edgecolor="k", linewidth = .5,
        label="recall"
    )

    # add a neat custom gridz
    custom_grid_0_n(ax1)
    custom_grid_0_n(ax2, ticks_only=True)

    # initialise the ticks on the x axis, use the positions created above (=x)
    x_ticks_arr = x
    x_ticks = target_names

    # plot what is left: legend, title, the limits, add the x-ticks, tighten
    # the plot
    ax1.legend(framealpha=1, loc="upper left")
    ax2.legend(framealpha=1, loc="upper right")
    ax1.set_ylim(0, 1.2)
    ax2.set_ylim(0, 1.2)
    ax1.set_title(f"{run_name}\n{datapack_name}\n{config['Script']}")
    ax1.set_xticks(x_ticks_arr)
    ax1.set_xticklabels(x_ticks)
    ax1.set_ylabel("precision")
    ax2.set_ylabel("recall")
    ax1.set_xlabel("class")
    f.tight_layout()

    return(f)


def plot_f1_score_support_per_class(Y_max_prob, pass_args):
    """plot the f1-scores and the support for all classes represented in the
    given data.
    Note, that if the number of different classes is hugh, this plot will be
    overloaded and not the weapon of choice.
    Remember to include a "plt.show()" at the end of the script that uses this
    function, to keep the plot open.

    Args:
        Y_max_prob (tuple(np.array)): contains the true- and pred- labels
        pass_args (tuple): contains 3 arguments that describe how the given
                           predictions were derived:
                                run_name (str): the name of the run, in which
                                                the model has it's origin
                                datapack_name (str): the name of the datapack
                                                     that's currently analysed
                                config (dict): config dict from the run
    
    Returns:
        f (matplotlib.figure): the created figure is returned, sothat it can
                               get saved or whatsoever
    """
    # unpack the bundled inputs
    Y_true, Y_pred = Y_max_prob
    run_name, datapack_name, config = pass_args
    
    target_names = [str(i) for i in list(get_classes_from_Y((Y_true, Y_pred)))]
    class_rep = classification_report(
        Y_true, Y_pred, target_names=target_names, output_dict=True,
        zero_division=0)
    
    f1_scores = [class_rep[c]["f1-score"] for c in target_names]
    supports = [class_rep[c]["support"] for c in target_names]

    # initialise the plot
    f, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # how many groups of bars will exist
    bar_groups = len(target_names)
    # map those to x-values. the "span_till" value, combined with the width will
    # mainly dictate how packed or spacious the bars are
    span_till = bar_groups-1
    # initialise the spacing
    x = np.linspace(0, span_till, bar_groups)
    # define values for the bar's layout
    width = .27
    gap = .035
    alpha_val = 1
    # plot the bars with the macro values
    ax1.bar(
        x-width/2-gap/2 , f1_scores, width, color="#33BFFF",
        alpha=alpha_val, zorder=10, edgecolor="k", linewidth=.5,
        label="f1_score"
    )
    # plot the bars with the weighted values
    ax2.bar(
        x+width/2+gap/2 , supports, width, color="#B3C7D0",
        alpha=alpha_val, zorder=10, edgecolor="k", linewidth = .5,
        label="support"
    )

    # add a neat custom grids
    custom_grid_0_n(ax1)
    custom_grid_0_n(ax2, n=max(supports), tick_decimals=0, ticks_only=True)

    # initialise the ticks on the x axis, use the positions created above (=x)
    x_ticks_arr = x
    x_ticks = target_names

    # plot what is left: legend, title, the limits, add the x-ticks, tighten
    # the plot
    ax1.legend(framealpha=1, loc="upper left")
    ax2.legend(framealpha=1, loc="upper right")
    ax1.set_ylim(0, 1.2)
    ax2.set_ylim(0, 1.2*max(supports))
    ax1.set_title(f"{run_name}\n{datapack_name}\n{config['Script']}")
    ax1.set_xticks(x_ticks_arr)
    ax1.set_xticklabels(x_ticks)
    ax1.set_ylabel("f1-score")
    ax2.set_ylabel("support")
    ax1.set_xlabel("class")
    f.tight_layout()

    return(f)


def plot_metrics(Y_max_prob, pass_args):
    """helper function to plot some general model metrics for a classification
    model.
    Remember to include a "plt.show()" at the end of the script that uses this
    function, to keep the plot open.

    Args:
        Y_max_prob (tuple(np.array)): contains the true- and pred- labels
        pass_args (tuple): contains 3 arguments that describe how the given
                           predictions were derived:
                                run_name (str): the name of the run, in which
                                                the model has it's origin
                                datapack_name (str): the name of the datapack
                                                     that's currently analysed
                                config (dict): config dict from the run
    
    Returns:
        f (matplotlib.figure): the created figure is returned, sothat it can
                               get saved or whatsoever
    """
    # unpack the bundled inputs
    Y_true, Y_pred = Y_max_prob
    run_name, datapack_name, config = pass_args
    
    # calculate all values that are to be displayes. var-names self-explanatory
    tot_acc = accuracy_score(Y_true, Y_pred)
    support = Y_true.shape[0]
    metr_macro = precision_recall_fscore_support(
        Y_true, Y_pred, average="macro", zero_division=0
    )
    metr_weighted = precision_recall_fscore_support(
        Y_true, Y_pred, average="weighted", zero_division=0
    )
    
    # initialise the plot
    f, ax1 = plt.subplots()

    # how many groups of bars will exist
    bar_groups = 4
    # map those to x-values. the "span_till" value, combined with the width will
    # mainly dictate how packed or spacious the bars are
    span_till = 1.8
    # set the position for the accuracy bar custom, to tighten the space there
    # as this bar stands alone. =1 means same space, =0 means on top of the last
    shrink_space_to_acc = .9
    x = np.linspace(0, span_till, bar_groups-1)
    x = np.append(x, x[-1]+(x[1])*shrink_space_to_acc)
    # define values for the bar's layout
    width = 0.33
    gap = 0.025
    alpha_val = 1
    # plot the bars with the macro values
    ax1.bar(
        x[:-1]-width/2-gap/2 , metr_macro[:-1], width, color="lightsteelblue",
        alpha=alpha_val, zorder=10, hatch="/", edgecolor="k", linewidth=.5,
        label="macro"
    )
    # plot the bars with the weighted values
    ax1.bar(
        x[:-1]+width/2+gap/2 , metr_weighted[:-1], width, color="darkgrey",
        alpha=alpha_val, zorder=10, hatch="\\", edgecolor="k", linewidth = .5,
        label="weighted"
    )
    # plot the accuracy bar
    ax1.bar(
        x[-1], tot_acc, 1.1*width, color="grey", alpha=alpha_val, zorder=10,
        edgecolor="k", linewidth=.5
    )
    # add the values of each bar on top of it
    add_values_above_bars([ax1])

    # create the box in the top left that display's the support of the data
    # shown
    props = dict(
        boxstyle="round,pad=.7", alpha=.8, facecolor="w", edgecolor='grey'
    )
    # seperate thousands with ","
    box_text = f"support: {support:,}"
    # plot textbox
    ax1.text(-.38, 1.09, box_text, fontsize=10, bbox=props)

    # initialise the ticks on the x axis, use the positions created above (=x)
    x_ticks_arr = x
    x_ticks = ["precision", "recall", "f-score", "accuracy"]

    # add a neat grid that includes minor grid lines and ranges from 0 to 1
    custom_grid_0_n(ax1)

    # plot what is left: legend, title, the limits, add the x-ticks, tighten
    # the plot
    ax1.legend(framealpha=1, loc="upper right")
    ax1.set_ylim(0, 1.2)
    ax1.set_title(f"{run_name}\n{datapack_name}\n{config['Script']}")
    ax1.set_xticks(x_ticks_arr)
    ax1.set_xticklabels(x_ticks)
    f.tight_layout()

    return(f)


def plot_conf_matr(Y_max_prob, pass_args, normalize=False, count=False):
    """helper function to plot a neat confusion matrix.
    Remember to include a "plt.show()" at the end of the script that uses this
    function, to keep the plot open.

    Args:
        Y_max_prob (tuple(np.array)): contains the true- and pred- labels
        pass_args (tuple): contains 3 arguments that describe how the given
                           predictions were derived:
                                run_name (str): the name of the run, in which
                                                the model has it's origin
                                datapack_name (str): the name of the datapack
                                                     that's currently analysed
                                config (dict): config dict from the run
        normalize (bool): wether the confusion matrix shall be normalized or not
        count (bool): if activated, an extra column with an instance count is
                      added
    
    Returns:
        f (matplotlib.figure): the created figure is returned, sothat it can
                               get saved or whatsoever
    """
    # unpack the bundled inputs
    Y_true, Y_pred = Y_max_prob
    run_name, datapack_name, config = pass_args

    # get the confusion matrix
    conf_mat = confusion_matrix(Y_true, Y_pred)

    # count the instances if desired
    if count:
        instances_n = np.expand_dims(np.sum(conf_mat, axis=1), axis=1)

    # normalize it if desired
    if normalize:
        conf_mat = np.around(
            conf_mat / conf_mat.astype(np.float).sum(axis=1, keepdims=True),
            decimals=2
        )
        cbar_txt = "recall"
    else:
        cbar_txt = "instances_n"

    # get the maximum value in the matrix or set to 1 if the matrix got
    # normalized. used to adjust colors and cbar-range
    if normalize:
        max_val = 1
    else:
        max_val = np.amax(conf_mat)

    # if an instance count column is added, the confusion matrix is adopted here
    if count:
        # add the instance count
        conf_mat = np.concatenate((conf_mat, instances_n), axis=1)
        # mask those instances, to use a special color for this column
        mask = np.zeros(conf_mat.shape)
        mask[:,-1] = 1
        conf_mat_mod = np.ma.masked_array(conf_mat, mask=mask)
    else:
        conf_mat_mod = conf_mat

    # set up the plot
    f, ax1 = plt.subplots()
    cmap = mpl.colormaps["Blues"]
    cmap.set_bad("grey", .3)

    # plot the matrix
    im = ax1.imshow(conf_mat_mod, cmap=cmap, vmin=0, vmax=max_val*1.2)

    # initialise the colorbar
    cbar = ax1.figure.colorbar(im, ax=ax1)
    cbar.ax.set_ylabel(cbar_txt, rotation=-90, va="bottom")
    cbar.ax.set_ylim([0, max_val])

    # get the (unique) sorted labels that appear in the Y-data
    classes = get_classes_from_Y((Y_true, Y_pred))

    # add the right classes on the axes
    if count:
        # create the extra tick for the instance-count column
        x_ticks_arr = np.arange(len(classes)+1)
        x_ticks = list(classes)
        x_ticks.append("inst_n")
    else:
        # use the pure classes for the x-ticks if no instance counts are added
        x_ticks_arr = np.arange(len(classes))
        x_ticks = classes
    ax1.set_xticks(x_ticks_arr)
    ax1.set_yticks(np.arange(len(classes)))
    ax1.set_xticklabels(x_ticks)
    ax1.set_yticklabels(classes)

    # Loop over data dimensions and create text annotations inside the fields
    annotate_im_matrix(conf_mat, ax1)

    # set labels and titel
    ax1.set_title(f"{run_name}\n{datapack_name}\n{config['Script']}")
    ax1.set_ylabel("True-label")
    ax1.set_xlabel("Pred-label")

    # tighten the output
    if count:
        f.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:
        f.tight_layout(rect=[0, 0, .93, 1])

    return(f)


def terminal_prints(Y_max_prob):
    """print metrics, confusion matrix and more to the terminal

    Args:
        Y_max_prob (tuple(np.array)): contains the true- and pred- labels
    """
    # unpack the bundled inputs
    Y_true, Y_pred = Y_max_prob

    # show of the models accuracy
    tot_acc = accuracy_score(Y_true, Y_pred)
    print(f"\nModel accuracy:\n{tot_acc:.3f}")

    # process the confusion matrix
    conf_mat = confusion_matrix(Y_true, Y_pred)
    print(f"\nConfucion Matrix:\n{conf_mat}")

    # get classification report
    target_names = [str(i) for i in list(get_classes_from_Y((Y_true, Y_pred)))]
    class_rep = classification_report(
        Y_true, Y_pred, target_names=target_names, zero_division=0
    )
    print(f"\nClassification report:\n{class_rep}")

    # print per class accuracys. remember, this is the same as the per-class-
    # recall
    accuracy_sto = []
    for i, (cur_class, class_result) in enumerate(zip(target_names, conf_mat)):
        # extract relevant informations from the confusion matrix
        instances_this_class = sum(class_result)
        true_positives = class_result[i]
        per_class_accuracy = true_positives / instances_this_class
        # bundle and save them
        sto = (cur_class, per_class_accuracy)
        accuracy_sto.append(sto)
    print("Per-Class-Accuracys:\nClass   Accuracy")
    [print("{:5}{:11.2f}".format(int(c), a)) for c, a in accuracy_sto]


if __name__ == "__main__":
    pass
